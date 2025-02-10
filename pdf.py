import os
import re
import logging
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

import requests
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import pypdf
import numpy as np
import cv2
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from a .env file if present
load_dotenv()

class PDFExtractor:
    """
    A utility class for extracting text from PDF files.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the PDFExtractor with the given configuration.
        Le paramètre 'company_number' doit être présent dans config.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Vérifier que le numéro de compagnie est fourni
        self.company_number = config.get("company_number")
        if not self.company_number:
            raise ValueError("Le numéro de compagnie doit être fourni dans la configuration.")
        
        # Les dossiers d'entrée restent tels quels, mais le dossier de sortie est renommé pour inclure le numéro de compagnie.
        self.input_dir = Path(config['input_dir'])
        original_output_dir = config['output_dir']
        self.output_dir = Path(f"{original_output_dir}_{self.company_number}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load API keys from the config or from a file
        self.api_keys = Queue()
        api_keys_file = config.get('api_keys_file', 'api_keys.txt')

        try:
            with open(api_keys_file, 'r') as f:
                for line in f:
                    key = line.strip()
                    if key:
                        self.api_keys.put(key)
            if self.api_keys.empty():
                raise ValueError("No API keys loaded from file.")
        except Exception as e:
            if logger:
                logger.error(f"Error loading API keys: {str(e)}")
            else:
                print(f"Error loading API keys: {str(e)}")
            raise

        self.max_workers = config.get('max_workers', 5)
        self.initial_dpi = config.get('initial_dpi', 300)
        self.retry_dpi = config.get('retry_dpi', 200)

        # LLM config
        self.llm_config = config.get('llm', {})
        self.chunk_split = config.get('chunk_split', 4000)

        # OCR config
        ocr_cfg = config.get('ocr', {})
        self.ocr_languages = ocr_cfg.get('languages', 'eng')

        # Image processing config
        img_cfg = config.get('image_processing', {})
        self.blur_kernel_size = img_cfg.get('blur_kernel_size', 5)

    def preprocess_image(self, image):
        """
        Preprocess an image to enhance OCR accuracy.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if image is None:
            raise ValueError("Empty or corrupted image.")

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, h=30)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Adaptive Threshold
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return binary
        except Exception as e:
            raise ValueError(f"Error during image preprocessing: {str(e)}")

    def convert_pdf_to_images(self, pdf_path, dpi):
        """
        Convert a PDF to images at the given DPI.
        """
        try:
            self.logger.info(f"Converting {pdf_path.name} to images with DPI={dpi}")
            images = convert_from_path(pdf_path, dpi=dpi, fmt='jpeg', thread_count=1)
            self.logger.info(f"Success converting {pdf_path.name} with DPI={dpi}")
            return images
        except Exception as e:
            self.logger.error(f"Error converting {pdf_path.name} at DPI={dpi}: {str(e)}")
            return None

    def extract_text_with_ocr(self, pdf_path):
        """
        Extract text from PDF via OCR.
        """
        # First attempt
        images = self.convert_pdf_to_images(pdf_path, self.initial_dpi)
        if images is None:
            # Second attempt
            images = self.convert_pdf_to_images(pdf_path, self.retry_dpi)
            if images is None:
                self.logger.error(f"Failed to convert {pdf_path.name} to images.")
                return None

        ocr_texts = []
        for i, image in enumerate(images, 1):
            self.logger.info(f"Performing OCR on page {i}/{len(images)} for {pdf_path.name}")
            try:
                processed_img = self.preprocess_image(image)
            except Exception as e:
                self.logger.error(f"Preprocessing failed on page {i} of {pdf_path.name}: {str(e)}")
                ocr_texts.append("")
                continue

            try:
                text = pytesseract.image_to_string(
                    processed_img,
                    lang=self.ocr_languages,
                    config='--psm 1'
                )
                # If result is too short, try another config
                if len(text.strip()) < 100:
                    self.logger.info(f"Insufficient OCR output on page {i} => retry with different config")
                    text = pytesseract.image_to_string(
                        processed_img,
                        lang=self.ocr_languages,
                        config='--psm 3 --oem 1'
                    )
                ocr_texts.append(text)
            except Exception as e:
                self.logger.error(f"OCR failed on page {i}: {str(e)}")
                ocr_texts.append("")
        return ocr_texts

    def extract_text_with_pypdf_per_page(self, pdf_path, page_num):
        """
        Extract text from a single PDF page using PyPDF.
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                if page_num < 1 or page_num > len(reader.pages):
                    self.logger.error(f"Invalid page number {page_num} in {pdf_path.name}")
                    return ''
                page = reader.pages[page_num - 1]
                text = page.extract_text() or ''
                self.logger.info(f"PyPDF extracted {len(text)} chars from page {page_num}.")
                return text
        except Exception as e:
            self.logger.error(f"PyPDF error on page {page_num} of {pdf_path.name}: {str(e)}")
            return ''

    def get_api_key(self):
        """
        Retrieve an API key from the queue.
        """
        try:
            return self.api_keys.get(timeout=10)
        except Empty:
            self.logger.error("No API keys available.")
            return None

    def process_with_gpt(self, content):
        """
        Process the content with GPT.
        """
        system_prompt = {
            "role": "system",
            "content": self.llm_config.get('system_prompt', '')
        }

        api_key = self.get_api_key()
        if not api_key:
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.llm_config.get('model', 'gpt-4'),
            "messages": [
                system_prompt,
                {"role": "user", "content": content}
            ],
            "temperature": self.llm_config.get('temperature', 0),
            "max_tokens": self.llm_config.get('max_tokens', 16000),
            "top_p": self.llm_config.get('top_p', 1),
            "frequency_penalty": self.llm_config.get('frequency_penalty', 0),
            "presence_penalty": self.llm_config.get('presence_penalty', 0)
        }

        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            processed_content = response.json()['choices'][0]['message']['content']
            time.sleep(1)
            return processed_content
        except Exception as e:
            self.logger.error(f"GPT API error: {str(e)}")
            return None
        finally:
            if api_key:
                self.api_keys.put(api_key)

    def split_content(self, content, max_length=4000):
        """
        Split content into smaller chunks.
        """
        try:
            paragraphs = content.split('\n\n')
            parts = []
            current_part = ""

            for para in paragraphs:
                if len(current_part) + len(para) + 2 > max_length:
                    if current_part:
                        parts.append(current_part.strip())
                    current_part = para + '\n\n'
                else:
                    current_part += para + '\n\n'

            if current_part.strip():
                parts.append(current_part.strip())

            return parts
        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            return [content]

    def process_single_page(self, document_name, page_num, page_text):
        """
        Process a single page of text, handling its parts with GPT.
        Returns the processed content for the entire page.
        """
        self.logger.info(f"Processing doc={document_name}, page={page_num}")
        
        parts = self.split_content(page_text, max_length=self.chunk_split)
        page_content = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx, part in enumerate(parts, 1):
                futures.append((
                    executor.submit(self.process_with_gpt, part),
                    idx
                ))
            
            # Collect all processed parts for this page
            for future, part_num in futures:
                try:
                    processed_part = future.result()
                    if processed_part:
                        page_content.append(processed_part)
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num} part {part_num}: {str(e)}")
        
        return '\n'.join(page_content)

    def process_pdf(self, pdf_path):
        """
        Process a single PDF page by page and create one consolidated output file.
        """
        document_name = pdf_path.stem
        self.logger.info(f"Starting processing of {pdf_path.name}")

        # OCR
        ocr_texts = self.extract_text_with_ocr(pdf_path)
        if ocr_texts is None:
            self.logger.error(f"OCR extraction failed for {pdf_path.name}")
            return False

        output_file_name = self.output_dir / f"{document_name}.txt"
        
        # Create/open the output file
        with open(output_file_name, 'w', encoding='utf-8') as f:
            f.write(f"📄 Document ID: {document_name}\n\n")
        
        # Process each page sequentially
        for page_num, ocr_text in enumerate(ocr_texts, 1):
            self.logger.info(f"Processing page {page_num}/{len(ocr_texts)} of {pdf_path.name}")
            
            # Get page text (OCR or PyPDF)
            if ocr_text and len(ocr_text.strip()) >= 100:
                page_text = ocr_text
                self.logger.info(f"OCR succeeded for page {page_num}")
            else:
                self.logger.info(f"Insufficient OCR => fallback to PyPDF for page {page_num}")
                page_text = self.extract_text_with_pypdf_per_page(pdf_path, page_num)

            if not page_text.strip():
                self.logger.warning(f"No text extracted for page {page_num}")
                continue

            # Process the page
            try:
                processed_page = self.process_single_page(document_name, page_num, page_text)
                
                # Append the processed page to the output file
                with open(output_file_name, 'a', encoding='utf-8') as f:
                    f.write(f"--- Page {page_num} ---\n\n")
                    f.write(f"{processed_page}\n\n")
                
                self.logger.info(f"Successfully processed and saved page {page_num}")
                
            except Exception as e:
                self.logger.error(f"Error processing page {page_num}: {str(e)}")
                continue

        self.logger.info(f"Completed processing of {pdf_path.name}")
        return True

    def process_all_pdfs(self):
        """
        Process all PDFs in self.input_dir.
        """
        pdf_files = list(self.input_dir.glob('*.pdf'))
        total_files = len(pdf_files)
        if total_files == 0:
            self.logger.warning(f"No PDF found in: {self.input_dir}")
            return

        self.logger.info(f"Starting processing of {total_files} PDF(s) in '{self.input_dir}'")

        with tqdm(total=total_files, desc="Traitement des PDFs", unit="pdf") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pdf = {executor.submit(self.process_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    try:
                        success = future.result()
                        if success:
                            self.logger.info(f"Successfully processed {pdf_path.name}")
                        else:
                            self.logger.warning(f"Failed to process {pdf_path.name}")
                    except Exception as e:
                        self.logger.error(f"Exception while processing {pdf_path.name}: {str(e)}")
                    finally:
                        pbar.update(1)

        self.logger.info(f"Completed. Processed {total_files} PDF file(s).")
