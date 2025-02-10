import re
import json
import logging
import threading
import requests
import numpy as np
from pathlib import Path
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed


class EmbeddingProcessor:
    """
    Classe pour traiter les embeddings de texte en utilisant l'API OpenAI.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        company_number,  # Nouveau paramètre
        openai_api_keys,
        verbose=False,
        logger=None,
        chunk_size=1200,
        overlap_size=100,
        embedding_model="text-embedding-ada-002",
        system_prompt=None,
        llm_max_tokens=200
    ):
        self.input_dir = Path(input_dir)
        if not company_number:
            raise ValueError("Le numéro de compagnie doit être fourni.")
        self.company_number = company_number
        # Le dossier de sortie intègre le numéro de compagnie
        self.output_dir = Path(f"{output_dir}_{company_number}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_embeddings = []
        self.all_metadata = []

        self.openai_api_keys = openai_api_keys
        self.headers_cycle = cycle([
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            for key in self.openai_api_keys
        ])
        self.lock = threading.Lock()

        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.embedding_model = embedding_model

        self.system_prompt = system_prompt or (
            "You are an expert analyst. The following text is an excerpt from a larger document. "
            "Your task is to provide context for the next section by referencing the overall document content. "
            "Ensure the context helps in better understanding the excerpt."
        )
        self.llm_max_tokens = llm_max_tokens

    def chunk_text(self, text):
        try:
            tokens = text.split()
            if len(tokens) <= self.chunk_size:
                return [text]

            chunks = []
            step = self.chunk_size - self.overlap_size
            for i in range(0, len(tokens), step):
                chunk = ' '.join(tokens[i:i+self.chunk_size])
                chunks.append(chunk)

            # Fusionner le dernier chunk si celui-ci est trop petit
            if len(chunks) > 1 and len(tokens[-step:]) < (self.chunk_size // 2):
                last_chunk = ' '.join(tokens[-self.chunk_size:])
                chunks[-1] = last_chunk

            return chunks
        except Exception as e:
            self.logger.error(f"Erreur lors du découpage: {str(e)}")
            return [text]

    def get_contextualized_chunk(self, chunk, full_text, headers, document_name, page_num, chunk_id):
        try:
            system_prompt = {
                "role": "system",
                "content": self.system_prompt
            }
            user_prompt = {
                "role": "user",
                "content": (
                    f"Document: {full_text}\n\n"
                    f"Chunk: {chunk}\n\n"
                    f"Please provide context for this excerpt in French."
                )
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [system_prompt, user_prompt],
                "temperature": 0,
                "max_tokens": self.llm_max_tokens,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }

            if self.verbose:
                self.logger.info(f"Appel LLM pour {document_name} page {page_num}, chunk {chunk_id}")

            self.logger.info(f"🔗 Appel GPT-4 pour {document_name} page {page_num} chunk {chunk_id}")
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"Erreur contextualisation: {str(e)}")
            return None

    def get_embedding(self, text, headers, document_name, page_num, chunk_id):
        try:
            payload = {
                "input": text,
                "model": self.embedding_model,
                "encoding_format": "float"
            }
            if self.verbose:
                self.logger.info(f"Appel Embedding pour {document_name}, page {page_num}, chunk {chunk_id}")

            self.logger.info(f"🔗 Appel Embedding API pour {document_name} page {page_num} chunk {chunk_id}")
            response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            self.logger.error(f"Erreur embedding: {str(e)}")
            return None
        finally:
            if headers.get("Authorization"):
                # Remettre la clé dans le cycle
                self.headers_cycle = cycle([
                    {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    }
                    for key in self.openai_api_keys
                ])

    def process_chunk(self, chunk_info):
        try:
            txt_file_path, chunk_id, chunk, full_text, doc_name, page_num = chunk_info

            with self.lock:
                headers = next(self.headers_cycle)

            context = self.get_contextualized_chunk(chunk, full_text, headers, doc_name, page_num, chunk_id)
            if not context:
                return None, None

            combined_text = f"{context}\n\nContext:\n{chunk}"
            embedding = self.get_embedding(combined_text, headers, doc_name, page_num, chunk_id)
            if embedding:
                metadata = {
                    "filename": txt_file_path.name,
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "text_raw": chunk,
                    "context": context,
                    "text": combined_text
                }
                return embedding, metadata
            return None, None
        except Exception as e:
            self.logger.error(f"Erreur traitement chunk: {str(e)}")
            return None, None

    def process_file(self, txt_file_path):
        try:
            self.logger.info(f"📂 Traitement fichier: {txt_file_path}")
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

            chunks = self.chunk_text(full_text)
            match = re.search(r'_page_(\d+)', txt_file_path.stem)
            if match:
                page_num = int(match.group(1))
            else:
                page_num = 1

            chunk_infos = [
                (txt_file_path, i, chunk, full_text, txt_file_path.stem, page_num)
                for i, chunk in enumerate(chunks, 1)
            ]
            return chunk_infos
        except Exception as e:
            self.logger.error(f"Erreur traitement fichier {txt_file_path}: {str(e)}")
            return []

    def process_all_files(self, max_workers=10):
        try:
            txt_files = list(self.input_dir.glob('*.txt'))
            total_files = len(txt_files)
            self.logger.info(f"📢 Début traitement de {total_files} fichiers dans '{self.input_dir}'")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for txt_file_path in txt_files:
                    chunk_infos = self.process_file(txt_file_path)
                    for chunk_info in chunk_infos:
                        futures.append(executor.submit(self.process_chunk, chunk_info))

                for future in as_completed(futures):
                    embedding, metadata = future.result()
                    if embedding and metadata:
                        self.all_embeddings.append(embedding)
                        self.all_metadata.append(metadata)

            if self.all_embeddings:
                chunks_json_path = self.output_dir / "chunks.json"
                with open(chunks_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump({"metadata": self.all_metadata}, json_file, ensure_ascii=False, indent=4)

                embeddings_npy_path = self.output_dir / "embeddings.npy"
                np.save(embeddings_npy_path, np.array(self.all_embeddings))

                self.logger.info(f"✅ Fichiers créés: {chunks_json_path} et {embeddings_npy_path}")
        except Exception as e:
            self.logger.error(f"Erreur globale: {str(e)}")
            raise
