import logging
import os
from pathlib import Path
from pdf import PDFExtractor  # On importe la classe PDFExtractor depuis pdf.py

def extract_text_from_pdfs(
    input_dir: str,
    output_dir: str,
    company_number: str,
    api_keys: list = None,
    api_keys_file: str = "api_keys.txt",
    max_workers: int = 5,
    initial_dpi: int = 300,
    retry_dpi: int = 200,
    llm_model: str = "gpt-4o-mini",
    llm_system_prompt: str = "",
    llm_temperature: float = 0,
    llm_max_tokens: int = 16000,
    llm_top_p: float = 1,
    llm_frequency_penalty: float = 0,
    llm_presence_penalty: float = 0,
    ocr_languages: str = "fra+eng",
    blur_kernel_size: int = 5,
    log_file: str = "pdf_crawler.log",
    log_level: str = "INFO",
    chunk_split: int = 4000
):
    """
    Extrait le texte depuis les PDFs (input_dir) grâce à l'OCR et GPT, puis
    enregistre le résultat dans output_dir.

    :param input_dir: Dossier contenant les PDF
    :param output_dir: Dossier où stocker les fichiers .txt (le numéro de compagnie sera ajouté au nom)
    :param company_number: Numéro de compagnie à intégrer dans les noms de dossiers et fichiers.
    :param api_keys: Liste de clés OpenAI (ex: ["sk-AAA", "sk-BBB"])
    :param api_keys_file: Fichier contenant les clés si api_keys == None
    :param max_workers: Nombre de threads
    :param initial_dpi: DPI initial PDF->image
    :param retry_dpi: DPI de secours
    :param llm_model: Modèle LLM (ex: "gpt-4o-mini")
    :param llm_system_prompt: Prompt Système pour GPT
    :param llm_temperature: Float, ex: 0
    :param llm_max_tokens: Nombre max de tokens
    :param llm_top_p: top-p
    :param llm_frequency_penalty: pénalité de fréquence
    :param llm_presence_penalty: pénalité de présence
    :param ocr_languages: Langues Tesseract ("fra+eng", etc.)
    :param blur_kernel_size: Param pour le traitement d'images
    :param log_file: Fichier logs
    :param log_level: Niveau log (DEBUG, INFO...)
    :param chunk_split: Taille max des morceaux de texte envoyés au LLM

    :return: None
    """
    # Vérifier que le numéro de compagnie est fourni
    if not company_number:
        raise ValueError("Le numéro de compagnie doit être fourni.")

    # 1) Configurer le logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Niveau de log invalide: {log_level}")

    # Modifier le nom du fichier de log pour inclure le numéro de compagnie
    base, ext = os.path.splitext(log_file)
    log_file = f"{base}_{company_number}{ext}"

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 2) Vérifier existence du input_dir
    if not Path(input_dir).exists():
        logger.error(f"Le dossier PDF spécifié n'existe pas : {input_dir}")
        return

    # 3) Gérer les clés API
    #    Si api_keys est None, on lit api_keys_file
    if not api_keys:
        if os.path.isfile(api_keys_file):
            with open(api_keys_file, 'r', encoding='utf-8') as f:
                api_keys = [line.strip() for line in f if line.strip()]
        if not api_keys:
            logger.error("Aucune clé API OpenAI disponible.")
            return

    # 4) Construire un config dict pour PDFExtractor, en y ajoutant company_number
    config = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "company_number": company_number,
        "api_keys_file": api_keys_file,
        "max_workers": max_workers,
        "initial_dpi": initial_dpi,
        "retry_dpi": retry_dpi,
        "llm": {
            "model": llm_model,
            "system_prompt": llm_system_prompt,
            "temperature": llm_temperature,
            "max_tokens": llm_max_tokens,
            "top_p": llm_top_p,
            "frequency_penalty": llm_frequency_penalty,
            "presence_penalty": llm_presence_penalty
        },
        "ocr": {
            "languages": ocr_languages
        },
        "image_processing": {
            "blur_kernel_size": blur_kernel_size
        },
        "logging": {
            "log_file": log_file,
            "log_level": log_level
        },
        "chunk_split": chunk_split
    }

    # 5) Instancier la classe PDFExtractor et lancer le traitement
    extractor = PDFExtractor(config=config, logger=logger)
    extractor.process_all_pdfs()
