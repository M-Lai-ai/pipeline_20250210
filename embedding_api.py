import logging
import os
from pathlib import Path

from embedding_processor import EmbeddingProcessor

def generate_embeddings(
    input_dir: str,
    output_dir: str,
    company_number: str,  # Nouveau paramètre
    api_keys: list = None,
    api_keys_file: str = "api_keys.txt",
    chunk_size: int = 900,
    overlap_size: int = 100,
    embedding_model: str = "text-embedding-ada-002",
    system_prompt: str = None,
    llm_max_tokens: int = 500,
    logging_level: str = "INFO",
    logging_file: str = "embedding_processor.log",
    verbose: bool = True,
    max_workers: int = 20
):
    """
    Génération d'embeddings SDK.
    """
    # Vérifier que le numéro de compagnie est fourni
    if not company_number:
        raise ValueError("Le numéro de compagnie doit être fourni.")

    # Configuration du logging
    numeric_level = getattr(logging, logging_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Niveau de logging invalide: {logging_level}")

    # Modifier le nom du fichier de log pour y intégrer le numéro de compagnie
    base, ext = os.path.splitext(logging_file)
    logging_file = f"{base}_{company_number}{ext}"

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Vérification des clés API
    if not api_keys and not os.path.exists(api_keys_file):
        raise ValueError("Aucune clé API fournie et fichier de clés introuvable")
        
    if not api_keys:
        with open(api_keys_file, 'r') as f:
            api_keys = [line.strip() for line in f if line.strip()]

    # Création du processeur d'embedding en passant le numéro de compagnie
    processor = EmbeddingProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        company_number=company_number,
        openai_api_keys=api_keys,
        verbose=verbose,
        logger=logger,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        embedding_model=embedding_model,
        system_prompt=system_prompt,
        llm_max_tokens=llm_max_tokens
    )

    # Traitement des fichiers
    processor.process_all_files(max_workers=max_workers)
