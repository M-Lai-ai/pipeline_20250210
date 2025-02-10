mport logging
import os
from pathlib import Path
from document_converter import DocumentConverter

def convert_documents(
    input_dir: str,
    output_dir: str,
    company_number: str,
    max_workers: int = 4,
    log_file: str = "document_converter.log",
    log_level: str = "INFO"
):
    """
    Convertit des .doc / .docx en PDF (SDK style).

    :param input_dir: Dossier contenant les .doc/.docx
    :param output_dir: Dossier où sauvegarder les PDF (le numéro de compagnie sera ajouté au nom)
    :param company_number: Numéro de compagnie à intégrer dans les noms de dossiers et fichiers.
    :param max_workers: Nombre de workers (threads)
    :param log_file: Nom du fichier de log
    :param log_level: Niveau de log (DEBUG, INFO, WARNING, ...)
    """
    if not company_number:
        raise ValueError("Le numéro de compagnie doit être fourni.")

    # Configurer le logger
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
    logger = logging.getLogger('DocumentConverter')

    # Vérifier l'existence du input_dir
    if not Path(input_dir).exists():
        logger.error(f"Le répertoire {input_dir} n'existe pas.")
        return

    # Créer le dictionnaire de configuration en y ajoutant le numéro de compagnie
    config = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'company_number': company_number,
        'max_workers': max_workers
    }

    converter = DocumentConverter(config=config, logger=logger)
    converter.convert_documents()
