import os
import re
import logging
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from docx2pdf import convert
import subprocess
import sys

# Détection du système d'exploitation
IS_WINDOWS = sys.platform.startswith('win')
IS_MAC = sys.platform.startswith('darwin')

if IS_WINDOWS:
    try:
        import win32com.client
    except ImportError:
        print("⚠️ La bibliothèque pywin32 n'est pas installée. Installez-la avec 'pip install pywin32'.")
        sys.exit(1)


class DocumentConverter:
    """
    Classe pour convertir des fichiers .doc et .docx en PDF.
    """

    def __init__(self, config, logger=None):
        """
        Initialise le DocumentConverter avec les répertoires et paramètres.
        Le paramètre "company_number" doit être présent dans la configuration.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Vérifier que le numéro de compagnie est fourni
        self.company_number = config.get("company_number")
        if not self.company_number:
            raise ValueError("Le numéro de compagnie doit être fourni dans la configuration.")
        
        self.input_dir = Path(config['input_dir'])
        original_output_dir = config['output_dir']
        # Intégrer le numéro de compagnie dans le nom du dossier de sortie
        self.output_dir = Path(f"{original_output_dir}_{self.company_number}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = config.get('max_workers', 4)

        # Lance LibreOffice en mode headless si on n'est pas sur Windows
        if not IS_WINDOWS:
            self.start_libreoffice()

    def start_libreoffice(self):
        """
        Lance LibreOffice en mode headless si on n'est pas sur Windows.
        """
        try:
            self.libreoffice_process = subprocess.Popen([
                'soffice', '--headless',
                '--accept=socket,host=localhost,port=2002;urp;',
                '--norestore', '--nolockcheck'
            ])
            self.logger.info("📢 LibreOffice démarré en mode headless.")
            time.sleep(5)
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du démarrage de LibreOffice : {str(e)}")
            sys.exit(1)

    def stop_libreoffice(self):
        """
        Ferme le processus LibreOffice.
        """
        try:
            self.libreoffice_process.terminate()
            self.libreoffice_process.wait(timeout=10)
            self.logger.info("📢 LibreOffice arrêté.")
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'arrêt de LibreOffice : {str(e)}")

    def convert_documents(self):
        """
        Convertit tous les .doc/.docx du input_dir en .pdf dans output_dir.
        """
        try:
            doc_files = [f for ext in ['*.doc', '*.DOC', '*.docx', '*.DOCX']
                         for f in self.input_dir.glob(ext)]
            total_files = len(doc_files)
            self.logger.info(f"📢 Début de la conversion de {total_files} fichiers.")

            if total_files == 0:
                self.logger.warning(f"Aucun fichier DOC ou DOCX dans : {self.input_dir}")
                return

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_doc = {
                    executor.submit(self.convert_single_document, doc_file): doc_file
                    for doc_file in doc_files
                }

                for future in as_completed(future_to_doc):
                    doc_file = future_to_doc[future]
                    try:
                        success = future.result()
                        if success:
                            self.logger.info(f"✅ Conversion réussie: {doc_file.name}")
                        else:
                            self.logger.warning(f"⚠️ Conversion échouée: {doc_file.name}")
                    except Exception as e:
                        self.logger.error(f"❌ Exception sur {doc_file.name}: {str(e)}")

            self.logger.info("🎉 Toutes les conversions sont terminées.")
        finally:
            if not IS_WINDOWS:
                self.stop_libreoffice()

    def convert_single_document(self, doc_path):
        """
        Convertit un fichier unique en PDF.
        """
        try:
            output_pdf = self.output_dir / f"{doc_path.stem}.pdf"

            if doc_path.suffix.lower() == '.docx':
                convert(str(doc_path), str(output_pdf))
            elif doc_path.suffix.lower() == '.doc':
                if IS_WINDOWS:
                    word = win32com.client.Dispatch('Word.Application')
                    word.Visible = False
                    doc = word.Documents.Open(str(doc_path))
                    doc.SaveAs(str(output_pdf), FileFormat=17)
                    doc.Close()
                    word.Quit()
                else:
                    subprocess.run(['unoconv', '-f', 'pdf', '-o', str(output_pdf), str(doc_path)], check=True)
            else:
                self.logger.warning(f"🔍 Format non supporté: {doc_path.name}")
                return False

            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Erreur conversion {doc_path.name} avec unoconv: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Erreur conversion {doc_path.name}: {str(e)}")
            return False
