# web_crawler.py

import os
import re
import sys
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from colorama import init, Fore, Style
from tqdm import tqdm
import html2text

from rich.traceback import install

# Initialize Rich modules for improved tracebacks
install()

# Initialize Colorama for colored terminal text
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    SYMBOLS = {
        'INFO': '✔',
        'WARNING': '⚠',
        'ERROR': '✘',
        'CRITICAL': '✘'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        symbol = self.SYMBOLS.get(record.levelname, '')
        header = f"{color}{symbol} {self.formatTime(record, self.datefmt)}#"
        level = f"- {record.levelname}#"
        message = f"- {record.getMessage()}#"
        return f"{header}\n{level}\n{message}"


class MovingIndicator(threading.Thread):
    def __init__(self, delay=0.1, length=20):
        super().__init__()
        self.delay = delay
        self.length = length
        self.running = False
        self.position = self.length - 1
        self.direction = -1

    def run(self):
        self.running = True
        while self.running:
            line = [' '] * self.length
            if 0 <= self.position < self.length:
                line[self.position] = '#'
            indicator = ''.join(line) + '##'
            sys.stdout.write(f"\r{indicator}")
            sys.stdout.flush()
            self.position += self.direction
            if self.position == 0 or self.position == self.length - 1:
                self.direction *= -1
            time.sleep(self.delay)

    def stop(self):
        self.running = False
        self.join()
        sys.stdout.write('\r' + ' ' * (self.length + 2) + '\r')
        sys.stdout.flush()


class WebCrawler:
    def __init__(
        self,
        start_url,
        max_depth=3,
        use_playwright=False,
        excluded_paths=None,
        download_extensions=None,
        language_pattern=None,
        company_number=None  # Nouveau paramètre
    ):
        if company_number is None:
            raise ValueError("Le numéro de compagnie doit être fourni.")
        self.company_number = company_number

        self.start_url = start_url
        self.max_depth = max_depth
        self.use_playwright = use_playwright
        self.visited_pages = set()
        self.downloaded_files = set()
        self.domain = urlparse(start_url).netloc

        self.excluded_paths = excluded_paths or ['selecteur-de-produits']
        self.download_extensions = download_extensions or {
            'PDF': ['.pdf'],
            'Image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'Doc': ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        }

        self.language_pattern = language_pattern

        # Le dossier de base est généré automatiquement avec le numéro de compagnie et un timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"crawler_output_{company_number}_{timestamp}"

        self.create_directories()
        self.setup_logging()
        self.stats = defaultdict(int)

        self.all_downloadable_exts = set(
            ext for exts in self.download_extensions.values() for ext in exts
        )

        self.content_type_mapping = {
            'PDF': {'application/pdf': '.pdf'},
            'Image': {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/svg+xml': '.svg',
            },
            'Doc': {
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'application/vnd.ms-excel': '.xls',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
                'application/vnd.ms-powerpoint': '.ppt',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            }
        }

        self.session = self.setup_session()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.body_width = 0
        self.html_converter.ignore_images = True
        self.html_converter.single_line_break = False

        if self.use_playwright:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()

        self.indicator = MovingIndicator(length=20)

    def get_folder_path(self, folder_key):
        return os.path.join(self.base_dir, f"{folder_key}")

    def setup_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        })
        return session

    def create_directories(self):
        for folder in ['content', 'PDF', 'Image', 'Doc', 'logs']:
            os.makedirs(self.get_folder_path(folder), exist_ok=True)

    def setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = ColoredFormatter(log_format)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.get_folder_path('logs'), 'crawler.log'), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logging.info(f"Crawler started with language pattern: {self.language_pattern}")

    # ... (Le reste du code de crawling reste inchangé)

    def crawl(self):
        start_time = time.time()
        logging.info(f"🚀 Starting crawl of {self.start_url}")
        self.indicator.start()
        try:
            self.extract_urls(self.start_url)
            # Extraction de contenu, génération de rapport, etc.
        except Exception as e:
            logging.error(f"⚠ Critical error during crawling: {str(e)}")
        finally:
            self.indicator.stop()
            # Nettoyage (playwright, fichiers téléchargés, etc.)
