import re
from web_crawler import WebCrawler

def run_crawler(
    start_url: str,
    max_depth: int = 3,
    use_playwright: bool = False,
    excluded_paths: list = None,
    download_extensions: dict = None,
    language_pattern: str = None,
    base_dir: str = "crawler_output",
    company_number: str = None  # Nouveau paramètre obligatoire
):
    """
    Run the WebCrawler with direct function arguments (SDK style).

    :param start_url: URL of the website to start crawling.
    :param max_depth: Maximum depth to explore links.
    :param use_playwright: Whether to use Playwright for JavaScript rendering.
    :param excluded_paths: List of URL path segments to exclude.
    :param download_extensions: Dict of extension arrays by file type.
    :param language_pattern: Regex pattern (string) or None to filter language.
    :param base_dir: Base output directory for the crawler.
    :param company_number: Le numéro de compagnie à intégrer dans les noms de dossiers.
    
    :return: None (crawler results are saved to disk).
    """
    if excluded_paths is None:
        excluded_paths = ['selecteur-de-produits']

    if download_extensions is None:
        download_extensions = {
            'PDF': ['.pdf'],
            'Image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'Doc': ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        }
    
    if company_number is None:
        raise ValueError("Le numéro de compagnie doit être fourni.")

    # Compiler l'expression régulière si donnée
    pattern_compiled = re.compile(language_pattern) if language_pattern else None

    crawler = WebCrawler(
        start_url=start_url,
        max_depth=max_depth,
        use_playwright=use_playwright,
        excluded_paths=excluded_paths,
        download_extensions=download_extensions,
        language_pattern=pattern_compiled,
        base_dir=base_dir,
        company_number=company_number  # Passage du numéro de compagnie
    )

    crawler.crawl()
