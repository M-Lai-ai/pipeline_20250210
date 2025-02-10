from crawler_api import run_crawler

run_crawler(
    start_url="https://www.exemple.com",
    max_depth=3,
    company_number="12345",
    language_pattern="fr",  # ou toute autre regex souhaitée
)


from pdf_api import extract_text_from_pdfs

extract_text_from_pdfs(
    company_number="12345",
    log_level="INFO"
)

from document_api import convert_documents

convert_documents(
    company_number="12345",
    max_workers=4,
    log_file="document_converter.log",
    log_level="INFO"
)


from embedding_api import generate_embeddings

generate_embeddings(
    company_number="12345",
    logging_level="INFO",
    logging_file="embedding_processor.log",
    verbose=True,
    max_workers=20
)
