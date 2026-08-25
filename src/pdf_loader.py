import os

import fitz


def load_pdf(pdf_path):
    """
    Load a PDF and extract text page by page.
    """
    documents = []
    source_name = os.path.basename(pdf_path)

    with fitz.open(pdf_path) as pdf_document:
        for page_number, page in enumerate(pdf_document, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    {
                        "text": text,
                        "page": page_number,
                        "source": source_name,
                    }
                )

    return documents
