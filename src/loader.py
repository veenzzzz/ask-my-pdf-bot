import os
import pdfplumber


class PDFLoader:
    """
    Extract text from a PDF while preserving page information.
    """

    def load(self, pdf_path):
        """
        Load a PDF and return one dictionary per page.
        """

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        documents = []

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                text = page.extract_text()

                if not text:
                    continue

                documents.append({
                    "text": text.strip(),
                    "page": page_number,
                    "source": os.path.basename(pdf_path)
                })

        return documents