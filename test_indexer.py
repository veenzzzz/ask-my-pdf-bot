from src.indexer import PDFIndexer


pdf_path = "data/sample.pdf"


indexer = PDFIndexer()

result = indexer.index_pdf(pdf_path)

print("\nRESULT")
print("==============================")

print("Pages:", result["pages"])
print("Chunks:", result["chunks"])
print("Embeddings:", result["embeddings"])