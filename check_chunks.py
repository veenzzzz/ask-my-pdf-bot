import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents


pdf_path = "data/sample.pdf"

documents = load_pdf(pdf_path)
chunks = split_documents(documents)

print("Total chunks:", len(chunks))

print("\n==============================")
print("CHUNK DISTRIBUTION")
print("==============================")

for i, chunk in enumerate(chunks):
    print(
        f"Chunk {i + 1}: "
        f"Page {chunk['page']} | "
        f"{len(chunk['text'])} characters"
    )


print("\n==============================")
print("LAST 5 CHUNKS")
print("==============================")

for chunk in chunks[-5:]:
    print("\n------------------------------")
    print("Page:", chunk["page"])
    print(chunk["text"][:1000])