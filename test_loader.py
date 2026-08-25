import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


# -----------------------------
# 1. Load PDF
# -----------------------------

pdf_path = "data/sample.pdf"

documents = load_pdf(pdf_path)

print("Pages extracted:", len(documents))


# -----------------------------
# 2. Split PDF into chunks
# -----------------------------

chunks = split_documents(documents)

print("Total chunks:", len(chunks))


# -----------------------------
# 3. Create embeddings
# -----------------------------

texts = [chunk["text"] for chunk in chunks]

embedding_model = EmbeddingModel()

embeddings = embedding_model.create_embeddings(texts)

print("Embeddings created:", len(embeddings))


# -----------------------------
# 4. Create FAISS index
# -----------------------------

vector_store = VectorStore()

vector_store.create_index(
    embeddings,
    chunks
)


# -----------------------------
# 5. Save vector store
# -----------------------------

vector_store.save(
    "vectorstore/index.faiss",
    "vectorstore/chunks.pkl"
)

print("Vector store saved successfully.")


# -----------------------------
# 6. Test search
# -----------------------------

query = "What is artificial intelligence?"

query_embedding = embedding_model.create_embeddings(
    [query]
)[0]

results = vector_store.search(
    query_embedding,
    top_k=3
)


# -----------------------------
# 7. Display search results
# -----------------------------

print("\n===== SEARCH RESULTS =====")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Page:", result["page"])
    print("Similarity Score:", result["score"])
    print("Text:")
    print(result["text"][:500])