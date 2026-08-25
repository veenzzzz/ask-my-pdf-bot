import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


# -----------------------------
# 1. Load existing vector store
# -----------------------------

vector_store = VectorStore()

vector_store.load(
    "vectorstore/index.faiss",
    "vectorstore/chunks.pkl"
)


# -----------------------------
# 2. Load embedding model
# -----------------------------

embedding_model = EmbeddingModel()


# -----------------------------
# 3. Ask a question
# -----------------------------

query = "What is artificial intelligence?"


# -----------------------------
# 4. Convert question to vector
# -----------------------------

query_embedding = embedding_model.create_embeddings(
    [query]
)[0]


# -----------------------------
# 5. Search FAISS
# -----------------------------

results = vector_store.search(
    query_embedding,
    top_k=3
)


# -----------------------------
# 6. Display results
# -----------------------------

print("\n===== RETRIEVED DOCUMENTS =====")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("-----------------------------")
    print("Page:", result["page"])
    print("Similarity Score:", result["score"])
    print("Text:")
    print(result["text"][:500])