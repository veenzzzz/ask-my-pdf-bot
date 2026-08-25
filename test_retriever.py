import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.retriever import Retriever


# Create retriever
retriever = Retriever()


# Ask a question
query = "What is artificial intelligence?"


# Retrieve relevant chunks
results = retriever.retrieve(
    query,
    top_k=3
)


# Display results
print("\n===== RETRIEVED CHUNKS =====")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("==========================")

    print("Page:", result["page"])
    print("Similarity Score:", result["score"])

    print("\nContent:")
    print(result["text"][:700])