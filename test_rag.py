import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.rag_pipeline import RAGPipeline

print("Initializing RAG pipeline...")
rag = RAGPipeline()

# Hardcoded question for non-interactive test execution
question = "What technologies are used in the RAG chatbot?"
print(f"\nQuestion: {question}")

result = rag.ask(
    question,
    top_k=5
)

print("\n" + "=" * 50)
print("ANSWER")
print("=" * 50)
print(result.get("answer", "No answer generated."))

print("\n" + "=" * 50)
print("SOURCES")
print("=" * 50)

sources = result.get("sources", [])

if not sources:
    print("No sources found.")
else:
    for index, source in enumerate(sources, start=1):
        print("\n" + "-" * 40)
        print(f"Source {index}")
        print(f"Page: {source.get('page', 'Unknown')}")
        print(f"Source: {source.get('source', 'Unknown')}")
        # Retrieve score from source instead of result
        print(f"Similarity Score: {source.get('score', 'Not available')}")
        print("\nText:")
        print(source.get("text", "No text available.")[:500] + "...")