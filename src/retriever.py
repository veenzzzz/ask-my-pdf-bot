from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    """
    Retrieve the most relevant chunks from the FAISS vector store.
    """

    def __init__(
        self,
        index_path="vectorstore/index.faiss",
        chunks_path="vectorstore/chunks.pkl",
        similarity_threshold=0.30,
    ):

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore()

        self.vector_store.load(
            index_path,
            chunks_path
        )

        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query,
        top_k=5
    ):

        # -----------------------------------
        # Validate query
        # -----------------------------------

        if not query or not query.strip():
            return []

        # -----------------------------------
        # Create query embedding
        # -----------------------------------

        query_embedding = (
            self.embedding_model
            .create_embeddings([query])[0]
        )

        # -----------------------------------
        # Search FAISS
        # -----------------------------------

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            score_threshold=self.similarity_threshold
        )

        return results