import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around the sentence-transformers embedding model.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts):
        if not texts:
            embedding_size = self.model.get_sentence_embedding_dimension()
            return np.empty((0, embedding_size), dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype("float32")
