import faiss
import numpy as np
import pickle


class VectorStore:
    """
    FAISS vector store using cosine similarity.
    """

    def __init__(self):
        self.index = None
        self.chunks = []

    # --------------------------------------------------
    # CREATE INDEX
    # --------------------------------------------------

    def create_index(self, embeddings, chunks):

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        if embeddings.size == 0:
            raise ValueError(
                "Cannot create FAISS index: no embeddings provided."
            )

        # Normalize vectors
        # After normalization, Inner Product = cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.chunks = chunks

        print(
            "FAISS cosine-similarity index created."
        )

        print(
            "Number of vectors:",
            self.index.ntotal
        )

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(
        self,
        query_embedding,
        top_k=5,
        score_threshold=0.25
    ):

        if self.index is None:
            raise ValueError(
                "FAISS index has not been created or loaded."
            )

        query_embedding = np.array(
            [query_embedding],
            dtype="float32"
        )

        # Normalize query vector
        faiss.normalize_L2(query_embedding)

        # Do not request more results than available
        k = min(
            top_k,
            self.index.ntotal
        )

        if k == 0:
            return []

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            # FAISS can return -1 when no result exists
            if index == -1:
                continue

            score = float(score)

            # Ignore weak similarity matches
            if score < score_threshold:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "text": chunk.get(
                        "text",
                        ""
                    ),
                    "page": chunk.get(
                        "page",
                        "Unknown"
                    ),
                    "source": chunk.get(
                        "source",
                        "Unknown document"
                    ),
                    "score": score
                }
            )

        return results

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    def save(
        self,
        index_path,
        chunks_path
    ):

        if self.index is None:
            raise ValueError(
                "Cannot save an empty FAISS index."
            )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            chunks_path,
            "wb"
        ) as file:

            pickle.dump(
                self.chunks,
                file
            )

        print(
            "Vector store saved."
        )

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    def load(
        self,
        index_path,
        chunks_path
    ):

        self.index = faiss.read_index(
            index_path
        )

        with open(
            chunks_path,
            "rb"
        ) as file:

            self.chunks = pickle.load(
                file
            )

        print(
            "Vector store loaded."
        )

        print(
            "Number of vectors:",
            self.index.ntotal
        )

        print(
            "Number of chunks:",
            len(self.chunks)
        )