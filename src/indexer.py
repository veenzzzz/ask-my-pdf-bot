import os

from src.loader import PDFLoader
from src.chunker import split_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class PDFIndexer:
    """
    Complete PDF indexing pipeline.

    PDF
      ↓
    Text extraction
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    FAISS vector store
    """

    def __init__(
        self,
        index_path="vectorstore/index.faiss",
        chunks_path="vectorstore/chunks.pkl"
    ):

        self.index_path = index_path
        self.chunks_path = chunks_path

        self.loader = PDFLoader()

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore()

    def index_pdf(self, pdf_path):

        print("\n==============================")
        print("STARTING PDF INDEXING")
        print("==============================")

        # -----------------------------------
        # 1. Load PDF
        # -----------------------------------

        print("\nLoading PDF...")

        documents = self.loader.load(
            pdf_path
        )

        print(
            "Pages loaded:",
            len(documents)
        )

        if not documents:

            raise ValueError(
                "No text could be extracted "
                "from the PDF."
            )

        # -----------------------------------
        # 2. Create chunks
        # -----------------------------------

        print("\nCreating chunks...")

        chunks = split_documents(
            documents
        )

        print(
            "Chunks created:",
            len(chunks)
        )

        if not chunks:

            raise ValueError(
                "No chunks were created "
                "from the PDF."
            )

        # -----------------------------------
        # 3. Create embeddings
        # -----------------------------------

        print("\nCreating embeddings...")

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_model
            .create_embeddings(texts)
        )

        print(
            "Embeddings created:",
            len(embeddings)
        )

        # -----------------------------------
        # 4. Create FAISS index
        # -----------------------------------

        print("\nCreating FAISS index...")

        self.vector_store.create_index(
            embeddings,
            chunks
        )

        # -----------------------------------
        # 5. Create directories
        # -----------------------------------

        os.makedirs(
            os.path.dirname(
                self.index_path
            ),
            exist_ok=True
        )

        # -----------------------------------
        # 6. Save vector store
        # -----------------------------------

        print("\nSaving vector store...")

        self.vector_store.save(
            self.index_path,
            self.chunks_path
        )

        print("\n==============================")
        print("PDF INDEXING COMPLETED")
        print("==============================")

        return {
            "pages": len(documents),
            "chunks": len(chunks),
            "embeddings": len(embeddings)
        }