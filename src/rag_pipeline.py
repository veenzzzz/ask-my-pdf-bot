from src.retriever import Retriever
from src.llm import LLM


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Flow:
        Question
            ↓
        Retriever
            ↓
        FAISS vector search
            ↓
        Relevant PDF chunks
            ↓
        Context
            ↓
        LLM
            ↓
        Answer + Sources
    """

    def __init__(self):
        print("Initializing RAG pipeline...")

        self.retriever = Retriever()
        self.llm = LLM()

        # Stores conversation history
        self.chat_history = []

    def ask(
        self,
        question,
        top_k=5,
        chat_history=None
    ):
        """
        Ask a question about the indexed PDF.

        Parameters
        ----------
        question : str
            User's question.

        top_k : int
            Number of relevant chunks to retrieve.

        chat_history : list, optional
            Previous conversation messages.

        Returns
        -------
        dict
            answer
            sources
            context
        """

        # -----------------------------------------
        # 1. Validate question
        # -----------------------------------------

        if not question or not question.strip():

            return {
                "answer": "Please enter a question.",
                "sources": [],
                "context": ""
            }

        question = question.strip()

        # -----------------------------------------
        # 2. Retrieve relevant chunks
        # -----------------------------------------

        try:

            results = self.retriever.retrieve(
                question,
                top_k=top_k
            )

        except Exception as e:

            return {
                "answer": f"Error during document retrieval: {str(e)}",
                "sources": [],
                "context": ""
            }

        # -----------------------------------------
        # 3. Check whether anything was retrieved
        # -----------------------------------------

        if not results:

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the provided document."
                ),
                "sources": [],
                "context": ""
            }

        # -----------------------------------------
        # 4. Build context
        # -----------------------------------------

        context_parts = []

        for result in results:

            page = result.get(
                "page",
                "Unknown"
            )

            source = result.get(
                "source",
                "Unknown document"
            )

            text = result.get(
                "text",
                ""
            )

            if not text:
                continue

            context_parts.append(
                f"[Source: {source} | Page: {page}]\n{text}"
            )

        context = "\n\n".join(context_parts)

        # -----------------------------------------
        # 5. Use supplied chat history if available
        # -----------------------------------------

        if chat_history is None:
            chat_history = self.chat_history

        # -----------------------------------------
        # 6. Generate answer using LLM
        # -----------------------------------------

        try:

            answer = self.llm.generate_answer(
                question=question,
                context=context,
                chat_history=chat_history
            )

        except TypeError:

            # Compatibility fallback for an older LLM class
            answer = self.llm.generate_answer(
                question=question,
                context=context
            )

        except Exception as e:

            return {
                "answer": (
                    f"An error occurred while generating "
                    f"the answer: {str(e)}"
                ),
                "sources": results,
                "context": context
            }

        # -----------------------------------------
        # 7. Save conversation
        # -----------------------------------------

        self.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        self.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # -----------------------------------------
        # 8. Return complete RAG result
        # -----------------------------------------

        return {
            "answer": answer,
            "sources": results,
            "context": context
        }

    def clear_history(self):
        """
        Clear conversation history.
        """

        self.chat_history = []

    def get_history(self):
        """
        Return current conversation history.
        """

        return self.chat_history