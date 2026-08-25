import os
from dotenv import load_dotenv

load_dotenv()

from google import genai


class LLM:
    """
    Gemini LLM wrapper for the RAG pipeline.
    """

    def __init__(
        self,
        model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    ):

        self.model_name = model_name

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Please add your Gemini API key "
                "to the environment variables."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate_answer(
        self,
        question,
        context,
        chat_history=None
    ):
        """
        Generate a grounded answer using
        only the retrieved PDF context.
        """

        if not context or not context.strip():

            return (
                "I could not find relevant information "
                "in the provided document."
            )

        # -----------------------------------------
        # Conversation history
        # -----------------------------------------

        history_text = ""

        if chat_history:

            history_parts = []

            for message in chat_history:

                role = message.get(
                    "role",
                    "user"
                )

                content = message.get(
                    "content",
                    ""
                )

                if content:

                    history_parts.append(
                        f"{role.upper()}: {content}"
                    )

            history_text = "\n".join(
                history_parts
            )

        # -----------------------------------------
        # Grounded RAG prompt
        # -----------------------------------------

        prompt = f"""
You are an AI assistant that answers questions
about a provided PDF document.

IMPORTANT RULES:

1. Answer ONLY using information contained
   in the provided context.

2. Do NOT use outside knowledge.

3. Do NOT invent, guess, or assume information.

4. If the answer cannot be found in the context,
   say exactly:

"I could not find this information in the
provided document."

5. Keep the answer clear and concise.

6. When useful, mention the relevant page number
   from the context.

7. Conversation history may be used to understand
   what the user means by words such as "it",
   "they", or "that project", but the actual answer
   must still come from the provided PDF context.

-------------------------
CONVERSATION HISTORY
-------------------------

{history_text}

-------------------------
PDF CONTEXT
-------------------------

{context}

-------------------------
USER QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""

        # -----------------------------------------
        # Generate response
        # -----------------------------------------

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        answer = getattr(
            response,
            "text",
            None
        )

        if not answer:

            return (
                "I could not generate an answer "
                "from the provided document."
            )

        return answer.strip()