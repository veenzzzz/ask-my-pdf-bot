import os
import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.indexer import PDFIndexer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ask My PDF Bot",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "rag" not in st.session_state:

    st.session_state.rag = RAGPipeline()


if "indexer" not in st.session_state:

    st.session_state.indexer = PDFIndexer()


if "messages" not in st.session_state:

    st.session_state.messages = []


if "uploaded_pdf" not in st.session_state:

    st.session_state.uploaded_pdf = None


if "document_info" not in st.session_state:

    st.session_state.document_info = {}


# ============================================================
# OBJECT REFERENCES
# ============================================================

rag = st.session_state.rag
indexer = st.session_state.indexer


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("PDF Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    st.divider()

    # --------------------------------------------------------
    # DOCUMENT STATUS
    # --------------------------------------------------------

    st.subheader("Document Status")

    if st.session_state.uploaded_pdf:

        st.success("Document Ready")

        st.write(
            f"**File:** "
            f"{st.session_state.uploaded_pdf}"
        )

        document_info = (
            st.session_state.document_info
        )

        if document_info:

            if "pages" in document_info:

                st.write(
                    f"**Pages:** "
                    f"{document_info['pages']}"
                )

            if "chunks" in document_info:

                st.write(
                    f"**Chunks:** "
                    f"{document_info['chunks']}"
                )

            if "embeddings" in document_info:

                st.write(
                    f"**Embeddings:** "
                    f"{document_info['embeddings']}"
                )

    else:

        st.warning(
            "No document loaded"
        )

    st.divider()

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        if hasattr(
            st.session_state.rag,
            "clear_history"
        ):

            st.session_state.rag.clear_history()

        st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Ask My PDF Bot'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF and ask questions about its contents.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PDF UPLOAD AND INDEXING
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Only process a new PDF
    # --------------------------------------------------------

    if (
        st.session_state.uploaded_pdf
        != uploaded_file.name
    ):

        os.makedirs(
            "data",
            exist_ok=True
        )

        pdf_path = os.path.join(
            "data",
            uploaded_file.name
        )

        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        try:

            with open(
                pdf_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

        except Exception as e:

            st.error(
                f"Could not save PDF: {e}"
            )

            st.stop()


        # ----------------------------------------------------
        # Index PDF
        # ----------------------------------------------------

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                result = indexer.index_pdf(
                    pdf_path
                )

            except Exception as e:

                st.error(
                    f"PDF indexing failed: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # Save document information
        # ----------------------------------------------------

        if isinstance(
            result,
            dict
        ):

            st.session_state.document_info = (
                result
            )

        else:

            st.session_state.document_info = {}


        # ----------------------------------------------------
        # Reload RAG
        #
        # This makes sure the Retriever loads the
        # newly-created FAISS index.
        # ----------------------------------------------------

        try:

            st.session_state.rag = (
                RAGPipeline()
            )

            rag = st.session_state.rag

        except Exception as e:

            st.error(
                f"Could not reload RAG pipeline: {e}"
            )

            st.stop()


        # ----------------------------------------------------
        # Save uploaded filename
        # ----------------------------------------------------

        st.session_state.uploaded_pdf = (
            uploaded_file.name
        )


        # ----------------------------------------------------
        # Clear previous conversation
        # ----------------------------------------------------

        st.session_state.messages = []

        if hasattr(
            st.session_state.rag,
            "clear_history"
        ):

            st.session_state.rag.clear_history()


        # ----------------------------------------------------
        # Success message
        # ----------------------------------------------------

        st.success(
            "PDF processed successfully!"
        )


        # ----------------------------------------------------
        # Display indexing information
        # ----------------------------------------------------

        if isinstance(
            result,
            dict
        ):

            pages = result.get(
                "pages"
            )

            chunks = result.get(
                "chunks"
            )

            embeddings = result.get(
                "embeddings"
            )

            information = []

            if pages is not None:

                information.append(
                    f"Pages: {pages}"
                )

            if chunks is not None:

                information.append(
                    f"Chunks: {chunks}"
                )

            if embeddings is not None:

                information.append(
                    f"Embeddings: {embeddings}"
                )

            if information:

                st.info(
                    " | ".join(information)
                )


    else:

        st.success(
            f"Loaded PDF: "
            f"{uploaded_file.name}"
        )


else:

    # --------------------------------------------------------
    # No PDF uploaded
    # --------------------------------------------------------

    if not st.session_state.uploaded_pdf:

        st.info(
            "Upload a PDF from the sidebar to begin."
        )


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "user"
    )

    content = message.get(
        "content",
        ""
    )

    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )

        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        sources = message.get(
            "sources",
            []
        )

        if (
            role == "assistant"
            and sources
        ):

            st.markdown(
                "### Sources"
            )

            for i, source in enumerate(
                sources,
                start=1
            ):

                page = source.get(
                    "page",
                    "Unknown"
                )

                source_name = source.get(
                    "source",
                    "Unknown document"
                )

                score = source.get(
                    "score"
                )

                text = source.get(
                    "text",
                    ""
                )

                with st.expander(
                    f"Source {i} — Page {page}"
                ):

                    st.write(
                        f"**Document:** "
                        f"{source_name}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{page}"
                    )

                    if score is not None:

                        st.write(
                            f"**Similarity Score:** "
                            f"{float(score):.4f}"
                        )

                    st.markdown(
                        "**Retrieved content:**"
                    )

                    st.write(
                        text
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your PDF..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if question:

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not st.session_state.uploaded_pdf:

        st.warning(
            "Please upload a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the PDF..."
        ):

            try:

                result = rag.ask(
                    question=question,
                    top_k=5
                )

            except Exception as e:

                st.error(
                    f"Error generating answer: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # Get answer
        # ----------------------------------------------------

        answer = result.get(
            "answer",
            "I could not generate an answer."
        )


        # ----------------------------------------------------
        # Get sources
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            []
        )


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            st.markdown(
                "### Sources"
            )

            for i, source in enumerate(
                sources,
                start=1
            ):

                page = source.get(
                    "page",
                    "Unknown"
                )

                source_name = source.get(
                    "source",
                    "Unknown document"
                )

                score = source.get(
                    "score"
                )

                text = source.get(
                    "text",
                    ""
                )

                with st.expander(
                    f"Source {i} — Page {page}"
                ):

                    st.write(
                        f"**Document:** "
                        f"{source_name}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{page}"
                    )

                    if score is not None:

                        st.write(
                            f"**Similarity Score:** "
                            f"{float(score):.4f}"
                        )

                    st.markdown(
                        "**Retrieved content:**"
                    )

                    st.write(
                        text
                    )


        # ----------------------------------------------------
        # Save assistant response
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )