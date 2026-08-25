from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split PDF pages into smaller text chunks
    while preserving page and source metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []

    for document in documents:

        text = document.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        page = document.get(
            "page",
            "Unknown"
        )

        source = document.get(
            "source",
            "Unknown document"
        )

        chunk_texts = splitter.split_text(
            text
        )

        for chunk_text in chunk_texts:

            cleaned_text = chunk_text.strip()

            if not cleaned_text:
                continue

            chunks.append({
                "text": cleaned_text,
                "page": page,
                "source": source
            })

    return chunks