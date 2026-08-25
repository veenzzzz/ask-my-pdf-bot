from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split extracted PDF pages into smaller chunks
    while preserving page and source metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []

    chunk_id = 1

    for document in documents:

        text = document.get("text", "").strip()

        if not text:
            continue

        page = document.get("page", "Unknown")
        source = document.get("source", "Unknown")

        split_chunks = splitter.split_text(text)

        for chunk_text in split_chunks:

            cleaned_text = chunk_text.strip()

            if not cleaned_text:
                continue

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": cleaned_text,
                    "page": page,
                    "source": source,
                }
            )

            chunk_id += 1

    return chunks