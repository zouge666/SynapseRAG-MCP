from __future__ import annotations

from observability.dashboard.services.data_service import DataService, image_exists


def render() -> None:
    import streamlit as st

    st.title("Data Browser")
    service = DataService()
    collections = service.list_collections()
    collection = st.sidebar.selectbox("Collection", collections)
    documents = service.list_documents(collection)
    if not documents:
        st.info("No ingested documents found.")
        return

    st.dataframe(_document_rows(documents), hide_index=True, use_container_width=True)
    labels = [_document_label(document) for document in documents]
    selected_label = st.sidebar.selectbox("Document", labels)
    selected_document = documents[labels.index(selected_label)]
    detail = service.get_document_detail(selected_document["doc_id"])
    document = detail["document"]
    chunks = detail["chunks"]
    images = detail["images"]

    left, middle, right = st.columns(3)
    left.metric("Chunks", document["chunk_count"])
    middle.metric("Images", document["image_count"])
    right.metric("Collection", document["collection"])
    st.subheader(document["source_path"])
    st.json(document["metadata"], expanded=False)

    st.subheader("Chunks")
    for chunk in chunks:
        with st.expander(_chunk_label(chunk)):
            st.write(chunk["text"])
            st.json(chunk["metadata"], expanded=False)
            chunk_images = _chunk_images(chunk, images)
            if chunk_images:
                _render_images(st, chunk_images)

    if images:
        st.subheader("Images")
        _render_images(st, images)


def _document_rows(documents: list[dict]) -> list[dict]:
    return [
        {
            "source_path": document["source_path"],
            "collection": document["collection"],
            "chunks": document["chunk_count"],
            "images": document["image_count"],
            "processed_at": document["processed_at"],
        }
        for document in documents
    ]


def _document_label(document: dict) -> str:
    return f"{document['source_path']} ({document['chunk_count']} chunks)"


def _chunk_label(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    index = metadata.get("chunk_index", "")
    return f"{chunk.get('id', '')} {index}".strip()


def _chunk_images(chunk: dict, images: list[dict]) -> list[dict]:
    metadata = chunk.get("metadata", {})
    refs = metadata.get("image_refs", [])
    if not isinstance(refs, list) or not refs:
        return []
    return [image for image in images if image.get("image_id") in refs]


def _render_images(st: object, images: list[dict]) -> None:
    for image in images:
        path = image.get("file_path")
        if image_exists(image):
            st.image(path, caption=image.get("image_id"))
        else:
            st.write({"image_id": image.get("image_id"), "file_path": path})
