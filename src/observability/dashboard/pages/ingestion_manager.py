from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from ingestion import IngestionPipeline
from observability.dashboard import runtime
from observability.dashboard.services.data_service import DataService
from observability.dashboard.services.session_context import SESSION_COLLECTION
from observability.logger import write_trace


NEW_COLLECTION = "__create_new__"
COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def render() -> None:
    import streamlit as st

    from observability.dashboard import runtime

    if runtime.is_public():
        _render_public(st)
        return

    st.title("Ingestion Manager")
    service = DataService()
    collections = service.list_collections()

    with st.container(border=True):
        st.markdown("**Upload & Ingest**")
        notice = st.session_state.pop("ingestion_notice", None)
        if notice:
            level, message = notice
            getattr(st, level, st.info)(message)
        selection = st.selectbox(
            "Collection",
            [*collections, NEW_COLLECTION],
            format_func=lambda value: "＋ Create a new collection…" if value == NEW_COLLECTION else value,
        )
        new_name = ""
        if selection == NEW_COLLECTION:
            new_name = st.text_input("New collection name", placeholder="e.g. product_manuals")
            collection = new_name.strip()
            if collection and not _valid_collection_name(collection):
                st.error("Collection names must start with a letter or digit and may only contain letters, digits, dots, dashes and underscores (max 64 chars).")
                collection = ""
            elif collection:
                st.caption(f"The collection `{collection}` is created automatically once the first file is ingested.")
        else:
            collection = selection
        uploaded_file = st.file_uploader("File", type=["pdf"])
        force = st.checkbox("Force reprocess", value=False)
        ready = uploaded_file is not None and bool(collection)
        if st.button("Ingest", type="primary", disabled=not ready):
            progress = st.progress(0.0)
            status = st.empty()
            try:
                result = _run_ingestion(uploaded_file, collection, force, service.settings, progress, status)
            except Exception as error:
                st.error(str(error))
            else:
                if result.get("skipped") or result.get("status") == "skipped":
                    st.session_state["ingestion_notice"] = (
                        "info",
                        "The file content matches the version already in the index, so it was not reprocessed. "
                        "The existing index can be queried directly; tick Force reprocess to parse it again.",
                    )
                else:
                    st.session_state["ingestion_notice"] = (
                        "success",
                        f"Ingestion complete: the file was parsed, chunked and written to collection `{collection}`.",
                    )
                st.session_state["ingestion_last_result"] = result
                st.rerun()

    last_result = st.session_state.pop("ingestion_last_result", None)
    if last_result:
        with st.expander("Processing details"):
            st.json(last_result, expanded=False)

    st.subheader("Documents")
    documents = service.list_documents(collection)
    if not documents:
        st.info("No ingested documents found.")
        return
    st.dataframe(_document_rows(documents), hide_index=True, use_container_width=True)
    for document in documents:
        left, right = st.columns([5, 1])
        left.write(document["source_path"])
        if right.button("Delete", key=f"delete-{document['source_path']}"):
            result = _delete_document(service, document["source_path"], collection)
            st.toast(f"Deleted {result['source_path']}")
            st.rerun()


def _render_public(st: Any) -> None:
    from observability.dashboard.public_pages.upload_page import process_upload

    st.title("Ingestion Manager")
    settings = runtime.require_settings(st)
    if settings is None:
        return
    session = runtime.current_session(st)
    st.caption(f"Session expires in {session.remaining_seconds() // 60} min. One PDF per session: a new upload replaces the previous document.")

    uploaded = st.file_uploader("PDF file (max 10 MB, 100 pages)", type=["pdf"])
    if uploaded is not None and st.button("Ingest", type="primary"):
        process_upload(st, session, uploaded)

    service = DataService(settings)
    documents = service.list_documents(SESSION_COLLECTION)
    st.subheader("Documents")
    if not documents:
        st.info("No ingested documents found.")
        return
    st.dataframe(_document_rows(documents), hide_index=True, use_container_width=True)
    for document in documents:
        left, right = st.columns([5, 1])
        left.write(runtime.mask(document["source_path"]))
        if right.button("Delete", key=f"delete-{document['source_path']}"):
            result = _delete_document(service, document["source_path"], SESSION_COLLECTION)
            st.toast(f"Deleted {runtime.mask(result['source_path'])}")
            st.rerun()


def _valid_collection_name(name: str) -> bool:
    return bool(COLLECTION_NAME.match(name))


def _run_ingestion(
    uploaded_file: Any,
    collection: str,
    force: bool,
    settings: Any,
    progress_widget: Any,
    status_widget: Any,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    source_path = _save_uploaded_file(uploaded_file)
    active_pipeline = pipeline or IngestionPipeline(settings)
    result = active_pipeline.run(
        source_path,
        collection=collection,
        force=force,
        on_progress=_progress_callback(progress_widget, status_widget),
    )
    trace = getattr(result, "trace", None)
    if isinstance(trace, dict):
        trace_path = getattr(getattr(settings, "observability", None), "trace_path", "logs/traces.jsonl")
        write_trace(trace, path=trace_path)
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def _save_uploaded_file(uploaded_file: Any, upload_dir: Path | None = None) -> str:
    target_dir = upload_dir or Path(tempfile.mkdtemp(prefix="synapserag_upload_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    name = Path(getattr(uploaded_file, "name", "upload.pdf")).name or "upload.pdf"
    target_path = target_dir / name
    data = uploaded_file.getbuffer() if hasattr(uploaded_file, "getbuffer") else uploaded_file.read()
    target_path.write_bytes(bytes(data))
    return str(target_path)


def _progress_callback(progress_widget: Any, status_widget: Any) -> Callable[[str, int, int], None]:
    def update(stage: str, current: int, total: int) -> None:
        value = current / total if total else 0.0
        progress_widget.progress(value)
        status_widget.write(_progress_message(stage, current, total))

    return update


def _progress_message(stage: str, current: int, total: int) -> str:
    if stage == "skipped":
        return f"Detected an identical file, skipped duplicate ingestion ({current}/{total})"
    labels = {
        "integrity": "Checking whether the file was processed before",
        "load": "Reading PDF content",
        "image_store": "Saving document images",
        "split": "Splitting text",
        "transform": "Refining chunks",
        "encode": "Generating embeddings",
        "store": "Writing to the index",
    }
    return f"{labels.get(stage, stage)} ({current}/{total})"


def _delete_document(service: DataService, source_path: str, collection: str) -> dict[str, Any]:
    result = service.document_manager.delete_document(source_path, collection)
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def _document_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_path": runtime.mask(document["source_path"]),
            "collection": document["collection"],
            "chunks": document["chunk_count"],
            "images": document["image_count"],
            "processed_at": document["processed_at"],
        }
        for document in documents
    ]
