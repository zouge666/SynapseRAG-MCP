from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from ingestion import IngestionPipeline
from observability.dashboard.services.data_service import DataService
from observability.logger import write_trace


def render() -> None:
    import streamlit as st

    st.title("Ingestion Manager")
    service = DataService()
    collections = service.list_collections()
    collection = st.selectbox("Collection", collections)
    uploaded_file = st.file_uploader("File", type=["pdf"])
    force = st.checkbox("Force reprocess", value=False)

    if uploaded_file is not None and st.button("Ingest", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()
        try:
            result = _run_ingestion(uploaded_file, collection, force, service.settings, progress, status)
        except Exception as error:
            st.error(str(error))
        else:
            if result.get("skipped") or result.get("status") == "skipped":
                st.info(
                    "文件内容与已入库版本一致，因此没有重复处理。"
                    "现有索引可直接用于查询；如需重新解析，请勾选 Force reprocess 后再次摄取。"
                )
            else:
                st.success("摄取完成：文件已解析、切分并写入当前 Collection。")
            with st.expander("查看处理详情"):
                st.json(result, expanded=False)

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
        return f"检测到相同文件，已跳过重复摄取（{current}/{total}）"
    labels = {
        "integrity": "正在检查文件是否已处理",
        "load": "正在读取 PDF 内容",
        "image_store": "正在保存文档图片",
        "split": "正在切分文本",
        "transform": "正在整理文本块",
        "encode": "正在生成向量",
        "store": "正在写入索引",
    }
    return f"{labels.get(stage, stage)}（{current}/{total}）"


def _delete_document(service: DataService, source_path: str, collection: str) -> dict[str, Any]:
    result = service.document_manager.delete_document(source_path, collection)
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def _document_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
