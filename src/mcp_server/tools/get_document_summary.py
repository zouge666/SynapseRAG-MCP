from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.settings import load_settings
from mcp_server.protocol_handler import ToolDefinition


class GetDocumentSummaryError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    title: str
    summary: str
    tags: list[str] = field(default_factory=list)
    created_at: str | None = None
    source_path: str | None = None
    chunk_count: int = 0
    collection: str | None = None
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "source_path": self.source_path,
            "chunk_count": self.chunk_count,
            "collection": self.collection,
            "chunk_ids": list(self.chunk_ids),
        }


def get_document_summary(
    doc_id: str,
    collection: str | None = None,
    settings: object | None = None,
    settings_path: str = "config/settings.yaml",
    persist_path: str | Path | None = None,
) -> dict[str, Any]:
    active_doc_id = _require_text(doc_id, "doc_id")
    active_settings = settings if settings is not None else None
    if persist_path is None and active_settings is None:
        active_settings = load_settings(settings_path)
    root = Path(persist_path) if persist_path is not None else Path(str(_setting(_setting(active_settings, "vector_store", {}), "persist_path", "data/db/chroma")))
    records = [record for record in _load_records(root, collection) if _matches_doc_id(record, active_doc_id)]
    if not records:
        return _not_found(active_doc_id)
    summary = _summarize(active_doc_id, records)
    return _build_response(summary)


def get_document_summary_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise GetDocumentSummaryError("arguments must be a mapping")
    return get_document_summary(doc_id=arguments.get("doc_id"), collection=arguments.get("collection"))


def get_document_summary_tool_definition(
    settings: object | None = None,
    persist_path: str | Path | None = None,
) -> ToolDefinition:
    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise GetDocumentSummaryError("arguments must be a mapping")
        return get_document_summary(
            doc_id=arguments.get("doc_id"),
            collection=arguments.get("collection"),
            settings=settings,
            persist_path=persist_path,
        )

    return ToolDefinition(
        name="get_document_summary",
        description="Get a local document title, summary, tags, and metadata by doc_id.",
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "collection": {"type": "string"},
            },
            "required": ["doc_id"],
        },
        handler=handle,
    )


def _load_records(root: Path, collection: str | None) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    paths = [_collection_path(root, collection)] if collection else _record_paths(root)
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        data = _load_json(path)
        collection_name = _clean_text(data.get("collection")) or path.stem
        raw_records = data.get("records", [])
        if not isinstance(raw_records, list):
            raise GetDocumentSummaryError("vector metadata records must be a list")
        for item in raw_records:
            if isinstance(item, dict):
                record = dict(item)
                record["collection"] = collection_name
                records.append(record)
    return records


def _record_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise GetDocumentSummaryError("vector metadata path must be a file or directory")
    return sorted(root.glob("*.json"), key=lambda item: item.name)


def _collection_path(root: Path, collection: str) -> Path:
    if root.is_file():
        return root
    return root / f"{_safe_collection_name(collection)}.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise GetDocumentSummaryError("vector metadata cache must be valid JSON") from error
    if not isinstance(data, dict):
        raise GetDocumentSummaryError("vector metadata cache must be an object")
    return data


def _matches_doc_id(record: dict[str, Any], doc_id: str) -> bool:
    target = _normalize_id(doc_id)
    metadata = _metadata(record)
    candidates = [record.get("id"), metadata.get("chunk_id")]
    for key in ("doc_id", "document_id", "source_ref", "source_path", "source", "file_path"):
        value = metadata.get(key)
        candidates.append(value)
        if isinstance(value, str):
            path = Path(value)
            candidates.extend([path.name, path.stem])
    return any(_normalize_id(candidate) == target for candidate in candidates if isinstance(candidate, str))


def _summarize(doc_id: str, records: list[dict[str, Any]]) -> DocumentSummary:
    ordered = sorted(records, key=_record_sort_key)
    metadata_items = [_metadata(record) for record in ordered]
    source_path = _first_text(metadata_items, "source_path", "source", "file_path")
    title = _first_text(metadata_items, "document_title", "title", "heading", "section_title") or _fallback_title(doc_id, source_path)
    summary = _document_summary(metadata_items) or _chunk_summary(metadata_items) or _text_summary(ordered)
    tags = _unique_tags(metadata_items)
    created_at = _first_text(metadata_items, "created_at", "created_time", "mtime", "modified_at", "date")
    chunk_ids = [_clean_text(_metadata(record).get("chunk_id")) or _clean_text(record.get("id")) for record in ordered]
    chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
    return DocumentSummary(
        doc_id=doc_id,
        title=title,
        summary=summary or "No summary available.",
        tags=tags,
        created_at=created_at or None,
        source_path=source_path or None,
        chunk_count=len(ordered),
        collection=_clean_text(ordered[0].get("collection")) or None,
        chunk_ids=chunk_ids,
    )


def _record_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    metadata = _metadata(record)
    chunk_index = metadata.get("chunk_index")
    index = chunk_index if isinstance(chunk_index, int) else 0
    return (index, _clean_text(record.get("id")))


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _first_text(items: list[dict[str, Any]], *keys: str) -> str:
    for item in items:
        for key in keys:
            value = _clean_text(item.get(key))
            if value:
                return value
    return ""


def _document_summary(items: list[dict[str, Any]]) -> str:
    return _first_text(items, "document_summary", "doc_summary")


def _chunk_summary(items: list[dict[str, Any]]) -> str:
    summaries = _unique_texts(item.get("summary") for item in items)
    return _truncate(" ".join(summaries[:3]), 800)


def _text_summary(records: list[dict[str, Any]]) -> str:
    texts = _unique_texts(record.get("text") for record in records)
    return _truncate(" ".join(texts[:2]), 800)


def _unique_tags(items: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for item in items:
        for tag in _tag_values(item.get("tags")):
            if tag not in tags:
                tags.append(tag)
    return tags


def _tag_values(value: Any) -> list[str]:
    raw = re.split(r"[,;|]", value) if isinstance(value, str) else value
    if not isinstance(raw, list):
        return []
    tags = []
    for item in raw:
        tag = _clean_text(item).lower()
        if tag:
            tags.append(tag)
    return tags


def _unique_texts(values: object) -> list[str]:
    texts: list[str] = []
    if not isinstance(values, list | tuple | set):
        values = list(values) if hasattr(values, "__iter__") and not isinstance(values, str | bytes | dict) else [values]
    for value in values:
        text = _clean_text(value)
        if text and text not in texts:
            texts.append(text)
    return texts


def _fallback_title(doc_id: str, source_path: str) -> str:
    if source_path:
        return Path(source_path).stem or doc_id
    return doc_id


def _not_found(doc_id: str) -> dict[str, Any]:
    return {
        "isError": True,
        "content": [{"type": "text", "text": f"Document not found: {doc_id}"}],
        "structuredContent": {"error": {"code": "document_not_found", "doc_id": doc_id}},
    }


def _build_response(summary: DocumentSummary) -> dict[str, Any]:
    lines = [summary.title, "", summary.summary]
    if summary.tags:
        lines.extend(["", f"Tags: {', '.join(summary.tags)}"])
    if summary.source_path:
        lines.append(f"Source: {summary.source_path}")
    return {"content": [{"type": "text", "text": "\n".join(lines)}], "structuredContent": {"document": summary.to_dict()}}


def _require_text(value: Any, name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise GetDocumentSummaryError(f"{name} must be a non-empty string")
    return text


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _truncate(text: str, limit: int) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" .,:;") or text[:limit].rstrip(" .,:;")


def _normalize_id(value: str) -> str:
    return _clean_text(value).replace("\\", "/").strip("./").lower()


def _setting(source: object, name: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _safe_collection_name(collection: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection).strip("._")
    return value or "default"
