import json
from pathlib import Path

import pytest

from mcp_server.tools import default_tools
from mcp_server.tools.get_document_summary import (
    GetDocumentSummaryError,
    get_document_summary,
    get_document_summary_tool_definition,
)


def write_store(root: Path, collection: str, records: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{collection}.json"
    path.write_text(json.dumps({"collection": collection, "records": records}), encoding="utf-8")
    return path


def record(
    record_id: str,
    text: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {"id": record_id, "vector": [1.0, 0.0], "text": text, "metadata": metadata}


def test_get_document_summary_returns_metadata_from_vector_records(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    write_store(
        root,
        "docs",
        [
            record(
                "vec-1",
                "Alpha chunk text",
                {
                    "chunk_id": "chunk-1",
                    "source_path": "docs/guide.pdf",
                    "title": "Guide",
                    "summary": "Explains the first part.",
                    "tags": ["RAG", "MCP"],
                    "chunk_index": 1,
                    "created_at": "2026-05-22",
                },
            ),
            record(
                "vec-0",
                "Intro chunk text",
                {
                    "chunk_id": "chunk-0",
                    "source_path": "docs/guide.pdf",
                    "summary": "Introduces the guide.",
                    "tags": "rag;summary",
                    "chunk_index": 0,
                },
            ),
        ],
    )

    response = get_document_summary("guide", persist_path=root)

    document = response["structuredContent"]["document"]
    assert response["content"][0]["type"] == "text"
    assert "Guide" in response["content"][0]["text"]
    assert document["doc_id"] == "guide"
    assert document["title"] == "Guide"
    assert document["summary"] == "Introduces the guide. Explains the first part."
    assert document["tags"] == ["rag", "summary", "mcp"]
    assert document["created_at"] == "2026-05-22"
    assert document["source_path"] == "docs/guide.pdf"
    assert document["chunk_count"] == 2
    assert document["collection"] == "docs"
    assert document["chunk_ids"] == ["chunk-0", "chunk-1"]


def test_get_document_summary_matches_document_id_metadata(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    write_store(
        root,
        "default",
        [
            record(
                "vec-1",
                "Stored document text",
                {
                    "document_id": "doc-123",
                    "source_path": "docs/source.md",
                    "document_summary": "Whole document summary.",
                    "title": "Source Notes",
                    "tags": ["notes"],
                },
            )
        ],
    )

    response = get_document_summary("doc-123", persist_path=root)

    document = response["structuredContent"]["document"]
    assert document["title"] == "Source Notes"
    assert document["summary"] == "Whole document summary."
    assert document["tags"] == ["notes"]


def test_get_document_summary_filters_by_collection(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    write_store(root, "alpha", [record("vec-a", "Alpha text", {"source_path": "docs/shared.pdf", "title": "Alpha"})])
    write_store(root, "beta", [record("vec-b", "Beta text", {"source_path": "docs/shared.pdf", "title": "Beta"})])

    response = get_document_summary("shared.pdf", collection="beta", persist_path=root)

    document = response["structuredContent"]["document"]
    assert document["title"] == "Beta"
    assert document["collection"] == "beta"


def test_get_document_summary_returns_tool_error_for_missing_doc_id(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    write_store(root, "docs", [record("vec-1", "Alpha", {"source_path": "docs/a.pdf"})])

    response = get_document_summary("missing", persist_path=root)

    assert response == {
        "isError": True,
        "content": [{"type": "text", "text": "Document not found: missing"}],
        "structuredContent": {"error": {"code": "document_not_found", "doc_id": "missing"}},
    }


def test_get_document_summary_rejects_empty_doc_id(tmp_path: Path) -> None:
    with pytest.raises(GetDocumentSummaryError, match="doc_id must be a non-empty string"):
        get_document_summary("", persist_path=tmp_path)


def test_get_document_summary_tool_definition_uses_bound_persist_path(tmp_path: Path) -> None:
    root = tmp_path / "chroma"
    write_store(root, "docs", [record("vec-1", "Alpha", {"source_path": "docs/a.pdf", "title": "Alpha"})])
    tool = get_document_summary_tool_definition(persist_path=root)

    response = tool.handler({"doc_id": "a"})

    assert tool.name == "get_document_summary"
    assert tool.input_schema["required"] == ["doc_id"]
    assert response["structuredContent"]["document"]["title"] == "Alpha"


def test_default_tools_registers_get_document_summary() -> None:
    tools = default_tools()

    assert "get_document_summary" in tools
    assert "list_collections" in tools
    assert "query_knowledge_hub" in tools
