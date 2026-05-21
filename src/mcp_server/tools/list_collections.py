from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_server.protocol_handler import ToolDefinition


class ListCollectionsError(ValueError):
    pass


@dataclass(frozen=True)
class CollectionInfo:
    name: str
    document_count: int = 0
    total_bytes: int = 0
    documents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "document_count": self.document_count,
            "total_bytes": self.total_bytes,
            "documents": list(self.documents),
        }


def list_collections(root: str | Path = "data/documents") -> dict[str, Any]:
    root_path = Path(root)
    if not root_path.exists():
        return _build_response([])
    if not root_path.is_dir():
        raise ListCollectionsError("collections root must be a directory")
    collections = [
        _scan_collection(path)
        for path in sorted(root_path.iterdir(), key=lambda item: item.name)
        if path.is_dir() and not path.name.startswith(".")
    ]
    return _build_response(collections)


def list_collections_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ListCollectionsError("arguments must be a mapping")
    return list_collections()


def list_collections_tool_definition(root: str | Path = "data/documents") -> ToolDefinition:
    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ListCollectionsError("arguments must be a mapping")
        return list_collections(root)

    return ToolDefinition(
        name="list_collections",
        description="List available local knowledge hub document collections.",
        input_schema={"type": "object", "properties": {}},
        handler=handle,
    )


def _scan_collection(path: Path) -> CollectionInfo:
    documents = [
        file.relative_to(path).as_posix()
        for file in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
        if file.is_file() and not _has_hidden_part(file.relative_to(path))
    ]
    total_bytes = sum((path / document).stat().st_size for document in documents)
    return CollectionInfo(
        name=path.name,
        document_count=len(documents),
        total_bytes=total_bytes,
        documents=documents,
    )


def _has_hidden_part(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _build_response(collections: list[CollectionInfo]) -> dict[str, Any]:
    structured = {"collections": [collection.to_dict() for collection in collections]}
    if not collections:
        return {"content": [{"type": "text", "text": "No document collections found."}], "structuredContent": structured}
    lines = [f"Found {len(collections)} document collection(s):"]
    lines.extend(
        f"- {collection.name}: {collection.document_count} document(s), {collection.total_bytes} bytes"
        for collection in collections
    )
    return {"content": [{"type": "text", "text": "\n".join(lines)}], "structuredContent": structured}
