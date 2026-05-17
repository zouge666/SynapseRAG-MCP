from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]
SparseVector = dict[str, float]


class CoreTypeError(ValueError):
    pass


def image_placeholder(image_id: str) -> str:
    if not isinstance(image_id, str) or not image_id:
        raise CoreTypeError("image_id must be a non-empty string")
    return f"[IMAGE: {image_id}]"


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id("Document.id", self.id)
        _validate_text("Document.text", self.text)
        _validate_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": _json_copy(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        _validate_mapping("Document", data)
        return cls(id=data.get("id"), text=data.get("text"), metadata=data.get("metadata", {}))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> Document:
        return cls.from_dict(_json_loads("Document", data))


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: Metadata
    start_offset: int
    end_offset: int
    source_ref: str | None = None

    def __post_init__(self) -> None:
        _validate_id("Chunk.id", self.id)
        _validate_text("Chunk.text", self.text)
        _validate_metadata(self.metadata)
        _validate_offsets(self.start_offset, self.end_offset)
        if self.source_ref is not None and not isinstance(self.source_ref, str):
            raise CoreTypeError("Chunk.source_ref must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _json_copy(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        _validate_mapping("Chunk", data)
        return cls(
            id=data.get("id"),
            text=data.get("text"),
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
            source_ref=data.get("source_ref"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> Chunk:
        return cls.from_dict(_json_loads("Chunk", data))


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    metadata: Metadata
    dense_vector: list[float] | None = None
    sparse_vector: SparseVector | None = None

    def __post_init__(self) -> None:
        _validate_id("ChunkRecord.id", self.id)
        _validate_text("ChunkRecord.text", self.text)
        _validate_metadata(self.metadata)
        _validate_dense_vector(self.dense_vector)
        _validate_sparse_vector(self.sparse_vector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _json_copy(self.metadata),
            "dense_vector": list(self.dense_vector) if self.dense_vector is not None else None,
            "sparse_vector": dict(self.sparse_vector) if self.sparse_vector is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkRecord:
        _validate_mapping("ChunkRecord", data)
        return cls(
            id=data.get("id"),
            text=data.get("text"),
            metadata=data.get("metadata", {}),
            dense_vector=data.get("dense_vector"),
            sparse_vector=data.get("sparse_vector"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> ChunkRecord:
        return cls.from_dict(_json_loads("ChunkRecord", data))


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id("RetrievalResult.chunk_id", self.chunk_id)
        if not isinstance(self.score, int | float):
            raise CoreTypeError("RetrievalResult.score must be a number")
        _validate_text("RetrievalResult.text", self.text)
        if not isinstance(self.metadata, dict):
            raise CoreTypeError("metadata must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": float(self.score),
            "text": self.text,
            "metadata": _json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalResult:
        _validate_mapping("RetrievalResult", data)
        return cls(
            chunk_id=data.get("chunk_id"),
            score=data.get("score"),
            text=data.get("text"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> RetrievalResult:
        return cls.from_dict(_json_loads("RetrievalResult", data))


def _validate_mapping(name: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise CoreTypeError(f"{name} must be a mapping")


def _validate_id(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise CoreTypeError(f"{name} must be a non-empty string")


def _validate_text(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise CoreTypeError(f"{name} must be a string")


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise CoreTypeError("metadata must be a mapping")
    source_path = metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise CoreTypeError("metadata.source_path must be a non-empty string")
    images = metadata.get("images", [])
    if images is None:
        return
    if not isinstance(images, list):
        raise CoreTypeError("metadata.images must be a list")
    for index, image in enumerate(images):
        _validate_image_metadata(index, image)


def _validate_image_metadata(index: int, image: Any) -> None:
    if not isinstance(image, dict):
        raise CoreTypeError(f"metadata.images[{index}] must be a mapping")
    for key in ("id", "path"):
        if not isinstance(image.get(key), str) or not image.get(key):
            raise CoreTypeError(f"metadata.images[{index}].{key} must be a non-empty string")
    for key in ("text_offset", "text_length"):
        value = image.get(key)
        if not isinstance(value, int) or value < 0:
            raise CoreTypeError(f"metadata.images[{index}].{key} must be a non-negative integer")
    if "page" in image and (not isinstance(image["page"], int) or image["page"] < 0):
        raise CoreTypeError(f"metadata.images[{index}].page must be a non-negative integer")
    if "position" in image and not isinstance(image["position"], dict):
        raise CoreTypeError(f"metadata.images[{index}].position must be a mapping")


def _validate_offsets(start_offset: Any, end_offset: Any) -> None:
    if not isinstance(start_offset, int) or start_offset < 0:
        raise CoreTypeError("Chunk.start_offset must be a non-negative integer")
    if not isinstance(end_offset, int) or end_offset < start_offset:
        raise CoreTypeError("Chunk.end_offset must be an integer greater than or equal to start_offset")


def _validate_dense_vector(vector: Any) -> None:
    if vector is None:
        return
    if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
        raise CoreTypeError("ChunkRecord.dense_vector must be a numeric list or None")


def _validate_sparse_vector(vector: Any) -> None:
    if vector is None:
        return
    if not isinstance(vector, dict) or not all(isinstance(key, str) and isinstance(value, int | float) for key, value in vector.items()):
        raise CoreTypeError("ChunkRecord.sparse_vector must be a mapping of string to number or None")


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _json_loads(name: str, data: str) -> dict[str, Any]:
    if not isinstance(data, str):
        raise CoreTypeError(f"{name} JSON must be a string")
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as error:
        raise CoreTypeError(f"{name} JSON must be valid") from error
    _validate_mapping(name, parsed)
    return parsed
