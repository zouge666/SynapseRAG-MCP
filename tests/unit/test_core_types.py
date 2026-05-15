import json

import pytest

from core import Chunk, ChunkRecord, CoreTypeError, Document, image_placeholder


def metadata() -> dict:
    return {
        "source_path": "docs/sample.pdf",
        "title": "Sample",
        "images": [
            {
                "id": "abc_1_0",
                "path": "data/images/default/abc_1_0.png",
                "page": 1,
                "text_offset": 12,
                "text_length": len(image_placeholder("abc_1_0")),
                "position": {"x": 10, "y": 20, "width": 100, "height": 80},
            }
        ],
    }


def test_document_serializes_to_stable_dict_and_json() -> None:
    document = Document(id="doc-1", text=f"hello {image_placeholder('abc_1_0')}", metadata=metadata())

    data = document.to_dict()
    restored = Document.from_json(document.to_json())

    assert data == {
        "id": "doc-1",
        "text": "hello [IMAGE: abc_1_0]",
        "metadata": metadata(),
    }
    assert json.loads(document.to_json()) == data
    assert restored == document


def test_chunk_serializes_offsets_and_source_ref() -> None:
    chunk = Chunk(
        id="doc-1_0001_abcd",
        text="hello",
        metadata=metadata(),
        start_offset=0,
        end_offset=5,
        source_ref="doc-1",
    )

    restored = Chunk.from_dict(chunk.to_dict())

    assert chunk.to_dict() == {
        "id": "doc-1_0001_abcd",
        "text": "hello",
        "metadata": metadata(),
        "start_offset": 0,
        "end_offset": 5,
        "source_ref": "doc-1",
    }
    assert restored == chunk


def test_chunk_record_serializes_dense_and_sparse_vectors() -> None:
    record = ChunkRecord(
        id="doc-1_0001_abcd",
        text="hello",
        metadata=metadata(),
        dense_vector=[1, 2.5],
        sparse_vector={"hello": 1, "world": 0.5},
    )

    restored = ChunkRecord.from_json(record.to_json())

    assert record.to_dict() == {
        "id": "doc-1_0001_abcd",
        "text": "hello",
        "metadata": metadata(),
        "dense_vector": [1, 2.5],
        "sparse_vector": {"hello": 1, "world": 0.5},
    }
    assert restored == ChunkRecord(
        id="doc-1_0001_abcd",
        text="hello",
        metadata=metadata(),
        dense_vector=[1, 2.5],
        sparse_vector={"hello": 1, "world": 0.5},
    )


def test_metadata_requires_source_path() -> None:
    with pytest.raises(CoreTypeError, match="metadata.source_path"):
        Document(id="doc-1", text="hello", metadata={})


def test_metadata_images_contract_is_validated() -> None:
    bad_metadata = {"source_path": "docs/sample.pdf", "images": [{"id": "img", "path": "image.png", "text_offset": -1, "text_length": 5}]}

    with pytest.raises(CoreTypeError, match="text_offset"):
        Document(id="doc-1", text="hello", metadata=bad_metadata)


def test_image_placeholder_uses_standard_format() -> None:
    assert image_placeholder("abc_1_0") == "[IMAGE: abc_1_0]"

    with pytest.raises(CoreTypeError, match="image_id"):
        image_placeholder("")


def test_chunk_offsets_must_be_ordered() -> None:
    with pytest.raises(CoreTypeError, match="end_offset"):
        Chunk(id="chunk-1", text="hello", metadata={"source_path": "docs/sample.pdf"}, start_offset=5, end_offset=4)
