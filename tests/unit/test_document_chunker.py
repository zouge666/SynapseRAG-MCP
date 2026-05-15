from types import SimpleNamespace

from core import Chunk, Document, image_placeholder
from core.settings import load_settings
from ingestion.chunking import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return list(self.settings.parts)


def document() -> Document:
    first_image = image_placeholder("img-1")
    second_image = image_placeholder("img-2")
    return Document(
        id="doc-1",
        text=f"Alpha text {first_image}\n\nBeta text\n\nGamma text {second_image}",
        metadata={
            "source_path": "docs/sample.pdf",
            "doc_type": "pdf",
            "title": "Sample",
            "images": [
                {
                    "id": "img-1",
                    "path": "data/images/doc-1/img-1.png",
                    "page": 1,
                    "text_offset": 11,
                    "text_length": len(first_image),
                    "position": {},
                },
                {
                    "id": "img-2",
                    "path": "data/images/doc-1/img-2.png",
                    "page": 2,
                    "text_offset": 45,
                    "text_length": len(second_image),
                    "position": {},
                },
            ],
        },
    )


def fake_chunker(parts: list[str]) -> DocumentChunker:
    settings = SimpleNamespace(splitter=SimpleNamespace(provider="fake", parts=parts))
    return DocumentChunker(settings, splitter=FakeSplitter(settings.splitter))


def test_document_chunker_converts_splitter_parts_to_chunks() -> None:
    chunker = fake_chunker(["Alpha text [IMAGE: img-1]", "Beta text"])

    chunks = chunker.split_document(document())

    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.text for chunk in chunks] == ["Alpha text [IMAGE: img-1]", "Beta text"]
    assert [chunk.source_ref for chunk in chunks] == ["doc-1", "doc-1"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len("Alpha text [IMAGE: img-1]")
    assert chunks[1].start_offset > chunks[0].end_offset


def test_chunk_ids_are_unique_and_deterministic() -> None:
    chunker = fake_chunker(["Alpha text [IMAGE: img-1]", "Beta text", "Gamma text [IMAGE: img-2]"])

    first = chunker.split_document(document())
    second = chunker.split_document(document())

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
    assert first[0].id.startswith("doc-1_0000_")
    assert first[1].id.startswith("doc-1_0001_")


def test_metadata_is_inherited_with_chunk_index() -> None:
    chunker = fake_chunker(["Beta text"])

    chunk = chunker.split_document(document())[0]

    assert chunk.metadata["source_path"] == "docs/sample.pdf"
    assert chunk.metadata["doc_type"] == "pdf"
    assert chunk.metadata["title"] == "Sample"
    assert chunk.metadata["chunk_index"] == 0
    assert "images" not in chunk.metadata
    assert "image_refs" not in chunk.metadata


def test_image_metadata_is_distributed_only_to_referencing_chunks() -> None:
    chunker = fake_chunker(["Alpha text [IMAGE: img-1]", "Beta text", "Gamma text [IMAGE: img-2] [IMAGE: img-1]"])

    chunks = chunker.split_document(document())

    assert chunks[0].metadata["image_refs"] == ["img-1"]
    assert [image["id"] for image in chunks[0].metadata["images"]] == ["img-1"]
    assert "images" not in chunks[1].metadata
    assert "image_refs" not in chunks[1].metadata
    assert chunks[2].metadata["image_refs"] == ["img-2", "img-1"]
    assert [image["id"] for image in chunks[2].metadata["images"]] == ["img-2", "img-1"]


def test_unknown_image_placeholder_keeps_ref_without_image_metadata() -> None:
    chunker = fake_chunker(["Alpha text [IMAGE: missing]"])

    chunk = chunker.split_document(document())[0]

    assert chunk.metadata["image_refs"] == ["missing"]
    assert chunk.metadata["images"] == []


def test_output_chunks_are_serializable() -> None:
    chunker = fake_chunker(["Alpha text [IMAGE: img-1]"])

    chunk = chunker.split_document(document())[0]

    assert Chunk.from_json(chunk.to_json()) == chunk


def test_real_recursive_splitter_is_config_driven() -> None:
    settings = SimpleNamespace(splitter=SimpleNamespace(provider="recursive", chunk_size=20, chunk_overlap=0))
    chunker = DocumentChunker(settings)
    doc = Document(id="doc-2", text="alpha beta gamma delta epsilon zeta", metadata={"source_path": "docs/sample.pdf"})

    chunks = chunker.split_document(doc)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in chunks)


def test_default_settings_expose_splitter_config() -> None:
    settings = load_settings("config/settings.yaml")

    assert settings.splitter.provider == "recursive"
    assert settings.splitter.chunk_size == 1000
