import base64
from pathlib import Path

import pymupdf
import pytest

from core import Document
from libs.loader import BaseLoader, LoaderError, PdfLoader


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def write_pdf(path: Path, text: str, image_bytes: bytes | None = None) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    if image_bytes is not None:
        page.insert_image(pymupdf.Rect(72, 90, 73, 91), stream=image_bytes)
    document.save(path, deflate=True)
    document.close()


def test_pdf_loader_implements_base_loader_contract() -> None:
    loader = PdfLoader()

    assert isinstance(loader, BaseLoader)


def test_pdf_loader_loads_simple_pdf_as_document(tmp_path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    write_pdf(pdf_path, "Hello PDF")
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    document = loader.load(str(pdf_path))

    assert isinstance(document, Document)
    assert document.text == "Hello PDF"
    assert document.metadata["source_path"] == str(pdf_path)
    assert document.metadata["doc_type"] == "pdf"
    assert document.metadata["title"] == "simple"
    assert document.metadata["images"] == []
    assert len(document.id) == 64
    assert document.id == document.metadata["file_hash"]


def test_pdf_loader_extracts_images_and_inserts_placeholders(tmp_path) -> None:
    pdf_path = tmp_path / "with_images.pdf"
    write_pdf(pdf_path, "Look here", image_bytes=PNG_BYTES)
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    document = loader.load(str(pdf_path))

    image = document.metadata["images"][0]
    assert document.text.startswith("Look here\n[IMAGE: ")
    assert image["id"] in document.text
    assert image["text_offset"] == len("Look here\n")
    assert image["text_length"] == len(f"[IMAGE: {image['id']}]")
    assert image["page"] == 0
    assert image["position"] == {}
    assert Path(image["path"]).read_bytes()


def test_pdf_loader_extracts_text_from_compressed_content_stream(tmp_path) -> None:
    pdf_path = tmp_path / "compressed.pdf"
    write_pdf(pdf_path, "Hello compressed PDF")
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    document = loader.load(str(pdf_path))

    assert document.text == "Hello compressed PDF"


def test_pdf_loader_rejects_pdf_without_extractable_text(tmp_path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    with pytest.raises(LoaderError, match="OCR may be required"):
        loader.load(str(pdf_path))


def test_pdf_loader_reports_missing_file() -> None:
    loader = PdfLoader()

    with pytest.raises(LoaderError, match="pdf file not found"):
        loader.load("missing.pdf")


def test_pdf_loader_degrades_when_image_write_fails(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "with_bad_image.pdf"
    write_pdf(pdf_path, "Look here", image_bytes=PNG_BYTES)
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    def fail_write(self, data: bytes) -> int:
        raise OSError("cannot write")

    monkeypatch.setattr(Path, "write_bytes", fail_write)

    document = loader.load(str(pdf_path))

    assert document.text == "Look here"
    assert document.metadata["images"] == []
