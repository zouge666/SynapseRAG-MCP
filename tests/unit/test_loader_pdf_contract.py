from pathlib import Path

import pytest

from core import Document
from libs.loader import BaseLoader, LoaderError, PdfLoader


def write_pdf(path: Path, text: str, image_bytes: bytes | None = None) -> None:
    content = [
        "%PDF-1.4",
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R >> endobj",
        f"4 0 obj << /Length {len(text) + 40} >> stream",
        f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET",
        "endstream endobj",
    ]
    if image_bytes is not None:
        content.extend(
            [
                "5 0 obj << /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 >> stream",
                image_bytes.decode("latin-1"),
                "endstream endobj",
            ]
        )
    content.extend(["trailer << /Root 1 0 R >>", "%%EOF"])
    path.write_bytes("\n".join(content).encode("latin-1"))


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
    write_pdf(pdf_path, "Look here", image_bytes=b"fake-png-bytes")
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    document = loader.load(str(pdf_path))

    image = document.metadata["images"][0]
    assert document.text.startswith("Look here\n[IMAGE: ")
    assert image["id"] in document.text
    assert image["text_offset"] == len("Look here\n")
    assert image["text_length"] == len(f"[IMAGE: {image['id']}]")
    assert image["page"] == 0
    assert image["position"] == {}
    assert Path(image["path"]).read_bytes() == b"fake-png-bytes"


def test_pdf_loader_supports_tj_arrays(tmp_path) -> None:
    pdf_path = tmp_path / "array.pdf"
    pdf_path.write_text("%PDF-1.4\nstream\nBT [(Hello) 120 ( world)] TJ ET\nendstream\n%%EOF", encoding="latin-1")
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    document = loader.load(str(pdf_path))

    assert document.text == "Hello world"


def test_pdf_loader_reports_missing_file() -> None:
    loader = PdfLoader()

    with pytest.raises(LoaderError, match="pdf file not found"):
        loader.load("missing.pdf")


def test_pdf_loader_degrades_when_image_write_fails(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "with_bad_image.pdf"
    write_pdf(pdf_path, "Look here", image_bytes=b"fake-png-bytes")
    loader = PdfLoader(image_root=str(tmp_path / "images"))

    def fail_write(self, data: bytes) -> int:
        raise OSError("cannot write")

    monkeypatch.setattr(Path, "write_bytes", fail_write)

    document = loader.load(str(pdf_path))

    assert document.text == "Look here"
    assert document.metadata["images"] == []
