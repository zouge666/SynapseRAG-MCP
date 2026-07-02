from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 100
MAX_TEXT_CHARS = 500_000
PDF_MAGIC = b"%PDF-"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.\-一-鿿]+")


class UploadRejected(ValueError):
    pass


def validate_pdf_upload(name: str, data: bytes) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise UploadRejected("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected("The file is larger than 10 MB. Please upload a smaller PDF.")
    if not bytes(data[:5]) == PDF_MAGIC:
        raise UploadRejected("Only PDF files are supported.")
    try:
        import pymupdf

        with pymupdf.open(stream=bytes(data), filetype="pdf") as document:
            page_count = document.page_count
            text_chars = 0
            for page in document:
                text_chars += len(page.get_text())
                if text_chars > MAX_TEXT_CHARS:
                    break
    except UploadRejected:
        raise
    except Exception:
        raise UploadRejected("The file could not be parsed as a PDF. Please upload a valid PDF.") from None
    if page_count > MAX_PAGES:
        raise UploadRejected(f"The PDF has {page_count} pages; the limit is {MAX_PAGES}.")
    if text_chars > MAX_TEXT_CHARS:
        raise UploadRejected("The PDF contains too much text to process in a temporary session.")
    return {"pages": page_count, "text_chars": text_chars}


def sanitize_display_name(name: str) -> str:
    base = Path(str(name or "document.pdf")).name
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "document.pdf"
    return cleaned[:120]


def save_upload_securely(uploads_dir: Path, data: bytes) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / f"{secrets.token_hex(12)}.pdf"
    target.write_bytes(bytes(data))
    return target
