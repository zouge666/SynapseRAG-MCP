from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pymupdf

from core import Document, image_placeholder
from libs.loader.base_loader import BaseLoader, LoaderError


class PdfLoader(BaseLoader):
    def __init__(self, image_root: str = "data/images") -> None:
        self.image_root = Path(image_root)

    def load(self, path: str) -> Document:
        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise LoaderError(f"pdf file not found: {path}")
        data = pdf_path.read_bytes()
        doc_hash = hashlib.sha256(data).hexdigest()
        text = self._extract_text(data)
        images = self._extract_images(data, doc_hash)
        text, images = self._append_image_placeholders(text, images)
        metadata = {
            "source_path": str(pdf_path),
            "doc_type": "pdf",
            "title": pdf_path.stem,
            "file_hash": doc_hash,
            "images": images,
        }
        return Document(id=doc_hash, text=text, metadata=metadata)

    def _extract_text(self, data: bytes) -> str:
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                pages = [page.get_text("text", sort=True).strip() for page in document]
        except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
            raise LoaderError(f"pdf text extraction failed: {error}") from error

        text = "\n\n".join(page for page in pages if page).strip()
        if not text:
            raise LoaderError("pdf contains no extractable text; OCR may be required")
        return text

    def _extract_images(self, data: bytes, doc_hash: str) -> list[dict[str, Any]]:
        try:
            return self._extract_images_strict(data, doc_hash)
        except OSError:
            return []

    def _extract_images_strict(self, data: bytes, doc_hash: str) -> list[dict[str, Any]]:
        image_dir = self.image_root / doc_hash
        images: list[dict[str, Any]] = []
        for index, match in enumerate(re.finditer(rb"<<[^>]*?/Subtype\s*/Image[^>]*?>>\s*stream\r?\n?(.*?)\r?\n?endstream", data, flags=re.DOTALL)):
            image_id = f"{doc_hash}_{index}"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{image_id}.png"
            image_path.write_bytes(match.group(1).strip())
            images.append(
                {
                    "id": image_id,
                    "path": str(image_path),
                    "page": 0,
                    "text_offset": 0,
                    "text_length": 0,
                    "position": {},
                }
            )
        return images

    def _append_image_placeholders(self, text: str, images: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        if not images:
            return text, []
        parts = [text] if text else []
        updated_images = []
        for image in images:
            placeholder = image_placeholder(image["id"])
            offset = len("\n".join(parts))
            if parts:
                offset += 1
            parts.append(placeholder)
            updated = dict(image)
            updated["text_offset"] = offset
            updated["text_length"] = len(placeholder)
            updated_images.append(updated)
        return "\n".join(parts), updated_images
