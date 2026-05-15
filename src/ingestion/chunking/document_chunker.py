from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from core import Chunk, Document
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class DocumentChunker:
    image_ref_pattern = re.compile(r"\[IMAGE:\s*([^\]]+?)\s*\]")

    def __init__(self, settings: object, splitter: BaseSplitter | None = None) -> None:
        self.settings = settings
        self.splitter = splitter or SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        parts = self.splitter.split_text(document.text)
        chunks: list[Chunk] = []
        cursor = 0
        for index, text in enumerate(parts):
            start_offset = self._start_offset(document.text, text, cursor)
            end_offset = start_offset + len(text)
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, index, text),
                    text=text,
                    metadata=self._inherit_metadata(document, index, text),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )
            cursor = end_offset
        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{digest}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> dict[str, Any]:
        metadata = self._copy_metadata(document.metadata)
        document_images = metadata.pop("images", [])
        metadata["chunk_index"] = chunk_index
        image_refs = self._image_refs(chunk_text)
        if image_refs:
            image_by_id = {image["id"]: image for image in document_images if isinstance(image, dict) and isinstance(image.get("id"), str)}
            metadata["image_refs"] = image_refs
            metadata["images"] = [self._copy_metadata(image_by_id[image_id]) for image_id in image_refs if image_id in image_by_id]
        return metadata

    def _image_refs(self, text: str) -> list[str]:
        refs: list[str] = []
        for match in self.image_ref_pattern.finditer(text):
            image_id = match.group(1).strip()
            if image_id and image_id not in refs:
                refs.append(image_id)
        return refs

    def _start_offset(self, source_text: str, chunk_text: str, cursor: int) -> int:
        start = source_text.find(chunk_text, cursor)
        if start >= 0:
            return start
        start = source_text.find(chunk_text)
        if start >= 0:
            return start
        return cursor

    def _copy_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(metadata, ensure_ascii=False))
