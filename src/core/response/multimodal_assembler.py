from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from core import RetrievalResult


class MultimodalAssemblerError(ValueError):
    pass


class MultimodalAssembler:
    def assemble(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, Any]]:
        if not isinstance(retrieval_results, list) or not all(isinstance(result, RetrievalResult) for result in retrieval_results):
            raise MultimodalAssemblerError("retrieval_results must be a list of RetrievalResult")
        content = []
        seen = set()
        for result in retrieval_results:
            for image in self._result_images(result):
                key = (image["id"], image["path"])
                if key in seen:
                    continue
                seen.add(key)
                item = self._image_content(image)
                if item is not None:
                    content.append(item)
        return content

    def _result_images(self, result: RetrievalResult) -> list[dict[str, str]]:
        refs = self._image_refs(result.metadata)
        images = self._images_by_id(result.metadata)
        result_images = []
        for image_id in refs:
            image = images.get(image_id)
            if image is None:
                continue
            path = image.get("path")
            if not isinstance(path, str) or not path:
                continue
            result_images.append(
                {
                    "id": image_id,
                    "path": path,
                    "mime_type": self._mime_type(path, image),
                }
            )
        return result_images

    def _image_refs(self, metadata: dict[str, Any]) -> list[str]:
        refs = metadata.get("image_refs", [])
        if not isinstance(refs, list):
            return []
        result = []
        for ref in refs:
            if isinstance(ref, str) and ref and ref not in result:
                result.append(ref)
        return result

    def _images_by_id(self, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        images = metadata.get("images", [])
        if not isinstance(images, list):
            return {}
        return {image["id"]: image for image in images if isinstance(image, dict) and isinstance(image.get("id"), str)}

    def _image_content(self, image: dict[str, str]) -> dict[str, Any] | None:
        path = Path(image["path"])
        if not path.is_file():
            return None
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        if not data:
            return None
        return {"type": "image", "mimeType": image["mime_type"], "data": data}

    def _mime_type(self, path: str, image: dict[str, Any]) -> str:
        for key in ("mimeType", "mime_type", "content_type"):
            value = image.get(key)
            if isinstance(value, str) and value.startswith("image/"):
                return value
        guessed, _ = mimetypes.guess_type(path)
        return guessed if isinstance(guessed, str) and guessed.startswith("image/") else "image/png"
