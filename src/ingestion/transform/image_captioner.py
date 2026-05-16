from __future__ import annotations

from pathlib import Path
from typing import Any

from core import Chunk
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory
from ingestion.transform.base_transform import BaseTransform


class ImageCaptioner(BaseTransform):
    default_prompt_path = "config/prompts/image_captioning.txt"
    default_prompt = "Describe the image faithfully for retrieval. Include visible text, chart meaning, and relevant context."

    def __init__(self, settings: object, vision_llm: BaseVisionLLM | None = None, prompt_path: str | None = None, prompt_template: str | None = None) -> None:
        self.settings = settings
        self.captioner_settings = self._image_captioner_settings(settings)
        self.enabled = bool(self._setting("enabled", False))
        self.prompt_path = prompt_path or self._setting("prompt_path", self.default_prompt_path)
        self.prompt_template = prompt_template or self._load_prompt(self.prompt_path)
        self.vision_llm = vision_llm

    def transform(self, chunks: list[Chunk], trace: object | None = None) -> list[Chunk]:
        captioned = []
        for chunk in chunks:
            captioned.append(self._caption_chunk(chunk, trace))
        return captioned

    def _caption_chunk(self, chunk: Chunk, trace: object | None = None) -> Chunk:
        image_refs = self._image_refs(chunk)
        if not image_refs:
            self._record(trace, "image_captioner", {"chunk_id": chunk.id, "method": "none", "count": 0})
            return chunk
        metadata = dict(chunk.metadata)
        if not self.enabled:
            metadata["has_unprocessed_images"] = True
            metadata["image_captioned_by"] = "disabled"
            self._record(trace, "image_captioner", {"chunk_id": chunk.id, "method": "disabled", "count": len(image_refs)})
            return self._replace_chunk(chunk, metadata)
        captions, errors = self._caption_images(chunk, trace)
        if captions:
            metadata["image_captions"] = captions
            metadata["has_unprocessed_images"] = len(captions) < len(image_refs)
            metadata["image_captioned_by"] = "vision_llm"
            if errors:
                metadata["image_caption_errors"] = errors
            self._record(trace, "image_captioner", {"chunk_id": chunk.id, "method": "vision_llm", "count": len(captions)})
            return self._replace_chunk(chunk, metadata)
        metadata["has_unprocessed_images"] = True
        metadata["image_captioned_by"] = "fallback"
        if errors:
            metadata["image_caption_errors"] = errors
        self._record(trace, "image_captioner", {"chunk_id": chunk.id, "method": "fallback", "count": len(image_refs)})
        return self._replace_chunk(chunk, metadata)

    def _caption_images(self, chunk: Chunk, trace: object | None = None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        captions = []
        errors = []
        images = self._images_by_id(chunk)
        for image_id in self._image_refs(chunk):
            image = images.get(image_id)
            if image is None:
                errors.append({"image_id": image_id, "reason": "missing image metadata"})
                continue
            image_path = image.get("path")
            if not isinstance(image_path, str) or not image_path:
                errors.append({"image_id": image_id, "reason": "missing image path"})
                continue
            caption, error = self._caption_image(chunk, image_id, image_path, trace)
            if caption:
                captions.append({"image_id": image_id, "caption": caption})
            else:
                errors.append({"image_id": image_id, "reason": error or "empty vision response"})
        return captions, errors

    def _caption_image(self, chunk: Chunk, image_id: str, image_path: str, trace: object | None = None) -> tuple[str | None, str | None]:
        try:
            vision_llm = self.vision_llm or LLMFactory.create_vision_llm(self.settings)
            prompt = self.prompt_template.format(text=chunk.text, image_id=image_id, source_path=chunk.metadata.get("source_path", ""))
            response = vision_llm.chat_with_image(prompt, image_path, trace=trace)
            text = response.text.strip()
            return (text, None) if text else (None, "empty vision response")
        except Exception as error:
            return None, type(error).__name__

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        path = Path(prompt_path or self.default_prompt_path)
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            return text or self.default_prompt
        return self.default_prompt

    def _image_refs(self, chunk: Chunk) -> list[str]:
        refs = chunk.metadata.get("image_refs", [])
        if not isinstance(refs, list):
            return []
        result = []
        for ref in refs:
            if isinstance(ref, str) and ref and ref not in result:
                result.append(ref)
        return result

    def _images_by_id(self, chunk: Chunk) -> dict[str, dict[str, Any]]:
        images = chunk.metadata.get("images", [])
        if not isinstance(images, list):
            return {}
        return {image["id"]: image for image in images if isinstance(image, dict) and isinstance(image.get("id"), str)}

    def _replace_chunk(self, chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _image_captioner_settings(self, settings: object) -> object:
        ingestion = getattr(settings, "ingestion", None)
        if ingestion is not None:
            return getattr(ingestion, "image_captioner", ingestion)
        if isinstance(settings, dict):
            return settings.get("ingestion", {}).get("image_captioner", settings.get("image_captioner", {}))
        return getattr(settings, "image_captioner", settings)

    def _setting(self, name: str, default: object) -> object:
        if isinstance(self.captioner_settings, dict):
            return self.captioner_settings.get(name, default)
        return getattr(self.captioner_settings, name, default)

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
