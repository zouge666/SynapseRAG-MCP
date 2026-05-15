from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core import Chunk
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from ingestion.transform.base_transform import BaseTransform


class ChunkRefiner(BaseTransform):
    default_prompt_path = "config/prompts/chunk_refinement.txt"
    default_prompt = "Rewrite the chunk as a clean retrieval unit while preserving facts.\n\n{text}"

    def __init__(self, settings: object, llm: BaseLLM | None = None, prompt_path: str | None = None) -> None:
        self.settings = settings
        self.refiner_settings = self._chunk_refiner_settings(settings)
        self.use_llm = bool(self._setting("use_llm", False))
        self.prompt_path = prompt_path or self._setting("prompt_path", self.default_prompt_path)
        self.prompt_template = self._load_prompt(self.prompt_path)
        self.llm = llm

    def transform(self, chunks: list[Chunk], trace: object | None = None) -> list[Chunk]:
        refined = []
        for chunk in chunks:
            refined.append(self._refine_chunk(chunk, trace))
        return refined

    def _refine_chunk(self, chunk: Chunk, trace: object | None = None) -> Chunk:
        try:
            rule_text = self._rule_based_refine(chunk.text)
            metadata = dict(chunk.metadata)
            metadata["refined_by"] = "rule"
            metadata["refinement_changed"] = rule_text != chunk.text
            if self.use_llm:
                llm_text, reason = self._llm_refine(rule_text, trace)
                if llm_text:
                    metadata["refined_by"] = "llm"
                    metadata["refinement_changed"] = llm_text != chunk.text
                    self._record(trace, "chunk_refiner", {"chunk_id": chunk.id, "method": "llm"})
                    return self._replace_chunk(chunk, llm_text, metadata)
                metadata["fallback_reason"] = reason or "llm returned empty response"
            self._record(trace, "chunk_refiner", {"chunk_id": chunk.id, "method": "rule"})
            return self._replace_chunk(chunk, rule_text, metadata)
        except Exception as error:
            metadata = dict(chunk.metadata)
            metadata["refined_by"] = "original"
            metadata["fallback_reason"] = type(error).__name__
            self._record(trace, "chunk_refiner", {"chunk_id": chunk.id, "method": "original", "error": type(error).__name__})
            return self._replace_chunk(chunk, chunk.text, metadata)

    def _rule_based_refine(self, text: str) -> str:
        blocks = re.split(r"(```.*?```)", text, flags=re.DOTALL)
        refined = []
        for block in blocks:
            if block.startswith("```") and block.endswith("```"):
                refined.append(block.strip())
            else:
                refined.append(self._refine_plain_text(block))
        return "\n\n".join(part for part in refined if part).strip()

    def _llm_refine(self, text: str, trace: object | None = None) -> tuple[str | None, str | None]:
        try:
            llm = self.llm or LLMFactory.create(self.settings)
            prompt = self.prompt_template.format(text=text)
            result = llm.chat([{"role": "user", "content": prompt}]).strip()
            return (result, None) if result else (None, "empty llm response")
        except Exception as error:
            return None, type(error).__name__

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        path = Path(prompt_path or self.default_prompt_path)
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            return text if "{text}" in text else f"{text}\n\n{{text}}"
        return self.default_prompt

    def _refine_plain_text(self, text: str) -> str:
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
        text = re.sub(r"</?(?:div|span|section|article|p|br|html|body)[^>]*>", " ", text)
        text = re.sub(r"\b([A-Za-z]+)-\s+([a-z]+)", r"\1\2", text)
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if self._drop_line(stripped):
                continue
            lines.append(self._clean_line(stripped))
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _drop_line(self, line: str) -> bool:
        lowered = line.lower()
        if re.fullmatch(r"[-_=*]{3,}", line):
            return True
        if re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", lowered):
            return True
        if lowered in {"confidential", "company confidential"}:
            return True
        if lowered.startswith("header:") or lowered.startswith("footer:"):
            return True
        return False

    def _clean_line(self, line: str) -> str:
        heading_prefix = ""
        heading = re.match(r"^(#{1,6}\s+)(.*)$", line)
        if heading:
            heading_prefix = heading.group(1)
            line = heading.group(2)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", line)
        line = re.sub(r"(?<!_)_(?!_)(.*?)(?<!_)_(?!_)", r"\1", line)
        line = re.sub(r"[ \t]{2,}", " ", line)
        return f"{heading_prefix}{line.strip()}"

    def _replace_chunk(self, chunk: Chunk, text: str, metadata: dict[str, Any]) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _chunk_refiner_settings(self, settings: object) -> object:
        ingestion = getattr(settings, "ingestion", None)
        if ingestion is not None:
            return getattr(ingestion, "chunk_refiner", ingestion)
        if isinstance(settings, dict):
            return settings.get("ingestion", {}).get("chunk_refiner", settings.get("chunk_refiner", {}))
        return getattr(settings, "chunk_refiner", settings)

    def _setting(self, name: str, default: object) -> object:
        if isinstance(self.refiner_settings, dict):
            return self.refiner_settings.get(name, default)
        return getattr(self.refiner_settings, name, default)

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
