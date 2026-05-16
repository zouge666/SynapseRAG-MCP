from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from core import Chunk
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from ingestion.transform.base_transform import BaseTransform


class MetadataEnricher(BaseTransform):
    default_prompt = (
        "Return JSON metadata for this retrieval chunk with keys title, summary, tags. "
        "Tags must be a short list of topic strings.\n\n{text}"
    )
    stop_words = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "because",
        "between",
        "for",
        "from",
        "into",
        "that",
        "the",
        "this",
        "with",
        "will",
        "your",
    }

    def __init__(self, settings: object, llm: BaseLLM | None = None, prompt_template: str | None = None) -> None:
        self.settings = settings
        self.enricher_settings = self._metadata_enricher_settings(settings)
        self.use_llm = bool(self._setting("use_llm", False))
        self.llm = llm
        self.prompt_template = prompt_template or self.default_prompt

    def transform(self, chunks: list[Chunk], trace: object | None = None) -> list[Chunk]:
        enriched = []
        for chunk in chunks:
            enriched.append(self._enrich_chunk(chunk, trace))
        return enriched

    def _enrich_chunk(self, chunk: Chunk, trace: object | None = None) -> Chunk:
        rule_metadata = self._rule_metadata(chunk)
        metadata = dict(chunk.metadata)
        metadata.update(rule_metadata)
        metadata["metadata_enriched_by"] = "rule"
        if self.use_llm:
            llm_metadata, reason = self._llm_metadata(chunk.text, trace)
            if llm_metadata is not None:
                metadata.update(llm_metadata)
                metadata["metadata_enriched_by"] = "llm"
                self._record(trace, "metadata_enricher", {"chunk_id": chunk.id, "method": "llm"})
                return self._replace_chunk(chunk, metadata)
            metadata["metadata_enrichment_fallback_reason"] = reason or "llm returned invalid metadata"
        self._record(trace, "metadata_enricher", {"chunk_id": chunk.id, "method": "rule"})
        return self._replace_chunk(chunk, metadata)

    def _rule_metadata(self, chunk: Chunk) -> dict[str, Any]:
        title = self._title(chunk)
        summary = self._summary(chunk.text)
        tags = self._tags(chunk.text, chunk.metadata)
        return {
            "title": title or "Untitled chunk",
            "summary": summary or title or "No summary available",
            "tags": tags or ["general"],
        }

    def _llm_metadata(self, text: str, trace: object | None = None) -> tuple[dict[str, Any] | None, str | None]:
        try:
            llm = self.llm or LLMFactory.create(self.settings)
            prompt = self.prompt_template.format(text=text)
            parsed = self._parse_llm_response(llm.chat([{"role": "user", "content": prompt}]))
            return (parsed, None) if parsed is not None else (None, "invalid llm response")
        except Exception as error:
            return None, type(error).__name__

    def _parse_llm_response(self, response: str) -> dict[str, Any] | None:
        data = self._json_from_response(response)
        if data is None:
            return None
        title = self._clean_text_value(data.get("title"))
        summary = self._clean_text_value(data.get("summary"))
        tags = self._clean_tags(data.get("tags"))
        if not title or not summary or not tags:
            return None
        return {"title": title, "summary": summary, "tags": tags}

    def _json_from_response(self, response: str) -> dict[str, Any] | None:
        if not isinstance(response, str) or not response.strip():
            return None
        text = response.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = match.group(0) if match else text
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _title(self, chunk: Chunk) -> str:
        for key in ("title", "heading", "section_title"):
            value = self._clean_text_value(chunk.metadata.get(key))
            if value:
                return value
        heading = re.search(r"^\s*#{1,6}\s+(.+?)\s*$", chunk.text, flags=re.MULTILINE)
        if heading:
            return self._truncate(self._clean_heading(heading.group(1)), 80)
        first_line = next((line.strip() for line in chunk.text.splitlines() if line.strip()), "")
        sentence = re.split(r"(?<=[.!?。！？])\s+", first_line)[0]
        return self._truncate(self._clean_heading(sentence), 80)

    def _summary(self, text: str) -> str:
        plain = self._plain_text(text)
        sentences = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+", plain) if item.strip()]
        if sentences:
            return self._truncate(" ".join(sentences[:2]), 240)
        return self._truncate(plain, 240)

    def _tags(self, text: str, metadata: dict[str, Any]) -> list[str]:
        existing = self._clean_tags(metadata.get("tags"))
        if existing:
            return existing[:8]
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
        counts = Counter(word for word in words if word not in self.stop_words)
        return [word for word, _ in counts.most_common(8)]

    def _plain_text(self, text: str) -> str:
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text)
        text = re.sub(r"\[[^\]]+]\([^)]*\)", " ", text)
        text = re.sub(r"[#>*_`~-]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _clean_heading(self, text: str) -> str:
        text = re.sub(r"[*_`#]+", " ", text)
        return re.sub(r"\s+", " ", text).strip(" -:")

    def _clean_text_value(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value).strip()

    def _clean_tags(self, value: Any) -> list[str]:
        if isinstance(value, str):
            raw = re.split(r"[,;|]", value)
        elif isinstance(value, list):
            raw = value
        else:
            return []
        tags = []
        for item in raw:
            if not isinstance(item, str):
                continue
            tag = re.sub(r"\s+", " ", item.strip().lower())
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _truncate(self, text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0].rstrip(" .,:;") or text[:limit].rstrip(" .,:;")

    def _replace_chunk(self, chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _metadata_enricher_settings(self, settings: object) -> object:
        ingestion = getattr(settings, "ingestion", None)
        if ingestion is not None:
            return getattr(ingestion, "metadata_enricher", ingestion)
        if isinstance(settings, dict):
            return settings.get("ingestion", {}).get("metadata_enricher", settings.get("metadata_enricher", {}))
        return getattr(settings, "metadata_enricher", settings)

    def _setting(self, name: str, default: object) -> object:
        if isinstance(self.enricher_settings, dict):
            return self.enricher_settings.get(name, default)
        return getattr(self.enricher_settings, name, default)

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
