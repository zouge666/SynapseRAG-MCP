from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class QueryProcessorError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessedQuery:
    query: str
    normalized_query: str
    keywords: list[str]
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query:
            raise QueryProcessorError("query must be a non-empty string")
        if not isinstance(self.normalized_query, str):
            raise QueryProcessorError("normalized_query must be a string")
        if not isinstance(self.keywords, list) or not self.keywords or not all(isinstance(keyword, str) and keyword for keyword in self.keywords):
            raise QueryProcessorError("keywords must be a non-empty list of strings")
        if not isinstance(self.filters, dict):
            raise QueryProcessorError("filters must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "keywords": list(self.keywords),
            "filters": json.loads(json.dumps(self.filters, ensure_ascii=False)),
        }


class QueryProcessor:
    token_pattern = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]")
    filter_pattern = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<sep>[:=])(?P<value>\"[^\"]+\"|'[^']+'|[^\s,;]+)")
    default_stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }

    def __init__(self, stop_words: set[str] | None = None, filter_keys: set[str] | None = None) -> None:
        self.stop_words = {word.lower() for word in (stop_words if stop_words is not None else self.default_stop_words)}
        self.filter_keys = {key.lower() for key in filter_keys} if filter_keys is not None else None

    def process(self, query: str, filters: dict[str, Any] | None = None, trace: object | None = None) -> ProcessedQuery:
        if not isinstance(query, str) or not query.strip():
            raise QueryProcessorError("query must be a non-empty string")
        normalized = self._normalize(query)
        inline_filters, query_without_filters = self._extract_filters(normalized)
        merged_filters = self._merge_filters(inline_filters, filters)
        keywords = self._keywords(query_without_filters)
        if not keywords:
            keywords = self._keywords(" ".join(self._filter_text(value) for value in merged_filters.values()))
        if not keywords:
            keywords = self._keywords(normalized)
        result = ProcessedQuery(
            query=query,
            normalized_query=query_without_filters or normalized,
            keywords=keywords,
            filters=merged_filters,
        )
        self._record(trace, "query_processor", {"keyword_count": len(result.keywords), "filters": dict(result.filters)})
        return result

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip()).lower()

    def _extract_filters(self, query: str) -> tuple[dict[str, Any], str]:
        filters: dict[str, Any] = {}
        spans = []
        for match in self.filter_pattern.finditer(query):
            key = match.group("key").lower()
            if self.filter_keys is not None and key not in self.filter_keys:
                continue
            value = self._filter_value(match.group("value"))
            filters[key] = value
            spans.append(match.span())
        if not spans:
            return filters, query
        parts = []
        cursor = 0
        for start, end in spans:
            parts.append(query[cursor:start])
            cursor = end
        parts.append(query[cursor:])
        cleaned = re.sub(r"\s+", " ", " ".join(parts)).strip(" ,;")
        return filters, cleaned

    def _filter_value(self, value: str) -> Any:
        value = value.strip().strip("\"'")
        if "," in value:
            items = [item.strip().strip("\"'") for item in value.split(",")]
            return [item for item in items if item]
        if value.isdigit():
            return int(value)
        if value in {"true", "false"}:
            return value == "true"
        return value

    def _filter_text(self, value: Any) -> str:
        if isinstance(value, list):
            return " ".join(self._filter_text(item) for item in value)
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def _merge_filters(self, inline_filters: dict[str, Any], explicit_filters: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(inline_filters)
        if explicit_filters is None:
            return merged
        if not isinstance(explicit_filters, dict):
            raise QueryProcessorError("filters must be a mapping")
        for key, value in explicit_filters.items():
            if not isinstance(key, str) or not key:
                raise QueryProcessorError("filter keys must be non-empty strings")
            if value is not None:
                merged[key] = value
        return merged

    def _keywords(self, query: str) -> list[str]:
        keywords = []
        for token in self.token_pattern.findall(query):
            if token in self.stop_words or token in keywords:
                continue
            keywords.append(token)
        return keywords

    def _record(self, trace: object | None, name: str, details: dict[str, Any]) -> None:
        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(name, details)
