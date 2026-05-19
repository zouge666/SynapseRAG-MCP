from __future__ import annotations

import re
from typing import Any

from core import RetrievalResult
from core.response.citation_generator import Citation, CitationGenerator


class ResponseBuilderError(ValueError):
    pass


class ResponseBuilder:
    def __init__(self, citation_generator: CitationGenerator | None = None) -> None:
        self.citation_generator = citation_generator or CitationGenerator()

    def build(self, retrieval_results: list[RetrievalResult], query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ResponseBuilderError("query must be a non-empty string")
        if not isinstance(retrieval_results, list) or not all(isinstance(result, RetrievalResult) for result in retrieval_results):
            raise ResponseBuilderError("retrieval_results must be a list of RetrievalResult")
        citations = self.citation_generator.generate(retrieval_results)
        if not citations:
            answer = f"未找到与「{query}」相关的内容。"
            return self._tool_result(answer, answer, [])
        markdown = self._markdown(query, retrieval_results, citations)
        return self._tool_result(markdown, markdown, [citation.to_dict() for citation in citations])

    def _markdown(self, query: str, results: list[RetrievalResult], citations: list[Citation]) -> str:
        lines = [f"找到 {len(results)} 个与「{query}」相关的片段：", ""]
        for result, citation in zip(results, citations):
            lines.append(f"[{citation.id}] {self._summary(result.text)}")
            lines.append(self._source_line(citation))
            lines.append("")
        return "\n".join(lines).strip()

    def _source_line(self, citation: Citation) -> str:
        parts = [f"来源: {citation.source}"]
        if citation.page is not None:
            parts.append(f"页码: {citation.page}")
        parts.append(f"score: {citation.score:.4f}")
        return "；".join(parts)

    def _summary(self, text: str, limit: int = 240) -> str:
        value = re.sub(r"\s+", " ", text).strip()
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."

    def _tool_result(self, text: str, answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": {"answer": answer, "citations": citations},
        }
