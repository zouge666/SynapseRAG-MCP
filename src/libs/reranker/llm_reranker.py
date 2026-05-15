from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.settings import RerankSettings
from libs.llm.base_llm import BaseLLM
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


class LLMRerankerError(RuntimeError):
    def __init__(self, message: str, fallback: bool = True) -> None:
        super().__init__(message)
        self.fallback = fallback


@dataclass(frozen=True)
class LLMRerankFallback:
    reason: str
    candidates: list[RerankCandidate]


class LLMReranker(BaseReranker):
    default_prompt_path = "config/prompts/rerank.txt"

    def __init__(
        self,
        settings: RerankSettings,
        llm: BaseLLM | None = None,
        prompt_text: str | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        super().__init__(settings)
        self.llm = llm
        self.prompt_text = prompt_text
        self.prompt_path = Path(prompt_path or self.default_prompt_path)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []
        if not isinstance(query, str) or not query:
            raise LLMRerankerError("llm reranker validation error: query must be a non-empty string")
        if self.llm is None:
            raise LLMRerankerError("llm reranker fallback: llm client is required")
        active_candidates = list(candidates[: self.settings.top_m])
        messages = [{"role": "user", "content": self._prompt(query, active_candidates)}]
        try:
            response = self.llm.chat(messages)
        except Exception as error:
            raise LLMRerankerError(f"llm reranker fallback: {type(error).__name__}") from error
        ranked_ids = self._parse_ranked_ids(response, active_candidates)
        ranked = self._ranked_candidates(ranked_ids, active_candidates)
        remaining = [candidate for candidate in active_candidates if candidate.id not in ranked_ids]
        overflow = list(candidates[self.settings.top_m :])
        return ranked + remaining + overflow

    def fallback(self, candidates: list[RerankCandidate], reason: str) -> LLMRerankFallback:
        return LLMRerankFallback(reason=reason, candidates=list(candidates))

    def _prompt(self, query: str, candidates: list[RerankCandidate]) -> str:
        payload = {
            "query": query,
            "candidates": [
                {"id": candidate.id, "text": candidate.text, "score": candidate.score, "metadata": candidate.metadata}
                for candidate in candidates
            ],
            "output_schema": {"ranked_ids": ["candidate_id"]},
        }
        return f"{self._prompt_text()}\n\nReturn only JSON matching the output_schema.\n\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"

    def _prompt_text(self) -> str:
        if self.prompt_text is not None:
            return self.prompt_text
        if not self.prompt_path.exists():
            raise LLMRerankerError(f"llm reranker prompt error: prompt file not found: {self.prompt_path}")
        text = self.prompt_path.read_text(encoding="utf-8").strip()
        if not text:
            raise LLMRerankerError("llm reranker prompt error: prompt is empty")
        return text

    def _parse_ranked_ids(self, response: str, candidates: list[RerankCandidate]) -> list[str]:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as error:
            raise LLMRerankerError("llm reranker schema error: response must be JSON") from error
        if not isinstance(parsed, Mapping):
            raise LLMRerankerError("llm reranker schema error: response must be object")
        ranked_ids = parsed.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not all(isinstance(item, str) for item in ranked_ids):
            raise LLMRerankerError("llm reranker schema error: ranked_ids must be string list")
        known_ids = {candidate.id for candidate in candidates}
        seen: set[str] = set()
        for ranked_id in ranked_ids:
            if ranked_id not in known_ids:
                raise LLMRerankerError(f"llm reranker schema error: unknown candidate id: {ranked_id}")
            if ranked_id in seen:
                raise LLMRerankerError(f"llm reranker schema error: duplicate candidate id: {ranked_id}")
            seen.add(ranked_id)
        return ranked_ids

    def _ranked_candidates(self, ranked_ids: Sequence[str], candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        by_id = {candidate.id: candidate for candidate in candidates}
        return [by_id[ranked_id] for ranked_id in ranked_ids]


RerankerFactory.register_provider("llm", LLMReranker)
