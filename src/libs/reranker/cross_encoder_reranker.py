from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.settings import RerankSettings
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


CrossEncoderScorer = Callable[[str, list[RerankCandidate]], list[float]]


class CrossEncoderRerankerError(RuntimeError):
    def __init__(self, message: str, fallback: bool = True) -> None:
        super().__init__(message)
        self.fallback = fallback


@dataclass(frozen=True)
class CrossEncoderRerankFallback:
    reason: str
    candidates: list[RerankCandidate]


class CrossEncoderReranker(BaseReranker):
    def __init__(self, settings: RerankSettings, scorer: CrossEncoderScorer | None = None) -> None:
        super().__init__(settings)
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []
        if not isinstance(query, str) or not query:
            raise CrossEncoderRerankerError("cross_encoder validation error: query must be a non-empty string")
        if self.scorer is None:
            raise CrossEncoderRerankerError("cross_encoder fallback: scorer is required")
        active_candidates = list(candidates[: self.settings.top_m])
        try:
            scores = self.scorer(query, active_candidates)
        except TimeoutError as error:
            raise CrossEncoderRerankerError("cross_encoder fallback: timeout") from error
        except Exception as error:
            raise CrossEncoderRerankerError(f"cross_encoder fallback: {type(error).__name__}") from error
        self._validate_scores(scores, active_candidates)
        ranked = [
            RerankCandidate(
                id=candidate.id,
                text=candidate.text,
                score=float(score),
                metadata=dict(candidate.metadata),
            )
            for candidate, score in zip(active_candidates, scores)
        ]
        ranked.sort(key=lambda candidate: (-candidate.score, candidate.id))
        return ranked + list(candidates[self.settings.top_m :])

    def fallback(self, candidates: list[RerankCandidate], reason: str) -> CrossEncoderRerankFallback:
        return CrossEncoderRerankFallback(reason=reason, candidates=list(candidates))

    def _validate_scores(self, scores: list[float], candidates: list[RerankCandidate]) -> None:
        if not isinstance(scores, list):
            raise CrossEncoderRerankerError("cross_encoder schema error: scores must be list")
        if len(scores) != len(candidates):
            raise CrossEncoderRerankerError("cross_encoder schema error: score count mismatch")
        if not all(isinstance(score, int | float) for score in scores):
            raise CrossEncoderRerankerError("cross_encoder schema error: scores must be numeric")


RerankerFactory.register_provider("cross_encoder", CrossEncoderReranker)
