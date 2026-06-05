from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase


class EvalRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class EvalCaseResult:
    query: str
    expected_chunk_ids: list[str]
    expected_sources: list[str]
    retrieved_ids: list[str]
    retrieved_sources: list[str]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_sources": list(self.expected_sources),
            "retrieved_ids": list(self.retrieved_ids),
            "retrieved_sources": list(self.retrieved_sources),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class EvalReport:
    metrics: dict[str, float]
    details: list[EvalCaseResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return float(self.metrics.get("hit_rate", 0.0))

    @property
    def mrr(self) -> float:
        return float(self.metrics.get("mrr", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "metrics": dict(self.metrics),
            "details": [detail.to_dict() for detail in self.details],
        }


class EvalRunner:
    def __init__(
        self,
        settings: object,
        hybrid_search: object,
        evaluator: BaseEvaluator,
        top_k: int | None = None,
    ) -> None:
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.top_k = top_k

    def run(self, test_set_path: str | Path) -> EvalReport:
        cases = self._load_cases(test_set_path)
        details = [self._run_case(case) for case in cases]
        return EvalReport(metrics=self._aggregate_metrics(details), details=details)

    def _run_case(self, raw_case: dict[str, Any]) -> EvalCaseResult:
        query = self._required_text(raw_case, "query")
        expected_chunk_ids = self._text_list(raw_case, "expected_chunk_ids", "golden_ids")
        expected_sources = self._text_list(raw_case, "expected_sources")
        filters = self._filters(raw_case)
        results = self.hybrid_search.search(query, top_k=self._top_k(raw_case), filters=filters)
        self._validate_results(results)
        retrieved_ids = [result.chunk_id for result in results]
        retrieved_sources = [self._source(result) for result in results]
        evaluation_case = EvaluationCase(
            query=query,
            retrieved_ids=retrieved_ids,
            golden_ids=expected_chunk_ids,
            metadata={
                "expected_sources": expected_sources,
                "generated_answer": self._optional_text(raw_case, "generated_answer", "answer"),
                "ground_truth": self._optional_text(raw_case, "expected_answer", "ground_truth", "reference"),
                "retrieved_contexts": [result.text for result in results],
                "retrieved_sources": retrieved_sources,
            },
        )
        metrics = self.evaluator.evaluate(evaluation_case)
        return EvalCaseResult(
            query=query,
            expected_chunk_ids=expected_chunk_ids,
            expected_sources=expected_sources,
            retrieved_ids=retrieved_ids,
            retrieved_sources=retrieved_sources,
            metrics={key: float(value) for key, value in metrics.items()},
        )

    def _load_cases(self, test_set_path: str | Path) -> list[dict[str, Any]]:
        path = Path(test_set_path)
        if not path.exists():
            raise EvalRunnerError(f"golden test set not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or not isinstance(data.get("test_cases"), list):
            raise EvalRunnerError("golden test set must contain a test_cases list")
        cases = data["test_cases"]
        if not cases:
            raise EvalRunnerError("golden test set must contain at least one test case")
        if not all(isinstance(case, dict) for case in cases):
            raise EvalRunnerError("each golden test case must be a mapping")
        return cases

    def _aggregate_metrics(self, details: list[EvalCaseResult]) -> dict[str, float]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for detail in details:
            for key, value in detail.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {key: round(totals[key] / counts[key], 6) for key in sorted(totals)}

    def _top_k(self, raw_case: dict[str, Any]) -> int:
        value = raw_case.get("top_k", self.top_k)
        if isinstance(value, int) and value > 0:
            return value
        retrieval = self._setting(self.settings, "retrieval", {})
        configured = self._setting(retrieval, "top_k_final", None)
        if isinstance(configured, int) and configured > 0:
            return configured
        raise EvalRunnerError("retrieval.top_k_final must be a positive integer")

    def _filters(self, raw_case: dict[str, Any]) -> dict[str, Any]:
        filters = raw_case.get("filters", {})
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise EvalRunnerError("test case filters must be a mapping")
        return dict(filters)

    def _source(self, result: RetrievalResult) -> str:
        source = result.metadata.get("source_path") or result.metadata.get("source") or ""
        return str(source)

    def _validate_results(self, results: object) -> None:
        if not isinstance(results, list) or not all(isinstance(result, RetrievalResult) for result in results):
            raise EvalRunnerError("hybrid_search.search must return a list of RetrievalResult")

    def _required_text(self, raw_case: dict[str, Any], key: str) -> str:
        value = raw_case.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EvalRunnerError(f"test case {key} must be a non-empty string")
        return value

    def _optional_text(self, raw_case: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = raw_case.get(key)
            if value is not None:
                return str(value)
        return ""

    def _text_list(self, raw_case: dict[str, Any], *keys: str) -> list[str]:
        for key in keys:
            value = raw_case.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
            if isinstance(value, tuple):
                return [str(item) for item in value]
            if value is not None:
                return [str(value)]
        return []

    def _setting(self, source: object, name: str, default: Any) -> Any:
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)
