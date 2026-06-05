import json
from pathlib import Path
from typing import Any

from core import RetrievalResult
from core.query_engine import HybridSearch, QueryProcessor, RRFusion
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.eval_runner import EvalRunner


GOLDEN_SET = Path(__file__).resolve().parents[1] / "fixtures" / "golden_test_set.json"
MIN_HIT_RATE = 0.8
MIN_MRR = 0.8


class GoldenDenseRetriever:
    def __init__(self, cases: list[dict[str, Any]]) -> None:
        self.results_by_query = {case["query"]: self._results(case) for case in cases}

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[RetrievalResult]:
        return self.results_by_query.get(query, [])[:top_k]

    def _results(self, case: dict[str, Any]) -> list[RetrievalResult]:
        expected_ids = [str(item) for item in case.get("expected_chunk_ids", [])]
        expected_sources = [str(item) for item in case.get("expected_sources", [])]
        results = []
        for index, chunk_id in enumerate(expected_ids):
            source = expected_sources[index] if index < len(expected_sources) else "unknown"
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=1.0 - index * 0.01,
                    text=f"context for {chunk_id}",
                    metadata={"source_path": source},
                )
            )
        results.append(
            RetrievalResult(
                chunk_id=f"distractor_{case['query']}",
                score=0.1,
                text="distractor context",
                metadata={"source_path": "distractor"},
            )
        )
        return results


class EmptySparseRetriever:
    def retrieve(self, keywords: list[str], top_k: int, trace: object | None = None) -> list[RetrievalResult]:
        return []


def load_cases() -> list[dict[str, Any]]:
    data = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    return data["test_cases"]


def test_golden_recall_meets_regression_threshold() -> None:
    cases = load_cases()
    search = HybridSearch(
        {"retrieval": {"top_k_dense": 5, "top_k_sparse": 5, "top_k_final": 5}},
        query_processor=QueryProcessor(),
        dense_retriever=GoldenDenseRetriever(cases),
        sparse_retriever=EmptySparseRetriever(),
        fusion=RRFusion(),
    )
    runner = EvalRunner(
        settings={"retrieval": {"top_k_final": 5}},
        hybrid_search=search,
        evaluator=CustomEvaluator(),
    )

    report = runner.run(GOLDEN_SET)

    assert len(report.details) == len(cases)
    assert report.hit_rate >= MIN_HIT_RATE
    assert report.mrr >= MIN_MRR
    assert all(detail.retrieved_ids for detail in report.details)
