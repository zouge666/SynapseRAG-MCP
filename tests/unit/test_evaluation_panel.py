from dataclasses import dataclass

from core import RetrievalResult
from observability.dashboard.pages import evaluation_panel
from observability.evaluation.eval_runner import EvalCaseResult, EvalReport


@dataclass
class EvaluationConfig:
    backends: list[str]


@dataclass
class RetrievalConfig:
    top_k_final: int


@dataclass
class VectorConfig:
    persist_path: str
    collection: str


@dataclass
class SettingsStub:
    evaluation: EvaluationConfig
    retrieval: RetrievalConfig
    vector_store: VectorConfig


class FakeSearch:
    def search(self, query: str, top_k: int, filters: dict | None = None, trace: object | None = None) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="chunk-a",
                score=1.0,
                text="context",
                metadata={"source_path": "docs/a.pdf"},
            )
        ]


def settings(backends: list[str] | None = None) -> SettingsStub:
    return SettingsStub(
        evaluation=EvaluationConfig(backends=backends or ["custom_metrics"]),
        retrieval=RetrievalConfig(top_k_final=7),
        vector_store=VectorConfig(persist_path="missing", collection="default"),
    )


def test_panel_helpers_format_metrics_and_details() -> None:
    report = EvalReport(
        metrics={"mrr": 0.5, "hit_rate": 1.0},
        details=[
            EvalCaseResult(
                query="q",
                expected_chunk_ids=["chunk-a"],
                expected_sources=["docs/a.pdf"],
                retrieved_ids=["chunk-a"],
                retrieved_sources=["docs/a.pdf"],
                metrics={"hit_rate": 1.0, "mrr": 0.5},
            )
        ],
    )

    assert evaluation_panel._metric_rows(report.metrics) == [
        {"metric": "hit_rate", "value": 1.0},
        {"metric": "mrr", "value": 0.5},
    ]
    assert evaluation_panel._detail_rows(report) == [
        {
            "query": "q",
            "expected_chunk_ids": "chunk-a",
            "retrieved_ids": "chunk-a",
            "expected_sources": "docs/a.pdf",
            "retrieved_sources": "docs/a.pdf",
            "hit_rate": 1.0,
            "mrr": 0.5,
        }
    ]


def test_panel_reads_backends_and_top_k_from_settings() -> None:
    active_settings = settings(["custom", "ragas"])

    assert evaluation_panel._configured_backends(active_settings) == ["custom", "ragas"]
    assert evaluation_panel._default_top_k(active_settings) == 7


def test_run_dashboard_evaluation_uses_selected_backends_and_search(monkeypatch, tmp_path) -> None:
    test_set = tmp_path / "golden.json"
    test_set.write_text(
        '{"test_cases":[{"query":"q","expected_chunk_ids":["chunk-a"],"expected_sources":["docs/a.pdf"]}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_panel, "_search", lambda active_settings: FakeSearch())

    report = evaluation_panel.run_dashboard_evaluation(settings(), ["custom"], str(test_set), 3)

    assert report.hit_rate == 1.0
    assert report.mrr == 1.0
    assert report.details[0].retrieved_ids == ["chunk-a"]
