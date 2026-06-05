from __future__ import annotations

from pathlib import Path
from typing import Any

from core.query_engine import HybridSearch
from core.settings import EvaluationSettings, load_settings
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.eval_runner import EvalReport, EvalRunner


DEFAULT_TEST_SET = "tests/fixtures/golden_test_set.json"


def render() -> None:
    import streamlit as st

    st.title("Evaluation")
    try:
        settings = load_settings("config/settings.yaml")
    except Exception as error:
        st.error(f"Failed to load settings: {error}")
        return

    configured_backends = _configured_backends(settings)
    selected_backends = st.sidebar.multiselect("Backends", configured_backends, default=configured_backends)
    test_set_path = st.sidebar.text_input("Golden test set", DEFAULT_TEST_SET)
    top_k = st.sidebar.number_input("Top K", min_value=1, value=_default_top_k(settings), step=1)

    if st.sidebar.button("Run Evaluation"):
        try:
            report = run_dashboard_evaluation(settings, selected_backends, test_set_path, int(top_k))
        except Exception as error:
            st.error(f"Evaluation failed: {error}")
            return
        _render_report(st, report)
        return

    st.info("Choose evaluation backends and run the golden test set.")


def run_dashboard_evaluation(
    settings: Any,
    backends: list[str],
    test_set_path: str,
    top_k: int,
) -> EvalReport:
    evaluator_settings = EvaluationSettings(enabled=True, backends=backends or _configured_backends(settings))
    evaluator = EvaluatorFactory.create(evaluator_settings)
    runner = EvalRunner(settings=settings, hybrid_search=_search(settings), evaluator=evaluator, top_k=top_k)
    return runner.run(test_set_path)


def _render_report(st: Any, report: EvalReport) -> None:
    metrics = report.metrics
    columns = st.columns(4)
    columns[0].metric("Hit Rate", _metric_value(metrics, "hit_rate"))
    columns[1].metric("MRR", _metric_value(metrics, "mrr"))
    columns[2].metric("Cases", len(report.details))
    columns[3].metric("Metrics", len(metrics))

    if metrics:
        st.subheader("Metrics")
        st.dataframe(_metric_rows(metrics), hide_index=True, use_container_width=True)

    rows = _detail_rows(report)
    if rows:
        st.subheader("Query Details")
        st.dataframe(rows, hide_index=True, use_container_width=True)

    st.json(report.to_dict(), expanded=False)


def _configured_backends(settings: Any) -> list[str]:
    evaluation = getattr(settings, "evaluation", None)
    backends = getattr(evaluation, "backends", None) if evaluation is not None else None
    if isinstance(backends, list) and backends:
        return [str(backend) for backend in backends]
    return ["custom_metrics"]


def _default_top_k(settings: Any) -> int:
    retrieval = getattr(settings, "retrieval", None)
    value = getattr(retrieval, "top_k_final", None) if retrieval is not None else None
    return value if isinstance(value, int) and value > 0 else 5


def _search(settings: Any) -> Any:
    if not _has_ingested_data(settings):
        return EmptySearch()
    return HybridSearch(settings)


class EmptySearch:
    def search(self, query: str, top_k: int, filters: dict | None = None, trace: object | None = None) -> list:
        return []


def _has_ingested_data(settings: Any) -> bool:
    vector_store = getattr(settings, "vector_store", None)
    persist_path = getattr(vector_store, "persist_path", "data/db/chroma") if vector_store is not None else "data/db/chroma"
    collection = getattr(vector_store, "collection", "default") if vector_store is not None else "default"
    if (Path(str(persist_path)) / f"{_safe_collection_name(str(collection))}.json").exists():
        return True
    return (Path("data/db/bm25") / "index.pkl").exists()


def _safe_collection_name(collection: str) -> str:
    value = "".join(character if character.isalnum() or character in "_.-" else "_" for character in collection).strip("._")
    return value or "default"


def _metric_value(metrics: dict[str, float], key: str) -> str:
    value = metrics.get(key, 0.0)
    return f"{float(value):.3f}"


def _metric_rows(metrics: dict[str, float]) -> list[dict[str, Any]]:
    return [{"metric": key, "value": round(float(value), 6)} for key, value in sorted(metrics.items())]


def _detail_rows(report: EvalReport) -> list[dict[str, Any]]:
    rows = []
    for detail in report.details:
        row: dict[str, Any] = {
            "query": detail.query,
            "expected_chunk_ids": ", ".join(detail.expected_chunk_ids),
            "retrieved_ids": ", ".join(detail.retrieved_ids),
            "expected_sources": ", ".join(detail.expected_sources),
            "retrieved_sources": ", ".join(source for source in detail.retrieved_sources if source),
        }
        for key, value in sorted(detail.metrics.items()):
            row[key] = round(float(value), 6)
        rows.append(row)
    return rows
