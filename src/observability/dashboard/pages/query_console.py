from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core import RetrievalResult
from core.query_engine import HybridSearch, Reranker
from core.settings import load_settings
from core.trace import TraceContext
from observability.dashboard.services.data_service import DataService
from observability.logger import write_trace


TraceWriter = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class DashboardQueryResult:
    question: str
    results: list[RetrievalResult]
    trace: TraceContext
    rerank_applied: bool


def render() -> None:
    import streamlit as st

    st.title("Knowledge Base Query")
    try:
        settings = load_settings("config/settings.yaml")
        collections = DataService(settings).list_collections()
    except Exception as error:
        st.error(f"Failed to load query service: {error}")
        return

    collection = st.selectbox("Collection", collections)
    question = st.text_area("Question", placeholder="例如：海盐桂花拿铁的售价是多少？")
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=settings.retrieval.top_k_final, step=1)

    if not st.button("Ask", type="primary"):
        st.info("Enter a question to retrieve the most relevant passages from the selected collection.")
        return
    if not isinstance(question, str) or not question.strip():
        st.warning("Please enter a question.")
        return

    try:
        with st.spinner("Searching the knowledge base..."):
            query_result = run_dashboard_query(settings, question, collection, int(top_k))
    except Exception as error:
        st.error(f"Query failed: {error}")
        return

    if not query_result.results:
        st.warning("No relevant passages found. Confirm that the document was ingested into this collection.")
        return

    best = query_result.results[0]
    st.subheader("Best matching passage")
    st.write(best.text)
    st.caption(_source_caption(best))

    left, middle = st.columns(2)
    left.metric("Results", len(query_result.results))
    middle.metric("Rerank", "applied" if query_result.rerank_applied else "disabled")

    st.subheader("Retrieved passages")
    for index, result in enumerate(query_result.results, start=1):
        with st.expander(_result_label(index, result)):
            st.write(result.text)
            st.json(result.metadata, expanded=False)

    st.caption("The current local configuration returns retrieved passages. Generative answers require an enabled LLM with valid credentials.")


def run_dashboard_query(
    settings: Any,
    question: str,
    collection: str,
    top_k: int,
    search: Any | None = None,
    reranker: Any | None = None,
    trace_writer: TraceWriter = write_trace,
) -> DashboardQueryResult:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    trace = TraceContext(trace_type="query", metadata={"query": question, "collection": collection})
    try:
        active_search = search or HybridSearch(settings)
        candidates = active_search.search(question, top_k=top_k, filters={"collection": collection}, trace=trace)
        active_reranker = reranker or Reranker(settings)
        results = active_reranker.rerank(question, candidates, trace=trace)
        rerank_applied = _rerank_enabled(settings) and bool(candidates)
        trace.finish("success")
    except Exception as error:
        trace.record_stage("query.error", {"error": str(error)})
        trace.finish("failed")
        trace_writer(trace.to_dict(), path=_trace_path(settings))
        raise

    trace_writer(trace.to_dict(), path=_trace_path(settings))
    return DashboardQueryResult(question=question, results=results, trace=trace, rerank_applied=rerank_applied)


def _rerank_enabled(settings: Any) -> bool:
    rerank = settings.get("rerank", {}) if isinstance(settings, dict) else getattr(settings, "rerank", None)
    return bool(rerank.get("enabled", False)) if isinstance(rerank, dict) else bool(getattr(rerank, "enabled", False))


def _trace_path(settings: Any) -> str:
    observability = settings.get("observability", {}) if isinstance(settings, dict) else getattr(settings, "observability", None)
    value = observability.get("trace_path") if isinstance(observability, dict) else getattr(observability, "trace_path", None)
    return value if isinstance(value, str) and value else "logs/traces.jsonl"


def _source_caption(result: RetrievalResult) -> str:
    source = result.metadata.get("source_path") or result.metadata.get("source") or "unknown"
    chunk_index = result.metadata.get("chunk_index", "-")
    return f"Source: {source} · Chunk: {chunk_index} · Score: {result.score:.4f}"


def _result_label(index: int, result: RetrievalResult) -> str:
    source = result.metadata.get("source_path") or result.metadata.get("source") or "unknown"
    return f"{index}. {source} · score={result.score:.4f}"
