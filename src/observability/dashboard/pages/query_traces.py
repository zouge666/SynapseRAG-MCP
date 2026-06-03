from __future__ import annotations

from observability.dashboard.services.trace_service import TraceService


def render() -> None:
    import streamlit as st

    st.title("Query Traces")
    service = TraceService()
    keyword = st.sidebar.text_input("Search")
    traces = service.search_query_traces(keyword)
    if not traces:
        st.info("No query traces found.")
        return

    st.dataframe(_trace_rows(traces), hide_index=True, use_container_width=True)
    labels = [_trace_label(trace) for trace in traces]
    selected_label = st.sidebar.selectbox("Trace", labels)
    trace = traces[labels.index(selected_label)]
    summary = service.summary_for_trace(trace)
    left, middle, right = st.columns(3)
    left.metric("Status", summary.status)
    middle.metric("Elapsed ms", summary.total_elapsed_ms)
    right.metric("Stages", len(trace.get("stages", [])))
    st.json(summary.metadata, expanded=False)

    waterfall_rows = service.query_waterfall_rows(trace)
    if waterfall_rows:
        st.subheader("Stage Timing")
        st.bar_chart(waterfall_rows, x="elapsed_ms", y="stage")
        st.dataframe(_stage_rows(waterfall_rows), hide_index=True, use_container_width=True)

    comparison_rows = service.retrieval_comparison_rows(trace)
    if comparison_rows:
        st.subheader("Dense vs Sparse")
        st.dataframe(comparison_rows, hide_index=True, use_container_width=True)

    rerank_rows = service.rerank_rows(trace)
    if rerank_rows:
        st.subheader("Rerank")
        for row in rerank_rows:
            with st.expander(row["stage"]):
                st.metric("Elapsed ms", row["elapsed_ms"])
                st.json(row["details"], expanded=False)

    st.subheader("Stage Details")
    for row in service.stage_rows(trace):
        with st.expander(row["stage"]):
            st.metric("Elapsed ms", row["elapsed_ms"])
            st.write(row["method"])
            st.json(row["details"], expanded=False)


def _trace_rows(traces: list[dict]) -> list[dict]:
    rows = []
    for trace in traces:
        metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
        rows.append(
            {
                "trace_id": trace.get("trace_id", ""),
                "query": metadata.get("query", ""),
                "status": trace.get("status", ""),
                "started_at": trace.get("started_at", ""),
                "finished_at": trace.get("finished_at", ""),
                "elapsed_ms": trace.get("total_elapsed_ms", trace.get("duration_ms", 0)),
            }
        )
    return rows


def _trace_label(trace: dict) -> str:
    metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
    query = metadata.get("query") or trace.get("trace_id", "")
    return f"{query} {trace.get('status', '')}".strip()


def _stage_rows(rows: list[dict]) -> list[dict]:
    return [{"stage": row["stage"], "elapsed_ms": row["elapsed_ms"], "method": row["method"]} for row in rows]
