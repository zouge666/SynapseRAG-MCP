from __future__ import annotations

import html

from observability.dashboard.components import search_box
from observability.dashboard.services.trace_service import TraceService


def render() -> None:
    import streamlit as st

    st.title("Ingestion Traces")
    service = TraceService()
    if not service.ingestion_traces():
        st.info("No ingestion traces found.")
        return

    with st.container(border=True):
        st.markdown("**Trace filters**")
        pending = st.session_state.get("ingestion_traces_search")
        match_count = len(service.search_ingestion_traces(pending)) if isinstance(pending, str) and pending else -1
        keyword = search_box(
            key="ingestion_traces_search",
            label="Search",
            placeholder="Filter by file, collection, or status…",
            match_count=match_count,
        )
        traces = service.search_ingestion_traces(keyword)
        if keyword and not traces:
            st.markdown(
                f'<span style="color:#d93025;font-weight:600">✕ No traces match “{html.escape(keyword)}” — clear the search to see all traces.</span>',
                unsafe_allow_html=True,
            )
        elif keyword:
            st.caption(f"{len(traces)} matching trace{'s' if len(traces) != 1 else ''}")
        labels = [_trace_label(trace) for trace in traces]
        selected_label = st.selectbox("Trace", labels) if traces else None
    if not traces:
        return

    st.dataframe(_trace_rows(traces), hide_index=True, use_container_width=True)
    trace = traces[labels.index(selected_label)]
    summary = service.summary_for_trace(trace)
    left, middle, right = st.columns(3)
    left.metric("Status", summary.status)
    middle.metric("Elapsed ms", summary.total_elapsed_ms)
    right.metric("Stages", len(trace.get("stages", [])))
    st.json(summary.metadata, expanded=False)

    waterfall_rows = service.ingestion_waterfall_rows(trace)
    if waterfall_rows:
        st.subheader("Stage Timing")
        st.bar_chart(waterfall_rows, x="elapsed_ms", y="stage")
        st.dataframe(_stage_rows(waterfall_rows), hide_index=True, use_container_width=True)

    st.subheader("Stage Details")
    for row in service.stage_rows(trace):
        with st.expander(row["stage"]):
            st.metric("Elapsed ms", row["elapsed_ms"])
            st.write(row["method"])
            st.json(row["details"], expanded=False)


def _trace_rows(traces: list[dict]) -> list[dict]:
    return [
        {
            "trace_id": trace.get("trace_id", ""),
            "status": trace.get("status", ""),
            "source_path": trace.get("metadata", {}).get("source_path", "") if isinstance(trace.get("metadata"), dict) else "",
            "collection": trace.get("metadata", {}).get("collection", "") if isinstance(trace.get("metadata"), dict) else "",
            "started_at": trace.get("started_at", ""),
            "finished_at": trace.get("finished_at", ""),
            "elapsed_ms": trace.get("total_elapsed_ms", trace.get("duration_ms", 0)),
        }
        for trace in traces
    ]


def _trace_label(trace: dict) -> str:
    metadata = trace.get("metadata", {}) if isinstance(trace.get("metadata"), dict) else {}
    source = metadata.get("source_path") or trace.get("trace_id", "")
    return f"{source} {trace.get('status', '')}".strip()


def _stage_rows(rows: list[dict]) -> list[dict]:
    return [{"stage": row["stage"], "elapsed_ms": row["elapsed_ms"], "method": row["method"]} for row in rows]
