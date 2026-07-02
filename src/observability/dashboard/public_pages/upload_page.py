from __future__ import annotations

from ingestion import IngestionPipeline
from observability.dashboard import runtime
from observability.dashboard.services.rate_limiter import upload_allowed
from observability.dashboard.services.session_context import SESSION_COLLECTION, sanitize_error
from observability.dashboard.services.upload_guard import UploadRejected, save_upload_securely, sanitize_display_name, validate_pdf_upload
from observability.logger import write_trace


def _client_upload_key(st, session) -> str:
    try:
        headers = st.context.headers
        forwarded = headers.get("X-Forwarded-For", "") if headers else ""
    except Exception:
        forwarded = ""
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"session:{session.session_id}"


def _upload_allowed(st, session) -> bool:
    session_key = f"session:{session.session_id}"
    if not upload_allowed(runtime.upload_limiter, session_key):
        return False
    ip_key = _client_upload_key(st, session)
    if ip_key != session_key and not upload_allowed(runtime.upload_limiter, ip_key):
        return False
    return True


def process_upload(st, session, uploaded) -> None:
    if not _upload_allowed(st, session):
        st.error("Upload limit reached (3 per hour). Please try again later.")
        return
    data = bytes(uploaded.getbuffer())
    try:
        stats = validate_pdf_upload(uploaded.name, data)
    except UploadRejected as error:
        st.error(str(error))
        return

    session.reset_workspace()
    target = save_upload_securely(session.paths.uploads, data)
    display_name = sanitize_display_name(uploaded.name)
    try:
        with st.spinner("Parsing and indexing the PDF..."):
            result = IngestionPipeline(session.settings).run(target, collection=SESSION_COLLECTION)
        trace = getattr(result, "trace", None)
        if isinstance(trace, dict):
            write_trace(trace, path=session.settings.observability.trace_path)
    except Exception as error:
        st.error(sanitize_error(f"Processing failed: {runtime.mask(str(error))}", [session.settings.llm.api_key]))
        return
    finally:
        target.unlink(missing_ok=True)

    st.session_state["upload_display_name"] = display_name
    chunk_count = getattr(result, "chunk_count", None)
    st.success(f"'{display_name}' indexed: {chunk_count} chunks, {stats['pages']} pages. Ask questions on the Query or LLM page.")
