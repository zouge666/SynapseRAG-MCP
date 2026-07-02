from __future__ import annotations

from observability.dashboard.public_pages.common import require_active_session
from observability.dashboard.services.data_service import DataService


def render() -> None:
    import streamlit as st

    st.title("Session")
    session = require_active_session(st)
    if session is None:
        return

    remaining = session.remaining_seconds()
    st.metric("Time remaining", f"{remaining // 60} min {remaining % 60} s")
    st.caption("The countdown is absolute: it is not extended by activity. Data is deleted when it reaches zero.")
    st.write(f"Mode: **{session.mode}**")

    try:
        documents = DataService(session.settings).list_documents()
        st.write(f"Indexed documents: **{len(documents)}**")
    except Exception:
        st.write("Indexed documents: **0**")

    if st.button("Delete my files and session data now", type="primary"):
        session.destroy()
        st.session_state.pop("session_context", None)
        st.success("All session data was deleted. You can start a new session on the Start page.")
