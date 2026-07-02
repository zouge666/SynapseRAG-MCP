from __future__ import annotations

from core.settings import EmbeddingSettings, LLMSettings, Settings, load_settings
from observability.dashboard.public_pages.common import PRIVACY_NOTICE, owner_embedding_settings, owner_llm_settings, verify_admin_password
from observability.dashboard.services.session_context import SessionContext


def render() -> None:
    import streamlit as st

    st.title("SynapseRAG — temporary RAG workspace")
    st.caption(PRIVACY_NOTICE)

    session = st.session_state.get("session_context")
    if session is not None and not session.is_expired():
        st.success(f"Active {session.mode} session — {session.remaining_seconds() // 60} min remaining.")
        if st.button("End session and start over"):
            session.destroy()
            st.session_state.pop("session_context", None)
            st.rerun()
        return

    guest_tab, admin_tab = st.tabs(["Guest", "Admin"])

    with guest_tab:
        st.markdown("Start a temporary workspace, then add your own LLM key on the Settings page to enable LLM answers. Without a key, retrieval still works with local search.")
        if st.button("Start guest session", type="primary"):
            _start_session(st, "guest")

    with admin_tab:
        st.markdown("Admin sign-in uses the server-side LLM configuration automatically.")
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        if st.button("Sign in as admin"):
            if verify_admin_password(username, password, st.secrets):
                _start_session(st, "admin")
            else:
                st.error("Invalid credentials.")


def _resolve_admin_llm(secrets_map, base: Settings) -> LLMSettings | None:
    owner = owner_llm_settings(secrets_map)
    if owner is not None:
        return owner
    return base.llm if base.llm.api_key else None


def _start_session(st, mode: str) -> None:
    base = load_settings("config/settings.yaml")
    llm: LLMSettings | None = None
    embedding: EmbeddingSettings | None = None
    if mode == "admin":
        llm = _resolve_admin_llm(st.secrets, base)
        embedding = owner_embedding_settings(st.secrets)
    session = SessionContext.create(base, mode, llm=llm, embedding=embedding)
    st.session_state["session_context"] = session
    if mode == "admin":
        st.success("Signed in. Server-side LLM configured automatically." if llm else "Signed in. No server-side LLM key configured; local search only.")
    else:
        st.success("Guest session started. Add your LLM key on the Settings page to enable LLM answers.")
    st.rerun()
