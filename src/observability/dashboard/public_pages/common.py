from __future__ import annotations

import hmac
from typing import Any, Mapping

from core.settings import EmbeddingSettings, LLMSettings
from observability.dashboard.services.session_context import SessionContext, SessionExpiredError


ADMIN_USERNAME = "admin"

PRIVACY_NOTICE = (
    "Privacy: uploaded content and temporary indexes are kept for at most one hour, "
    "and may be deleted earlier when the service restarts. Do not upload sensitive documents."
)


def verify_admin_password(username: str, password: str, secrets_map: Mapping[str, Any]) -> bool:
    expected = secrets_map.get("ADMIN_PASSWORD")
    if not isinstance(expected, str) or not expected:
        return False
    if username != ADMIN_USERNAME:
        return False
    return hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8"))


def owner_llm_settings(secrets_map: Mapping[str, Any]) -> LLMSettings | None:
    provider = secrets_map.get("OWNER_LLM_PROVIDER")
    model = secrets_map.get("OWNER_LLM_MODEL")
    base_url = secrets_map.get("OWNER_LLM_BASE_URL")
    api_key = secrets_map.get("OWNER_LLM_API_KEY")
    if not all(isinstance(value, str) and value for value in (provider, model, api_key)):
        return None
    return LLMSettings(provider=provider, model=model, base_url=base_url if isinstance(base_url, str) else "", api_key=api_key)


def owner_embedding_settings(secrets_map: Mapping[str, Any]) -> EmbeddingSettings | None:
    provider = secrets_map.get("OWNER_EMBEDDING_PROVIDER")
    model = secrets_map.get("OWNER_EMBEDDING_MODEL")
    if not all(isinstance(value, str) and value for value in (provider, model)):
        return None
    base_url = secrets_map.get("OWNER_EMBEDDING_BASE_URL")
    api_key = secrets_map.get("OWNER_EMBEDDING_API_KEY")
    dimensions = secrets_map.get("OWNER_EMBEDDING_DIMENSIONS")
    try:
        parsed_dimensions = int(dimensions)
    except (TypeError, ValueError):
        parsed_dimensions = None
    return EmbeddingSettings(
        provider=provider,
        model=model,
        dimensions=parsed_dimensions,
        base_url=base_url if isinstance(base_url, str) else "",
        api_key=api_key if isinstance(api_key, str) else "",
    )


def require_active_session(st: Any) -> SessionContext | None:
    session = st.session_state.get("session_context")
    if session is None:
        st.warning("Start a guest session or sign in as admin on the Start page first.")
        return None
    try:
        session.require_active()
    except SessionExpiredError:
        st.session_state.pop("session_context", None)
        st.warning("Your session expired after one hour and its data was deleted. Start a new session to continue.")
        return None
    return session
