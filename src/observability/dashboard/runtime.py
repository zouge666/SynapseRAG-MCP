from __future__ import annotations

import contextvars
import os
import re
from typing import Any

from core.settings import Settings, load_settings
from observability.dashboard.services.cleanup_service import CleanupService
from observability.dashboard.services.rate_limiter import RateLimiter, owner_llm_allowed
from observability.dashboard.services.session_context import SessionContext, SessionExpiredError


PUBLIC_ENV_VAR = "SYNAPSERAG_PUBLIC"

MASK = "•••"
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:)?(?:/[^\s/,()\[\]{}'\"]+)+/([^\s/,()\[\]{}'\"]+)")
_SECRET_TOKEN = re.compile(r"\b(sk-[A-Za-z0-9_-]{3,}|[A-Za-z0-9_-]{24,})\b")

upload_limiter = RateLimiter()
owner_llm_limiter = RateLimiter()
_cleanup_service = CleanupService()

_admin_view: contextvars.ContextVar[bool] = contextvars.ContextVar("synapserag_admin_view", default=False)


def start_public_housekeeping() -> None:
    _cleanup_service.ensure_cleanup_thread(interval_seconds=60)
    _cleanup_service.cleanup_expired()


def is_public() -> bool:
    return os.environ.get(PUBLIC_ENV_VAR) == "1"


def current_session(st: Any) -> SessionContext | None:
    session = st.session_state.get("session_context")
    return session if isinstance(session, SessionContext) else None


def require_settings(st: Any) -> Settings | None:
    if not is_public():
        return load_settings("config/settings.yaml")
    _admin_view.set(False)
    session = current_session(st)
    if session is None:
        st.warning("Start a guest session or sign in as admin on the Start page first.")
        return None
    try:
        session.require_active()
    except SessionExpiredError:
        st.session_state.pop("session_context", None)
        st.warning("Your session expired after one hour and its data was deleted. Start a new session to continue.")
        return None
    _admin_view.set(session.mode == "admin")
    return session.settings


def owner_llm_quota_ok(st: Any) -> bool:
    session = current_session(st)
    if session is None or session.mode != "admin":
        return True
    return owner_llm_allowed(owner_llm_limiter)


def llm_missing_notice(st: Any, settings: Settings) -> bool:
    if not is_public() or settings.llm.api_key:
        return False
    st.info("LLM is not configured for this session. Add your provider, base URL and API key on the Settings page to enable LLM answers.")
    return True


def admin_view() -> bool:
    return _admin_view.get()


def mask(value: Any) -> str:
    text = str(value)
    if not is_public() or _admin_view.get():
        return text
    text = _ABSOLUTE_PATH.sub(lambda match: f"{MASK}/{match.group(1)}", text)
    return _SECRET_TOKEN.sub(MASK, text)


def mask_obj(value: Any) -> Any:
    if not is_public() or _admin_view.get():
        return value
    if isinstance(value, dict):
        return {key: mask_obj(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_obj(item) for item in value]
    if isinstance(value, str):
        return mask(value)
    return value
