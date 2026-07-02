from __future__ import annotations

from pathlib import Path
from typing import Any

_COMPONENT_DIR = Path(__file__).parent / "trace_search"
_component: Any = None
_component_unavailable = False


def search_box(*, key: str, label: str, placeholder: str, match_count: int) -> str:
    component = _get_component()
    if component is None:
        import streamlit as st

        return st.text_input(label, placeholder=placeholder, key=key) or ""
    result = component(label=label, placeholder=placeholder, match_count=match_count, key=key, default="")
    return result if isinstance(result, str) else ""


def _get_component() -> Any:
    global _component, _component_unavailable
    if _component is None and not _component_unavailable:
        try:
            import streamlit.components.v1 as components

            _component = components.declare_component("trace_search", path=str(_COMPONENT_DIR))
        except Exception:
            _component_unavailable = True
    return _component
