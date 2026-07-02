from __future__ import annotations

import copy
from typing import Any

from observability.dashboard.services.config_service import ConfigService, parse_env_reference
from observability.dashboard.services.llm_diagnostics import test_llm_connection


LLM_PROVIDERS = ["openai", "anthropic"]
EMBEDDING_PROVIDERS = ["local", "openai", "ollama"]
EMBEDDING_PROVIDER_LABELS = {
    "local": "local (hash, offline)",
    "openai": "online (OpenAI-compatible API)",
    "ollama": "ollama (local server)",
}
VECTOR_STORE_BACKENDS = ["chroma"]
SPLITTER_PROVIDERS = ["recursive"]
SPARSE_BACKENDS = ["bm25"]
FUSION_ALGORITHMS = ["rrf"]
RERANK_BACKENDS = ["none", "llm", "cross_encoder"]
KEY_REQUIRED_PROVIDERS = {"openai", "anthropic"}
KEY_REQUIRED_EMBEDDING_PROVIDERS = {"openai"}
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "ollama": "http://localhost:11434",
}
KEY_MASK = "••••••••"


def render() -> None:
    import streamlit as st

    st.title("Settings")
    service = ConfigService()
    try:
        raw = service.load_raw()
        settings = service.load()
    except Exception as error:
        st.error(f"Failed to load settings: {error}")
        return

    values: dict[str, Any] = {
        "_effective": {
            "llm.model": settings.llm.model,
            "llm.base_url": settings.llm.base_url,
            "llm.api_key": settings.llm.api_key,
            "embedding.model": settings.embedding.model,
            "embedding.base_url": settings.embedding.base_url,
            "embedding.api_key": settings.embedding.api_key,
        }
    }
    _render_pipeline_section(st, service, raw, values)
    _render_llm_section(st, service, raw, values)
    llm_ready = _llm_ready(raw, values)
    _render_embedding_section(st, service, raw, values)
    _render_ingestion_section(st, service, raw, values, llm_ready)
    _render_splitter_section(st, service, raw, values)
    _render_retrieval_section(st, service, raw, values)
    _render_rerank_section(st, service, raw, values, llm_ready)
    _render_storage_section(st, service, raw, values)


def _render_pipeline_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any]) -> None:
    with st.container(border=True):
        _section_title(st, "Pipeline", extra_class="sr-section-title--pipeline")
        st.caption("The green option is active. Click another option to switch, then save this section.")
        _saved_banner(st, "pipeline", "Pipeline settings saved.")
        values["llm_provider"] = _step_option(st, "LLM", LLM_PROVIDERS, raw, ("llm", "provider"))
        if values["llm_provider"] in KEY_REQUIRED_PROVIDERS and not _llm_ready(raw, values):
            st.warning(f"The {values['llm_provider']} provider needs an API key. Add it in the LLM section below, or set it in .env.")
        values["embedding_provider"] = _step_option(st, "Embedding", EMBEDDING_PROVIDERS, raw, ("embedding", "provider"), labels=EMBEDDING_PROVIDER_LABELS)
        if values["embedding_provider"] in KEY_REQUIRED_EMBEDDING_PROVIDERS and not values["_effective"].get("embedding.api_key"):
            st.warning(f"The {values['embedding_provider']} embedding provider needs an API key. Add it in the Embedding section below, or set it in .env.")
        values["vector_backend"] = _step_option(st, "Vector Store", VECTOR_STORE_BACKENDS, raw, ("vector_store", "backend"))
        values["splitter_provider"] = _step_option(st, "Splitter", SPLITTER_PROVIDERS, raw, ("splitter", "provider"))
        values["sparse_backend"] = _step_option(st, "Sparse Retrieval", SPARSE_BACKENDS, raw, ("retrieval", "sparse_backend"))
        values["fusion_algorithm"] = _step_option(st, "Fusion", FUSION_ALGORITHMS, raw, ("retrieval", "fusion_algorithm"))
        values["rerank_backend"] = _step_option(st, "Reranker", RERANK_BACKENDS, raw, ("rerank", "backend"))
        if values["rerank_backend"] == "llm" and not _llm_ready(raw, values):
            st.warning("The llm rerank backend calls the LLM during queries. Configure the LLM API key in the LLM section below first.")
        if st.button("Save", key="settings_save_pipeline", type="primary"):
            _save_section(st, service, "pipeline", values)


def _render_llm_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any]) -> None:
    with st.container(border=True):
        _section_title(st, "LLM")
        _saved_banner(st, "llm", "LLM settings saved.")
        test_result = st.session_state.pop("settings_llm_test_result", None)
        if test_result is not None:
            if test_result.ok:
                st.success(f"{test_result.summary}\n\n{test_result.detail}")
            else:
                st.error(f"{test_result.summary}\n\n{test_result.detail}")
        values["llm_model"] = _mapped_input(st, "Model", raw, ("llm", "model"), values["_effective"]["llm.model"], key="settings_llm_model")
        values["llm_base_url"] = _mapped_input(
            st,
            "Base URL",
            raw,
            ("llm", "base_url"),
            values["_effective"]["llm.base_url"],
            placeholder=DEFAULT_BASE_URLS.get(values["llm_provider"], ""),
            key="settings_llm_base_url",
        )
        values["llm_api_key"] = _secret_input(st, raw, ("llm", "api_key"), values["_effective"].get("llm.api_key", ""), "settings_llm_api_key")
        provider = values["llm_provider"]
        if provider in KEY_REQUIRED_PROVIDERS and not _llm_ready(raw, values):
            st.warning(f"The {provider} provider requires an API key. Save one here before enabling LLM-powered features.")
        if st.button("Test connection", key="settings_test_llm"):
            with st.spinner("Testing the LLM connection…"):
                result = test_llm_connection(
                    provider=provider,
                    model=values["llm_model"],
                    base_url=values["llm_base_url"],
                    api_key=values["llm_api_key"] or values["_effective"].get("llm.api_key", ""),
                )
            st.session_state["settings_llm_test_result"] = result
            st.rerun()
        if st.button("Save", key="settings_save_llm", type="primary"):
            _save_section(st, service, "llm", values)


def _render_embedding_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any]) -> None:
    with st.container(border=True):
        _section_title(st, "Embedding")
        st.caption("Separate from the LLM section: chat endpoints (OpenAI-compatible or Anthropic) do not necessarily serve embeddings, so the connection is configured here.")
        _saved_banner(st, "embedding", "Embedding settings saved.")
        values["embedding_model"] = _mapped_input(st, "Model", raw, ("embedding", "model"), values["_effective"]["embedding.model"], key="settings_embedding_model")
        values["embedding_dimensions"] = st.number_input("Dimensions", min_value=1, value=int(_current(raw, ("embedding", "dimensions")) or 128), step=1, key="settings_embedding_dimensions")
        if values["embedding_provider"] != "local":
            values["embedding_base_url"] = _mapped_input(
                st,
                "Base URL",
                raw,
                ("embedding", "base_url"),
                values["_effective"]["embedding.base_url"],
                placeholder=DEFAULT_BASE_URLS.get(values["embedding_provider"], ""),
                key="settings_embedding_base_url",
            )
        if values["embedding_provider"] in KEY_REQUIRED_EMBEDDING_PROVIDERS:
            values["embedding_api_key"] = _secret_input(st, raw, ("embedding", "api_key"), values["_effective"].get("embedding.api_key", ""), "settings_embedding_api_key")
        if st.button("Save", key="settings_save_embedding", type="primary"):
            _save_section(st, service, "embedding", values)


def _render_ingestion_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any], llm_ready: bool) -> None:
    with st.container(border=True):
        _section_title(st, "Ingestion LLM Assist")
        _saved_banner(st, "ingestion", "Ingestion settings saved.")
        values["use_chunk_refiner"] = st.checkbox("Refine chunks with LLM", value=bool(_current(raw, ("ingestion", "chunk_refiner", "use_llm"))), key="settings_use_chunk_refiner")
        values["use_metadata_enricher"] = st.checkbox("Enrich metadata with LLM", value=bool(_current(raw, ("ingestion", "metadata_enricher", "use_llm"))), key="settings_use_metadata_enricher")
        values["use_image_captioner"] = st.checkbox("Caption images with LLM", value=bool(_current(raw, ("ingestion", "image_captioner", "enabled"))), key="settings_use_image_captioner")
        if (values["use_chunk_refiner"] or values["use_metadata_enricher"] or values["use_image_captioner"]) and not llm_ready:
            st.warning("These features call the LLM during ingestion. Configure the LLM API key first.")
        if st.button("Save", key="settings_save_ingestion", type="primary"):
            _save_section(st, service, "ingestion", values)


def _render_splitter_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any]) -> None:
    with st.container(border=True):
        _section_title(st, "Splitter")
        _saved_banner(st, "splitter", "Splitter settings saved.")
        values["chunk_size"] = st.number_input("Chunk size", min_value=1, value=int(_current(raw, ("splitter", "chunk_size")) or 1000), step=1, key="settings_chunk_size")
        values["chunk_overlap"] = st.number_input("Chunk overlap", min_value=0, value=int(_current(raw, ("splitter", "chunk_overlap")) or 200), step=1, key="settings_chunk_overlap")
        if st.button("Save", key="settings_save_splitter", type="primary"):
            _save_section(st, service, "splitter", values)


def _render_retrieval_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any]) -> None:
    with st.container(border=True):
        _section_title(st, "Retrieval")
        _saved_banner(st, "retrieval", "Retrieval settings saved.")
        values["top_k_dense"] = st.number_input("Top K (dense)", min_value=1, value=int(_current(raw, ("retrieval", "top_k_dense")) or 20), step=1, key="settings_top_k_dense")
        values["top_k_sparse"] = st.number_input("Top K (sparse)", min_value=1, value=int(_current(raw, ("retrieval", "top_k_sparse")) or 20), step=1, key="settings_top_k_sparse")
        values["top_k_final"] = st.number_input("Top K (final)", min_value=1, value=int(_current(raw, ("retrieval", "top_k_final")) or 5), step=1, key="settings_top_k_final")
        if st.button("Save", key="settings_save_retrieval", type="primary"):
            _save_section(st, service, "retrieval", values)


def _render_rerank_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any], llm_ready: bool) -> None:
    with st.container(border=True):
        _section_title(st, "Rerank")
        _saved_banner(st, "rerank", "Rerank settings saved.")
        values["rerank_enabled"] = st.checkbox("Enable rerank", value=bool(_current(raw, ("rerank", "enabled"))), key="settings_rerank_enabled")
        values["rerank_top_m"] = st.number_input("Top M", min_value=1, value=int(_current(raw, ("rerank", "top_m")) or 30), step=1, key="settings_rerank_top_m")
        if values["rerank_enabled"] and values["rerank_backend"] == "llm" and not llm_ready:
            st.warning("The llm rerank backend calls the LLM. Configure the LLM API key first.")
        if st.button("Save", key="settings_save_rerank", type="primary"):
            _save_section(st, service, "rerank", values)


def _render_storage_section(st: Any, service: ConfigService, raw: dict[str, Any], values: dict[str, Any]) -> None:
    with st.container(border=True):
        _section_title(st, "Storage & Logs")
        _saved_banner(st, "storage", "Storage settings saved.")
        values["persist_path"] = st.text_input("Vector store path", value=_current(raw, ("vector_store", "persist_path")) or "", key="settings_persist_path")
        values["collection"] = st.text_input("Collection", value=_current(raw, ("vector_store", "collection")) or "default", key="settings_collection")
        values["log_path"] = st.text_input("Log path", value=_current(raw, ("observability", "log_path")) or "", key="settings_log_path")
        values["trace_path"] = st.text_input("Trace path", value=_current(raw, ("observability", "trace_path")) or "", key="settings_trace_path")
        if st.button("Save", key="settings_save_storage", type="primary"):
            _save_section(st, service, "storage", values)


def _save_section(st: Any, service: ConfigService, section: str, values: dict[str, Any]) -> None:
    try:
        service.save(_apply_section(service.load_raw(), section, values))
    except Exception as error:
        st.error(f"Failed to save settings: {error}")
        return
    st.session_state[f"settings_saved_{section}"] = True
    st.session_state.pop("settings_llm_api_key", None)
    st.session_state.pop("settings_embedding_api_key", None)
    st.session_state.pop("settings_llm_test_result", None)
    st.rerun()


def _saved_banner(st: Any, section: str, message: str) -> None:
    if st.session_state.pop(f"settings_saved_{section}", False):
        st.success(message)


def _section_title(st: Any, title: str, extra_class: str = "") -> None:
    classes = f"sr-section-title {extra_class}".strip()
    st.markdown(f'<div class="{classes}">{title}</div>', unsafe_allow_html=True)


def _llm_ready(raw: dict[str, Any], values: dict[str, Any]) -> bool:
    provider = values.get("llm_provider") or _current(raw, ("llm", "provider"))
    if provider not in KEY_REQUIRED_PROVIDERS:
        return True
    return bool(values.get("llm_api_key")) or bool(values.get("_effective", {}).get("llm.api_key"))


def _step_option(st: Any, label: str, options: list[str], raw: dict[str, Any], path: tuple[str, str], labels: dict[str, str] | None = None) -> str:
    current = _current(raw, path)
    choices = list(options)
    if current and current not in choices:
        choices.append(current)
    format_func = (lambda option: labels.get(option, option)) if labels else None
    selected = st.segmented_control(label, choices, default=current or choices[0], format_func=format_func, key=f"settings_step_{path[0]}_{path[1]}")
    return selected or current or choices[0]


def _mapped_input(st: Any, label: str, raw: dict[str, Any], path: tuple[str, str], effective: str, placeholder: str = "", key: str = "") -> str:
    raw_value = _current(raw, path)
    reference = parse_env_reference(raw_value)
    display = effective if reference else (raw_value or "")
    return st.text_input(label, value=display or "", placeholder=placeholder, key=key) or ""


def _secret_input(st: Any, raw: dict[str, Any], path: tuple[str, str], resolved: str, key: str) -> str:
    saved = _current(raw, path)
    reference = parse_env_reference(saved)
    configured = bool(resolved) or bool(isinstance(saved, str) and saved and not reference)
    value = st.text_input("API Key", type="password", value=KEY_MASK if configured else "", placeholder="Paste your API key", key=key)
    if not value or value == KEY_MASK:
        return ""
    return value


def _current(raw: dict[str, Any], path: tuple[str, ...]) -> Any:
    section: Any = raw
    for key in path:
        if not isinstance(section, dict):
            return None
        section = section.get(key)
    return section


def _ensure(section: dict[str, Any], key: str) -> dict[str, Any]:
    value = section.get(key)
    if not isinstance(value, dict):
        value = {}
        section[key] = value
    return value


def _apply_pipeline(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    _ensure(updated, "llm")["provider"] = values["llm_provider"]
    _ensure(updated, "embedding")["provider"] = values["embedding_provider"]
    _ensure(updated, "vector_store")["backend"] = values["vector_backend"]
    _ensure(updated, "splitter")["provider"] = values["splitter_provider"]
    retrieval = _ensure(updated, "retrieval")
    retrieval["sparse_backend"] = values["sparse_backend"]
    retrieval["fusion_algorithm"] = values["fusion_algorithm"]
    _ensure(updated, "rerank")["backend"] = values["rerank_backend"]


def _write_field(section: dict[str, Any], key: str, raw_section: dict[str, Any], new_value: str, effective_value: Any) -> None:
    current = raw_section.get(key)
    if effective_value is not None and isinstance(current, str) and parse_env_reference(current) and new_value == effective_value:
        return
    section[key] = new_value


def _apply_llm(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    llm = _ensure(updated, "llm")
    llm["provider"] = values["llm_provider"]
    raw_llm = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
    effective = values.get("_effective", {})
    _write_field(llm, "model", raw_llm, values["llm_model"], effective.get("llm.model"))
    _write_field(llm, "base_url", raw_llm, values["llm_base_url"], effective.get("llm.base_url"))
    if values["llm_api_key"]:
        llm["api_key"] = values["llm_api_key"]


def _apply_embedding(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    embedding = _ensure(updated, "embedding")
    embedding["provider"] = values["embedding_provider"]
    raw_embedding = raw.get("embedding") if isinstance(raw.get("embedding"), dict) else {}
    effective = values.get("_effective", {})
    _write_field(embedding, "model", raw_embedding, values["embedding_model"], effective.get("embedding.model"))
    embedding["dimensions"] = int(values["embedding_dimensions"])
    if values["embedding_provider"] != "local":
        _write_field(embedding, "base_url", raw_embedding, values["embedding_base_url"], effective.get("embedding.base_url"))
    if values.get("embedding_api_key"):
        embedding["api_key"] = values["embedding_api_key"]


def _apply_ingestion(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    ingestion = _ensure(updated, "ingestion")
    _ensure(ingestion, "chunk_refiner")["use_llm"] = bool(values["use_chunk_refiner"])
    _ensure(ingestion, "metadata_enricher")["use_llm"] = bool(values["use_metadata_enricher"])
    _ensure(ingestion, "image_captioner")["enabled"] = bool(values["use_image_captioner"])


def _apply_splitter(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    splitter = _ensure(updated, "splitter")
    splitter["provider"] = values["splitter_provider"]
    splitter["chunk_size"] = int(values["chunk_size"])
    splitter["chunk_overlap"] = int(values["chunk_overlap"])


def _apply_retrieval(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    retrieval = _ensure(updated, "retrieval")
    retrieval["sparse_backend"] = values["sparse_backend"]
    retrieval["fusion_algorithm"] = values["fusion_algorithm"]
    retrieval["top_k_dense"] = int(values["top_k_dense"])
    retrieval["top_k_sparse"] = int(values["top_k_sparse"])
    retrieval["top_k_final"] = int(values["top_k_final"])


def _apply_rerank(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    rerank = _ensure(updated, "rerank")
    rerank["enabled"] = bool(values["rerank_enabled"])
    rerank["backend"] = values["rerank_backend"]
    rerank["top_m"] = int(values["rerank_top_m"])


def _apply_storage(updated: dict[str, Any], values: dict[str, Any], raw: dict[str, Any]) -> None:
    vector_store = _ensure(updated, "vector_store")
    vector_store["backend"] = values["vector_backend"]
    vector_store["persist_path"] = values["persist_path"]
    vector_store["collection"] = values["collection"]
    observability = _ensure(updated, "observability")
    observability["log_path"] = values["log_path"]
    observability["trace_path"] = values["trace_path"]


SECTION_APPLIERS = {
    "pipeline": _apply_pipeline,
    "llm": _apply_llm,
    "embedding": _apply_embedding,
    "ingestion": _apply_ingestion,
    "splitter": _apply_splitter,
    "retrieval": _apply_retrieval,
    "rerank": _apply_rerank,
    "storage": _apply_storage,
}


def _apply_section(raw: dict[str, Any], section: str, values: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(raw)
    SECTION_APPLIERS[section](updated, values, raw)
    return updated


def _build_raw(raw: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(raw)
    for apply_section in SECTION_APPLIERS.values():
        apply_section(updated, values, raw)
    return updated
