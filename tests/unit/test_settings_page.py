import copy

import pytest

from core.settings import SettingsError
from observability.dashboard.pages.settings_page import (
    EMBEDDING_PROVIDERS,
    KEY_MASK,
    _apply_section,
    _build_raw,
    _llm_ready,
    _secret_input,
)
from observability.dashboard.services.config_service import ConfigService, parse_env_reference


BASE_RAW = {
    "app": {"name": "synapserag-mcp", "environment": "local"},
    "llm": {"provider": "openai", "model": "gpt-4o", "api_key": "sk-saved-123456"},
    "embedding": {"provider": "local", "model": "local-hash", "dimensions": 128},
    "vector_store": {"backend": "chroma", "persist_path": "data/db/chroma", "collection": "default"},
    "splitter": {"provider": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
    "ingestion": {
        "chunk_refiner": {"use_llm": False},
        "metadata_enricher": {"use_llm": False},
        "image_captioner": {"enabled": False},
    },
    "retrieval": {"sparse_backend": "bm25", "fusion_algorithm": "rrf", "top_k_dense": 20, "top_k_sparse": 20, "top_k_final": 5},
    "rerank": {"enabled": False, "backend": "none", "top_m": 30},
    "evaluation": {"enabled": False, "backends": ["custom_metrics"]},
    "observability": {"log_path": "logs/app.log", "trace_path": "logs/traces.jsonl"},
}


def make_values(**overrides):
    values = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "llm_base_url": "",
        "llm_api_key": "",
        "embedding_provider": "local",
        "embedding_model": "local-hash",
        "embedding_dimensions": 128,
        "embedding_base_url": "",
        "embedding_api_key": "",
        "vector_backend": "chroma",
        "splitter_provider": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "sparse_backend": "bm25",
        "fusion_algorithm": "rrf",
        "top_k_dense": 20,
        "top_k_sparse": 20,
        "top_k_final": 5,
        "rerank_backend": "none",
        "rerank_enabled": False,
        "rerank_top_m": 30,
        "use_chunk_refiner": False,
        "use_metadata_enricher": False,
        "use_image_captioner": False,
        "persist_path": "data/db/chroma",
        "collection": "default",
        "log_path": "logs/app.log",
        "trace_path": "logs/traces.jsonl",
    }
    values.update(overrides)
    return values


def test_build_raw_keeps_saved_api_key_when_input_is_empty() -> None:
    updated = _build_raw(BASE_RAW, make_values())

    assert updated["llm"]["api_key"] == "sk-saved-123456"


def test_build_raw_replaces_api_key_only_with_new_input() -> None:
    updated = _build_raw(BASE_RAW, make_values(llm_api_key="sk-new-abcdef"))

    assert updated["llm"]["api_key"] == "sk-new-abcdef"
    assert BASE_RAW["llm"]["api_key"] == "sk-saved-123456"


def test_build_raw_applies_pipeline_and_toggle_choices() -> None:
    values = make_values(
        llm_provider="anthropic",
        llm_model="claude-haiku-4-5-20251001",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        use_chunk_refiner=True,
        rerank_backend="llm",
        rerank_enabled=True,
        top_k_final=8,
    )

    updated = _build_raw(BASE_RAW, values)

    assert updated["llm"]["provider"] == "anthropic"
    assert updated["embedding"]["provider"] == "openai"
    assert updated["ingestion"]["chunk_refiner"]["use_llm"] is True
    assert updated["rerank"]["backend"] == "llm"
    assert updated["rerank"]["enabled"] is True
    assert updated["retrieval"]["top_k_final"] == 8


def test_apply_section_updates_only_the_target_section() -> None:
    updated = _apply_section(BASE_RAW, "llm", make_values(llm_model="gpt-4o-mini", top_k_final=99))

    assert updated["llm"]["model"] == "gpt-4o-mini"
    assert updated["retrieval"]["top_k_final"] == 5


def test_apply_section_keeps_saved_api_key_for_other_sections() -> None:
    updated = _apply_section(BASE_RAW, "retrieval", make_values(top_k_final=8))

    assert updated["retrieval"]["top_k_final"] == 8
    assert updated["llm"]["api_key"] == "sk-saved-123456"
    assert BASE_RAW["retrieval"]["top_k_final"] == 5


def test_apply_pipeline_section_updates_all_step_choices() -> None:
    updated = _apply_section(BASE_RAW, "pipeline", make_values(llm_provider="anthropic", rerank_backend="cross_encoder"))

    assert updated["llm"]["provider"] == "anthropic"
    assert updated["rerank"]["backend"] == "cross_encoder"
    assert updated["llm"]["model"] == "gpt-4o"


def test_apply_llm_preserves_env_reference_when_value_matches_effective() -> None:
    raw = copy.deepcopy(BASE_RAW)
    raw["llm"]["model"] = "${LLM_MODEL:-gpt-4o}"
    values = make_values(llm_model="gpt-4o")
    values["_effective"] = {"llm.model": "gpt-4o", "llm.base_url": ""}

    updated = _apply_section(raw, "llm", values)

    assert updated["llm"]["model"] == "${LLM_MODEL:-gpt-4o}"


def test_apply_llm_pins_literal_value_when_user_edits_mapped_field() -> None:
    raw = copy.deepcopy(BASE_RAW)
    raw["llm"]["model"] = "${LLM_MODEL:-gpt-4o}"
    values = make_values(llm_model="gpt-4o-mini")
    values["_effective"] = {"llm.model": "gpt-4o", "llm.base_url": ""}

    updated = _apply_section(raw, "llm", values)

    assert updated["llm"]["model"] == "gpt-4o-mini"


def test_embedding_providers_are_local_openai_ollama() -> None:
    assert EMBEDDING_PROVIDERS == ["local", "openai", "ollama"]


def test_apply_embedding_ollama_writes_base_url_without_api_key_field() -> None:
    values = make_values(embedding_provider="ollama", embedding_model="nomic-embed-text", embedding_base_url="http://localhost:11434")
    values.pop("embedding_api_key")

    updated = _apply_section(BASE_RAW, "embedding", values)

    assert updated["embedding"]["provider"] == "ollama"
    assert updated["embedding"]["base_url"] == "http://localhost:11434"
    assert "api_key" not in updated["embedding"]


def test_apply_embedding_local_skips_connection_fields() -> None:
    values = make_values(embedding_provider="local", embedding_model="local-hash", embedding_dimensions=256)
    values.pop("embedding_base_url")
    values.pop("embedding_api_key")

    updated = _apply_section(BASE_RAW, "embedding", values)

    assert updated["embedding"]["provider"] == "local"
    assert updated["embedding"]["dimensions"] == 256
    assert "base_url" not in updated["embedding"]
    assert "api_key" not in updated["embedding"]


def test_mask_secret_hides_the_middle_of_a_key() -> None:
    assert ConfigService.mask_secret("sk-abcdef123456") == "sk-a****3456"
    assert ConfigService.mask_secret("short") == "****"
    assert ConfigService.mask_secret("") == ""


def test_save_writes_valid_settings(tmp_path) -> None:
    settings_path = tmp_path / "settings.yaml"
    import yaml

    settings_path.write_text(yaml.safe_dump(BASE_RAW), encoding="utf-8")
    service = ConfigService(settings_path)

    service.save(_build_raw(BASE_RAW, make_values(llm_model="gpt-4o-mini")))

    assert service.load().llm.model == "gpt-4o-mini"


def test_save_rolls_back_when_validation_fails(tmp_path) -> None:
    settings_path = tmp_path / "settings.yaml"
    import yaml

    settings_path.write_text(yaml.safe_dump(BASE_RAW), encoding="utf-8")
    service = ConfigService(settings_path)

    with pytest.raises(SettingsError):
        service.save(_build_raw(BASE_RAW, make_values(llm_model="")))

    assert service.load().llm.model == "gpt-4o"


def test_parse_env_reference_extracts_variable_and_default() -> None:
    assert parse_env_reference("${LLM_MODEL:-deepseek-v4-flash}") == ("LLM_MODEL", "deepseek-v4-flash")
    assert parse_env_reference("${API_KEY}") == ("API_KEY", None)
    assert parse_env_reference("${EMBEDDING_BASE_URL:-}") == ("EMBEDDING_BASE_URL", "")
    assert parse_env_reference("deepseek-v4-flash") is None
    assert parse_env_reference("") is None
    assert parse_env_reference(None) is None
    assert parse_env_reference(42) is None


class _FakeSt:
    def __init__(self, text_value: str) -> None:
        self.text_value = text_value
        self.captured: dict = {}

    def text_input(self, label, **kwargs):
        self.captured = {"label": label, **kwargs}
        return self.text_value


def test_secret_input_shows_mask_dots_when_key_is_saved() -> None:
    st = _FakeSt(KEY_MASK)
    raw = {"llm": {"api_key": "sk-saved-123456"}}

    assert _secret_input(st, raw, ("llm", "api_key"), "sk-saved-123456", "k") == ""
    assert st.captured["value"] == KEY_MASK
    assert st.captured["type"] == "password"


def test_secret_input_shows_mask_dots_when_key_comes_from_env(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "sk-real")
    st = _FakeSt(KEY_MASK)
    raw = {"llm": {"api_key": "${API_KEY}"}}

    assert _secret_input(st, raw, ("llm", "api_key"), "sk-real", "k") == ""
    assert st.captured["value"] == KEY_MASK


def test_secret_input_is_empty_when_no_key_configured() -> None:
    st = _FakeSt("")

    assert _secret_input(st, {"llm": {}}, ("llm", "api_key"), "", "k") == ""
    assert st.captured["value"] == ""


def test_secret_input_returns_new_key_when_user_types_one() -> None:
    st = _FakeSt("sk-new-abcdef")
    raw = {"llm": {"api_key": "sk-saved-123456"}}

    assert _secret_input(st, raw, ("llm", "api_key"), "sk-saved-123456", "k") == "sk-new-abcdef"


def test_llm_ready_uses_resolved_key_not_raw_placeholder() -> None:
    raw = {"llm": {"provider": "openai", "api_key": "${API_KEY}"}}

    assert _llm_ready(raw, {"llm_provider": "openai", "_effective": {"llm.api_key": ""}}) is False
    assert _llm_ready(raw, {"llm_provider": "openai", "_effective": {"llm.api_key": "sk-x"}}) is True
    assert _llm_ready(raw, {"llm_provider": "openai", "_effective": {"llm.api_key": "sk-x"}, "llm_api_key": "sk-typed"}) is True
