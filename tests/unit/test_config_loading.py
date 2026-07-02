from pathlib import Path

import pytest
import yaml

from core.settings import Settings, SettingsError, _ensure_dotenv, _expand_env_vars, load_settings


FULL_CONFIG = """
app:
  name: synapserag-mcp
  environment: local
llm:
  provider: openai
  model: gpt-4o
embedding:
  provider: local
  model: local-hash
  dimensions: 128
vector_store:
  backend: chroma
  persist_path: data/db/chroma
splitter:
  provider: recursive
  chunk_size: 1000
  chunk_overlap: 200
ingestion:
  chunk_refiner:
    use_llm: false
  image_captioner:
    enabled: false
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 20
  top_k_sparse: 20
  top_k_final: 5
rerank:
  enabled: false
  backend: none
evaluation:
  enabled: false
  backends: []
observability:
  log_path: logs/app.log
  trace_path: logs/traces.jsonl
"""


def test_load_settings_reads_full_config(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(FULL_CONFIG, encoding="utf-8")

    settings = load_settings(str(config_path))

    assert isinstance(settings, Settings)
    assert settings.app.name == "synapserag-mcp"
    assert settings.embedding.provider == "local"
    assert settings.embedding.model == "local-hash"
    assert settings.embedding.dimensions == 128
    assert settings.vector_store.backend == "chroma"
    assert settings.splitter.provider == "recursive"
    assert settings.splitter.chunk_size == 1000
    assert settings.ingestion.chunk_refiner.use_llm is False
    assert settings.ingestion.image_captioner.enabled is False
    assert settings.retrieval.top_k_final == 5


def test_load_settings_parses_shipped_config() -> None:
    settings = load_settings("config/settings.yaml")

    assert isinstance(settings, Settings)
    assert settings.app.name
    assert settings.llm.provider
    assert settings.embedding.provider
    assert settings.vector_store.backend


def test_load_settings_reports_missing_required_field(tmp_path: Path) -> None:
    data = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    del data["embedding"]["provider"]
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(str(config_path))


def test_load_settings_reports_missing_file() -> None:
    with pytest.raises(SettingsError, match="settings file not found"):
        load_settings("config/missing.yaml")


def test_expand_env_vars_uses_inline_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert _expand_env_vars({"model": "${LLM_MODEL:-deepseek-v4-flash}"})["model"] == "deepseek-v4-flash"


def test_expand_env_vars_prefers_environment_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "custom-model")

    assert _expand_env_vars({"model": "${LLM_MODEL:-deepseek-v4-flash}"})["model"] == "custom-model"


def test_expand_env_vars_without_default_expands_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)

    assert _expand_env_vars("${API_KEY}") == ""


def test_ensure_dotenv_creates_template_only_when_missing(tmp_path: Path) -> None:
    target = tmp_path / ".env"

    _ensure_dotenv(target)
    assert target.is_file()
    assert "API_KEY" in target.read_text(encoding="utf-8")

    target.write_text("CUSTOM=1\n", encoding="utf-8")
    _ensure_dotenv(target)
    assert target.read_text(encoding="utf-8") == "CUSTOM=1\n"
