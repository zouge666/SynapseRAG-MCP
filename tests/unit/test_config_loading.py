from pathlib import Path

import pytest
import yaml

from core.settings import Settings, SettingsError, load_settings


def test_load_settings_reads_default_config() -> None:
    settings = load_settings("config/settings.yaml")

    assert isinstance(settings, Settings)
    assert settings.app.name == "synapserag-mcp"
    assert settings.embedding.provider == "openai"
    assert settings.vector_store.backend == "chroma"
    assert settings.retrieval.top_k_final == 5


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
