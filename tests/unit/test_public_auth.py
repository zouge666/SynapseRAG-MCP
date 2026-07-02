from core.settings import LLMSettings, Settings
from observability.dashboard.public_pages.common import owner_llm_settings, verify_admin_password
from observability.dashboard.public_pages.start_page import _resolve_admin_llm


SECRETS = {
    "ADMIN_PASSWORD": "correct-horse-battery-staple",
    "OWNER_LLM_PROVIDER": "openai",
    "OWNER_LLM_MODEL": "owner-model",
    "OWNER_LLM_BASE_URL": "https://api.example.com/v1",
    "OWNER_LLM_API_KEY": "sk-owner-secret-000",
}


def test_admin_password_accepts_correct_credentials() -> None:
    assert verify_admin_password("admin", "correct-horse-battery-staple", SECRETS) is True


def test_admin_password_rejects_wrong_password() -> None:
    assert verify_admin_password("admin", "wrong", SECRETS) is False


def test_admin_password_rejects_wrong_username_and_missing_secret() -> None:
    assert verify_admin_password("root", "correct-horse-battery-staple", SECRETS) is False
    assert verify_admin_password("admin", "anything", {}) is False


def test_owner_llm_settings_reads_secrets_without_exposing_key() -> None:
    llm = owner_llm_settings(SECRETS)

    assert llm is not None
    assert llm.provider == "openai"
    assert llm.model == "owner-model"
    assert llm.api_key == "sk-owner-secret-000"


def test_owner_llm_settings_missing_values_returns_none() -> None:
    assert owner_llm_settings({}) is None
    assert owner_llm_settings({"OWNER_LLM_PROVIDER": "openai"}) is None


def _base_settings_with_key(api_key: str) -> Settings:
    from types import SimpleNamespace

    return SimpleNamespace(llm=LLMSettings(provider="openai", model="base-model", base_url="https://api.deepseek.com/v1", api_key=api_key))


def test_resolve_admin_llm_prefers_secrets() -> None:
    llm = _resolve_admin_llm(SECRETS, _base_settings_with_key("sk-base"))
    assert llm is not None and llm.model == "owner-model"


def test_resolve_admin_llm_falls_back_to_base_config() -> None:
    llm = _resolve_admin_llm({}, _base_settings_with_key("sk-deepseek-from-env"))
    assert llm is not None and llm.api_key == "sk-deepseek-from-env"


def test_resolve_admin_llm_none_when_nothing_configured() -> None:
    assert _resolve_admin_llm({}, _base_settings_with_key("")) is None


def test_owner_embedding_settings_reads_full_config() -> None:
    from observability.dashboard.public_pages.common import owner_embedding_settings

    embedding = owner_embedding_settings({
        "OWNER_EMBEDDING_PROVIDER": "openai",
        "OWNER_EMBEDDING_MODEL": "BAAI/bge-m3",
        "OWNER_EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
        "OWNER_EMBEDDING_API_KEY": "sk-owner-emb-000",
        "OWNER_EMBEDDING_DIMENSIONS": "1024",
    })
    assert embedding is not None
    assert embedding.provider == "openai"
    assert embedding.model == "BAAI/bge-m3"
    assert embedding.dimensions == 1024
    assert embedding.api_key == "sk-owner-emb-000"


def test_owner_embedding_settings_missing_or_partial() -> None:
    from observability.dashboard.public_pages.common import owner_embedding_settings

    assert owner_embedding_settings({}) is None
    assert owner_embedding_settings({"OWNER_EMBEDDING_PROVIDER": "openai"}) is None
    embedding = owner_embedding_settings({"OWNER_EMBEDDING_PROVIDER": "openai", "OWNER_EMBEDDING_MODEL": "m"})
    assert embedding is not None and embedding.dimensions is None and embedding.api_key == ""
