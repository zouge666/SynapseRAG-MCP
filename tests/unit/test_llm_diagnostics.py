import pytest

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from observability.dashboard.services.llm_diagnostics import test_llm_connection as run_llm_test


class FakeLLM(BaseLLM):
    reply = "ok"
    error: Exception | None = None

    def chat(self, messages):
        if FakeLLM.error is not None:
            raise FakeLLM.error
        return FakeLLM.reply


@pytest.fixture
def fake_provider():
    LLMFactory.register_provider("fake", FakeLLM)
    FakeLLM.error = None
    FakeLLM.reply = "ok"
    yield
    LLMFactory.unregister_provider("fake")


def test_missing_api_key_short_circuits_without_network() -> None:
    result = run_llm_test(provider="openai", model="gpt-4o", api_key="")

    assert result.ok is False
    assert "API key" in result.summary


def test_missing_model_is_reported() -> None:
    result = run_llm_test(provider="openai", model="")

    assert result.ok is False
    assert "Model is empty" in result.summary


def test_successful_connection_reports_latency(fake_provider) -> None:
    result = run_llm_test(provider="fake", model="fake-1", api_key="sk-x")

    assert result.ok is True
    assert "fake-1" in result.summary
    assert result.latency_ms >= 0


def test_http_401_maps_to_auth_failure(fake_provider) -> None:
    FakeLLM.error = RuntimeError("fake http error: 401")

    result = run_llm_test(provider="fake", model="fake-1", api_key="sk-bad")

    assert result.ok is False
    assert "Authentication failed" in result.summary
    assert "401" in result.detail


def test_http_404_maps_to_not_found(fake_provider) -> None:
    FakeLLM.error = RuntimeError("fake http error: 404")

    result = run_llm_test(provider="fake", model="missing-model", api_key="sk-x")

    assert result.ok is False
    assert "not found" in result.summary.lower()


def test_connection_error_maps_to_unreachable(fake_provider) -> None:
    FakeLLM.error = RuntimeError("fake connection error: Connection refused")

    result = run_llm_test(provider="fake", model="fake-1", api_key="sk-x")

    assert result.ok is False
    assert "Cannot reach the server" in result.summary


def test_anthropic_missing_key_short_circuits() -> None:
    result = run_llm_test(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="")

    assert result.ok is False
    assert "API key" in result.summary


def test_unknown_provider_is_reported() -> None:
    result = run_llm_test(provider="nonexistent-provider", model="m", api_key="k")

    assert result.ok is False
    assert "unsupported" in result.summary.lower()
