from typing import Any

import pytest

from core.settings import LLMSettings
from libs.llm.azure_llm import AzureOpenAILLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAICompatibleLLMError, OpenAILLM


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"choices": [{"message": {"content": "ok"}}]}
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        return self.response


def test_factory_routes_openai_azure_and_deepseek() -> None:
    assert isinstance(LLMFactory.create(LLMSettings(provider="openai", model="gpt-4o")), OpenAILLM)
    assert isinstance(
        LLMFactory.create(
            LLMSettings(
                provider="azure",
                model="gpt-4o",
                azure_endpoint="https://example.openai.azure.com",
                deployment_name="chat",
            )
        ),
        AzureOpenAILLM,
    )
    assert isinstance(LLMFactory.create(LLMSettings(provider="deepseek", model="deepseek-chat")), DeepSeekLLM)


def test_openai_chat_uses_openai_compatible_payload() -> None:
    transport = FakeTransport()
    llm = OpenAILLM(
        LLMSettings(provider="openai", model="gpt-4o", api_key="secret"),
        transport=transport,
    )

    result = llm.chat([{"role": "user", "content": "hello"}])

    url, headers, payload, timeout = transport.calls[0]
    assert result == "ok"
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert payload == {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    assert timeout == 30.0


def test_azure_chat_uses_deployment_endpoint_and_api_key_header() -> None:
    transport = FakeTransport()
    llm = AzureOpenAILLM(
        LLMSettings(
            provider="azure",
            model="gpt-4o",
            api_key="secret",
            azure_endpoint="https://example.openai.azure.com/",
            api_version="2024-05-01-preview",
            deployment_name="chat deployment",
        ),
        transport=transport,
    )

    result = llm.chat([{"role": "user", "content": "hello"}])

    url, headers, payload, _ = transport.calls[0]
    assert result == "ok"
    assert url == "https://example.openai.azure.com/openai/deployments/chat%20deployment/chat/completions?api-version=2024-05-01-preview"
    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers
    assert payload["model"] == "gpt-4o"


def test_deepseek_uses_default_deepseek_base_url() -> None:
    transport = FakeTransport()
    llm = DeepSeekLLM(LLMSettings(provider="deepseek", model="deepseek-chat"), transport=transport)

    result = llm.chat([{"role": "user", "content": "hello"}])

    assert result == "ok"
    assert transport.calls[0][0] == "https://api.deepseek.com/v1/chat/completions"


def test_chat_validation_error_mentions_provider_and_error_type() -> None:
    llm = OpenAILLM(LLMSettings(provider="openai", model="gpt-4o"), transport=FakeTransport())

    with pytest.raises(OpenAICompatibleLLMError, match="openai validation error"):
        llm.chat([{"role": "user"}])


def test_chat_response_error_mentions_provider_and_error_type() -> None:
    llm = OpenAILLM(LLMSettings(provider="openai", model="gpt-4o"), transport=FakeTransport({"choices": []}))

    with pytest.raises(OpenAICompatibleLLMError, match="openai response error"):
        llm.chat([{"role": "user", "content": "hello"}])
