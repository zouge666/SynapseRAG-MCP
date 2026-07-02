from typing import Any

import pytest

from core.settings import LLMSettings
from libs.llm.anthropic_llm import AnthropicLLM, AnthropicLLMError
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAICompatibleLLMError, OpenAILLM


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"choices": [{"message": {"content": "ok"}}]}
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        return self.response


def test_factory_routes_openai_and_anthropic() -> None:
    assert isinstance(LLMFactory.create(LLMSettings(provider="openai", model="gpt-4o")), OpenAILLM)
    assert isinstance(LLMFactory.create(LLMSettings(provider="anthropic", model="claude-haiku-4-5-20251001")), AnthropicLLM)


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


def test_openai_chat_uses_custom_base_url_for_compatible_endpoints() -> None:
    transport = FakeTransport()
    llm = OpenAILLM(
        LLMSettings(provider="openai", model="deepseek-v4-flash", api_key="secret", base_url="https://api.deepseek.com/"),
        transport=transport,
    )

    result = llm.chat([{"role": "user", "content": "hello"}])

    assert result == "ok"
    assert transport.calls[0][0] == "https://api.deepseek.com/chat/completions"


def test_anthropic_chat_uses_messages_api_payload() -> None:
    transport = FakeTransport({"content": [{"type": "text", "text": "ok"}]})
    llm = AnthropicLLM(
        LLMSettings(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="secret"),
        transport=transport,
    )

    result = llm.chat([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}])

    url, headers, payload, timeout = transport.calls[0]
    assert result == "ok"
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["system"] == "be brief"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] > 0
    assert timeout == 30.0


def test_anthropic_chat_uses_custom_base_url_for_relays() -> None:
    transport = FakeTransport({"content": [{"type": "text", "text": "ok"}]})
    llm = AnthropicLLM(
        LLMSettings(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="secret", base_url="https://relay.example.com/"),
        transport=transport,
    )

    llm.chat([{"role": "user", "content": "hello"}])

    assert transport.calls[0][0] == "https://relay.example.com/v1/messages"


def test_anthropic_chat_requires_a_conversation_message() -> None:
    llm = AnthropicLLM(LLMSettings(provider="anthropic", model="claude-haiku-4-5-20251001"), transport=FakeTransport())

    with pytest.raises(AnthropicLLMError, match="at least one user or assistant message"):
        llm.chat([{"role": "system", "content": "be brief"}])


def test_anthropic_response_error_on_missing_text_blocks() -> None:
    llm = AnthropicLLM(
        LLMSettings(provider="anthropic", model="claude-haiku-4-5-20251001"),
        transport=FakeTransport({"content": [{"type": "tool_use", "name": "x"}]}),
    )

    with pytest.raises(AnthropicLLMError, match="no text content"):
        llm.chat([{"role": "user", "content": "hello"}])


def test_chat_validation_error_mentions_provider_and_error_type() -> None:
    llm = OpenAILLM(LLMSettings(provider="openai", model="gpt-4o"), transport=FakeTransport())

    with pytest.raises(OpenAICompatibleLLMError, match="openai validation error"):
        llm.chat([{"role": "user"}])


def test_chat_response_error_mentions_provider_and_error_type() -> None:
    llm = OpenAILLM(LLMSettings(provider="openai", model="gpt-4o"), transport=FakeTransport({"choices": []}))

    with pytest.raises(OpenAICompatibleLLMError, match="openai response error"):
        llm.chat([{"role": "user", "content": "hello"}])
