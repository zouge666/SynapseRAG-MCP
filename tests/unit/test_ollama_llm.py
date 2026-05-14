from typing import Any
from urllib.error import URLError

import pytest

from core.settings import LLMSettings
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import OllamaLLM, OllamaLLMError


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response or {"message": {"content": "local ok"}}
        self.error = error
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        if self.error:
            raise self.error
        return self.response


def test_factory_routes_ollama_provider() -> None:
    llm = LLMFactory.create(LLMSettings(provider="ollama", model="llama3"))

    assert isinstance(llm, OllamaLLM)


def test_ollama_chat_uses_default_endpoint_and_payload() -> None:
    transport = FakeTransport()
    llm = OllamaLLM(LLMSettings(provider="ollama", model="llama3"), transport=transport)

    result = llm.chat([{"role": "user", "content": "hello"}])

    url, headers, payload, timeout = transport.calls[0]
    assert result == "local ok"
    assert url == "http://localhost:11434/api/chat"
    assert headers == {"Content-Type": "application/json"}
    assert payload == {"model": "llama3", "messages": [{"role": "user", "content": "hello"}], "stream": False}
    assert timeout == 30.0


def test_ollama_chat_uses_configured_base_url() -> None:
    transport = FakeTransport()
    llm = OllamaLLM(
        LLMSettings(provider="ollama", model="llama3", base_url="http://127.0.0.1:11435/"),
        transport=transport,
    )

    result = llm.chat([{"role": "user", "content": "hello"}])

    assert result == "local ok"
    assert transport.calls[0][0] == "http://127.0.0.1:11435/api/chat"


def test_ollama_connection_error_is_readable_and_does_not_leak_config() -> None:
    transport = FakeTransport(error=URLError("connection refused"))
    llm = OllamaLLM(
        LLMSettings(provider="ollama", model="secret-model", api_key="secret-key", base_url="http://private-host:11434"),
        transport=transport,
    )

    with pytest.raises(OllamaLLMError) as error:
        llm.chat([{"role": "user", "content": "hello"}])

    message = str(error.value)
    assert "ollama connection error" in message
    assert "secret-key" not in message
    assert "private-host" not in message


def test_ollama_validation_error_is_readable() -> None:
    llm = OllamaLLM(LLMSettings(provider="ollama", model="llama3"), transport=FakeTransport())

    with pytest.raises(OllamaLLMError, match="ollama validation error"):
        llm.chat([{"role": "user"}])


def test_ollama_response_error_is_readable() -> None:
    llm = OllamaLLM(LLMSettings(provider="ollama", model="llama3"), transport=FakeTransport({"message": {}}))

    with pytest.raises(OllamaLLMError, match="ollama response error"):
        llm.chat([{"role": "user", "content": "hello"}])
