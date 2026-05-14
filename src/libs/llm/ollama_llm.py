from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


OllamaTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class OllamaLLMError(RuntimeError):
    pass


class OllamaLLM(BaseLLM):
    default_base_url = "http://localhost:11434"

    def __init__(self, settings: LLMSettings, transport: OllamaTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings)
        self.transport = transport or self._default_transport
        self.timeout = timeout

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self._validate_messages(messages)
        payload = {"model": self.settings.model, "messages": list(messages), "stream": False}
        response = self._send(self._chat_url(), {"Content-Type": "application/json"}, payload)
        return self._extract_content(response)

    def _chat_url(self) -> str:
        return f"{self._base_url()}/api/chat"

    def _base_url(self) -> str:
        return (self.settings.base_url or self.default_base_url).rstrip("/")

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport(url, headers, payload, self.timeout)
        except OllamaLLMError:
            raise
        except HTTPError as error:
            raise OllamaLLMError(f"ollama http error: {error.code}") from error
        except URLError as error:
            raise OllamaLLMError(f"ollama connection error: {error.reason}") from error
        except OSError as error:
            raise OllamaLLMError(f"ollama transport error: {type(error).__name__}") from error

    def _default_transport(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise OllamaLLMError("ollama response error: expected object")
        return parsed

    def _extract_content(self, response: dict[str, Any]) -> str:
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as error:
            raise OllamaLLMError("ollama response error: missing message.content") from error
        if not isinstance(content, str):
            raise OllamaLLMError("ollama response error: content must be string")
        return content

    def _validate_messages(self, messages: Sequence[Mapping[str, str]]) -> None:
        if isinstance(messages, (str, bytes)) or not messages:
            raise OllamaLLMError("ollama validation error: messages must be a non-empty sequence")
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise OllamaLLMError(f"ollama validation error: messages[{index}] must be a mapping")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise OllamaLLMError(f"ollama validation error: messages[{index}].role is required")
            if not isinstance(content, str):
                raise OllamaLLMError(f"ollama validation error: messages[{index}].content must be string")


LLMFactory.register_provider("ollama", OllamaLLM)
