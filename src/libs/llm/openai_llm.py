from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


LLMTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class OpenAICompatibleLLMError(RuntimeError):
    pass


class OpenAILLM(BaseLLM):
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, settings: LLMSettings, transport: LLMTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings)
        self.transport = transport or self._default_transport
        self.timeout = timeout

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self._validate_messages(messages)
        payload = {"model": self.settings.model, "messages": list(messages)}
        response = self._send(self._chat_url(), self._headers(), payload)
        return self._extract_content(response)

    def _chat_url(self) -> str:
        return f"{self._base_url()}/chat/completions"

    def _base_url(self) -> str:
        return (self.settings.base_url or self.default_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport(url, headers, payload, self.timeout)
        except OpenAICompatibleLLMError:
            raise
        except HTTPError as error:
            raise OpenAICompatibleLLMError(f"{self.settings.provider} http error: {error.code}") from error
        except URLError as error:
            raise OpenAICompatibleLLMError(f"{self.settings.provider} connection error: {error.reason}") from error
        except OSError as error:
            raise OpenAICompatibleLLMError(f"{self.settings.provider} transport error: {type(error).__name__}") from error

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
            raise OpenAICompatibleLLMError(f"{self.settings.provider} response error: expected object")
        return parsed

    def _extract_content(self, response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenAICompatibleLLMError(f"{self.settings.provider} response error: missing choices[0].message.content") from error
        if not isinstance(content, str):
            raise OpenAICompatibleLLMError(f"{self.settings.provider} response error: content must be string")
        return content

    def _validate_messages(self, messages: Sequence[Mapping[str, str]]) -> None:
        if isinstance(messages, (str, bytes)) or not messages:
            raise OpenAICompatibleLLMError(f"{self.settings.provider} validation error: messages must be a non-empty sequence")
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise OpenAICompatibleLLMError(f"{self.settings.provider} validation error: messages[{index}] must be a mapping")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise OpenAICompatibleLLMError(f"{self.settings.provider} validation error: messages[{index}].role is required")
            if not isinstance(content, str):
                raise OpenAICompatibleLLMError(f"{self.settings.provider} validation error: messages[{index}].content must be string")


LLMFactory.register_provider("openai", OpenAILLM)
