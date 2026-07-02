from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import LLMTransport


class AnthropicLLMError(RuntimeError):
    pass


class AnthropicLLM(BaseLLM):
    default_base_url = "https://api.anthropic.com"
    api_version = "2023-06-01"
    default_max_tokens = 1024

    def __init__(self, settings: LLMSettings, transport: LLMTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings)
        self.transport = transport or self._default_transport
        self.timeout = timeout

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self._validate_messages(messages)
        system, converted = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.default_max_tokens,
            "messages": converted,
        }
        if system:
            payload["system"] = system
        response = self._send(self._messages_url(), self._headers(), payload)
        return self._extract_content(response)

    def _messages_url(self) -> str:
        return f"{self._base_url()}/v1/messages"

    def _base_url(self) -> str:
        return (self.settings.base_url or self.default_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "anthropic-version": self.api_version}
        if self.settings.api_key:
            headers["x-api-key"] = self.settings.api_key
        return headers

    def _convert_messages(self, messages: Sequence[Mapping[str, str]]) -> tuple[str, list[dict[str, str]]]:
        system_parts: list[str] = []
        converted: list[dict[str, str]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                system_parts.append(message["content"])
                continue
            if role not in ("user", "assistant"):
                raise AnthropicLLMError(f"{self.settings.provider} validation error: unsupported role {role!r}")
            converted.append({"role": role, "content": message["content"]})
        if not converted:
            raise AnthropicLLMError(f"{self.settings.provider} validation error: at least one user or assistant message is required")
        return "\n\n".join(system_parts), converted

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        from urllib.error import HTTPError, URLError

        try:
            return self.transport(url, headers, payload, self.timeout)
        except AnthropicLLMError:
            raise
        except HTTPError as error:
            raise AnthropicLLMError(f"{self.settings.provider} http error: {error.code}") from error
        except URLError as error:
            raise AnthropicLLMError(f"{self.settings.provider} connection error: {error.reason}") from error
        except OSError as error:
            raise AnthropicLLMError(f"{self.settings.provider} transport error: {type(error).__name__}") from error

    def _default_transport(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        import json
        from urllib.request import Request, urlopen

        request = Request(url=url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise AnthropicLLMError(f"{self.settings.provider} response error: expected object")
        return parsed

    def _extract_content(self, response: dict[str, Any]) -> str:
        blocks = response.get("content")
        if not isinstance(blocks, list):
            raise AnthropicLLMError(f"{self.settings.provider} response error: missing content blocks")
        texts = [block.get("text") for block in blocks if isinstance(block, dict) and block.get("type") == "text"]
        texts = [text for text in texts if isinstance(text, str)]
        if not texts:
            raise AnthropicLLMError(f"{self.settings.provider} response error: no text content in response")
        return "".join(texts)

    def _validate_messages(self, messages: Sequence[Mapping[str, str]]) -> None:
        if isinstance(messages, (str, bytes)) or not messages:
            raise AnthropicLLMError(f"{self.settings.provider} validation error: messages must be a non-empty sequence")
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise AnthropicLLMError(f"{self.settings.provider} validation error: messages[{index}] must be a mapping")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise AnthropicLLMError(f"{self.settings.provider} validation error: messages[{index}].role is required")
            if not isinstance(content, str):
                raise AnthropicLLMError(f"{self.settings.provider} validation error: messages[{index}].content must be string")


LLMFactory.register_provider("anthropic", AnthropicLLM)
