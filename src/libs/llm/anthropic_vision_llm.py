from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.settings import LLMSettings
from libs.llm import vision_image
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMResponse
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import LLMTransport


class AnthropicVisionLLMError(RuntimeError):
    pass


class AnthropicVisionLLM(BaseVisionLLM):
    default_base_url = "https://api.anthropic.com"
    api_version = "2023-06-01"
    default_max_tokens = 1024

    def __init__(self, settings: LLMSettings, transport: LLMTransport | None = None, timeout: float = 30.0) -> None:
        super().__init__(settings)
        self.transport = transport or self._default_transport
        self.timeout = timeout

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> VisionLLMResponse:
        self._validate(text, image_path)
        try:
            encoded, mime = vision_image.prepare_image_base64(image_path, self.settings.max_image_size)
        except vision_image.VisionImageError as error:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision validation error: {error}") from error
        payload = self._payload(text, encoded, mime)
        response = self._send(self._messages_url(), self._headers(), payload)
        return VisionLLMResponse(text=self._extract_content(response), metadata={"provider": self.settings.provider, "model": self.settings.model})

    def _validate(self, text: str, image_path: str | bytes) -> None:
        if not isinstance(text, str) or not text:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision validation error: text must be a non-empty string")
        if not isinstance(image_path, str | bytes) or image_path in ("", b""):
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision validation error: image_path must be a non-empty string or bytes")
        if self.settings.max_image_size <= 0:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision validation error: max_image_size must be positive")

    def _payload(self, text: str, encoded: str, mime: str) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "max_tokens": self.default_max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}},
                        {"type": "text", "text": text},
                    ],
                }
            ],
        }

    def _messages_url(self) -> str:
        return f"{(self.settings.base_url or self.default_base_url).rstrip('/')}/v1/messages"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "anthropic-version": self.api_version}
        if self.settings.api_key:
            headers["x-api-key"] = self.settings.api_key
        return headers

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport(url, headers, payload, self.timeout)
        except AnthropicVisionLLMError:
            raise
        except TimeoutError as error:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision timeout error: request timed out") from error
        except HTTPError as error:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision http error: {error.code}") from error
        except URLError as error:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision connection error: {error.reason}") from error
        except OSError as error:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision transport error: {type(error).__name__}") from error

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
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision response error: expected object")
        return parsed

    def _extract_content(self, response: dict[str, Any]) -> str:
        blocks = response.get("content")
        if not isinstance(blocks, list):
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision response error: missing content blocks")
        texts = [block.get("text") for block in blocks if isinstance(block, dict) and block.get("type") == "text"]
        texts = [text for text in texts if isinstance(text, str)]
        if not texts:
            raise AnthropicVisionLLMError(f"{self.settings.provider} vision response error: no text content in response")
        return "".join(texts)


LLMFactory.register_vision_provider("anthropic", AnthropicVisionLLM)
