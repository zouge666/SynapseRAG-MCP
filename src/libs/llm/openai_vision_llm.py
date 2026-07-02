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


class OpenAIVisionLLMError(RuntimeError):
    pass


class OpenAIVisionLLM(BaseVisionLLM):
    default_base_url = "https://api.openai.com/v1"

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
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision validation error: {error}") from error
        payload = self._payload(text, f"data:{mime};base64,{encoded}")
        response = self._send(self._chat_url(), self._headers(), payload)
        return VisionLLMResponse(text=self._extract_content(response), metadata={"provider": self.settings.provider, "model": self.settings.model})

    def _validate(self, text: str, image_path: str | bytes) -> None:
        if not isinstance(text, str) or not text:
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision validation error: text must be a non-empty string")
        if not isinstance(image_path, str | bytes) or image_path in ("", b""):
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision validation error: image_path must be a non-empty string or bytes")
        if self.settings.max_image_size <= 0:
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision validation error: max_image_size must be positive")

    def _payload(self, text: str, image_url: str) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }

    def _chat_url(self) -> str:
        return f"{(self.settings.base_url or self.default_base_url).rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport(url, headers, payload, self.timeout)
        except OpenAIVisionLLMError:
            raise
        except TimeoutError as error:
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision timeout error: request timed out") from error
        except HTTPError as error:
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision http error: {error.code}") from error
        except URLError as error:
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision connection error: {error.reason}") from error
        except OSError as error:
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision transport error: {type(error).__name__}") from error

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
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision response error: expected object")
        return parsed

    def _extract_content(self, response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenAIVisionLLMError(f"{self.settings.provider} vision response error: missing choices[0].message.content") from error
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get("text") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
            if parts:
                return "\n".join(parts)
        raise OpenAIVisionLLMError(f"{self.settings.provider} vision response error: content must be string or text parts")


LLMFactory.register_vision_provider("openai", OpenAIVisionLLM)
