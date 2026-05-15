from __future__ import annotations

import base64
import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from core.settings import LLMSettings
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMResponse
from libs.llm.llm_factory import LLMFactory


VisionTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class AzureVisionLLMError(RuntimeError):
    pass


class AzureVisionLLM(BaseVisionLLM):
    def __init__(self, settings: LLMSettings, transport: VisionTransport | None = None, timeout: float = 30.0) -> None:
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
        payload = self._payload(text, image_path)
        response = self._send(self._chat_url(), self._headers(), payload)
        return VisionLLMResponse(text=self._extract_content(response), metadata={"provider": "azure", "model": self.settings.model})

    def _validate(self, text: str, image_path: str | bytes) -> None:
        if not self.settings.azure_endpoint:
            raise ValueError("azure vision validation error: azure_endpoint is required")
        if not isinstance(text, str) or not text:
            raise AzureVisionLLMError("azure vision validation error: text must be a non-empty string")
        if not isinstance(image_path, str | bytes) or image_path in ("", b""):
            raise AzureVisionLLMError("azure vision validation error: image_path must be a non-empty string or bytes")
        if self.settings.max_image_size <= 0:
            raise AzureVisionLLMError("azure vision validation error: max_image_size must be positive")

    def _chat_url(self) -> str:
        endpoint = self.settings.azure_endpoint.rstrip("/")
        deployment = quote(self.settings.deployment_name or self.settings.model, safe="")
        api_version = self.settings.api_version or "2024-02-15-preview"
        return f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["api-key"] = self.settings.api_key
        return headers

    def _payload(self, text: str, image_path: str | bytes) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": self._image_url(image_path)}},
                    ],
                }
            ],
        }

    def _image_url(self, image_path: str | bytes) -> str:
        if isinstance(image_path, bytes):
            image_bytes = self._resize_image(image_path, self._detect_mime(image_path))
            return self._data_url(image_bytes, self._detect_mime(image_bytes))
        if image_path.startswith("data:image/"):
            return image_path
        file_path = self._existing_path(image_path)
        if file_path is not None:
            image_bytes = file_path.read_bytes()
            mime = self._detect_mime(image_bytes, file_path.suffix)
            image_bytes = self._resize_image(image_bytes, mime)
            return self._data_url(image_bytes, self._detect_mime(image_bytes, file_path.suffix))
        try:
            image_bytes = base64.b64decode(image_path, validate=True)
        except ValueError as error:
            raise AzureVisionLLMError("azure vision validation error: image_path must be an existing path, bytes, data URL, or base64 string") from error
        mime = self._detect_mime(image_bytes)
        image_bytes = self._resize_image(image_bytes, mime)
        return self._data_url(image_bytes, self._detect_mime(image_bytes, mime))

    def _data_url(self, image_bytes: bytes, mime: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _resize_image(self, image_bytes: bytes, mime: str) -> bytes:
        try:
            from PIL import Image
        except ImportError:
            return image_bytes
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            max_size = self.settings.max_image_size
            if max(width, height) <= max_size:
                return image_bytes
            ratio = max_size / max(width, height)
            size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
            resized = image.resize(size, Image.Resampling.LANCZOS)
            output = BytesIO()
            fmt = "JPEG" if mime == "image/jpeg" else "PNG"
            if fmt == "JPEG" and resized.mode in {"RGBA", "P"}:
                resized = resized.convert("RGB")
            resized.save(output, format=fmt)
            return output.getvalue()

    def _detect_mime(self, image_bytes: bytes, fallback: str | None = None) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return "image/gif"
        if fallback in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if fallback == ".png":
            return "image/png"
        if fallback == ".gif":
            return "image/gif"
        if fallback and fallback.startswith("image/"):
            return fallback
        return "image/png"

    def _existing_path(self, image_path: str) -> Path | None:
        try:
            file_path = Path(image_path)
            return file_path if file_path.exists() else None
        except OSError:
            return None

    def _send(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.transport(url, headers, payload, self.timeout)
        except AzureVisionLLMError:
            raise
        except TimeoutError as error:
            raise AzureVisionLLMError("azure vision timeout error: request timed out") from error
        except HTTPError as error:
            raise AzureVisionLLMError(f"azure vision http error: {error.code}") from error
        except URLError as error:
            raise AzureVisionLLMError(f"azure vision connection error: {error.reason}") from error
        except OSError as error:
            raise AzureVisionLLMError(f"azure vision transport error: {type(error).__name__}") from error

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
            raise AzureVisionLLMError("azure vision response error: expected object")
        return parsed

    def _extract_content(self, response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise AzureVisionLLMError("azure vision response error: missing choices[0].message.content") from error
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get("text") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
            if parts:
                return "\n".join(parts)
        raise AzureVisionLLMError("azure vision response error: content must be string or text parts")


LLMFactory.register_vision_provider("azure", AzureVisionLLM)
