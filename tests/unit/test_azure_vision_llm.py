import base64
from typing import Any
from urllib.error import HTTPError

import pytest

from core.settings import LLMSettings, load_settings
from libs.llm.azure_vision_llm import AzureVisionLLM, AzureVisionLLMError
from libs.llm.llm_factory import LLMFactory


PNG_BYTES = b"\x89PNG\r\n\x1a\nimage-bytes"


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"choices": [{"message": {"content": "caption"}}]}
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        return self.response


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    LLMFactory.unregister_vision_provider("azure")
    yield
    LLMFactory.unregister_vision_provider("azure")


def azure_settings(max_image_size: int = 2048) -> LLMSettings:
    return LLMSettings(
        provider="azure",
        model="gpt-4o",
        api_key="secret",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-05-01-preview",
        deployment_name="vision deployment",
        max_image_size=max_image_size,
    )


def test_factory_routes_azure_vision_llm_from_llm_settings() -> None:
    vision_llm = LLMFactory.create_vision_llm(azure_settings())

    assert isinstance(vision_llm, AzureVisionLLM)


def test_factory_routes_azure_vision_llm_from_project_vision_settings(tmp_path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app:",
                "  name: synapserag-mcp",
                "llm:",
                "  provider: openai",
                "  model: gpt-4o",
                "vision_llm:",
                "  provider: azure",
                "  model: gpt-4o",
                "  api_key: secret",
                "  azure_endpoint: https://example.openai.azure.com",
                "  api_version: 2024-05-01-preview",
                "  deployment_name: vision",
                "  max_image_size: 1024",
                "embedding:",
                "  provider: openai",
                "  model: text-embedding-3-small",
                "vector_store:",
                "  backend: chroma",
                "  persist_path: data/db/chroma",
                "retrieval:",
                "  sparse_backend: bm25",
                "  fusion_algorithm: rrf",
                "  top_k_dense: 20",
                "  top_k_sparse: 20",
                "  top_k_final: 5",
                "rerank:",
                "  enabled: false",
                "  backend: none",
                "evaluation:",
                "  enabled: false",
                "  backends: []",
                "observability:",
                "  log_path: logs/app.log",
                "  trace_path: logs/traces.jsonl",
                "",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(str(config_path))

    vision_llm = LLMFactory.create_vision_llm(settings)

    assert isinstance(vision_llm, AzureVisionLLM)
    assert vision_llm.settings.provider == "azure"
    assert vision_llm.settings.max_image_size == 1024


def test_chat_with_image_uses_azure_endpoint_headers_and_path_payload(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(PNG_BYTES)
    transport = FakeTransport()
    vision_llm = AzureVisionLLM(azure_settings(), transport=transport)

    response = vision_llm.chat_with_image("describe it", str(image_path))

    url, headers, payload, timeout = transport.calls[0]
    content = payload["messages"][0]["content"]
    assert response.text == "caption"
    assert response.metadata == {"provider": "azure", "model": "gpt-4o"}
    assert url == "https://example.openai.azure.com/openai/deployments/vision%20deployment/chat/completions?api-version=2024-05-01-preview"
    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers
    assert payload["model"] == "gpt-4o"
    assert content[0] == {"type": "text", "text": "describe it"}
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"
    assert timeout == 30.0


def test_chat_with_image_accepts_base64_string() -> None:
    transport = FakeTransport()
    vision_llm = AzureVisionLLM(azure_settings(), transport=transport)
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")

    vision_llm.chat_with_image("caption", encoded)

    image_url = transport.calls[0][2]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{encoded}"


def test_chat_with_image_resizes_before_sending(monkeypatch) -> None:
    transport = FakeTransport()
    vision_llm = AzureVisionLLM(azure_settings(max_image_size=12), transport=transport)

    def fake_resize(image_bytes: bytes, mime: str) -> bytes:
        assert image_bytes == PNG_BYTES
        assert mime == "image/png"
        assert vision_llm.settings.max_image_size == 12
        return b"small"

    monkeypatch.setattr(vision_llm, "_resize_image", fake_resize)

    vision_llm.chat_with_image("caption", PNG_BYTES)

    image_url = transport.calls[0][2]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{base64.b64encode(b'small').decode('ascii')}"


def test_chat_with_image_wraps_timeout_error() -> None:
    def failing_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise TimeoutError()

    vision_llm = AzureVisionLLM(azure_settings(), transport=failing_transport)

    with pytest.raises(AzureVisionLLMError, match="timeout"):
        vision_llm.chat_with_image("caption", PNG_BYTES)


def test_chat_with_image_wraps_auth_http_error() -> None:
    def failing_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise HTTPError(url, 401, "Unauthorized", {}, None)

    vision_llm = AzureVisionLLM(azure_settings(), transport=failing_transport)

    with pytest.raises(AzureVisionLLMError, match="401"):
        vision_llm.chat_with_image("caption", PNG_BYTES)
