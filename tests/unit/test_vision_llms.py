import base64
from typing import Any
from urllib.error import HTTPError

import pytest

from core.settings import LLMSettings, load_settings
from libs.llm import vision_image
from libs.llm.anthropic_vision_llm import AnthropicVisionLLM, AnthropicVisionLLMError
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"choices": [{"message": {"content": "caption"}}]}
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        return self.response


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    LLMFactory.unregister_vision_provider("openai")
    LLMFactory.unregister_vision_provider("anthropic")
    yield
    LLMFactory.unregister_vision_provider("openai")
    LLMFactory.unregister_vision_provider("anthropic")


def openai_settings(max_image_size: int = 2048, base_url: str = "") -> LLMSettings:
    return LLMSettings(provider="openai", model="deepseek-v4-vision", api_key="secret", base_url=base_url, max_image_size=max_image_size)


def anthropic_settings(max_image_size: int = 2048) -> LLMSettings:
    return LLMSettings(provider="anthropic", model="claude-sonnet-5", api_key="secret", max_image_size=max_image_size)


def test_factory_routes_openai_vision_llm_from_llm_settings() -> None:
    vision_llm = LLMFactory.create_vision_llm(openai_settings())

    assert isinstance(vision_llm, OpenAIVisionLLM)


def test_factory_routes_anthropic_vision_llm_from_llm_settings() -> None:
    vision_llm = LLMFactory.create_vision_llm(anthropic_settings())

    assert isinstance(vision_llm, AnthropicVisionLLM)


def test_factory_routes_vision_llm_from_project_vision_settings(tmp_path) -> None:
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
                "  provider: anthropic",
                "  model: claude-sonnet-5",
                "  api_key: secret",
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

    assert isinstance(vision_llm, AnthropicVisionLLM)
    assert vision_llm.settings.provider == "anthropic"
    assert vision_llm.settings.max_image_size == 1024


def test_openai_vision_uses_chat_completions_with_data_url(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(PNG_BYTES)
    transport = FakeTransport()
    vision_llm = OpenAIVisionLLM(openai_settings(), transport=transport)

    response = vision_llm.chat_with_image("describe it", str(image_path))

    url, headers, payload, timeout = transport.calls[0]
    content = payload["messages"][0]["content"]
    assert response.text == "caption"
    assert response.metadata == {"provider": "openai", "model": "deepseek-v4-vision"}
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert payload["model"] == "deepseek-v4-vision"
    assert content[0] == {"type": "text", "text": "describe it"}
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"
    assert timeout == 30.0


def test_openai_vision_uses_custom_base_url_for_compatible_endpoints(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(PNG_BYTES)
    transport = FakeTransport()
    vision_llm = OpenAIVisionLLM(openai_settings(base_url="https://relay.example.com/v1/"), transport=transport)

    vision_llm.chat_with_image("describe it", str(image_path))

    assert transport.calls[0][0] == "https://relay.example.com/v1/chat/completions"


def test_openai_vision_accepts_base64_string() -> None:
    transport = FakeTransport()
    vision_llm = OpenAIVisionLLM(openai_settings(), transport=transport)
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")

    vision_llm.chat_with_image("caption", encoded)

    image_url = transport.calls[0][2]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{encoded}"


def test_openai_vision_accepts_data_url() -> None:
    transport = FakeTransport()
    vision_llm = OpenAIVisionLLM(openai_settings(), transport=transport)
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")

    vision_llm.chat_with_image("caption", f"data:image/png;base64,{encoded}")

    image_url = transport.calls[0][2]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{encoded}"


def test_openai_vision_resizes_before_sending(monkeypatch) -> None:
    transport = FakeTransport()
    vision_llm = OpenAIVisionLLM(openai_settings(max_image_size=12), transport=transport)

    def fake_resize(image_bytes: bytes, mime: str, max_size: int) -> bytes:
        assert image_bytes == PNG_BYTES
        assert mime == "image/png"
        assert max_size == 12
        return b"small"

    monkeypatch.setattr(vision_image, "resize_image", fake_resize)

    vision_llm.chat_with_image("caption", PNG_BYTES)

    image_url = transport.calls[0][2]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{base64.b64encode(b'small').decode('ascii')}"


def test_openai_vision_rejects_invalid_image_input() -> None:
    vision_llm = OpenAIVisionLLM(openai_settings(), transport=FakeTransport())

    with pytest.raises(OpenAIVisionLLMError, match="vision validation error"):
        vision_llm.chat_with_image("caption", "not an image at all !!!")


def test_openai_vision_wraps_timeout_error() -> None:
    def failing_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise TimeoutError()

    vision_llm = OpenAIVisionLLM(openai_settings(), transport=failing_transport)

    with pytest.raises(OpenAIVisionLLMError, match="timeout"):
        vision_llm.chat_with_image("caption", PNG_BYTES)


def test_openai_vision_wraps_auth_http_error() -> None:
    def failing_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise HTTPError(url, 401, "Unauthorized", {}, None)

    vision_llm = OpenAIVisionLLM(openai_settings(), transport=failing_transport)

    with pytest.raises(OpenAIVisionLLMError, match="401"):
        vision_llm.chat_with_image("caption", PNG_BYTES)


def test_anthropic_vision_uses_messages_api_with_base64_image_block() -> None:
    transport = FakeTransport({"content": [{"type": "text", "text": "caption"}]})
    vision_llm = AnthropicVisionLLM(anthropic_settings(), transport=transport)

    response = vision_llm.chat_with_image("describe it", PNG_BYTES)

    url, headers, payload, timeout = transport.calls[0]
    content = payload["messages"][0]["content"]
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")
    assert response.text == "caption"
    assert response.metadata == {"provider": "anthropic", "model": "claude-sonnet-5"}
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers
    assert payload["model"] == "claude-sonnet-5"
    assert payload["max_tokens"] > 0
    assert content[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}
    assert content[1] == {"type": "text", "text": "describe it"}
    assert timeout == 30.0


def test_anthropic_vision_uses_custom_base_url_for_relays() -> None:
    transport = FakeTransport({"content": [{"type": "text", "text": "caption"}]})
    vision_llm = AnthropicVisionLLM(
        LLMSettings(provider="anthropic", model="claude-sonnet-5", api_key="secret", base_url="https://relay.example.com/"),
        transport=transport,
    )

    vision_llm.chat_with_image("caption", PNG_BYTES)

    assert transport.calls[0][0] == "https://relay.example.com/v1/messages"


def test_anthropic_vision_response_error_on_missing_text_blocks() -> None:
    vision_llm = AnthropicVisionLLM(anthropic_settings(), transport=FakeTransport({"content": [{"type": "tool_use", "name": "x"}]}))

    with pytest.raises(AnthropicVisionLLMError, match="no text content"):
        vision_llm.chat_with_image("caption", PNG_BYTES)
