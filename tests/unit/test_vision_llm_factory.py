from pathlib import Path

import pytest

from core.settings import LLMSettings, load_settings
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMResponse
from libs.llm.llm_factory import LLMFactory


class FakeVisionLLM(BaseVisionLLM):
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> VisionLLMResponse:
        image_kind = "bytes" if isinstance(image_path, bytes) else "path"
        return VisionLLMResponse(
            text=f"{self.settings.provider}:{text}:{image_kind}",
            metadata={"model": self.settings.model, "image_kind": image_kind},
        )


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    LLMFactory.unregister_vision_provider("fake")
    LLMFactory.unregister_vision_provider("openai")
    yield
    LLMFactory.unregister_vision_provider("fake")
    LLMFactory.unregister_vision_provider("openai")


def test_vision_factory_creates_registered_provider_from_llm_settings() -> None:
    LLMFactory.register_vision_provider("fake", FakeVisionLLM)
    settings = LLMSettings(provider="fake", model="fake-vision")

    vision_llm = LLMFactory.create_vision_llm(settings)
    response = vision_llm.chat_with_image("describe", "image.png")

    assert isinstance(vision_llm, FakeVisionLLM)
    assert response == VisionLLMResponse(text="fake:describe:path", metadata={"model": "fake-vision", "image_kind": "path"})


def test_vision_factory_creates_registered_provider_from_project_settings() -> None:
    LLMFactory.register_vision_provider("openai", FakeVisionLLM)
    settings = load_settings("config/settings.yaml")

    vision_llm = LLMFactory.create_vision_llm(settings)
    response = vision_llm.chat_with_image("caption", b"image-bytes")

    assert isinstance(vision_llm, FakeVisionLLM)
    assert response.text == "openai:caption:bytes"
    assert response.metadata["model"] == settings.llm.model


def test_vision_llm_accepts_path_like_string_and_bytes_inputs(tmp_path: Path) -> None:
    LLMFactory.register_vision_provider("fake", FakeVisionLLM)
    image_path = tmp_path / "image.png"
    settings = LLMSettings(provider="fake", model="fake-vision")
    vision_llm = LLMFactory.create_vision_llm(settings)

    path_response = vision_llm.chat_with_image("describe", str(image_path))
    bytes_response = vision_llm.chat_with_image("describe", b"image-bytes")

    assert path_response.metadata["image_kind"] == "path"
    assert bytes_response.metadata["image_kind"] == "bytes"


def test_vision_factory_rejects_unknown_provider() -> None:
    settings = LLMSettings(provider="missing", model="missing-vision")

    with pytest.raises(ValueError, match="unsupported vision LLM provider: missing"):
        LLMFactory.create_vision_llm(settings)
