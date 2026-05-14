from collections.abc import Mapping, Sequence

import pytest

from core.settings import LLMSettings, load_settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        return f"{self.settings.provider}:{messages[-1]['content']}"


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    LLMFactory.unregister_provider("fake")
    yield
    LLMFactory.unregister_provider("fake")


def test_factory_creates_registered_provider_from_llm_settings() -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    settings = LLMSettings(provider="fake", model="fake-model")

    llm = LLMFactory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert llm.settings.model == "fake-model"
    assert llm.chat([{"role": "user", "content": "hello"}]) == "fake:hello"


def test_factory_creates_registered_provider_from_project_settings() -> None:
    LLMFactory.register_provider("openai", FakeLLM)
    settings = load_settings("config/settings.yaml")

    llm = LLMFactory.create(settings)

    assert isinstance(llm, FakeLLM)
    assert llm.settings.provider == "openai"


def test_factory_rejects_unknown_provider() -> None:
    settings = LLMSettings(provider="missing", model="missing-model")

    with pytest.raises(ValueError, match="unsupported LLM provider: missing"):
        LLMFactory.create(settings)
