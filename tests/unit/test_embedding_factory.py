import pytest

from core.settings import EmbeddingSettings, load_settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    EmbeddingFactory.unregister_provider("fake")
    EmbeddingFactory.unregister_provider("openai")
    yield
    EmbeddingFactory.unregister_provider("fake")
    EmbeddingFactory.unregister_provider("openai")


def test_factory_creates_registered_provider_from_embedding_settings() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)
    settings = EmbeddingSettings(provider="fake", model="fake-model", dimensions=2)

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings.model == "fake-model"
    assert embedding.embed(["hi", "there"]) == [[2.0, 0.0], [5.0, 1.0]]


def test_factory_creates_registered_provider_from_project_settings() -> None:
    EmbeddingFactory.register_provider("openai", FakeEmbedding)
    settings = load_settings("config/settings.yaml")

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings.provider == "openai"


def test_factory_rejects_unknown_provider() -> None:
    settings = EmbeddingSettings(provider="missing", model="missing-model")

    with pytest.raises(ValueError, match="unsupported embedding provider: missing"):
        EmbeddingFactory.create(settings)
