import pytest

from core.settings import EmbeddingSettings, load_settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.local_embedding import LocalEmbedding


class FakeEmbedding(BaseEmbedding):
    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


PROJECT_CONFIG = """
app:
  name: synapserag-mcp
llm:
  provider: openai
  model: gpt-4o
embedding:
  provider: local
  model: local-hash
  dimensions: 128
vector_store:
  backend: chroma
  persist_path: data/db/chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 20
  top_k_sparse: 20
  top_k_final: 5
rerank:
  enabled: false
  backend: none
evaluation:
  enabled: false
  backends: []
observability:
  log_path: logs/app.log
  trace_path: logs/traces.jsonl
"""


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    EmbeddingFactory.unregister_provider("fake")
    EmbeddingFactory.unregister_provider("openai")
    EmbeddingFactory.unregister_provider("local")
    yield
    EmbeddingFactory.unregister_provider("fake")
    EmbeddingFactory.unregister_provider("openai")
    EmbeddingFactory.unregister_provider("local")


def test_factory_creates_registered_provider_from_embedding_settings() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)
    settings = EmbeddingSettings(provider="fake", model="fake-model", dimensions=2)

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings.model == "fake-model"
    assert embedding.embed(["hi", "there"]) == [[2.0, 0.0], [5.0, 1.0]]


def test_factory_creates_registered_provider_from_project_settings(tmp_path) -> None:
    EmbeddingFactory.register_provider("local", FakeEmbedding)
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(PROJECT_CONFIG, encoding="utf-8")
    settings = load_settings(str(config_path))

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings.provider == "local"


def test_factory_loads_builtin_local_provider() -> None:
    settings = EmbeddingSettings(provider="local", model="local-hash", dimensions=4)

    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed(["alpha beta", "alpha"])

    assert isinstance(embedding, LocalEmbedding)
    assert len(vectors) == 2
    assert all(len(vector) == 4 for vector in vectors)
    assert vectors[0] != vectors[1]


def test_factory_rejects_unknown_provider() -> None:
    settings = EmbeddingSettings(provider="missing", model="missing-model")

    with pytest.raises(ValueError, match="unsupported embedding provider: missing"):
        EmbeddingFactory.create(settings)
