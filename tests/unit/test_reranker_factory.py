import pytest

from core.settings import RerankSettings, load_settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        return sorted(candidates, key=lambda candidate: candidate.score)


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
    RerankerFactory.unregister_provider("fake")
    yield
    RerankerFactory.unregister_provider("fake")


def test_none_reranker_keeps_original_order() -> None:
    settings = RerankSettings(enabled=False, backend="none")
    reranker = RerankerFactory.create(settings)
    candidates = [
        RerankCandidate(id="first", text="alpha", score=0.1),
        RerankCandidate(id="second", text="beta", score=0.9),
    ]

    ranked = reranker.rerank("query", candidates)

    assert isinstance(reranker, NoneReranker)
    assert ranked == candidates
    assert ranked is not candidates


def test_factory_creates_registered_backend_from_rerank_settings() -> None:
    RerankerFactory.register_provider("fake", FakeReranker)
    settings = RerankSettings(enabled=True, backend="fake")
    candidates = [
        RerankCandidate(id="high", text="alpha", score=0.9),
        RerankCandidate(id="low", text="beta", score=0.1),
    ]

    reranker = RerankerFactory.create(settings)
    ranked = reranker.rerank("query", candidates)

    assert isinstance(reranker, FakeReranker)
    assert [candidate.id for candidate in ranked] == ["low", "high"]


def test_factory_creates_none_backend_from_project_settings(tmp_path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(PROJECT_CONFIG, encoding="utf-8")
    settings = load_settings(str(config_path))

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, NoneReranker)
    assert reranker.settings.backend == "none"


def test_factory_rejects_unknown_backend() -> None:
    settings = RerankSettings(enabled=True, backend="missing")

    with pytest.raises(ValueError, match="unsupported reranker backend: missing"):
        RerankerFactory.create(settings)


class FakeLLM(BaseLLM):
    def chat(self, messages):
        return '{"ranked_ids": []}'


LLM_PROJECT_CONFIG = PROJECT_CONFIG.replace(
    """llm:
  provider: openai
  model: gpt-4o""",
    """llm:
  provider: fake
  model: fake-1
  api_key: x""",
).replace(
    """rerank:
  enabled: false
  backend: none""",
    """rerank:
  enabled: true
  backend: llm
  model: ''""",
)


def load_llm_rerank_settings(tmp_path, config: str = LLM_PROJECT_CONFIG):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(config, encoding="utf-8")
    return load_settings(str(config_path))


def test_factory_wires_llm_client_into_llm_reranker_from_project_settings(tmp_path) -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    try:
        settings = load_llm_rerank_settings(tmp_path)

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, LLMReranker)
        assert isinstance(reranker.llm, FakeLLM)
        assert reranker.llm.settings.model == "fake-1"
    finally:
        LLMFactory.unregister_provider("fake")


def test_factory_rerank_model_overrides_llm_model(tmp_path) -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    try:
        settings = load_llm_rerank_settings(tmp_path, LLM_PROJECT_CONFIG.replace("model: ''", "model: rerank-special"))

        reranker = RerankerFactory.create(settings)

        assert reranker.llm.settings.model == "rerank-special"
    finally:
        LLMFactory.unregister_provider("fake")


def test_factory_llm_backend_requires_api_key(tmp_path) -> None:
    settings = load_llm_rerank_settings(tmp_path, LLM_PROJECT_CONFIG.replace("api_key: x", "api_key: ''"))

    with pytest.raises(ValueError, match="requires an LLM API key"):
        RerankerFactory.create(settings)


def test_factory_normalizes_registered_backend_names() -> None:
    RerankerFactory.register_provider(" Fake ", FakeReranker)
    settings = RerankSettings(enabled=True, backend=" FAKE ")

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, FakeReranker)
    assert reranker.settings.backend == " FAKE "


def test_factory_rejects_empty_backend_name() -> None:
    settings = RerankSettings(enabled=True, backend=" ")

    with pytest.raises(ValueError, match="rerank.backend is required"):
        RerankerFactory.create(settings)


def test_unregister_provider_keeps_none_backend_available() -> None:
    RerankerFactory.unregister_provider("none")
    settings = RerankSettings(enabled=False, backend="none")

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, NoneReranker)


def test_none_reranker_preserves_candidate_shapes_for_empty_and_metadata() -> None:
    settings = RerankSettings(enabled=False, backend="none")
    reranker = RerankerFactory.create(settings)
    candidate = RerankCandidate(id="doc-1", text="alpha", score=0.7, metadata={"source": "one"})

    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", [candidate])[0] == candidate
    assert reranker.rerank("query", [candidate])[0].metadata == {"source": "one"}
