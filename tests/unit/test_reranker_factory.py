import pytest

from core.settings import RerankSettings, load_settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        return sorted(candidates, key=lambda candidate: candidate.score)


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


def test_factory_creates_none_backend_from_project_settings() -> None:
    settings = load_settings("config/settings.yaml")

    reranker = RerankerFactory.create(settings)

    assert isinstance(reranker, NoneReranker)
    assert reranker.settings.backend == "none"


def test_factory_rejects_unknown_backend() -> None:
    settings = RerankSettings(enabled=True, backend="missing")

    with pytest.raises(ValueError, match="unsupported reranker backend: missing"):
        RerankerFactory.create(settings)
