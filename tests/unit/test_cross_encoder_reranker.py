import pytest

from core.settings import RerankSettings
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker, CrossEncoderRerankerError
from libs.reranker.reranker_factory import RerankerFactory


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    RerankerFactory.unregister_provider("cross_encoder")
    yield
    RerankerFactory.unregister_provider("cross_encoder")


def candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="alpha", score=0.1, metadata={"source": "one"}),
        RerankCandidate(id="b", text="beta", score=0.2, metadata={"source": "two"}),
        RerankCandidate(id="c", text="gamma", score=0.3, metadata={"source": "three"}),
    ]


def test_factory_routes_cross_encoder_backend() -> None:
    reranker = RerankerFactory.create(RerankSettings(enabled=True, backend="cross_encoder", top_m=2))

    assert isinstance(reranker, CrossEncoderReranker)


def test_cross_encoder_reranker_sorts_by_mock_scores() -> None:
    calls = []

    def scorer(query: str, active_candidates: list[RerankCandidate]) -> list[float]:
        calls.append((query, active_candidates))
        return [0.2, 0.9, 0.1]

    reranker = CrossEncoderReranker(RerankSettings(enabled=True, backend="cross_encoder", top_m=3), scorer=scorer)

    ranked = reranker.rerank("find beta", candidates())

    assert [candidate.id for candidate in ranked] == ["b", "a", "c"]
    assert [candidate.score for candidate in ranked] == [0.9, 0.2, 0.1]
    assert ranked[0].metadata == {"source": "two"}
    assert calls[0][0] == "find beta"


def test_cross_encoder_respects_top_m_and_keeps_overflow_order() -> None:
    def scorer(query: str, active_candidates: list[RerankCandidate]) -> list[float]:
        return [0.9, 0.1]

    reranker = CrossEncoderReranker(RerankSettings(enabled=True, backend="cross_encoder", top_m=2), scorer=scorer)

    ranked = reranker.rerank("find alpha", candidates())

    assert [candidate.id for candidate in ranked] == ["a", "b", "c"]
    assert ranked[2] == candidates()[2]


def test_cross_encoder_rejects_missing_scorer_with_fallback_flag() -> None:
    reranker = CrossEncoderReranker(RerankSettings(enabled=True, backend="cross_encoder"))

    with pytest.raises(CrossEncoderRerankerError, match="fallback") as error:
        reranker.rerank("query", candidates())

    assert error.value.fallback is True


def test_cross_encoder_wraps_timeout_and_failure_as_fallback_errors() -> None:
    timeout = CrossEncoderReranker(
        RerankSettings(enabled=True, backend="cross_encoder"),
        scorer=lambda query, active_candidates: (_ for _ in ()).throw(TimeoutError("slow")),
    )
    failure = CrossEncoderReranker(
        RerankSettings(enabled=True, backend="cross_encoder"),
        scorer=lambda query, active_candidates: (_ for _ in ()).throw(RuntimeError("broken")),
    )

    with pytest.raises(CrossEncoderRerankerError, match="timeout") as timeout_error:
        timeout.rerank("query", candidates())

    with pytest.raises(CrossEncoderRerankerError, match="RuntimeError") as failure_error:
        failure.rerank("query", candidates())

    assert timeout_error.value.fallback is True
    assert failure_error.value.fallback is True


def test_cross_encoder_rejects_invalid_scores() -> None:
    wrong_count = CrossEncoderReranker(RerankSettings(enabled=True, backend="cross_encoder"), scorer=lambda query, active_candidates: [1.0])
    wrong_type = CrossEncoderReranker(RerankSettings(enabled=True, backend="cross_encoder"), scorer=lambda query, active_candidates: [1.0, "bad", 0.0])

    with pytest.raises(CrossEncoderRerankerError, match="score count mismatch"):
        wrong_count.rerank("query", candidates())

    with pytest.raises(CrossEncoderRerankerError, match="scores must be numeric"):
        wrong_type.rerank("query", candidates())


def test_cross_encoder_fallback_signal_keeps_candidates() -> None:
    original = candidates()
    reranker = CrossEncoderReranker(RerankSettings(enabled=True, backend="cross_encoder"))

    signal = reranker.fallback(original, "model unavailable")

    assert signal.reason == "model unavailable"
    assert signal.candidates == original
    assert signal.candidates is not original
