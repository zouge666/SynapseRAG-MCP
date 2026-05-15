from collections.abc import Mapping, Sequence

import pytest

from core.settings import LLMSettings, RerankSettings
from libs.llm.base_llm import BaseLLM
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.llm_reranker import LLMReranker, LLMRerankerError
from libs.reranker.reranker_factory import RerankerFactory


class FakeLLM(BaseLLM):
    def __init__(self, response: str = '{"ranked_ids": ["b", "a"]}', error: Exception | None = None) -> None:
        super().__init__(LLMSettings(provider="fake", model="fake"))
        self.response = response
        self.error = error
        self.messages: list[Sequence[Mapping[str, str]]] = []

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.messages.append(messages)
        if self.error:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    RerankerFactory.unregister_provider("llm")
    yield
    RerankerFactory.unregister_provider("llm")


def candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="alpha", score=0.7, metadata={"source": "one"}),
        RerankCandidate(id="b", text="beta", score=0.4, metadata={"source": "two"}),
        RerankCandidate(id="c", text="gamma", score=0.2, metadata={"source": "three"}),
    ]


def test_factory_routes_llm_reranker_backend() -> None:
    reranker = RerankerFactory.create(RerankSettings(enabled=True, backend="llm", top_m=2))

    assert isinstance(reranker, LLMReranker)


def test_llm_reranker_uses_prompt_and_orders_candidates() -> None:
    llm = FakeLLM('{"ranked_ids": ["b", "a"]}')
    reranker = LLMReranker(RerankSettings(enabled=True, backend="llm", top_m=2), llm=llm, prompt_text="Rank these chunks")

    ranked = reranker.rerank("find beta", candidates())

    assert [candidate.id for candidate in ranked] == ["b", "a", "c"]
    prompt = llm.messages[0][0]["content"]
    assert "Rank these chunks" in prompt
    assert '"query": "find beta"' in prompt
    assert '"ranked_ids"' in prompt
    assert '"id": "a"' in prompt


def test_llm_reranker_reads_default_prompt_file() -> None:
    llm = FakeLLM('{"ranked_ids": ["a"]}')
    reranker = LLMReranker(RerankSettings(enabled=True, backend="llm", top_m=1), llm=llm)

    ranked = reranker.rerank("find alpha", candidates())

    assert [candidate.id for candidate in ranked] == ["a", "b", "c"]
    assert "Rank the candidate chunks" in llm.messages[0][0]["content"]


def test_llm_reranker_rejects_non_json_response() -> None:
    reranker = LLMReranker(RerankSettings(enabled=True, backend="llm"), llm=FakeLLM("not json"), prompt_text="rank")

    with pytest.raises(LLMRerankerError, match="schema error") as error:
        reranker.rerank("query", candidates())

    assert error.value.fallback is True


def test_llm_reranker_rejects_unknown_and_duplicate_ids() -> None:
    unknown = LLMReranker(RerankSettings(enabled=True, backend="llm"), llm=FakeLLM('{"ranked_ids": ["missing"]}'), prompt_text="rank")
    duplicate = LLMReranker(RerankSettings(enabled=True, backend="llm"), llm=FakeLLM('{"ranked_ids": ["a", "a"]}'), prompt_text="rank")

    with pytest.raises(LLMRerankerError, match="unknown candidate id"):
        unknown.rerank("query", candidates())

    with pytest.raises(LLMRerankerError, match="duplicate candidate id"):
        duplicate.rerank("query", candidates())


def test_llm_reranker_wraps_llm_failure_as_fallback_error() -> None:
    reranker = LLMReranker(
        RerankSettings(enabled=True, backend="llm"),
        llm=FakeLLM(error=TimeoutError("slow")),
        prompt_text="rank",
    )

    with pytest.raises(LLMRerankerError, match="fallback") as error:
        reranker.rerank("query", candidates())

    assert error.value.fallback is True


def test_llm_reranker_fallback_signal_keeps_candidates() -> None:
    original = candidates()
    reranker = LLMReranker(RerankSettings(enabled=True, backend="llm"), prompt_text="rank")

    signal = reranker.fallback(original, "llm unavailable")

    assert signal.reason == "llm unavailable"
    assert signal.candidates == original
    assert signal.candidates is not original
