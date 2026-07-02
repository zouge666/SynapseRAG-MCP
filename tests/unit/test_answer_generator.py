from types import SimpleNamespace
from typing import Any

import pytest

from core import RetrievalResult
from core.response import AnswerGenerator, AnswerGeneratorError
from core.settings import LLMSettings
from core.trace import TraceContext
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    reply = "The latte costs 36 yuan [1]."
    error: Exception | None = None
    captured: list[Any] = []

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        FakeLLM.captured = []

    def chat(self, messages):
        FakeLLM.captured.append(list(messages))
        if FakeLLM.error is not None:
            raise FakeLLM.error
        return FakeLLM.reply


@pytest.fixture
def fake_llm() -> FakeLLM:
    FakeLLM.error = None
    FakeLLM.reply = "The latte costs 36 yuan [1]."
    return FakeLLM(LLMSettings(provider="fake", model="fake-1"))


@pytest.fixture
def results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.9,
            text="Osmanthus Oolong Latte (house signature) - 6.20",
            metadata={"source_path": "pdfs/menu.pdf", "chunk_index": 0, "page": 1},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            score=0.7,
            text="Oat milk adds 0.50 per drink.",
            metadata={"source_path": "pdfs/menu.pdf", "chunk_index": 1},
        ),
    ]


def make_generator(fake_llm: FakeLLM) -> AnswerGenerator:
    settings = SimpleNamespace(llm=LLMSettings(provider="fake", model="fake-1", api_key="x"))
    return AnswerGenerator(settings, llm=fake_llm)


def test_generate_returns_answer_with_citations(fake_llm, results) -> None:
    generated = make_generator(fake_llm).generate("latte price?", results)

    assert generated.answer == "The latte costs 36 yuan [1]."
    assert generated.provider == "fake"
    assert generated.model == "fake-1"
    assert [citation.id for citation in generated.citations] == [1, 2]
    assert generated.citations[0].source == "pdfs/menu.pdf"


def test_prompt_contains_numbered_context_and_question(fake_llm, results) -> None:
    make_generator(fake_llm).generate("latte price?", results)

    messages = FakeLLM.captured[0]
    assert messages[0]["role"] == "system"
    assert "[n]" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert "[1] source: pdfs/menu.pdf, page: 1" in user
    assert "[2] source: pdfs/menu.pdf" in user
    assert "Osmanthus Oolong Latte (house signature) - 6.20" in user
    assert user.rstrip().endswith("Question: latte price?")


def test_long_passages_are_truncated(fake_llm, results) -> None:
    generator = make_generator(fake_llm)
    generator.max_chunk_chars = 20

    generator.generate("latte price?", results)

    user = FakeLLM.captured[0][1]["content"]
    assert "Osmanthus Oolong ..." in user
    assert "house signature" not in user


def test_generate_records_trace_stage(fake_llm, results) -> None:
    trace = TraceContext(trace_type="query")

    make_generator(fake_llm).generate("latte price?", results, trace=trace)

    stage = next(stage for stage in trace.stages if stage["name"] == "generation")
    assert stage["details"]["provider"] == "fake"
    assert stage["details"]["model"] == "fake-1"
    assert stage["details"]["citations"] == 2
    assert stage["duration_ms"] is not None and stage["duration_ms"] >= 0


def test_generate_without_llm_uses_factory_and_registered_provider() -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    try:
        settings = SimpleNamespace(llm=LLMSettings(provider="fake", model="fake-1", api_key="x"))
        result = RetrievalResult(chunk_id="c", score=1.0, text="answer context", metadata={"source_path": "s"})
        generated = AnswerGenerator(settings).generate("q?", [result])
        assert generated.provider == "fake"
    finally:
        LLMFactory.unregister_provider("fake")


def test_generate_without_api_key_points_to_settings(results) -> None:
    settings = SimpleNamespace(llm=LLMSettings(provider="openai", model="gpt-4o", api_key=""))
    generator = AnswerGenerator(settings)

    with pytest.raises(AnswerGeneratorError, match="API key"):
        generator.generate("latte price?", results)


def test_generate_rejects_empty_results(fake_llm) -> None:
    with pytest.raises(AnswerGeneratorError, match="must not be empty"):
        make_generator(fake_llm).generate("latte price?", [])


def test_generate_rejects_empty_question(fake_llm, results) -> None:
    with pytest.raises(AnswerGeneratorError, match="non-empty string"):
        make_generator(fake_llm).generate("  ", results)


def test_llm_failure_is_wrapped(fake_llm, results) -> None:
    FakeLLM.error = RuntimeError("http 500")

    with pytest.raises(AnswerGeneratorError, match="generation failed"):
        make_generator(fake_llm).generate("latte price?", results)


def test_empty_llm_answer_is_rejected(fake_llm, results) -> None:
    FakeLLM.reply = "   "

    with pytest.raises(AnswerGeneratorError, match="empty answer"):
        make_generator(fake_llm).generate("latte price?", results)
