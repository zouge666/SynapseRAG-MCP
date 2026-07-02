from types import SimpleNamespace

import pytest

from core import RetrievalResult
from core.settings import LLMSettings
from observability.dashboard.pages.llm_chat import MAX_HISTORY_MESSAGES, decide_route, run_chat_turn


class FakeSearch:
    def search(self, query, top_k, filters=None, trace=None):
        trace.record_stage("hybrid_search", {"count": 1, "filters": filters or {}})
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.75,
                text="The sea salt osmanthus latte costs 36 CNY.",
                metadata={"source_path": "pdfs/menu.pdf", "chunk_index": 0},
            )
        ]


class EmptySearch:
    def search(self, query, top_k, filters=None, trace=None):
        return []


class ExplodingSearch:
    def search(self, query, top_k, filters=None, trace=None):
        raise AssertionError("search must not run on the chat route")


class FakeReranker:
    def rerank(self, query, candidates, trace=None):
        trace.record_stage("rerank", {"count": len(candidates), "enabled": False})
        return list(candidates)


class FakeLLM:
    def __init__(self, reply="It costs 36 CNY [1].", error=None):
        self.settings = LLMSettings(provider="fake", model="fake-1", api_key="x")
        self.reply = reply
        self.error = error
        self.captured = None

    def chat(self, messages):
        self.captured = list(messages)
        if self.error is not None:
            raise self.error
        return self.reply


def make_settings(api_key="x"):
    return SimpleNamespace(
        rerank=SimpleNamespace(enabled=False),
        retrieval=SimpleNamespace(top_k_final=5),
        observability=SimpleNamespace(trace_path="logs/test-traces.jsonl"),
        llm=LLMSettings(provider="fake", model="fake-1", api_key=api_key),
    )


def collect(written):
    return lambda trace, path: written.append((trace, path)) or trace


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("hello", "chat"),
        ("你好！", "chat"),
        ("Thanks!", "chat"),
        ("who are you?", "chat"),
        ("你能做什么", "chat"),
        ("ok", "chat"),
        ("What is the price of the sea salt osmanthus latte?", "rag"),
        ("帮我查一下拿铁的价格", "rag"),
    ],
)
def test_decide_route(message, expected) -> None:
    assert decide_route(message) == expected


def test_rag_turn_returns_answer_with_citations_and_trace() -> None:
    written = []

    turn = run_chat_turn(
        make_settings(),
        [],
        "What is the price of the sea salt osmanthus latte?",
        "default",
        llm=FakeLLM(),
        search=FakeSearch(),
        reranker=FakeReranker(),
        trace_writer=collect(written),
    )

    assert turn.route == "rag"
    assert turn.answer == "It costs 36 CNY [1]."
    assert [citation.id for citation in turn.citations] == [1]
    assert turn.error is None
    trace = written[0][0]
    assert trace["metadata"]["chat"] is True
    assert trace["metadata"]["route"] == "rag"
    assert [stage["name"] for stage in trace["stages"]] == ["hybrid_search", "rerank", "generation"]
    assert trace["status"] == "success"


def test_chat_turn_skips_retrieval_and_sends_capped_history() -> None:
    written = []
    llm = FakeLLM(reply="Hi there!")
    history = [{"role": "user" if index % 2 == 0 else "assistant", "content": f"msg-{index}"} for index in range(10)]

    turn = run_chat_turn(make_settings(), history, "hello", "default", llm=llm, search=ExplodingSearch(), trace_writer=collect(written))

    assert turn.route == "chat"
    assert turn.answer == "Hi there!"
    assert turn.error is None
    assert llm.captured[0]["role"] == "system"
    assert llm.captured[-1] == {"role": "user", "content": "hello"}
    assert len(llm.captured) == MAX_HISTORY_MESSAGES + 2
    assert [message["content"] for message in llm.captured[1:-1]] == [f"msg-{index}" for index in range(4, 10)]
    trace = written[0][0]
    assert trace["metadata"]["route"] == "chat"
    assert [stage["name"] for stage in trace["stages"]] == ["chat"]
    assert trace["stages"][0]["details"]["history_messages"] == MAX_HISTORY_MESSAGES


def test_mode_override_forces_route() -> None:
    written = []
    llm = FakeLLM(reply="你好！有什么可以帮你？")

    chat_turn = run_chat_turn(make_settings(), [], "hello", "default", mode="search", llm=FakeLLM(), search=FakeSearch(), reranker=FakeReranker(), trace_writer=collect(written))
    rag_turn = run_chat_turn(make_settings(), [], "What is the price of the latte?", "default", mode="chat", llm=llm, search=ExplodingSearch(), trace_writer=collect(written))

    assert chat_turn.route == "rag"
    assert rag_turn.route == "chat"
    assert rag_turn.answer == "你好！有什么可以帮你？"


def test_rag_turn_without_results_falls_back_to_chat_with_note() -> None:
    written = []

    turn = run_chat_turn(
        make_settings(),
        [],
        "Tell me about the seasonal menu",
        "default",
        llm=FakeLLM(reply="I could not find it in the knowledge base."),
        search=EmptySearch(),
        reranker=FakeReranker(),
        trace_writer=collect(written),
    )

    assert turn.route == "chat"
    assert turn.note is not None and "without the knowledge base" in turn.note
    assert turn.answer == "I could not find it in the knowledge base."
    assert "generation" not in [stage["name"] for stage in written[0][0]["stages"]]


def test_chat_turn_without_api_key_returns_error() -> None:
    written = []

    turn = run_chat_turn(make_settings(api_key=""), [], "hello", "default", search=ExplodingSearch(), trace_writer=collect(written))

    assert turn.error is not None and "API key" in turn.error
    assert turn.answer == ""
    assert written[0][0]["status"] == "failed"


def test_chat_turn_llm_failure_returns_error() -> None:
    written = []

    turn = run_chat_turn(make_settings(), [], "hello", "default", llm=FakeLLM(error=RuntimeError("http 500")), trace_writer=collect(written))

    assert turn.error is not None and "http 500" in turn.error
    assert "chat.error" in [stage["name"] for stage in written[0][0]["stages"]]


def test_chat_turn_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="message must be a non-empty string"):
        run_chat_turn(make_settings(), [], "  ", "default", trace_writer=collect([]))
    with pytest.raises(ValueError, match="unsupported chat mode"):
        run_chat_turn(make_settings(), [], "hi", "default", mode="guess", trace_writer=collect([]))
