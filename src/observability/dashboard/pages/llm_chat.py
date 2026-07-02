from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from core.response import AnswerGenerator, AnswerGeneratorError, Citation
from core.trace import TraceContext
from libs.llm.llm_factory import LLMFactory
from observability.dashboard.pages.query_console import _dimension_warning, _retrieve, _trace_path
from observability.dashboard.services.data_service import DataService
from observability.logger import write_trace


TraceWriter = Callable[..., dict[str, Any]]

MAX_HISTORY_MESSAGES = 6

CHAT_SYSTEM_PROMPT = (
    "You are the SynapseRAG assistant. Answer briefly and directly. "
    "If the user asks about documents in the knowledge base, say that a search is needed instead of inventing facts. "
    "Respond in the same language as the user."
)

_SMALL_TALK = {
    "hi", "hello", "hey", "hiya", "yo", "howdy", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thx", "bye", "goodbye", "see you", "ok", "okay", "yes", "no",
    "你好", "您好", "嗨", "哈喽", "在吗", "早上好", "下午好", "晚上好", "谢谢", "感谢", "多谢",
    "再见", "拜拜", "好的", "好", "嗯", "行",
}

_META_PATTERNS = (
    "who are you", "what are you", "your name", "what can you do", "help me",
    "你是谁", "你叫什么", "你能做什么", "你会什么", "介绍一下你自己", "帮助",
)


@dataclass(frozen=True)
class ChatTurn:
    message: str
    route: str
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    note: str | None = None
    error: str | None = None


def decide_route(message: str) -> str:
    normalized = re.sub(r"[\s\W_]+", " ", message.lower()).strip()
    compact = normalized.replace(" ", "")
    if not compact:
        return "chat"
    if compact in {item.replace(" ", "") for item in _SMALL_TALK} or normalized in _SMALL_TALK:
        return "chat"
    if any(pattern in normalized or pattern.replace(" ", "") in compact for pattern in _META_PATTERNS):
        return "chat"
    if len(compact) <= 2:
        return "chat"
    return "rag"


def run_chat_turn(
    settings: Any,
    history: list[dict[str, str]],
    message: str,
    collection: str,
    mode: str = "auto",
    llm: Any | None = None,
    search: Any | None = None,
    reranker: Any | None = None,
    trace_writer: TraceWriter = write_trace,
) -> ChatTurn:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if mode not in {"auto", "search", "chat"}:
        raise ValueError(f"unsupported chat mode: {mode}")
    route = decide_route(message) if mode == "auto" else ("rag" if mode == "search" else "chat")
    trace = TraceContext(
        trace_type="query",
        metadata={"query": message, "collection": collection, "chat": True, "mode": mode, "route": route},
    )
    turn = _run_turn(settings, history, message, collection, route, trace, llm, search, reranker)
    trace_writer(trace.to_dict(), path=_trace_path(settings))
    return turn


def _run_turn(settings, history, message, collection, route, trace, llm, search, reranker) -> ChatTurn:
    if route == "rag":
        try:
            results, _ = _retrieve(settings, message, collection, settings.retrieval.top_k_final, search, reranker, trace)
        except Exception as error:
            trace.record_stage("chat.error", {"error": str(error)})
            trace.finish("failed")
            return ChatTurn(message=message, route="rag", error=f"retrieval failed: {error}")
        if results:
            try:
                generated = AnswerGenerator(settings, llm=llm).generate(message, results, trace=trace)
                trace.finish("success")
                return ChatTurn(
                    message=message,
                    route="rag",
                    answer=generated.answer,
                    citations=generated.citations,
                    provider=generated.provider,
                    model=generated.model,
                )
            except AnswerGeneratorError as error:
                trace.record_stage("generation.error", {"error": str(error)})
                trace.finish("failed")
                return ChatTurn(message=message, route="rag", error=str(error))
        return _chat_reply(settings, history, message, trace, llm, note=f"No matching passages in '{collection}'; answered without the knowledge base.")
    return _chat_reply(settings, history, message, trace, llm)


def _chat_reply(settings, history, message, trace, llm, note: str | None = None) -> ChatTurn:
    client = llm
    if client is None:
        api_key = getattr(settings.llm, "api_key", "")
        if not api_key:
            error = "LLM API key is not configured; add one on the Settings page or in .env"
            trace.record_stage("chat.error", {"error": error})
            trace.finish("failed")
            return ChatTurn(message=message, route="chat", note=note, error=error)
        try:
            client = LLMFactory.create(settings.llm)
        except Exception as error:
            trace.record_stage("chat.error", {"error": str(error)})
            trace.finish("failed")
            return ChatTurn(message=message, route="chat", note=note, error=f"llm init failed: {error}")
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": message})
    started = perf_counter()
    try:
        reply = client.chat(messages)
    except Exception as error:
        trace.record_stage("chat.error", {"error": str(error)})
        trace.finish("failed")
        return ChatTurn(message=message, route="chat", note=note, error=f"chat failed: {error}")
    duration_ms = round((perf_counter() - started) * 1000, 3)
    if not isinstance(reply, str) or not reply.strip():
        trace.record_stage("chat.error", {"error": "llm returned an empty reply"})
        trace.finish("failed")
        return ChatTurn(message=message, route="chat", note=note, error="llm returned an empty reply")
    trace.record_stage(
        "chat",
        {
            "provider": client.settings.provider,
            "model": client.settings.model,
            "history_messages": len(messages) - 2,
            "reply_chars": len(reply),
        },
        duration_ms=duration_ms,
    )
    trace.finish("success")
    return ChatTurn(
        message=message,
        route="chat",
        answer=reply,
        provider=client.settings.provider,
        model=client.settings.model,
        note=note,
    )


def render() -> None:
    import streamlit as st

    from observability.dashboard import runtime

    st.title("LLM Chat")
    try:
        settings = runtime.require_settings(st)
        if settings is None:
            return
        service = DataService(settings)
        collections = service.list_collections()
    except Exception as error:
        st.error(f"Failed to load chat service: {error}")
        return
    if runtime.llm_missing_notice(st, settings):
        return

    collection = st.selectbox("Collection", collections)
    dimension_warning = _dimension_warning(settings, service.collection_dimensions(collection))
    if dimension_warning:
        st.warning(dimension_warning)
    mode_label = st.radio(
        "Retrieval mode",
        ["Auto", "Search", "Chat"],
        horizontal=True,
        help="Auto routes small talk to plain chat and knowledge questions to RAG search, decided by code (no extra LLM call).",
    )
    mode = mode_label.lower() if isinstance(mode_label, str) else "auto"
    if st.button("Clear chat"):
        st.session_state["llm_chat_history"] = []

    history: list[dict[str, Any]] = st.session_state.setdefault("llm_chat_history", [])
    for item in history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                _render_turn_meta(st, item)

    prompt = st.chat_input("Ask the knowledge base, or just chat...")
    if not prompt:
        if not history:
            st.info("Ask about your ingested documents, or say hi. Auto mode only calls the LLM once per message.")
        return

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not runtime.owner_llm_quota_ok(st):
            st.error("The server's shared LLM quota is exhausted (30/hour, 200/day). Please try again later.")
            return
        with st.spinner("Thinking..."):
            try:
                turn = run_chat_turn(settings, _llm_history(history[:-1]), prompt, collection, mode)
            except Exception as error:
                turn = ChatTurn(message=prompt, route=mode if mode != "auto" else "rag", error=str(error))
        if turn.error:
            st.error(runtime.mask(turn.error))
        else:
            st.markdown(turn.answer)
        _render_turn_meta(st, _turn_meta(turn))

    history.append({"role": "assistant", "content": turn.answer or turn.error or "", **_turn_meta(turn)})


def _llm_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": item["role"], "content": item["content"]} for item in history if item.get("content")]


def _turn_meta(turn: ChatTurn) -> dict[str, Any]:
    from observability.dashboard import runtime

    return {
        "route": turn.route,
        "provider": turn.provider,
        "model": turn.model,
        "note": turn.note,
        "error": runtime.mask(turn.error) if turn.error else None,
        "citations": list(dict.fromkeys(runtime.mask(f"[{citation.id}] {citation.source}") for citation in turn.citations)),
    }


def _render_turn_meta(st: Any, meta: dict[str, Any]) -> None:
    route = meta.get("route")
    if not route:
        return
    label = "RAG search" if route == "rag" else "chat"
    parts = [f"route: {label}"]
    if meta.get("provider"):
        parts.append(str(meta["provider"]))
    if meta.get("model"):
        parts.append(str(meta["model"]))
    if meta.get("citations"):
        parts.append(" · ".join(meta["citations"]))
    st.caption(" · ".join(parts))
    if meta.get("note"):
        st.caption(str(meta["note"]))
