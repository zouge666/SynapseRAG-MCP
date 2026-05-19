from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.query_engine import HybridSearch, Reranker
from core.response import ResponseBuilder
from core.settings import load_settings
from core.trace import TraceContext
from mcp_server.protocol_handler import ToolDefinition


SearchFactory = Callable[[object], HybridSearch]
RerankerFactory = Callable[[object], Reranker]


class QueryKnowledgeHubError(ValueError):
    pass


def query_knowledge_hub(
    query: str,
    top_k: int = 5,
    collection: str | None = None,
    settings: object | None = None,
    settings_path: str = "config/settings.yaml",
    search_factory: SearchFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
    response_builder: ResponseBuilder | None = None,
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise QueryKnowledgeHubError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise QueryKnowledgeHubError("top_k must be a positive integer")
    active_settings = settings or load_settings(settings_path)
    builder = response_builder or ResponseBuilder()
    if search_factory is None and not _has_ingested_data(active_settings, collection):
        return builder.build([], query)
    filters = {"collection": collection} if collection else {}
    trace = TraceContext(trace_type="query", metadata={"query": query, "filters": filters, "tool": "query_knowledge_hub"})
    search = search_factory(active_settings) if search_factory is not None else HybridSearch(active_settings)
    candidates = search.search(query, top_k=top_k, filters=filters, trace=trace)
    if not candidates:
        trace.finish("success")
        return builder.build([], query)
    reranker = reranker_factory(active_settings) if reranker_factory is not None else Reranker(active_settings)
    results = reranker.rerank(query, candidates, trace=trace)
    trace.finish("success")
    return builder.build(results, query)


def query_knowledge_hub_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise QueryKnowledgeHubError("arguments must be a mapping")
    return query_knowledge_hub(
        query=arguments.get("query"),
        top_k=arguments.get("top_k", 5),
        collection=arguments.get("collection"),
    )


def query_knowledge_hub_tool_definition(
    search_factory: SearchFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
    settings: object | None = None,
) -> ToolDefinition:
    def handle(arguments: dict[str, Any]) -> dict[str, Any]:
        return query_knowledge_hub(
            query=arguments.get("query"),
            top_k=arguments.get("top_k", 5),
            collection=arguments.get("collection"),
            settings=settings,
            search_factory=search_factory,
            reranker_factory=reranker_factory,
        )

    return ToolDefinition(
        name="query_knowledge_hub",
        description="Search the local knowledge hub with hybrid retrieval and reranking.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "collection": {"type": "string"},
            },
            "required": ["query"],
        },
        handler=handle,
    )


def _has_ingested_data(settings: object, collection: str | None) -> bool:
    return _vector_store_has_records(settings, collection) or _bm25_has_index(settings)


def _vector_store_has_records(settings: object, collection: str | None) -> bool:
    vector_store = _setting(settings, "vector_store", {})
    persist_path = _setting(vector_store, "persist_path", "data/db/chroma")
    active_collection = collection or _setting(vector_store, "collection", "default")
    store_path = Path(str(persist_path)) / f"{_safe_collection_name(str(active_collection))}.json"
    if not store_path.exists():
        return False
    try:
        with store_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return True
    records = data.get("records") if isinstance(data, dict) else None
    return isinstance(records, list) and bool(records)


def _bm25_has_index(settings: object) -> bool:
    ingestion = _setting(settings, "ingestion", {})
    path = _setting(ingestion, "bm25_path", "data/db/bm25")
    return (Path(str(path)) / "index.pkl").exists()


def _setting(source: object, name: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _safe_collection_name(collection: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection).strip("._")
    return value or "default"
