from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core import RetrievalResult
from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, RRFusion, Reranker, SparseRetriever
from core.settings import load_settings
from core.trace import TraceContext


SearchFactory = Callable[[object], HybridSearch]
RerankerFactory = Callable[[object], Reranker]


@dataclass(frozen=True)
class QueryScriptResult:
    query: str
    candidates: list[RetrievalResult]
    results: list[RetrievalResult]
    trace: TraceContext
    rerank_applied: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="query.py")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collection")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--settings", default="config/settings.yaml")
    return parser


def run_query(
    query: str,
    top_k: int,
    collection: str | None,
    no_rerank: bool,
    settings_path: str,
    search_factory: SearchFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
) -> QueryScriptResult:
    if not isinstance(top_k, int) or top_k <= 0:
        raise QueryScriptError("top_k must be a positive integer")
    settings = load_settings(settings_path)
    filters = {"collection": collection} if collection else {}
    trace = TraceContext(trace_type="query", metadata={"query": query, "filters": filters})
    if search_factory is None and not has_ingested_data(settings, collection):
        trace.record_stage("query.preflight", {"has_data": False, "collection": collection or ""})
        trace.finish("success")
        return QueryScriptResult(query=query, candidates=[], results=[], trace=trace, rerank_applied=False)
    search = search_factory(settings) if search_factory is not None else build_search(settings)
    candidates = search.search(query, top_k=top_k, filters=filters, trace=trace)
    if no_rerank or not candidates:
        if no_rerank:
            trace.record_stage("reranker", {"skipped": True, "count": len(candidates)})
        trace.finish("success")
        return QueryScriptResult(query=query, candidates=candidates, results=list(candidates), trace=trace, rerank_applied=False)
    reranker = reranker_factory(settings) if reranker_factory is not None else Reranker(settings)
    results = reranker.rerank(query, candidates, trace=trace)
    trace.finish("success")
    return QueryScriptResult(query=query, candidates=candidates, results=results, trace=trace, rerank_applied=True)


def build_search(settings: object) -> HybridSearch:
    return HybridSearch(
        settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(settings),
        sparse_retriever=SparseRetriever(settings),
        fusion=RRFusion(),
    )


def has_ingested_data(settings: object, collection: str | None = None) -> bool:
    return _vector_store_has_records(settings, collection) or _bm25_has_index(settings)


def main(
    argv: Sequence[str] | None = None,
    search_factory: SearchFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_query(
            query=args.query,
            top_k=args.top_k,
            collection=args.collection,
            no_rerank=args.no_rerank,
            settings_path=args.settings,
            search_factory=search_factory,
            reranker_factory=reranker_factory,
        )
    except Exception as error:
        print(f"query failed: {error}", file=sys.stderr)
        return 1
    if args.verbose:
        print("Candidates before rerank:")
        print_results(result.candidates)
        print("")
        print("Final results:")
    print_results(result.results)
    if args.verbose:
        print("")
        print_trace(result.trace)
    return 0


class QueryScriptError(ValueError):
    pass


def print_results(results: list[RetrievalResult]) -> None:
    if not results:
        print("未找到相关文档，请先运行 ingest.py 摄取数据")
        return
    for index, result in enumerate(results, start=1):
        metadata = result.metadata
        source = metadata.get("source_path") or metadata.get("source") or "unknown"
        page = metadata.get("page", metadata.get("page_number", "-"))
        print(f"{index}. score={result.score:.4f} source={source} page={page}")
        print(f"   {summarize_text(result.text)}")


def print_trace(trace: TraceContext) -> None:
    print("Trace:")
    for stage in trace.stages:
        details = json.dumps(stage["details"], ensure_ascii=False, sort_keys=True)
        print(f"- {stage['name']}: {details}")


def summarize_text(text: str, limit: int = 160) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


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


if __name__ == "__main__":
    raise SystemExit(main())
