import pytest

from core.query_engine import ProcessedQuery, QueryProcessor, QueryProcessorError
from core.trace import TraceContext


def test_process_extracts_keywords_from_query() -> None:
    result = QueryProcessor().process("What is hybrid retrieval for Azure OpenAI?")

    assert result.query == "What is hybrid retrieval for Azure OpenAI?"
    assert result.normalized_query == "what is hybrid retrieval for azure openai?"
    assert result.keywords == ["what", "hybrid", "retrieval", "azure", "openai"]
    assert result.filters == {}


def test_process_removes_stop_words_and_deduplicates_keywords() -> None:
    result = QueryProcessor().process("The vector store and the vector index")

    assert result.keywords == ["vector", "store", "index"]


def test_process_parses_inline_filters() -> None:
    result = QueryProcessor().process("collection:docs doc_type=pdf explain chunking")

    assert result.keywords == ["explain", "chunking"]
    assert result.filters == {"collection": "docs", "doc_type": "pdf"}
    assert result.normalized_query == "explain chunking"


def test_process_supports_quoted_and_typed_filter_values() -> None:
    result = QueryProcessor().process("collection:'team docs' page=12 include_images=true")

    assert result.filters == {"collection": "team docs", "page": 12, "include_images": True}
    assert result.keywords == ["team", "docs", "12", "true"]


def test_explicit_filters_override_inline_filters() -> None:
    result = QueryProcessor().process("collection:docs summarize ingestion", filters={"collection": "notes", "doc_type": "pdf"})

    assert result.keywords == ["summarize", "ingestion"]
    assert result.filters == {"collection": "notes", "doc_type": "pdf"}


def test_filter_key_allowlist_ignores_unknown_inline_filters() -> None:
    result = QueryProcessor(filter_keys={"collection"}).process("collection:docs url:https://example.test retrieval")

    assert result.filters == {"collection": "docs"}
    assert "url" in result.keywords


def test_process_supports_chinese_keywords() -> None:
    result = QueryProcessor().process("检索 系统 collection:docs")

    assert result.keywords == ["检", "索", "系", "统"]
    assert result.filters == {"collection": "docs"}


def test_processed_query_serializes_to_dict() -> None:
    result = QueryProcessor().process("collection:docs alpha beta")

    assert result.to_dict() == {
        "query": "collection:docs alpha beta",
        "normalized_query": "alpha beta",
        "keywords": ["alpha", "beta"],
        "filters": {"collection": "docs"},
    }


def test_invalid_query_raises() -> None:
    with pytest.raises(QueryProcessorError, match="query"):
        QueryProcessor().process("  ")


def test_invalid_filters_raise() -> None:
    with pytest.raises(QueryProcessorError, match="filters"):
        QueryProcessor().process("alpha", filters=["bad"])


def test_trace_records_query_processor_stage() -> None:
    trace = TraceContext(trace_type="query")

    QueryProcessor().process("collection:docs alpha beta", trace=trace)

    assert trace.stages[0]["name"] == "query_processor"
    assert trace.stages[0]["details"] == {"keyword_count": 2, "filters": {"collection": "docs"}}


def test_query_processor_can_be_imported_from_package() -> None:
    from core.query_engine import QueryProcessor as ExportedQueryProcessor

    assert ExportedQueryProcessor is QueryProcessor


def test_processed_query_requires_keywords() -> None:
    with pytest.raises(QueryProcessorError, match="keywords"):
        ProcessedQuery(query="alpha", normalized_query="alpha", keywords=[], filters={})
