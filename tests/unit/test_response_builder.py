import pytest

from core import RetrievalResult
from core.response import CitationGenerator, CitationGeneratorError, ResponseBuilder, ResponseBuilderError


def result(chunk_id: str, score: float, text: str, page: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": f"docs/{chunk_id}.pdf", "page": page},
    )


def test_citation_generator_extracts_source_page_chunk_and_score() -> None:
    citations = CitationGenerator().generate([result("a", 0.92, "alpha text", page=5)])

    assert citations[0].to_dict() == {
        "id": 1,
        "source": "docs/a.pdf",
        "page": 5,
        "chunk_id": "a",
        "score": 0.92,
        "text": "alpha text",
    }


def test_response_builder_returns_markdown_content_and_structured_citations() -> None:
    response = ResponseBuilder().build(
        [
            result("a", 0.92, "alpha retrieval text", page=5),
            result("b", 0.81, "beta retrieval text", page=7),
        ],
        "retrieval",
    )

    text = response["content"][0]["text"]
    assert response["content"][0]["type"] == "text"
    assert "[1] alpha retrieval text" in text
    assert "[2] beta retrieval text" in text
    assert response["structuredContent"]["citations"][0]["source"] == "docs/a.pdf"
    assert response["structuredContent"]["citations"][1]["chunk_id"] == "b"


def test_response_builder_returns_friendly_empty_result() -> None:
    response = ResponseBuilder().build([], "missing")

    assert response["content"] == [{"type": "text", "text": "未找到与「missing」相关的内容。"}]
    assert response["structuredContent"] == {"answer": "未找到与「missing」相关的内容。", "citations": []}


def test_response_builder_validates_inputs() -> None:
    with pytest.raises(ResponseBuilderError, match="query"):
        ResponseBuilder().build([], "")
    with pytest.raises(ResponseBuilderError, match="retrieval_results"):
        ResponseBuilder().build(["bad"], "query")
    with pytest.raises(CitationGeneratorError, match="retrieval_results"):
        CitationGenerator().generate(["bad"])
