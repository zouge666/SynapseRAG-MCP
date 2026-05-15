import os

import pytest

from core import Chunk
from core.settings import load_settings
from ingestion.transform import ChunkRefiner
from libs.llm.llm_factory import LLMFactory


def chunk(text: str) -> Chunk:
    return Chunk(id="chunk-1", text=text, metadata={"source_path": "docs/noisy.pdf"}, start_offset=0, end_offset=len(text))


@pytest.mark.integration
def test_chunk_refiner_real_llm_refines_when_enabled() -> None:
    if os.environ.get("RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("RUN_REAL_LLM_TESTS is not enabled")
    settings = load_settings("config/settings.yaml")
    llm = LLMFactory.create(settings)
    refiner = ChunkRefiner({"chunk_refiner": {"use_llm": True}}, llm=llm)

    refined = refiner.transform([chunk("Header: draft\n\nThis   chunk explains retrieval.\n\nPage 1")])[0]

    assert refined.metadata["refined_by"] in {"llm", "rule"}
    assert "Header:" not in refined.text
    assert "Page 1" not in refined.text


@pytest.mark.integration
def test_chunk_refiner_invalid_llm_falls_back() -> None:
    class BrokenLLM:
        def chat(self, messages):
            raise RuntimeError("invalid model")

    refiner = ChunkRefiner({"chunk_refiner": {"use_llm": True}}, llm=BrokenLLM())

    refined = refiner.transform([chunk("Header: draft\n\nImportant   content\n\nPage 1")])[0]

    assert refined.metadata["refined_by"] == "rule"
    assert refined.metadata["fallback_reason"] == "RuntimeError"
    assert refined.text == "Important content"
