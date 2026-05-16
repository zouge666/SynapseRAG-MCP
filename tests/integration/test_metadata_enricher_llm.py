import os

import pytest

from core import Chunk
from core.settings import load_settings
from ingestion.transform import MetadataEnricher
from libs.llm.llm_factory import LLMFactory


def chunk(text: str) -> Chunk:
    return Chunk(id="chunk-1", text=text, metadata={"source_path": "docs/metadata.pdf"}, start_offset=0, end_offset=len(text))


@pytest.mark.integration
def test_metadata_enricher_real_llm_enriches_when_enabled() -> None:
    if os.environ.get("RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("RUN_REAL_LLM_TESTS is not enabled")
    settings = load_settings("config/settings.yaml")
    llm = LLMFactory.create(settings)
    enricher = MetadataEnricher({"metadata_enricher": {"use_llm": True}}, llm=llm)

    enriched = enricher.transform([chunk("Hybrid retrieval combines dense vectors, sparse BM25 scores, and optional reranking.")])[0]

    assert enriched.metadata["title"]
    assert enriched.metadata["summary"]
    assert enriched.metadata["tags"]
    assert enriched.metadata["metadata_enriched_by"] in {"llm", "rule"}


@pytest.mark.integration
def test_metadata_enricher_invalid_llm_falls_back() -> None:
    class BrokenLLM:
        def chat(self, messages):
            raise RuntimeError("invalid model")

    enricher = MetadataEnricher({"metadata_enricher": {"use_llm": True}}, llm=BrokenLLM())

    enriched = enricher.transform([chunk("# Storage Layer\n\nVector upsert writes chunk metadata and embeddings.")])[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert enriched.metadata["metadata_enrichment_fallback_reason"] == "RuntimeError"
    assert enriched.metadata["title"] == "Storage Layer"
