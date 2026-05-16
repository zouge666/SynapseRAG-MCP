from types import SimpleNamespace

from core import Chunk
from core.settings import load_settings
from core.trace import TraceContext
from ingestion.transform import BaseTransform, MetadataEnricher
from libs.llm.base_llm import BaseLLM


class FakeLLM(BaseLLM):
    def __init__(self, response: str = '{"title":"Semantic Retrieval","summary":"Explains hybrid search metadata.","tags":["retrieval","metadata"]}', fail: bool = False) -> None:
        super().__init__(SimpleNamespace(provider="fake", model="fake"))
        self.response = response
        self.fail = fail
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        if self.fail:
            raise RuntimeError("llm failed")
        return self.response


def settings(use_llm: bool = False):
    return SimpleNamespace(ingestion=SimpleNamespace(metadata_enricher=SimpleNamespace(use_llm=use_llm)))


def chunk(text: str = "# Retrieval Pipeline\n\nHybrid search combines dense vectors and sparse BM25 scoring.", chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={"source_path": "docs/sample.pdf"}, start_offset=0, end_offset=len(text), source_ref="doc-1")


def test_metadata_enricher_is_base_transform() -> None:
    enricher = MetadataEnricher(settings())

    assert isinstance(enricher, BaseTransform)


def test_rule_mode_adds_required_metadata() -> None:
    enriched = MetadataEnricher(settings()).transform([chunk()])[0]

    assert enriched.metadata["title"] == "Retrieval Pipeline"
    assert enriched.metadata["summary"]
    assert enriched.metadata["tags"]
    assert enriched.metadata["metadata_enriched_by"] == "rule"


def test_rule_mode_uses_existing_title_and_tags() -> None:
    original = Chunk(
        id="chunk-1",
        text="Content about storage.",
        metadata={"source_path": "docs/sample.pdf", "title": "Storage Notes", "tags": ["Vector Store", "Chroma"]},
        start_offset=0,
        end_offset=22,
    )

    enriched = MetadataEnricher(settings()).transform([original])[0]

    assert enriched.metadata["title"] == "Storage Notes"
    assert enriched.metadata["tags"] == ["vector store", "chroma"]


def test_rule_mode_preserves_chunk_identity_offsets_and_source_ref() -> None:
    original = chunk()

    enriched = MetadataEnricher(settings()).transform([original])[0]

    assert enriched.id == original.id
    assert enriched.text == original.text
    assert enriched.start_offset == original.start_offset
    assert enriched.end_offset == original.end_offset
    assert enriched.source_ref == original.source_ref


def test_transform_returns_same_number_of_chunks() -> None:
    chunks = [chunk("Alpha retrieval text."), chunk("Beta storage text.", "chunk-2")]

    enriched = MetadataEnricher(settings()).transform(chunks)

    assert len(enriched) == 2


def test_transform_records_trace_stage_for_rule_mode() -> None:
    trace = TraceContext()

    MetadataEnricher(settings()).transform([chunk()], trace=trace)

    assert trace.stages[0]["name"] == "metadata_enricher"
    assert trace.stages[0]["details"]["method"] == "rule"


def test_llm_mode_calls_llm_and_uses_json_metadata() -> None:
    llm = FakeLLM()
    enricher = MetadataEnricher(settings(use_llm=True), llm=llm)

    enriched = enricher.transform([chunk()])[0]

    assert enriched.metadata["title"] == "Semantic Retrieval"
    assert enriched.metadata["summary"] == "Explains hybrid search metadata."
    assert enriched.metadata["tags"] == ["retrieval", "metadata"]
    assert enriched.metadata["metadata_enriched_by"] == "llm"
    assert "{text}" not in llm.messages[0][0]["content"]


def test_llm_mode_parses_json_inside_markdown_fence() -> None:
    response = '```json\n{"title":"Chunk Title","summary":"Chunk summary.","tags":["Chunk","LLM"]}\n```'

    enriched = MetadataEnricher(settings(use_llm=True), llm=FakeLLM(response)).transform([chunk()])[0]

    assert enriched.metadata["title"] == "Chunk Title"
    assert enriched.metadata["tags"] == ["chunk", "llm"]


def test_llm_empty_response_falls_back_to_rule() -> None:
    enriched = MetadataEnricher(settings(use_llm=True), llm=FakeLLM("")).transform([chunk()])[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert enriched.metadata["metadata_enrichment_fallback_reason"] == "invalid llm response"
    assert enriched.metadata["title"] == "Retrieval Pipeline"


def test_llm_exception_falls_back_to_rule() -> None:
    enriched = MetadataEnricher(settings(use_llm=True), llm=FakeLLM(fail=True)).transform([chunk()])[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert enriched.metadata["metadata_enrichment_fallback_reason"] == "RuntimeError"
    assert enriched.metadata["title"] == "Retrieval Pipeline"


def test_llm_mode_records_trace_stage() -> None:
    trace = TraceContext()

    MetadataEnricher(settings(use_llm=True), llm=FakeLLM()).transform([chunk()], trace=trace)

    assert trace.stages[0]["details"]["method"] == "llm"


def test_dict_settings_enable_llm() -> None:
    enricher = MetadataEnricher({"metadata_enricher": {"use_llm": True}}, llm=FakeLLM())

    assert enricher.use_llm is True


def test_default_settings_expose_metadata_enricher_config() -> None:
    settings_obj = load_settings("config/settings.yaml")

    assert settings_obj.ingestion.metadata_enricher.use_llm is False
