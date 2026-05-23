import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import Chunk
from core.settings import load_settings
from core.trace import TraceContext
from ingestion.transform import BaseTransform, ChunkRefiner
from libs.llm.base_llm import BaseLLM


class FakeLLM(BaseLLM):
    def __init__(self, response: str = "clean llm text", fail: bool = False) -> None:
        super().__init__(SimpleNamespace(provider="fake", model="fake"))
        self.response = response
        self.fail = fail
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        if self.fail:
            raise RuntimeError("llm failed")
        return self.response


def settings(use_llm: bool = False, prompt_path: str = "config/prompts/chunk_refinement.txt"):
    return SimpleNamespace(ingestion=SimpleNamespace(chunk_refiner=SimpleNamespace(use_llm=use_llm, prompt_path=prompt_path)))


def chunk(text: str = "Alpha text", chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={"source_path": "docs/sample.pdf"}, start_offset=0, end_offset=len(text), source_ref="doc-1")


def fixture_cases():
    return json.loads(Path("tests/fixtures/noisy_chunks.json").read_text(encoding="utf-8"))


def case(case_id: str) -> dict:
    return next(item for item in fixture_cases() if item["id"] == case_id)


def test_chunk_refiner_is_base_transform() -> None:
    refiner = ChunkRefiner(settings())

    assert isinstance(refiner, BaseTransform)


def test_rule_refines_typical_noise_scenario() -> None:
    item = case("typical_noise_scenario")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_rule_refines_ocr_errors() -> None:
    item = case("ocr_errors")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_rule_removes_page_headers_and_footers() -> None:
    item = case("page_header_footer")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_rule_collapses_excessive_whitespace() -> None:
    item = case("excessive_whitespace")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_rule_removes_format_markers() -> None:
    item = case("format_markers")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_rule_preserves_clean_text() -> None:
    item = case("clean_text")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_rule_preserves_code_block_spacing() -> None:
    item = case("code_blocks")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_rule_refines_mixed_noise() -> None:
    item = case("mixed_noise")

    assert ChunkRefiner(settings())._rule_based_refine(item["input"]) == item["expected"]


def test_transform_returns_same_number_of_chunks() -> None:
    chunks = [chunk("Alpha"), chunk("Beta", "chunk-2")]

    refined = ChunkRefiner(settings()).transform(chunks)

    assert len(refined) == 2


def test_transform_marks_rule_metadata() -> None:
    refined = ChunkRefiner(settings()).transform([chunk("Alpha   beta")])[0]

    assert refined.metadata["refined_by"] == "rule"
    assert refined.metadata["refinement_changed"] is True


def test_transform_keeps_chunk_identity_offsets_and_source_ref() -> None:
    original = chunk("Alpha   beta")

    refined = ChunkRefiner(settings()).transform([original])[0]

    assert refined.id == original.id
    assert refined.start_offset == original.start_offset
    assert refined.end_offset == original.end_offset
    assert refined.source_ref == original.source_ref


def test_transform_preserves_existing_metadata() -> None:
    original = Chunk(id="chunk-1", text="Alpha", metadata={"source_path": "docs/sample.pdf", "title": "Sample"}, start_offset=0, end_offset=5)

    refined = ChunkRefiner(settings()).transform([original])[0]

    assert refined.metadata["title"] == "Sample"


def test_transform_records_trace_stage_for_rule_mode() -> None:
    trace = TraceContext()

    ChunkRefiner(settings()).transform([chunk("Alpha")], trace=trace)

    assert trace.stages[0]["name"] == "chunk_refiner"
    assert trace.stages[0]["details"]["method"] == "rule"


def test_llm_mode_calls_llm_with_prompt() -> None:
    llm = FakeLLM("LLM clean text")
    refiner = ChunkRefiner(settings(use_llm=True), llm=llm)

    refined = refiner.transform([chunk("Alpha   beta")])[0]

    assert refined.text == "LLM clean text"
    assert "{text}" not in llm.messages[0][0]["content"]


def test_llm_mode_marks_llm_metadata() -> None:
    refined = ChunkRefiner(settings(use_llm=True), llm=FakeLLM("LLM clean text")).transform([chunk("Alpha")])[0]

    assert refined.metadata["refined_by"] == "llm"


def test_llm_mode_records_trace_stage() -> None:
    trace = TraceContext()

    ChunkRefiner(settings(use_llm=True), llm=FakeLLM("LLM clean text")).transform([chunk("Alpha")], trace=trace)

    assert trace.stages[0]["details"]["method"] == "llm"


def test_llm_empty_response_falls_back_to_rule() -> None:
    refined = ChunkRefiner(settings(use_llm=True), llm=FakeLLM("")).transform([chunk("Alpha   beta")])[0]

    assert refined.text == "Alpha beta"
    assert refined.metadata["refined_by"] == "rule"
    assert refined.metadata["fallback_reason"] == "empty llm response"


def test_llm_exception_falls_back_to_rule() -> None:
    refined = ChunkRefiner(settings(use_llm=True), llm=FakeLLM(fail=True)).transform([chunk("Alpha   beta")])[0]

    assert refined.text == "Alpha beta"
    assert refined.metadata["refined_by"] == "rule"
    assert refined.metadata["fallback_reason"] == "RuntimeError"


def test_prompt_loader_reads_file_with_text_placeholder(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Clean this:\n{text}", encoding="utf-8")

    refiner = ChunkRefiner(settings(prompt_path=str(prompt_path)))

    assert refiner.prompt_template == "Clean this:\n{text}"


def test_prompt_loader_appends_text_placeholder_when_missing(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Clean this", encoding="utf-8")

    refiner = ChunkRefiner(settings(prompt_path=str(prompt_path)))

    assert refiner.prompt_template == "Clean this\n\n{text}"


def test_prompt_loader_uses_fallback_when_missing(tmp_path) -> None:
    refiner = ChunkRefiner(settings(prompt_path=str(tmp_path / "missing.txt")))

    assert "{text}" in refiner.prompt_template


def test_dict_settings_disable_llm() -> None:
    refiner = ChunkRefiner({"chunk_refiner": {"use_llm": False}})

    assert refiner.use_llm is False


def test_dict_settings_enable_llm() -> None:
    refiner = ChunkRefiner({"chunk_refiner": {"use_llm": True}}, llm=FakeLLM("ok"))

    assert refiner.use_llm is True


def test_transform_continues_when_one_chunk_fails(monkeypatch) -> None:
    refiner = ChunkRefiner(settings())
    calls = {"count": 0}

    def fail_once(text: str) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("bad chunk")
        return text

    monkeypatch.setattr(refiner, "_rule_based_refine", fail_once)

    refined = refiner.transform([chunk("Alpha"), chunk("Beta", "chunk-2")])

    assert refined[0].metadata["refined_by"] == "original"
    assert refined[1].metadata["refined_by"] == "rule"


def test_rule_removes_html_comments() -> None:
    text = "Alpha <!-- hidden --> beta"

    assert ChunkRefiner(settings())._rule_based_refine(text) == "Alpha beta"


def test_rule_preserves_markdown_heading_structure() -> None:
    text = "# **Heading**\n\nBody   text"

    assert ChunkRefiner(settings())._rule_based_refine(text) == "# Heading\n\nBody text"


def test_trace_context_finish_serializes_state() -> None:
    trace = TraceContext(trace_type="ingestion", metadata={"source_path": "docs/sample.pdf"})
    trace.record_stage("chunk_refiner", {"count": 1})

    data = trace.finish()

    assert data["status"] == "success"
    assert data["trace_type"] == "ingestion"
    assert data["metadata"]["source_path"] == "docs/sample.pdf"
    assert data["stages"][0]["name"] == "chunk_refiner"


def test_default_settings_expose_chunk_refiner_config() -> None:
    settings_obj = load_settings("config/settings.yaml")

    assert settings_obj.ingestion.chunk_refiner.use_llm is False
    assert settings_obj.ingestion.chunk_refiner.prompt_path == "config/prompts/chunk_refinement.txt"
