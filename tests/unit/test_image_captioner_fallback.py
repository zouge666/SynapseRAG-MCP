from types import SimpleNamespace

from core import Chunk
from core.settings import load_settings
from core.trace import TraceContext
from ingestion.transform import BaseTransform, ImageCaptioner
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMResponse


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, response: str = "A chart showing retrieval quality.", fail: bool = False) -> None:
        super().__init__(SimpleNamespace(provider="fake", model="fake-vision"))
        self.response = response
        self.fail = fail
        self.calls = []

    def chat_with_image(self, text: str, image_path: str | bytes, trace: object | None = None) -> VisionLLMResponse:
        self.calls.append({"text": text, "image_path": image_path, "trace": trace})
        if self.fail:
            raise RuntimeError("vision failed")
        return VisionLLMResponse(text=self.response, metadata={"provider": "fake"})


def settings(enabled: bool = False, prompt_path: str = "config/prompts/image_captioning.txt"):
    return SimpleNamespace(ingestion=SimpleNamespace(image_captioner=SimpleNamespace(enabled=enabled, prompt_path=prompt_path)))


def chunk_with_image(text: str = "Alpha [IMAGE: img-1] text") -> Chunk:
    return Chunk(
        id="chunk-1",
        text=text,
        metadata={
            "source_path": "docs/sample.pdf",
            "image_refs": ["img-1"],
            "images": [
                {
                    "id": "img-1",
                    "path": "data/images/doc-1/img-1.png",
                    "text_offset": 6,
                    "text_length": 14,
                }
            ],
        },
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_image_captioner_is_base_transform() -> None:
    captioner = ImageCaptioner(settings())

    assert isinstance(captioner, BaseTransform)


def test_chunk_without_image_refs_is_unchanged() -> None:
    original = Chunk(id="chunk-1", text="Plain text", metadata={"source_path": "docs/sample.pdf"}, start_offset=0, end_offset=10)

    captioned = ImageCaptioner(settings(enabled=True), vision_llm=FakeVisionLLM()).transform([original])[0]

    assert captioned is original


def test_disabled_mode_keeps_image_refs_and_marks_unprocessed() -> None:
    original = chunk_with_image()

    captioned = ImageCaptioner(settings(enabled=False), vision_llm=FakeVisionLLM()).transform([original])[0]

    assert captioned.metadata["image_refs"] == ["img-1"]
    assert "image_captions" not in captioned.metadata
    assert captioned.metadata["has_unprocessed_images"] is True
    assert captioned.metadata["image_captioned_by"] == "disabled"


def test_enabled_mode_generates_caption_metadata() -> None:
    vision_llm = FakeVisionLLM("An architecture diagram with ingestion stages.")

    captioned = ImageCaptioner(settings(enabled=True), vision_llm=vision_llm).transform([chunk_with_image()])[0]

    assert captioned.metadata["image_captions"] == [{"image_id": "img-1", "caption": "An architecture diagram with ingestion stages."}]
    assert captioned.metadata["has_unprocessed_images"] is False
    assert captioned.metadata["image_captioned_by"] == "vision_llm"
    assert vision_llm.calls[0]["image_path"] == "data/images/doc-1/img-1.png"


def test_enabled_mode_preserves_chunk_identity_offsets_and_text() -> None:
    original = chunk_with_image()

    captioned = ImageCaptioner(settings(enabled=True), vision_llm=FakeVisionLLM()).transform([original])[0]

    assert captioned.id == original.id
    assert captioned.text == original.text
    assert captioned.start_offset == original.start_offset
    assert captioned.end_offset == original.end_offset
    assert captioned.source_ref == original.source_ref


def test_enabled_mode_formats_prompt_with_chunk_context() -> None:
    vision_llm = FakeVisionLLM()

    ImageCaptioner(settings(enabled=True), vision_llm=vision_llm, prompt_template="Describe {image_id} from {source_path}: {text}").transform([chunk_with_image()])[0]

    assert "img-1" in vision_llm.calls[0]["text"]
    assert "docs/sample.pdf" in vision_llm.calls[0]["text"]
    assert "Alpha [IMAGE: img-1] text" in vision_llm.calls[0]["text"]


def test_vision_exception_falls_back_without_caption() -> None:
    captioned = ImageCaptioner(settings(enabled=True), vision_llm=FakeVisionLLM(fail=True)).transform([chunk_with_image()])[0]

    assert "image_captions" not in captioned.metadata
    assert captioned.metadata["has_unprocessed_images"] is True
    assert captioned.metadata["image_captioned_by"] == "fallback"
    assert captioned.metadata["image_caption_errors"] == [{"image_id": "img-1", "reason": "RuntimeError"}]


def test_missing_image_metadata_falls_back() -> None:
    original = Chunk(
        id="chunk-1",
        text="Alpha [IMAGE: missing]",
        metadata={"source_path": "docs/sample.pdf", "image_refs": ["missing"], "images": []},
        start_offset=0,
        end_offset=22,
    )

    captioned = ImageCaptioner(settings(enabled=True), vision_llm=FakeVisionLLM()).transform([original])[0]

    assert captioned.metadata["has_unprocessed_images"] is True
    assert captioned.metadata["image_caption_errors"] == [{"image_id": "missing", "reason": "missing image metadata"}]


def test_partial_caption_marks_remaining_images_unprocessed() -> None:
    original = Chunk(
        id="chunk-1",
        text="Alpha [IMAGE: img-1] [IMAGE: missing]",
        metadata={
            "source_path": "docs/sample.pdf",
            "image_refs": ["img-1", "missing"],
            "images": [{"id": "img-1", "path": "data/images/doc-1/img-1.png", "text_offset": 6, "text_length": 14}],
        },
        start_offset=0,
        end_offset=38,
    )

    captioned = ImageCaptioner(settings(enabled=True), vision_llm=FakeVisionLLM("Caption one")).transform([original])[0]

    assert captioned.metadata["image_captions"] == [{"image_id": "img-1", "caption": "Caption one"}]
    assert captioned.metadata["has_unprocessed_images"] is True
    assert captioned.metadata["image_caption_errors"] == [{"image_id": "missing", "reason": "missing image metadata"}]


def test_prompt_loader_reads_file(tmp_path) -> None:
    prompt_path = tmp_path / "image_prompt.txt"
    prompt_path.write_text("Describe exactly.", encoding="utf-8")

    captioner = ImageCaptioner(settings(prompt_path=str(prompt_path)))

    assert captioner.prompt_template == "Describe exactly."


def test_prompt_loader_uses_fallback_when_missing(tmp_path) -> None:
    captioner = ImageCaptioner(settings(prompt_path=str(tmp_path / "missing.txt")))

    assert "Describe the image" in captioner.prompt_template


def test_dict_settings_enable_captioner() -> None:
    captioner = ImageCaptioner({"image_captioner": {"enabled": True}}, vision_llm=FakeVisionLLM())

    assert captioner.enabled is True


def test_trace_records_enabled_caption_stage() -> None:
    trace = TraceContext()

    ImageCaptioner(settings(enabled=True), vision_llm=FakeVisionLLM()).transform([chunk_with_image()], trace=trace)

    assert trace.stages[0]["name"] == "image_captioner"
    assert trace.stages[0]["details"]["method"] == "vision_llm"


def test_default_settings_expose_image_captioner_config() -> None:
    settings_obj = load_settings("config/settings.yaml")

    assert settings_obj.ingestion.image_captioner.enabled is False
    assert settings_obj.ingestion.image_captioner.prompt_path == "config/prompts/image_captioning.txt"
