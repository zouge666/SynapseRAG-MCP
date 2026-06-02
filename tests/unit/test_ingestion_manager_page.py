from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from observability.dashboard.pages.ingestion_manager import _delete_document, _progress_callback, _run_ingestion, _save_uploaded_file


class FakeUploadedFile:
    name = "../sample.pdf"

    def getbuffer(self) -> bytes:
        return b"pdf"


class FakeProgress:
    def __init__(self) -> None:
        self.values = []

    def progress(self, value: float) -> None:
        self.values.append(value)


class FakeStatus:
    def __init__(self) -> None:
        self.values = []

    def write(self, value: str) -> None:
        self.values.append(value)


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def run(self, source_path: str, collection: str, force: bool = False, on_progress=None):
        self.calls.append({"source_path": source_path, "collection": collection, "force": force})
        if on_progress is not None:
            on_progress("load", 1, 2)
            on_progress("store", 2, 2)
        return FakeResult(source_path=source_path, collection=collection)


@dataclass(frozen=True)
class FakeResult:
    source_path: str
    collection: str

    def to_dict(self) -> dict:
        return {"source_path": self.source_path, "collection": self.collection, "status": "success"}


class FakeDeleteResult:
    def to_dict(self) -> dict:
        return {"source_path": "docs/a.pdf", "collection": "docs", "deleted": True}


class FakeDocumentManager:
    def __init__(self) -> None:
        self.calls = []

    def delete_document(self, source_path: str, collection: str) -> FakeDeleteResult:
        self.calls.append({"source_path": source_path, "collection": collection})
        return FakeDeleteResult()


def test_save_uploaded_file_writes_sanitized_name(tmp_path: Path) -> None:
    path = Path(_save_uploaded_file(FakeUploadedFile(), upload_dir=tmp_path))

    assert path == tmp_path / "sample.pdf"
    assert path.read_bytes() == b"pdf"


def test_progress_callback_updates_widgets() -> None:
    progress = FakeProgress()
    status = FakeStatus()
    callback = _progress_callback(progress, status)

    callback("split", 2, 4)

    assert progress.values == [0.5]
    assert status.values == ["split 2/4"]


def test_run_ingestion_saves_upload_and_passes_progress(tmp_path: Path) -> None:
    progress = FakeProgress()
    status = FakeStatus()
    pipeline = FakePipeline()

    result = _run_ingestion(FakeUploadedFile(), "docs", True, SimpleNamespace(), progress, status, pipeline=pipeline)

    assert result["status"] == "success"
    assert pipeline.calls[0]["collection"] == "docs"
    assert pipeline.calls[0]["force"] is True
    assert Path(pipeline.calls[0]["source_path"]).name == "sample.pdf"
    assert progress.values == [0.5, 1.0]
    assert status.values == ["load 1/2", "store 2/2"]


def test_delete_document_calls_document_manager() -> None:
    document_manager = FakeDocumentManager()
    service = SimpleNamespace(document_manager=document_manager)

    result = _delete_document(service, "docs/a.pdf", "docs")

    assert result["deleted"] is True
    assert document_manager.calls == [{"source_path": "docs/a.pdf", "collection": "docs"}]
