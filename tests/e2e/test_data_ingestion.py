from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ingestion import IngestionResult


INGEST_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"
SPEC = importlib.util.spec_from_file_location("ingest_script", INGEST_PATH)
assert SPEC is not None and SPEC.loader is not None
ingest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ingest)


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []
        self.seen = set()

    def run(self, source_path: str, collection: str = "default", force: bool = False, on_progress=None) -> IngestionResult:
        self.calls.append({"source_path": source_path, "collection": collection, "force": force})
        if on_progress is not None:
            on_progress("integrity", 1, 7)
            on_progress("store", 7, 7)
        skipped = source_path in self.seen and not force
        self.seen.add(source_path)
        return IngestionResult(
            source_path=source_path,
            collection=collection,
            file_hash=f"hash-{len(self.calls)}",
            status="skipped" if skipped else "success",
            chunk_count=0 if skipped else 2,
            vector_ids=[] if skipped else ["vec-1", "vec-2"],
            image_count=0 if skipped else 1,
            skipped=skipped,
        )


def write_settings(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "app:",
                "  name: test",
                "llm:",
                "  provider: fake",
                "  model: fake-llm",
                "embedding:",
                "  provider: fake",
                "  model: fake-embedding",
                "vector_store:",
                "  backend: fake",
                "  persist_path: data/db/vector",
                "splitter:",
                "  provider: recursive",
                "  chunk_size: 1000",
                "  chunk_overlap: 0",
                "retrieval:",
                "  sparse_backend: bm25",
                "  fusion_algorithm: rrf",
                "  top_k_dense: 20",
                "  top_k_sparse: 20",
                "  top_k_final: 5",
                "rerank:",
                "  enabled: false",
                "  backend: none",
                "evaluation:",
                "  enabled: false",
                "  backends: []",
                "observability:",
                "  log_path: logs/app.log",
                "  trace_path: logs/traces.jsonl",
            ]
        ),
        encoding="utf-8",
    )


def test_ingest_script_runs_file_path_and_prints_summary(tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / "settings.yaml"
    pdf_path = tmp_path / "sample.pdf"
    write_settings(settings_path)
    pdf_path.write_bytes(b"%PDF sample")
    fake = FakePipeline()

    code = ingest.main(["--path", str(pdf_path), "--collection", "docs", "--settings", str(settings_path)], pipeline_factory=lambda settings: fake)

    captured = capsys.readouterr()
    assert code == 0
    assert fake.calls == [{"source_path": str(pdf_path), "collection": "docs", "force": False}]
    assert "success:" in captured.out
    assert "chunks=2" in captured.out
    assert "sample.pdf: integrity 1/7" in captured.err


def test_ingest_script_discovers_pdf_files_in_directory(tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / "settings.yaml"
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    write_settings(settings_path)
    (docs_path / "b.pdf").write_bytes(b"%PDF b")
    (docs_path / "a.pdf").write_bytes(b"%PDF a")
    (docs_path / "notes.txt").write_text("ignore", encoding="utf-8")
    fake = FakePipeline()

    code = ingest.main(["--path", str(docs_path), "--collection", "docs", "--settings", str(settings_path)], pipeline_factory=lambda settings: fake)

    assert code == 0
    assert [Path(call["source_path"]).name for call in fake.calls] == ["a.pdf", "b.pdf"]
    assert capsys.readouterr().out.count("success:") == 2


def test_ingest_script_outputs_json_and_allows_incremental_skip(tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / "settings.yaml"
    pdf_path = tmp_path / "sample.pdf"
    write_settings(settings_path)
    pdf_path.write_bytes(b"%PDF sample")
    fake = FakePipeline()

    first = ingest.main(["--path", str(pdf_path), "--settings", str(settings_path), "--json"], pipeline_factory=lambda settings: fake)
    second = ingest.main(["--path", str(pdf_path), "--settings", str(settings_path), "--json"], pipeline_factory=lambda settings: fake)

    output = capsys.readouterr().out.strip().splitlines()
    assert first == 0
    assert second == 0
    assert json.loads(output[0])[0]["status"] == "success"
    assert json.loads(output[1])[0]["status"] == "skipped"


def test_ingest_script_force_reprocesses_existing_file(tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / "settings.yaml"
    pdf_path = tmp_path / "sample.pdf"
    write_settings(settings_path)
    pdf_path.write_bytes(b"%PDF sample")
    fake = FakePipeline()

    ingest.main(["--path", str(pdf_path), "--settings", str(settings_path)], pipeline_factory=lambda settings: fake)
    code = ingest.main(["--path", str(pdf_path), "--settings", str(settings_path), "--force"], pipeline_factory=lambda settings: fake)

    assert code == 0
    assert fake.calls[-1]["force"] is True
    assert "success:" in capsys.readouterr().out


def test_ingest_script_reports_missing_path(tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / "settings.yaml"
    write_settings(settings_path)

    code = ingest.main(["--path", str(tmp_path / "missing.pdf"), "--settings", str(settings_path)], pipeline_factory=lambda settings: FakePipeline())

    captured = capsys.readouterr()
    assert code == 1
    assert "path not found" in captured.err


def test_resolve_paths_rejects_directory_without_pdf(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("not a pdf", encoding="utf-8")

    try:
        ingest.resolve_paths(tmp_path)
    except ingest.IngestScriptError as error:
        assert "no pdf files" in str(error)
    else:
        raise AssertionError("expected IngestScriptError")
