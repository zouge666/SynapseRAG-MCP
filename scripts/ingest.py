from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.settings import load_settings
from ingestion import IngestionPipeline, IngestionResult


PipelineFactory = Callable[[object], IngestionPipeline]


class IngestScriptError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest.py")
    parser.add_argument("--path", required=True)
    parser.add_argument("--collection", default="default")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--json", action="store_true")
    return parser


def resolve_paths(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted(item for item in source.rglob("*.pdf") if item.is_file())
        if files:
            return files
        raise IngestScriptError(f"no pdf files found under: {source}")
    raise IngestScriptError(f"path not found: {source}")


def run_ingestion(
    paths: list[Path],
    collection: str,
    force: bool,
    settings_path: str,
    pipeline_factory: PipelineFactory | None = None,
    emit_progress: bool = True,
) -> list[IngestionResult]:
    settings = load_settings(settings_path)
    pipeline = pipeline_factory(settings) if pipeline_factory is not None else IngestionPipeline(settings)
    results = []
    for path in paths:
        active_path = str(path)
        progress = _progress_printer(active_path) if emit_progress else None
        results.append(pipeline.run(active_path, collection=collection, force=force, on_progress=progress))
    return results


def main(argv: Sequence[str] | None = None, pipeline_factory: PipelineFactory | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = resolve_paths(args.path)
        results = run_ingestion(
            paths,
            collection=args.collection,
            force=args.force,
            settings_path=args.settings,
            pipeline_factory=pipeline_factory,
            emit_progress=not args.json,
        )
    except Exception as error:
        print(f"ingest failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, sort_keys=True))
    else:
        _print_summary(results)
    return 0


def _progress_printer(path: str) -> Callable[[str, int, int], None]:
    def print_progress(stage: str, current: int, total: int) -> None:
        print(f"{path}: {stage} {current}/{total}", file=sys.stderr)

    return print_progress


def _print_summary(results: list[IngestionResult]) -> None:
    for result in results:
        status = "skipped" if result.skipped else result.status
        print(
            f"{status}: {result.source_path} "
            f"collection={result.collection} chunks={result.chunk_count} images={result.image_count}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
