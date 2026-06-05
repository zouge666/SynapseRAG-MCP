from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.query_engine import HybridSearch
from core.settings import load_settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.eval_runner import EvalReport, EvalRunner
from query import has_ingested_data


SearchFactory = Callable[[object], object]
EvaluatorBuilder = Callable[[object], BaseEvaluator]


class EmptyHybridSearch:
    def search(self, query: str, top_k: int, filters: dict | None = None, trace: object | None = None) -> list:
        return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate.py")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--test-set", default="tests/fixtures/golden_test_set.json")
    parser.add_argument("--top-k", type=int)
    return parser


def run_evaluation(
    settings_path: str,
    test_set_path: str,
    top_k: int | None = None,
    search_factory: SearchFactory | None = None,
    evaluator_builder: EvaluatorBuilder | None = None,
) -> EvalReport:
    settings = load_settings(settings_path)
    search = search_factory(settings) if search_factory is not None else build_search(settings)
    evaluator = evaluator_builder(settings) if evaluator_builder is not None else EvaluatorFactory.create(settings)
    return EvalRunner(settings=settings, hybrid_search=search, evaluator=evaluator, top_k=top_k).run(test_set_path)


def build_search(settings: object) -> object:
    if not has_ingested_data(settings):
        return EmptyHybridSearch()
    return HybridSearch(settings)


def main(
    argv: Sequence[str] | None = None,
    search_factory: SearchFactory | None = None,
    evaluator_builder: EvaluatorBuilder | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(
            settings_path=args.settings,
            test_set_path=args.test_set,
            top_k=args.top_k,
            search_factory=search_factory,
            evaluator_builder=evaluator_builder,
        )
    except Exception as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
