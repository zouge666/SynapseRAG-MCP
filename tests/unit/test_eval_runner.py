import json
import importlib.util
from pathlib import Path

from core import RetrievalResult
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.eval_runner import EvalRunner, EvalRunnerError


EVALUATE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_script", EVALUATE_SCRIPT)
evaluate_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_script)


class FakeSearch:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, top_k: int, filters: dict | None = None, trace: object | None = None) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters or {}})
        return self.results_by_query.get(query, [])


def write_test_set(path, cases: list[dict]) -> None:
    path.write_text(json.dumps({"test_cases": cases}, ensure_ascii=False), encoding="utf-8")


def result(chunk_id: str, source_path: str = "docs/a.pdf") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=1.0, text=f"text for {chunk_id}", metadata={"source_path": source_path})


def test_eval_runner_runs_golden_cases_and_aggregates_metrics(tmp_path) -> None:
    test_set = tmp_path / "golden.json"
    write_test_set(
        test_set,
        [
            {"query": "first", "expected_chunk_ids": ["a"], "expected_sources": ["docs/a.pdf"]},
            {"query": "second", "expected_chunk_ids": ["missing"], "top_k": 2, "filters": {"collection": "docs"}},
        ],
    )
    search = FakeSearch({"first": [result("a")], "second": [result("b", "docs/b.pdf")]})
    runner = EvalRunner(settings={"retrieval": {"top_k_final": 3}}, hybrid_search=search, evaluator=CustomEvaluator())

    report = runner.run(test_set)

    assert report.hit_rate == 0.5
    assert report.mrr == 0.5
    assert report.to_dict()["metrics"] == {"hit_rate": 0.5, "mrr": 0.5}
    assert report.details[0].retrieved_ids == ["a"]
    assert report.details[0].retrieved_sources == ["docs/a.pdf"]
    assert search.calls == [
        {"query": "first", "top_k": 3, "filters": {}},
        {"query": "second", "top_k": 2, "filters": {"collection": "docs"}},
    ]


def test_eval_runner_rejects_missing_test_cases(tmp_path) -> None:
    test_set = tmp_path / "golden.json"
    test_set.write_text(json.dumps({"items": []}), encoding="utf-8")
    runner = EvalRunner(settings={"retrieval": {"top_k_final": 3}}, hybrid_search=FakeSearch({}), evaluator=CustomEvaluator())

    try:
        runner.run(test_set)
    except EvalRunnerError as error:
        assert "test_cases" in str(error)
    else:
        raise AssertionError("expected EvalRunnerError")


def test_evaluate_script_main_outputs_metrics_with_injected_components(tmp_path, capsys) -> None:
    test_set = tmp_path / "golden.json"
    write_test_set(test_set, [{"query": "q", "expected_chunk_ids": ["a"]}])

    def search_factory(settings: object) -> FakeSearch:
        return FakeSearch({"q": [result("a")]})

    exit_code = evaluate_script.main(
        ["--settings", "config/settings.yaml", "--test-set", str(test_set)],
        search_factory=search_factory,
        evaluator_builder=lambda settings: CustomEvaluator(),
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["hit_rate"] == 1.0
    assert output["mrr"] == 1.0
