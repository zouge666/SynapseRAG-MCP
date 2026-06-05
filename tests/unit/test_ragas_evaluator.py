import pytest

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import EvaluationCase
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.ragas_evaluator import RagasEvaluator


def test_ragas_evaluator_returns_standard_metrics_with_mock_runner() -> None:
    observed: dict[str, object] = {}

    def runner(samples: list[dict[str, object]], metric_names: list[str]) -> dict[str, list[float]]:
        observed["samples"] = samples
        observed["metric_names"] = metric_names
        return {
            "faithfulness": [0.91],
            "answer_relevancy": [0.82],
            "context_precision": [0.73],
        }

    evaluator = RagasEvaluator(runner=runner)
    case = EvaluationCase(
        query="What does the dashboard show?",
        retrieved_ids=["chunk-1"],
        golden_ids=["chunk-1"],
        metadata={
            "generated_answer": "It shows query traces.",
            "ground_truth": "The dashboard shows query trace timing.",
            "retrieved_contexts": ["Query traces include timing and retrieval details."],
        },
    )

    metrics = evaluator.evaluate(case)

    assert metrics == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.82,
        "context_precision": 0.73,
    }
    assert observed["metric_names"] == ["faithfulness", "answer_relevancy", "context_precision"]
    assert observed["samples"] == [
        {
            "question": "What does the dashboard show?",
            "answer": "It shows query traces.",
            "contexts": ["Query traces include timing and retrieval details."],
            "ground_truth": "The dashboard shows query trace timing.",
            "user_input": "What does the dashboard show?",
            "response": "It shows query traces.",
            "retrieved_contexts": ["Query traces include timing and retrieval details."],
            "reference": "The dashboard shows query trace timing.",
        }
    ]


def test_ragas_evaluator_accepts_answer_relevance_alias() -> None:
    evaluator = RagasEvaluator(runner=lambda samples, metric_names: {"answer_relevance": 0.6})
    case = EvaluationCase(query="q", retrieved_ids=["ctx"], golden_ids=["ctx"])

    metrics = evaluator.evaluate(case)

    assert metrics == {"answer_relevancy": 0.6}


def test_ragas_evaluator_uses_retrieved_ids_as_context_fallback() -> None:
    observed: dict[str, object] = {}

    def runner(samples: list[dict[str, object]], metric_names: list[str]) -> dict[str, float]:
        observed["contexts"] = samples[0]["contexts"]
        return {"faithfulness": 1.0}

    evaluator = RagasEvaluator(runner=runner)
    case = EvaluationCase(query="q", retrieved_ids=["chunk-a", "chunk-b"], golden_ids=["chunk-a"])

    metrics = evaluator.evaluate(case)

    assert metrics == {"faithfulness": 1.0}
    assert observed["contexts"] == ["chunk-a", "chunk-b"]


def test_factory_creates_ragas_evaluator() -> None:
    evaluator = EvaluatorFactory.create(EvaluationSettings(enabled=True, backends=["ragas"]))

    assert isinstance(evaluator, RagasEvaluator)


def test_ragas_evaluator_raises_clear_import_error_without_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = RagasEvaluator()
    case = EvaluationCase(query="q", retrieved_ids=["ctx"], golden_ids=["ctx"])

    def load_missing_components() -> object:
        raise ImportError("missing ragas")

    monkeypatch.setattr(evaluator, "_load_ragas_components", load_missing_components)

    with pytest.raises(ImportError, match="install ragas and datasets"):
        evaluator.evaluate(case)
