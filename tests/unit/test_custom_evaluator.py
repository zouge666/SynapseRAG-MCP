import pytest

from core.settings import EvaluationSettings, load_settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class FakeEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        return {"fake_score": float(len(case.retrieved_ids))}


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    EvaluatorFactory.unregister_provider("fake")
    yield
    EvaluatorFactory.unregister_provider("fake")


def test_custom_evaluator_returns_hit_rate_and_mrr_for_first_hit() -> None:
    evaluator = CustomEvaluator()
    case = EvaluationCase(query="q", retrieved_ids=["a", "b", "c"], golden_ids=["b"])

    metrics = evaluator.evaluate(case)

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


def test_custom_evaluator_returns_zero_for_no_hit() -> None:
    evaluator = CustomEvaluator()
    case = EvaluationCase(query="q", retrieved_ids=["a", "b"], golden_ids=["c"])

    metrics = evaluator.evaluate(case)

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


def test_custom_evaluator_returns_zero_without_golden_ids() -> None:
    evaluator = CustomEvaluator()
    case = EvaluationCase(query="q", retrieved_ids=["a"], golden_ids=[])

    metrics = evaluator.evaluate(case)

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


def test_factory_creates_default_custom_evaluator_from_project_settings() -> None:
    settings = load_settings("config/settings.yaml")

    evaluator = EvaluatorFactory.create(settings)

    assert isinstance(evaluator, CustomEvaluator)


def test_factory_creates_registered_provider() -> None:
    EvaluatorFactory.register_provider("fake", FakeEvaluator)
    settings = EvaluationSettings(enabled=True, backends=["fake"])

    evaluator = EvaluatorFactory.create(settings)
    metrics = evaluator.evaluate(EvaluationCase(query="q", retrieved_ids=["a", "b"], golden_ids=["a"]))

    assert isinstance(evaluator, FakeEvaluator)
    assert metrics == {"fake_score": 2.0}


def test_factory_rejects_unknown_provider() -> None:
    settings = EvaluationSettings(enabled=True, backends=["missing"])

    with pytest.raises(ValueError, match="unsupported evaluator backend: missing"):
        EvaluatorFactory.create(settings)


def test_custom_evaluator_uses_first_retrieved_hit_with_multiple_golden_ids() -> None:
    evaluator = CustomEvaluator()
    case = EvaluationCase(query="q", retrieved_ids=["a", "b", "c", "d"], golden_ids=["c", "d"])

    metrics = evaluator.evaluate(case)

    assert metrics == {"hit_rate": 1.0, "mrr": 1 / 3}


def test_custom_evaluator_handles_duplicate_ids_without_changing_metric_shape() -> None:
    evaluator = CustomEvaluator()
    case = EvaluationCase(query="q", retrieved_ids=["a", "a", "b"], golden_ids=["b", "b"])

    metrics = evaluator.evaluate(case)

    assert set(metrics) == {"hit_rate", "mrr"}
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1 / 3
    assert all(isinstance(value, float) for value in metrics.values())


def test_evaluation_case_metadata_defaults_are_independent() -> None:
    first = EvaluationCase(query="q1", retrieved_ids=[], golden_ids=[])
    second = EvaluationCase(query="q2", retrieved_ids=[], golden_ids=[])

    first.metadata["source"] = "one"

    assert second.metadata == {}


def test_factory_normalizes_string_provider_name() -> None:
    evaluator = EvaluatorFactory.create(" CUSTOM ")

    assert isinstance(evaluator, CustomEvaluator)


def test_factory_rejects_empty_provider_name() -> None:
    with pytest.raises(ValueError, match="evaluation backend is required"):
        EvaluatorFactory.create(" ")
