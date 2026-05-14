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
