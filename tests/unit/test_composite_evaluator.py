import pytest

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.composite_evaluator import CompositeEvaluator


class HitEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        return {"hit_rate": 1.0 if case.retrieved_ids and case.golden_ids else 0.0}


class QualityEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        return {"faithfulness": 0.8, "answer_relevancy": 0.7}


class DuplicateHitEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        return {"hit_rate": 0.5}


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    for provider in ("hit_fake", "quality_fake", "duplicate_hit_fake"):
        EvaluatorFactory.unregister_provider(provider)
    yield
    for provider in ("hit_fake", "quality_fake", "duplicate_hit_fake"):
        EvaluatorFactory.unregister_provider(provider)


def test_composite_evaluator_merges_metrics_from_multiple_evaluators() -> None:
    evaluator = CompositeEvaluator([HitEvaluator(), QualityEvaluator()])
    case = EvaluationCase(query="q", retrieved_ids=["a"], golden_ids=["a"])

    metrics = evaluator.evaluate(case)

    assert metrics == {"hit_rate": 1.0, "faithfulness": 0.8, "answer_relevancy": 0.7}


def test_composite_evaluator_preserves_duplicate_metric_names() -> None:
    evaluator = CompositeEvaluator([HitEvaluator(), DuplicateHitEvaluator()])
    case = EvaluationCase(query="q", retrieved_ids=["a"], golden_ids=["a"])

    metrics = evaluator.evaluate(case)

    assert metrics == {"hit_rate": 1.0, "duplicatehit.hit_rate": 0.5}


def test_composite_evaluator_rejects_empty_evaluator_list() -> None:
    with pytest.raises(ValueError, match="requires at least one evaluator"):
        CompositeEvaluator([])


def test_factory_creates_composite_evaluator_for_multiple_backends() -> None:
    EvaluatorFactory.register_provider("hit_fake", HitEvaluator)
    EvaluatorFactory.register_provider("quality_fake", QualityEvaluator)
    settings = EvaluationSettings(enabled=True, backends=["hit_fake", "quality_fake"])

    evaluator = EvaluatorFactory.create(settings)
    metrics = evaluator.evaluate(EvaluationCase(query="q", retrieved_ids=["a"], golden_ids=["a"]))

    assert isinstance(evaluator, CompositeEvaluator)
    assert metrics == {"hit_rate": 1.0, "faithfulness": 0.8, "answer_relevancy": 0.7}


def test_factory_supports_custom_alias_in_composite_config() -> None:
    EvaluatorFactory.register_provider("quality_fake", QualityEvaluator)
    settings = EvaluationSettings(enabled=True, backends=["custom", "quality_fake"])

    evaluator = EvaluatorFactory.create(settings)
    metrics = evaluator.evaluate(EvaluationCase(query="q", retrieved_ids=["a", "b"], golden_ids=["b"]))

    assert isinstance(evaluator, CompositeEvaluator)
    assert metrics == {"hit_rate": 1.0, "mrr": 0.5, "faithfulness": 0.8, "answer_relevancy": 0.7}
