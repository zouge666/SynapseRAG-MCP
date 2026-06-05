from __future__ import annotations

from collections.abc import Callable

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.composite_evaluator import CompositeEvaluator
from observability.evaluation.ragas_evaluator import RagasEvaluator


EvaluatorBuilder = Callable[[object | None], BaseEvaluator]


class EvaluatorFactory:
    _providers: dict[str, EvaluatorBuilder] = {
        "custom": CustomEvaluator,
        "custom_metrics": CustomEvaluator,
        "ragas": RagasEvaluator,
    }

    @classmethod
    def register_provider(cls, provider: str, builder: EvaluatorBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        if key not in {"custom", "custom_metrics"}:
            cls._providers.pop(key, None)

    @classmethod
    def create(cls, settings: Settings | EvaluationSettings | str) -> BaseEvaluator:
        evaluation_settings = settings.evaluation if isinstance(settings, Settings) else settings
        if isinstance(evaluation_settings, str):
            return cls._create_provider(evaluation_settings, evaluation_settings)

        providers = evaluation_settings.backends or ["custom_metrics"]
        if len(providers) == 1:
            return cls._create_provider(providers[0], evaluation_settings)

        evaluators = [cls._create_provider(provider, evaluation_settings) for provider in providers]
        return CompositeEvaluator(evaluators, settings=evaluation_settings)

    @classmethod
    def _create_provider(cls, provider: str, settings: EvaluationSettings | str) -> BaseEvaluator:
        key = cls._normalize_provider(provider)
        builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported evaluator backend: {provider}")
        return builder(settings)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("evaluation backend is required")
        return key
