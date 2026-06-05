from __future__ import annotations

from collections.abc import Callable

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.ragas_evaluator import RagasEvaluator


EvaluatorBuilder = Callable[[object | None], BaseEvaluator]


class EvaluatorFactory:
    _providers: dict[str, EvaluatorBuilder] = {"custom_metrics": CustomEvaluator, "ragas": RagasEvaluator}

    @classmethod
    def register_provider(cls, provider: str, builder: EvaluatorBuilder) -> None:
        key = cls._normalize_provider(provider)
        cls._providers[key] = builder

    @classmethod
    def unregister_provider(cls, provider: str) -> None:
        key = cls._normalize_provider(provider)
        if key != "custom_metrics":
            cls._providers.pop(key, None)

    @classmethod
    def create(cls, settings: Settings | EvaluationSettings | str) -> BaseEvaluator:
        evaluation_settings = settings.evaluation if isinstance(settings, Settings) else settings
        provider = cls._provider(evaluation_settings)
        key = cls._normalize_provider(provider)
        builder = cls._providers.get(key)
        if builder is None:
            raise ValueError(f"unsupported evaluator backend: {provider}")
        return builder(evaluation_settings)

    @staticmethod
    def _provider(settings: EvaluationSettings | str) -> str:
        if isinstance(settings, str):
            return settings
        if settings.backends:
            return settings.backends[0]
        return "custom_metrics"

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        key = provider.strip().lower()
        if not key:
            raise ValueError("evaluation backend is required")
        return key
