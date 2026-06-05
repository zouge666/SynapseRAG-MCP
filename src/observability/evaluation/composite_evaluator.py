from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase


class CompositeEvaluator(BaseEvaluator):
    def __init__(self, evaluators: list[BaseEvaluator], settings: object | None = None) -> None:
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")
        super().__init__(settings)
        self.evaluators = list(evaluators)

    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        with ThreadPoolExecutor(max_workers=len(self.evaluators)) as executor:
            futures = [executor.submit(evaluator.evaluate, case, trace) for evaluator in self.evaluators]

        merged: dict[str, float] = {}
        for evaluator, future in zip(self.evaluators, futures, strict=True):
            for name, value in future.result().items():
                metric_name = self._metric_name(name, evaluator, merged)
                merged[metric_name] = float(value)
        return merged

    def _metric_name(self, name: str, evaluator: BaseEvaluator, merged: dict[str, float]) -> str:
        if name not in merged:
            return name

        prefix = evaluator.__class__.__name__.replace("Evaluator", "").lower() or "evaluator"
        candidate = f"{prefix}.{name}"
        if candidate not in merged:
            return candidate

        index = 2
        while f"{candidate}.{index}" in merged:
            index += 1
        return f"{candidate}.{index}"
