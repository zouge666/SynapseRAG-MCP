from __future__ import annotations

from collections.abc import Callable
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase


RagasRunner = Callable[[list[dict[str, Any]], list[str]], Any]


class RagasEvaluator(BaseEvaluator):
    metric_names = ("faithfulness", "answer_relevancy", "context_precision")

    def __init__(self, settings: object | None = None, runner: RagasRunner | None = None) -> None:
        super().__init__(settings)
        self.runner = runner

    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        sample = self._sample(case)
        runner = self.runner or self._run_ragas
        raw_result = runner([sample], list(self.metric_names))
        return self._normalize_metrics(raw_result)

    def _run_ragas(self, samples: list[dict[str, Any]], metric_names: list[str]) -> Any:
        try:
            dataset_type, ragas_evaluate, metrics_by_name = self._load_ragas_components()
        except ImportError as exc:
            raise ImportError(
                "RagasEvaluator requires optional dependencies: install ragas and datasets to use the ragas backend."
            ) from exc

        metrics = [metrics_by_name[name] for name in metric_names if name in metrics_by_name]
        dataset = dataset_type.from_list(samples)
        return ragas_evaluate(dataset, metrics=metrics)

    def _load_ragas_components(self) -> tuple[Any, Callable[..., Any], dict[str, Any]]:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        return (
            Dataset,
            ragas_evaluate,
            {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
            },
        )

    def _sample(self, case: EvaluationCase) -> dict[str, Any]:
        answer = self._metadata_text(case, "generated_answer", "answer", "response")
        ground_truth = self._metadata_text(case, "ground_truth", "reference", "expected_answer")
        contexts = self._metadata_list(case, "retrieved_contexts", "contexts", "context")

        return {
            "question": case.query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "user_input": case.query,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth,
        }

    def _metadata_text(self, case: EvaluationCase, *keys: str) -> str:
        for key in keys:
            value = case.metadata.get(key)
            if value is not None:
                return str(value)
        return ""

    def _metadata_list(self, case: EvaluationCase, *keys: str) -> list[str]:
        for key in keys:
            value = case.metadata.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
            if isinstance(value, tuple):
                return [str(item) for item in value]
            if value is not None:
                return [str(value)]
        return [str(item) for item in case.retrieved_ids]

    def _normalize_metrics(self, raw_result: Any) -> dict[str, float]:
        row = self._result_row(raw_result)
        aliases = {
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
            "answer_relevance": "answer_relevancy",
            "context_precision": "context_precision",
        }
        metrics: dict[str, float] = {}
        for raw_name, output_name in aliases.items():
            if raw_name in row:
                metrics[output_name] = float(row[raw_name])
        return metrics

    def _result_row(self, raw_result: Any) -> dict[str, Any]:
        if isinstance(raw_result, dict):
            return {key: self._first_value(value) for key, value in raw_result.items()}

        scores = getattr(raw_result, "scores", None)
        if isinstance(scores, list) and scores:
            return {key: self._first_value(value) for key, value in scores[0].items()}
        if isinstance(scores, dict):
            return {key: self._first_value(value) for key, value in scores.items()}

        to_pandas = getattr(raw_result, "to_pandas", None)
        if callable(to_pandas):
            frame = to_pandas()
            if len(frame.index) > 0:
                return frame.iloc[0].to_dict()

        return {}

    def _first_value(self, value: Any) -> Any:
        if isinstance(value, list | tuple):
            return value[0] if value else 0.0
        return value
