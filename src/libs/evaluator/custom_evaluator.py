from __future__ import annotations

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase


class CustomEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: object | None = None) -> dict[str, float]:
        retrieved = case.retrieved_ids
        golden = set(case.golden_ids)

        if not golden:
            return {"hit_rate": 0.0, "mrr": 0.0}

        hit_rate = 1.0 if any(item in golden for item in retrieved) else 0.0
        mrr = 0.0
        for index, item in enumerate(retrieved, start=1):
            if item in golden:
                mrr = 1.0 / index
                break

        return {"hit_rate": hit_rate, "mrr": mrr}
