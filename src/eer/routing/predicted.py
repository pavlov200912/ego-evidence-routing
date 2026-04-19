"""Predicted router — stub for Week 3."""

from __future__ import annotations


class PredictedRouter:
    """Lightweight text classifier that predicts the best tool from question text.

    To be implemented in Week 3.
    """

    def fit(self, questions: list[str], labels: list[str]) -> None:
        raise NotImplementedError("PredictedRouter not yet implemented (Week 3).")

    def predict(self, question: str) -> str:
        raise NotImplementedError("PredictedRouter not yet implemented (Week 3).")
