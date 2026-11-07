"""Validation-only coverage-risk calibration for abstaining QA models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CalibrationSummary:
    threshold: float
    coverage: float
    selective_risk: float
    abstention_rate: float
    ece: float


class CoverageRiskCalibrator:
    """Choose a confidence threshold on validation data only.

    Scores are model confidences in [0, 1], labels indicate whether the
    structured answer passed the task-specific correctness/evidence checks.
    """

    def __init__(self, *, target_risk: float = 0.10, bins: int = 10) -> None:
        if not 0.0 <= target_risk <= 1.0 or bins <= 0:
            raise ValueError("invalid calibration arguments")
        self.target_risk = target_risk
        self.bins = bins
        self.threshold: float | None = None

    def fit(self, confidences: Sequence[float], correct: Sequence[bool]) -> float:
        if len(confidences) != len(correct) or not confidences:
            raise ValueError("confidences and correct must be non-empty and aligned")
        pairs = sorted((self._clip(score), bool(label)) for score, label in zip(confidences, correct))
        candidates = sorted({score for score, _ in pairs} | {0.0, 1.0})
        feasible = []
        for threshold in candidates:
            accepted = [label for score, label in pairs if score >= threshold]
            if accepted and 1.0 - sum(accepted) / len(accepted) <= self.target_risk:
                feasible.append(threshold)
        self.threshold = max(feasible) if feasible else 1.0
        return self.threshold

    def predict(self, confidence: float) -> bool:
        if self.threshold is None:
            raise RuntimeError("fit must be called before predict")
        return self._clip(confidence) >= self.threshold

    def summarize(self, confidences: Sequence[float], correct: Sequence[bool]) -> CalibrationSummary:
        if self.threshold is None:
            raise RuntimeError("fit must be called before summarize")
        accepted = [i for i, score in enumerate(confidences) if self.predict(score)]
        coverage = len(accepted) / max(len(confidences), 1)
        risk = 1.0 - sum(bool(correct[i]) for i in accepted) / len(accepted) if accepted else 0.0
        ece = expected_calibration_error(confidences, correct, bins=self.bins)
        return CalibrationSummary(self.threshold, coverage, risk, 1.0 - coverage, ece)

    @staticmethod
    def _clip(value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("confidence must be finite")
        return min(1.0, max(0.0, float(value)))


def expected_calibration_error(confidences: Iterable[float], correct: Iterable[bool], *, bins: int = 10) -> float:
    confidences = list(confidences)
    correct = [bool(value) for value in correct]
    if len(confidences) != len(correct) or not confidences or bins <= 0:
        raise ValueError("invalid ECE inputs")
    total = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        bucket = [i for i, score in enumerate(confidences) if lo <= score < hi or (index == bins - 1 and score == hi)]
        if bucket:
            accuracy = sum(correct[i] for i in bucket) / len(bucket)
            confidence = sum(confidences[i] for i in bucket) / len(bucket)
            total += len(bucket) / len(confidences) * abs(accuracy - confidence)
    return total
