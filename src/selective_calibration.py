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
    risk_upper_bound: float


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
        self.reject_all = False

    def fit(self, confidences: Sequence[float], correct: Sequence[bool]) -> float:
        if len(confidences) != len(correct) or not confidences:
            raise ValueError("confidences and correct must be non-empty and aligned")
        pairs = sorted((self._clip(score), bool(label)) for score, label in zip(confidences, correct))
        candidates = sorted({score for score, _ in pairs} | {0.0, 1.0})
        feasible = []
        for threshold in candidates:
            accepted = [label for score, label in pairs if score >= threshold]
            errors = len(accepted) - sum(accepted)
            if accepted and self._wilson_upper(errors, len(accepted)) <= self.target_risk:
                feasible.append(threshold)
        # Select the lowest feasible threshold to maximize coverage while
        # respecting the requested validation risk.
        self.reject_all = not feasible
        self.threshold = min(feasible) if feasible else 1.0
        return self.threshold

    def predict(self, confidence: float) -> bool:
        if self.threshold is None:
            raise RuntimeError("fit must be called before predict")
        if self.reject_all:
            return False
        return self._clip(confidence) >= self.threshold

    def summarize(self, confidences: Sequence[float], correct: Sequence[bool]) -> CalibrationSummary:
        if self.threshold is None:
            raise RuntimeError("fit must be called before summarize")
        accepted = [i for i, score in enumerate(confidences) if self.predict(score)]
        coverage = len(accepted) / max(len(confidences), 1)
        risk = 1.0 - sum(bool(correct[i]) for i in accepted) / len(accepted) if accepted else 0.0
        errors = len(accepted) - sum(bool(correct[i]) for i in accepted)
        risk_upper = self._wilson_upper(errors, len(accepted)) if accepted else 0.0
        ece = expected_calibration_error(confidences, correct, bins=self.bins)
        return CalibrationSummary(self.threshold, coverage, risk, 1.0 - coverage, ece, risk_upper)

    @staticmethod
    def _wilson_upper(errors: int, count: int, *, z: float = 1.96) -> float:
        if count <= 0:
            return 0.0
        rate = errors / count
        z2 = z * z
        center = rate + z2 / (2 * count)
        radius = z * math.sqrt(rate * (1.0 - rate) / count + z2 / (4 * count * count))
        return min(1.0, (center + radius) / (1.0 + z2 / count))

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
