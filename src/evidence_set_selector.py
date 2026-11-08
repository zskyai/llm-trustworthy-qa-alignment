"""Coverage-aware listwise evidence selection for long-context QA."""
from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SECTION_PRIOR = {"results": 1.0, "conclusion": 0.95, "findings": 0.9, "discussion": 0.75, "methods": 0.55, "background": 0.35}


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(str(text)) if len(x) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _section_prior(name: str) -> float:
    lowered = str(name).lower()
    return max((value for key, value in _SECTION_PRIOR.items() if key in lowered), default=0.5)


class CoverageAwareEvidenceSelector:
    """Greedy listwise selector with explicit evidence diversity controls.

    The selector is extractive and never invents a paragraph or sentence ID.
    Weights can be fitted on a validation set by the caller; defaults are
    conservative and work with a frozen reranker.
    """

    def __init__(self, *, score_weight: float = 1.0, coverage_weight: float = 0.45, section_weight: float = 0.12, redundancy_penalty: float = 0.30, conflict_penalty: float = 0.10) -> None:
        values = (score_weight, coverage_weight, section_weight, redundancy_penalty, conflict_penalty)
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
            raise ValueError("selector weights must be finite and non-negative")
        self.weights = {"score": float(score_weight), "coverage": float(coverage_weight), "section": float(section_weight), "redundancy": float(redundancy_penalty), "conflict": float(conflict_penalty)}

    def _features(self, question: str, row: Mapping[str, object], covered: set[str], selected: Sequence[Mapping[str, object]]) -> dict[str, float]:
        question_tokens = _tokens(question)
        text_tokens = _tokens(str(row.get("text", "")))
        score = float(row.get("rerank_score", row.get("score", 0.0)))
        coverage_gain = len((question_tokens & text_tokens) - covered) / max(len(question_tokens), 1)
        redundancy = max((_jaccard(text_tokens, _tokens(str(previous.get("text", "")))) for previous in selected), default=0.0)
        lowered = str(row.get("text", "")).lower()
        conflict = float(any(marker in lowered for marker in (" no ", " not ", " without ", " failed ", " impossible ")))
        return {"score": score, "coverage_gain": coverage_gain, "section_prior": _section_prior(str(row.get("section_name", ""))), "redundancy": redundancy, "conflict": conflict}

    def select(self, question: str, candidates: Sequence[Mapping[str, object]], *, top_k: int = 3, max_per_paragraph: int = 2, min_score: float | None = None) -> list[dict[str, object]]:
        if not question.strip() or top_k < 1 or max_per_paragraph < 1:
            raise ValueError("question and selection budgets must be valid")
        ids = [str(row.get("id", "")) for row in candidates]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique and non-empty")
        remaining = [dict(row) for row in candidates]
        selected: list[dict[str, object]] = []
        covered: set[str] = set()
        counts: dict[str, int] = {}
        for _ in range(min(top_k, len(remaining))):
            best = None
            for row in remaining:
                parent = str(row.get("paragraph_id", row.get("parent_id", row.get("id"))))
                if counts.get(parent, 0) >= max_per_paragraph:
                    continue
                features = self._features(question, row, covered, selected)
                if min_score is not None and features["score"] < min_score:
                    continue
                utility = (self.weights["score"] * features["score"] + self.weights["coverage"] * features["coverage_gain"] + self.weights["section"] * features["section_prior"] - self.weights["redundancy"] * features["redundancy"] - self.weights["conflict"] * features["conflict"])
                key = (utility, features["coverage_gain"], features["score"], str(row["id"]))
                if best is None or key > best[0]:
                    best = (key, row, features, parent)
            if best is None:
                break
            _, row, features, parent = best
            selected.append({**row, "selection_score": float(best[0][0]), "selection_features": features, "coverage_after": len(covered | (_tokens(str(row.get("text", ""))) & _tokens(question))) / max(len(_tokens(question)), 1)})
            remaining.remove(row)
            counts[parent] = counts.get(parent, 0) + 1
            covered.update(_tokens(str(row.get("text", ""))) & _tokens(question))
        return selected
