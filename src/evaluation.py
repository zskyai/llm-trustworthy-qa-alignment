"""Task-level metrics for evidence-grounded answer predictions."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from .structured_answer import extract_inline_citations


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: object) -> list[str]:
    return _TOKEN_RE.findall(str(value or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted, gold = _tokens(prediction), _tokens(reference)
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    return 2.0 * overlap / (len(predicted) + len(gold)) if overlap else 0.0


def _refused(row: Mapping[str, object]) -> bool:
    text = str(row.get("prediction", row.get("final_answer", ""))).lower()
    return bool(row.get("abstained")) or "[no_answer]" in text or "insufficient evidence" in text or "cannot answer" in text


def _macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float | None:
    labels = sorted(set(gold) | set(predicted))
    if not labels:
        return None
    values = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(values) / len(values)


def evaluate_predictions(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Evaluate citation, refusal, answer, label and context-length slices."""

    if not rows:
        raise ValueError("rows must not be empty")
    f1_values: list[float] = []
    citation_total = citation_valid = citation_gold = citation_recalled = 0
    refusals = gold_refusals = true_refusals = 0
    supported_citations = citations_with_support_labels = 0
    labels_gold: list[str] = []
    labels_predicted: list[str] = []
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prediction = str(row.get("prediction", row.get("final_answer", "")))
        reference = str(row.get("gold_answer", row.get("reference", "")))
        f1 = token_f1(prediction, reference)
        f1_values.append(f1)
        valid_ids = set(str(item) for item in row.get("valid_citations", []))
        cited = set(extract_inline_citations(prediction))
        citation_total += len(cited)
        citation_valid += sum(item in valid_ids for item in cited)
        citation_gold += len(valid_ids)
        citation_recalled += sum(item in cited for item in valid_ids)
        supported_ids = row.get("supported_citation_ids")
        if supported_ids is not None:
            support_set = set(str(item) for item in supported_ids)
            citations_with_support_labels += len(cited)
            supported_citations += sum(item in support_set for item in cited)
        refused = _refused(row)
        gold_refusal = not bool(row.get("answerable", True))
        refusals += int(refused)
        gold_refusals += int(gold_refusal)
        true_refusals += int(refused and gold_refusal)
        if row.get("gold_label") is not None:
            labels_gold.append(str(row["gold_label"]).lower())
            labels_predicted.append(str(row.get("predicted_label", "")).lower())
        words = int(row.get("document_word_count", 0) or 0)
        bucket = "short" if words < 2048 else "medium" if words < 4096 else "long"
        buckets[bucket].append(f1)
    answerable = sum(bool(row.get("answerable", True)) for row in rows)
    unanswerable = len(rows) - answerable
    return {
        "count": len(rows),
        "mean_token_f1": sum(f1_values) / len(f1_values),
        "citation_precision": citation_valid / citation_total if citation_total else 0.0,
        "citation_id_validity": citation_valid / citation_total if citation_total else None,
        "citation_recall": citation_recalled / citation_gold if citation_gold else 0.0,
        "citation_support_rate": supported_citations / citations_with_support_labels if citations_with_support_labels else None,
        "refusal_precision": true_refusals / refusals if refusals else 0.0,
        "refusal_recall": true_refusals / gold_refusals if gold_refusals else None,
        "over_refusal_rate": sum(_refused(row) and bool(row.get("answerable", True)) for row in rows) / max(answerable, 1),
        "pubmedqa_macro_f1": _macro_f1(labels_gold, labels_predicted),
        "length_bucket_token_f1": {key: sum(values) / len(values) for key, values in buckets.items()},
    }
