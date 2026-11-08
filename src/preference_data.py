"""Auditable hard-negative construction for evidence-grounded preferences."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from .structured_answer import extract_inline_citations


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    error_type: str
    source_id: str
    metadata: dict[str, object] = field(default_factory=dict)


def _replace_citations(text: str, replacement: str) -> str:
    return re.sub(r"\[E[A-Za-z0-9_.:-]+\]", replacement, text)


def build_hard_negative(row: Mapping[str, object], error_type: str) -> PreferencePair:
    """Create a controlled rejected answer while preserving prompt and source ID."""

    prompt = str(row.get("prompt", ""))
    chosen = str(row.get("chosen", ""))
    source_id = str(row.get("record_id", row.get("document_id", "")))
    if not prompt or not chosen or not source_id:
        raise ValueError("row requires prompt, chosen, and record_id/document_id")
    if error_type == "invalid_citation":
        rejected = _replace_citations(chosen, "[E999]")
    elif error_type == "citation_removed":
        rejected = _replace_citations(chosen, "")
    elif error_type == "unsupported_extension":
        rejected = chosen.rstrip() + " This proves the opposite conclusion is always true."
    elif error_type == "over_refusal":
        rejected = "Insufficient evidence: the passages do not contain an explicit answer. [NO_ANSWER]"
    elif error_type == "polarity_flip":
        lowered = chosen.lower()
        if "yes" in lowered:
            rejected = re.sub(r"\byes\b", "No", chosen, count=1, flags=re.I)
        elif "no" in lowered:
            rejected = re.sub(r"\bno\b", "Yes", chosen, count=1, flags=re.I)
        else:
            rejected = "No. " + chosen
    else:
        raise ValueError(f"unsupported error_type: {error_type}")
    if rejected == chosen:
        raise ValueError("hard-negative transform did not change the answer")
    return PreferencePair(prompt, chosen, rejected, error_type, source_id, {"construction": "controlled_transform", "test_used": False})


def validate_preference_pair(pair: PreferencePair, valid_citations: set[str]) -> dict[str, object]:
    chosen_ids = set(extract_inline_citations(pair.chosen))
    rejected_ids = set(extract_inline_citations(pair.rejected))
    report = {
        "same_prompt": bool(pair.prompt.strip()),
        "different_completion": pair.chosen != pair.rejected,
        "chosen_citations_valid": chosen_ids <= valid_citations,
        "rejected_contains_invalid_citation": bool(rejected_ids - valid_citations),
        "error_type": pair.error_type,
        "source_id": pair.source_id,
    }
    report["metadata"] = dict(pair.metadata)
    return report


def audit_model_preference_candidate(
    row: Mapping[str, object],
    *,
    valid_citations: set[str],
    chosen_entailment: float,
    rejected_entailment: float,
    chosen_answer_match: bool,
    rejected_answer_match: bool = False,
    nli_threshold: float = 0.55,
) -> dict[str, object]:
    """Filter a pair produced by a real model run.

    A pair is accepted only when the chosen answer is source-supported and
    answer-consistent, while the rejected answer exhibits the recorded failure
    mode.  This makes DPO provenance explicit instead of treating every fluent
    alternative as a valid negative.
    """
    if not 0.0 <= nli_threshold <= 1.0 or not 0.0 <= chosen_entailment <= 1.0 or not 0.0 <= rejected_entailment <= 1.0:
        raise ValueError("entailment scores and threshold must be in [0, 1]")
    pair = build_hard_negative({"prompt": row.get("prompt", ""), "chosen": row.get("chosen", ""), "record_id": row.get("record_id", row.get("document_id", ""))}, str(row.get("error_type", "unknown"))) if row.get("error_type") in {"invalid_citation", "citation_removed", "unsupported_extension", "over_refusal", "polarity_flip"} else None
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    chosen_ids = set(extract_inline_citations(chosen))
    rejected_ids = set(extract_inline_citations(rejected))
    answerable = bool(row.get("answerable", True))
    checks = {
        "chosen_citations_valid": chosen_ids <= valid_citations,
        "chosen_entails_evidence": (not answerable) or chosen_entailment >= nli_threshold,
        "chosen_answer_consistent": (not answerable) or bool(chosen_answer_match),
        "rejected_is_distinct": chosen != rejected,
        "rejected_not_supported": (not answerable) or rejected_entailment < nli_threshold or not rejected_answer_match,
        "rejected_citation_failure_recorded": bool(rejected_ids - valid_citations) or str(row.get("error_type", "")) in {"unsupported_extension", "over_refusal", "polarity_flip", "citation_removed"},
    }
    accepted = all(checks.values())
    return {"accepted": accepted, "checks": checks, "chosen_entailment": chosen_entailment, "rejected_entailment": rejected_entailment, "error_type": row.get("error_type"), "source_id": row.get("record_id", row.get("document_id")), "test_used": False, "pair_metadata": pair.metadata if pair else {"construction": "model_output", "test_used": False}}
