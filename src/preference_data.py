"""Auditable hard-negative construction for evidence-grounded preferences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from .structured_answer import extract_inline_citations


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    error_type: str
    source_id: str


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
    return PreferencePair(prompt, chosen, rejected, error_type, source_id)


def validate_preference_pair(pair: PreferencePair, valid_citations: set[str]) -> dict[str, object]:
    chosen_ids = set(extract_inline_citations(pair.chosen))
    rejected_ids = set(extract_inline_citations(pair.rejected))
    return {
        "same_prompt": bool(pair.prompt.strip()),
        "different_completion": pair.chosen != pair.rejected,
        "chosen_citations_valid": chosen_ids <= valid_citations,
        "rejected_contains_invalid_citation": bool(rejected_ids - valid_citations),
        "error_type": pair.error_type,
        "source_id": pair.source_id,
    }
