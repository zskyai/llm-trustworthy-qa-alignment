"""Strict, serializable answer protocol for evidence-grounded QA."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvidenceCitation:
    evidence_id: str
    sentence_id: str | None = None
    quote: str | None = None


@dataclass(frozen=True)
class StructuredAnswer:
    final_answer: str
    citations: tuple[EvidenceCitation, ...]
    confidence: float
    abstained: bool = False
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.final_answer.strip():
            raise ValueError("final_answer must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.abstained and not self.refusal_reason:
            raise ValueError("abstained answers require refusal_reason")
        if not self.abstained and self.refusal_reason:
            raise ValueError("answerable responses cannot contain refusal_reason")

    def validate_citations(self, valid_ids: set[str]) -> tuple[str, ...]:
        invalid = tuple(citation.evidence_id for citation in self.citations if citation.evidence_id not in valid_ids)
        return invalid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"citations": [asdict(citation) for citation in self.citations]}


_CITATION_RE = re.compile(r"\[E([A-Za-z0-9_.:-]+)\]")


def parse_structured_answer(payload: str | Mapping[str, Any]) -> StructuredAnswer:
    """Parse JSON or a fenced JSON response and enforce the output contract."""

    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not valid JSON") from exc
    else:
        data = dict(payload)
    if not isinstance(data, Mapping):
        raise ValueError("structured answer must be a JSON object")
    citations = data.get("citations", [])
    if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
        raise ValueError("citations must be a list")
    parsed: list[EvidenceCitation] = []
    for citation in citations:
        if isinstance(citation, str):
            match = _CITATION_RE.fullmatch(citation.strip())
            if not match:
                raise ValueError(f"invalid citation: {citation}")
            parsed.append(EvidenceCitation("E" + match.group(1)))
        elif isinstance(citation, Mapping) and citation.get("evidence_id"):
            parsed.append(EvidenceCitation(str(citation["evidence_id"]), citation.get("sentence_id"), citation.get("quote")))
        else:
            raise ValueError("citation must be [E#] or an evidence object")
    return StructuredAnswer(
        final_answer=str(data.get("final_answer", "")),
        citations=tuple(parsed),
        confidence=float(data.get("confidence", 0.0)),
        abstained=bool(data.get("abstained", False)),
        refusal_reason=str(data["refusal_reason"]) if data.get("refusal_reason") else None,
    )


def extract_inline_citations(text: str) -> tuple[str, ...]:
    """Extract citation IDs for legacy free-form generations."""

    return tuple(dict.fromkeys("E" + match.group(1) for match in _CITATION_RE.finditer(text)))
