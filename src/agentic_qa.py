"""Bounded tool-using workflow for evidence-grounded question answering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .structured_answer import StructuredAnswer, parse_structured_answer


@dataclass(frozen=True)
class QAAgentResult:
    answer: StructuredAnswer
    evidence: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]
    attempts: int


class EvidenceQAAgent:
    """Coordinate retrieval, generation, citation checks, and bounded repair.

    ``retrieve`` and ``generate`` are adapters for the deployment's retriever
    and language model. ``support_score`` should use a domain NLI verifier.
    """

    def __init__(
        self,
        retrieve: Callable[[str, int], Sequence[Mapping[str, Any]]],
        generate: Callable[[str, Sequence[Mapping[str, Any]], str], str | Mapping[str, Any]],
        support_score: Callable[[StructuredAnswer, Sequence[Mapping[str, Any]]], float],
        *,
        max_retries: int = 1,
        top_k: int = 8,
        min_support: float = 0.5,
    ) -> None:
        if max_retries < 0 or top_k <= 0 or not 0.0 <= min_support <= 1.0:
            raise ValueError("invalid agent limits or support threshold")
        self.retrieve = retrieve
        self.generate = generate
        self.support_score = support_score
        self.max_retries = max_retries
        self.top_k = top_k
        self.min_support = min_support

    def run(self, question: str) -> QAAgentResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        evidence: dict[str, dict[str, Any]] = {}
        trace: list[dict[str, Any]] = []
        feedback = "Find passages that directly answer the question."

        for attempt in range(self.max_retries + 1):
            hits = list(self.retrieve(question, self.top_k * (attempt + 1)))
            previous = evidence
            evidence = {}
            for hit in hits:
                evidence.setdefault(str(hit["evidence_id"]), dict(hit))
            for key, value in previous.items():
                evidence.setdefault(key, value)
            current = list(evidence.values())
            trace.append({"step": "retrieve", "attempt": attempt, "hits": len(hits), "evidence_total": len(current)})

            raw = self.generate(question, current, feedback)
            try:
                answer = parse_structured_answer(raw)
                invalid = answer.validate_citations(set(evidence))
                if not answer.abstained and not answer.citations:
                    invalid = (*invalid, "missing_citation")
                support = float(self.support_score(answer, current)) if not answer.abstained and not invalid else 0.0
                if not math.isfinite(support) or not 0.0 <= support <= 1.0:
                    raise ValueError("support_score must be finite and in [0, 1]")
            except (TypeError, ValueError) as exc:
                answer, invalid, support = None, ("malformed_answer",), 0.0
                feedback = f"The previous response failed validation ({type(exc).__name__}). Return the required JSON schema and cite only supplied evidence IDs."

            trace.append({"step": "verify", "attempt": attempt, "invalid_citations": list(invalid), "support_score": support})
            if answer is not None and not invalid and (answer.abstained or support >= self.min_support):
                return QAAgentResult(answer, tuple(current), tuple(trace), attempt + 1)
            if attempt < self.max_retries:
                if invalid:
                    feedback = f"Invalid citation check: {', '.join(invalid)}. Cite only supplied evidence IDs and return a reasoned abstention if none supports the answer."
                else:
                    feedback = f"Evidence support score {support:.3f} is below {self.min_support:.3f}. Remove unsupported claims or abstain with a reason."

        refusal = StructuredAnswer(
            final_answer="I cannot provide a supported answer from the available evidence.",
            citations=(),
            confidence=0.0,
            abstained=True,
            refusal_reason="evidence_validation_failed",
        )
        trace.append({"step": "stop", "reason": "retry_budget_exhausted"})
        return QAAgentResult(refusal, tuple(evidence.values()), tuple(trace), self.max_retries + 1)
