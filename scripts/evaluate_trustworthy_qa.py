"""Simple evaluation placeholders for trustworthy QA outputs."""

from dataclasses import dataclass


@dataclass
class EvalResult:
    citation_precision: float
    unsupported_claim_rate: float
    refusal_accuracy: float


def evaluate_sample() -> EvalResult:
    return EvalResult(
        citation_precision=0.0,
        unsupported_claim_rate=0.0,
        refusal_accuracy=0.0,
    )


if __name__ == "__main__":
    print(evaluate_sample())
