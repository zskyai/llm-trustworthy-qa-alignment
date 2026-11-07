"""Evaluate prediction JSON produced by real_pipeline or a structured adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("prediction file must contain a JSON list")
    metrics = evaluate_predictions(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
