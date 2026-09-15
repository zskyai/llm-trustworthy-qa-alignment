"""Materialize preference pairs from official Qasper/PubMedQA data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from real_pipeline import pubmedqa_records, qasper_records, split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qasper", type=Path, required=True)
    p.add_argument("--pubmedqa", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    rows = qasper_records(a.qasper) + (pubmedqa_records(a.pubmedqa) if a.pubmedqa else [])
    train, valid = split(rows); a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("train", train), ("validation", valid)):
        (a.output_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in values) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "train": len(train), "validation": len(valid)}, indent=2))

if __name__ == "__main__":
    main()
