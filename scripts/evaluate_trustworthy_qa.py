"""Re-score a real prediction file produced by ``real_pipeline.py``."""
import argparse, json, re
from pathlib import Path

def main():
    p = argparse.ArgumentParser(); p.add_argument("--predictions", type=Path, required=True); p.add_argument("--output", type=Path)
    a = p.parse_args(); rows = json.loads(a.predictions.read_text(encoding="utf-8"));
    answerable = [r for r in rows if r.get("answerable")]
    unanswerable = [r for r in rows if not r.get("answerable")]
    def refused(r):
        t = r.get("prediction", "").lower(); return "[no_answer]" in t or "insufficient evidence" in t or "cannot answer" in t
    valid = 0; cited = 0
    for r in answerable:
        cites = re.findall(r"\[E(\d+)\]", r.get("prediction", "")); allowed = {x[1:] for x in r.get("valid_citations", [])}
        cited += bool(cites); valid += bool(cites) and all(c in allowed for c in cites)
    result = {"count": len(rows), "citation_precision": valid / cited if cited else 0.0,
              "unsupported_claim_rate": sum(bool(r.get("invalid_citation")) for r in rows) / max(len(rows), 1),
              "refusal_accuracy": sum(refused(r) for r in unanswerable) / len(unanswerable) if unanswerable else None,
              "answerable_non_refusal_rate": sum(not refused(r) for r in answerable) / len(answerable) if answerable else None}
    if a.output: a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
