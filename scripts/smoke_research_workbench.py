"""Exercise the Qasper API using a real official-train paper."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research_api import build_app
from src.research_workbench import QasperEvidenceWorkbench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasper", type=Path, default=Path("data/qasper_train.parquet"))
    args = parser.parse_args()
    raw = pq.read_table(args.qasper, columns=["id", "title", "abstract", "full_text", "qas"]).to_pylist()
    sample = next(
        paper for paper in raw
        if paper.get("qas", {}).get("question") and paper.get("full_text", {}).get("paragraphs")
    )
    question = sample["qas"]["question"][0]
    with tempfile.TemporaryDirectory() as directory:
        app = build_app(
            QasperEvidenceWorkbench(raw),
            history_db_path=Path(directory) / "smoke.sqlite3",
        )
        with TestClient(app) as client:
            health = client.get("/healthz").raise_for_status().json()
            search = client.get("/v1/papers", params={"query": sample["title"][:80]}).raise_for_status().json()
            assert any(item["paper_id"] == sample["id"] for item in search)
            response = client.post(
                "/v1/questions",
                json={"paper_id": sample["id"], "question": question, "top_k": 6},
            ).raise_for_status().json()
            assert response["result"]["paper_id"] == sample["id"]
            review = client.post(
                f"/v1/questions/{response['question_id']}/feedback",
                json={"helpful": True, "note": "Smoke test"},
            ).raise_for_status().json()
            assert len(review["feedback"]) == 1
            page = client.get("/").raise_for_status().text
            assert "Paper Evidence Desk" in page
    print({
        "status": health["status"],
        "official_train_papers": health["papers"],
        "searched_paper": sample["id"],
        "question_recorded": response["question_id"],
        "abstained": response["result"]["abstained"],
        "evidence_items": len(response["result"]["evidence"]),
        "feedback_saved": True,
    })


if __name__ == "__main__":
    main()
