"""Run the Qasper paper evidence desk against the official train corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research_api import build_app
from src.research_workbench import QasperEvidenceWorkbench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasper", type=Path, default=Path("data/qasper_train.parquet"))
    parser.add_argument("--history-db", type=Path, default=Path("results/paper_qa_history.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--nli-model", type=Path)
    parser.add_argument("--min-support", type=float, default=.5)
    parser.add_argument("--library", type=Path, default=Path("data/paper_library"))
    parser.add_argument("--pubmedqa", type=Path, help="Optional official ori_pqal JSON; index contexts only")
    parser.add_argument('--paragraph-reranker', type=Path, help='Optional local relevance Cross-Encoder; use global BM25 Top-24 candidates')
    parser.add_argument('--sentence-selector', type=Path, help='Optional local sentence relevance Cross-Encoder; replaces lexical sentence coverage')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    model_qa = None
    if bool(args.model) != bool(args.nli_model):
        parser.error("--model and --nli-model must be supplied together")
    if args.model:
        from src.local_grounded_qa import LocalGroundedQA
        model_qa = LocalGroundedQA(args.model, args.nli_model, min_support=args.min_support)
    workbench = QasperEvidenceWorkbench.from_parquet(args.qasper)
    if args.paragraph_reranker:
        import torch
        from src.paragraph_reranker import ParagraphCrossEncoder
        torch.set_num_threads(args.threads)
        workbench.paragraph_reranker = ParagraphCrossEncoder(args.paragraph_reranker)
    if args.sentence_selector:
        import torch
        from src.sentence_evidence_selector import SentenceEvidenceSelector
        torch.set_num_threads(args.threads)
        workbench.sentence_selector = SentenceEvidenceSelector(args.sentence_selector)
    if args.pubmedqa:
        workbench.load_pubmedqa(args.pubmedqa)
    app = build_app(
        workbench,
        history_db_path=args.history_db,
        model_qa=model_qa,
        library_path=args.library,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
