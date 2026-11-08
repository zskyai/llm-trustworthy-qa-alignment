"""HTTP application for the Qasper paper evidence desk."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .research_workbench import PaperQAStore, QasperEvidenceWorkbench


class AskRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    mode: Literal["extractive", "verified_model"] = "extractive"
    max_retries: int = Field(default=1, ge=0, le=2)


class FeedbackRequest(BaseModel):
    helpful: bool
    note: str = Field(default="", max_length=1000)


def build_app(
    workbench: QasperEvidenceWorkbench,
    *,
    history_db_path: str | Path = "results/paper_qa_history.sqlite3",
    model_qa=None,
    library_path: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Paper Evidence Desk", version="0.1.0")
    history = PaperQAStore(history_db_path)
    library = None
    if library_path is not None:
        from .paper_ingestion import PaperLibrary
        library = PaperLibrary(library_path)
        for row in library.rows():
            workbench.add_paper(row)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        counts = {}
        for paper in workbench.papers.values():
            kind = paper['source']['kind']
            counts[kind] = counts.get(kind, 0) + 1
        return {"status": "ok", "dataset": "Qasper official train corpus + local paper library", "papers": len(workbench.papers), "sources": counts, "verified_model_enabled": model_qa is not None}

    @app.get("/v1/papers")
    def search_papers(query: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            return workbench.search_papers(query, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/papers/{paper_id}")
    def get_paper(paper_id: str) -> dict[str, Any]:
        try:
            return workbench.get_paper(paper_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="paper not found") from exc

    @app.post("/v1/library/pdf")
    async def import_pdf(request: Request, filename: str = "paper.pdf"):
        if library is None:
            raise HTTPException(status_code=503, detail="PDF library is not configured")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="PDF exceeds the 10 MiB upload limit")
        try:
            # Parsing and disk I/O must not block the event loop.
            from starlette.concurrency import run_in_threadpool
            row, created = await run_in_threadpool(library.import_pdf, bytes(data), filename)
            paper = workbench.add_paper(row)
            return {"created": created, "paper": paper}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/library/{paper_id}/pdf")
    def original_pdf(paper_id: str):
        if library is None:
            raise HTTPException(status_code=404, detail="PDF library is not configured")
        try:
            return FileResponse(library.pdf_path(paper_id), media_type="application/pdf")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="PDF not found") from exc

    @app.post("/v1/questions")
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            if not request.question.strip():
                raise ValueError("question must not be empty")
            if request.mode == "verified_model":
                if model_qa is None:
                    raise HTTPException(status_code=503, detail="Start with --model and --nli-model to enable verified generation")
                result = model_qa.ask(workbench, request.paper_id, request.question, top_k=request.top_k, max_retries=request.max_retries)
            else:
                result = workbench.ask(request.paper_id, request.question, top_k=request.top_k)
            return history.save(request.paper_id, request.question, result)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="paper not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/questions")
    def list_questions(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200 or offset < 0:
            raise HTTPException(status_code=422, detail="limit must be 1..200 and offset non-negative")
        return history.list(limit, offset)

    @app.get("/v1/questions/{question_id}")
    def get_question(question_id: str) -> dict[str, Any]:
        try:
            return history.get(question_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="question not found") from exc

    @app.get("/v1/papers/{paper_id}/paragraphs/{paragraph_id}")
    def source_paragraph(paper_id: str, paragraph_id: str):
        try:
            return workbench.get_paragraph(paper_id, paragraph_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="source paragraph not found") from exc

    @app.get("/v1/questions/{question_id}/report")
    def report(question_id: str):
        try:
            item = history.get(question_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="question not found") from exc
        # Review candidates are exported separately from any training data.
        payload = {"schema_version": 1, "purpose": "research_review", "training_eligible": False, **item}
        return Response(json.dumps(payload, ensure_ascii=False, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="question-{item["question_id"]}.json"'})

    @app.post("/v1/questions/{question_id}/feedback")
    def submit_feedback(question_id: str, request: FeedbackRequest) -> dict[str, Any]:
        try:
            return history.add_feedback(question_id, request.helpful, request.note.strip())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="question not found") from exc

    return app
