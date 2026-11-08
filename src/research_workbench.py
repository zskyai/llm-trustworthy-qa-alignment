"""Paper-level evidence retrieval for the Qasper research workflow."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
import time
from functools import lru_cache
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hierarchical_retrieval import HierarchicalParagraphRetriever

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has", "have", "how", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with"}


def _tokens(text: object) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if token not in _STOPWORDS]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QasperEvidenceWorkbench:
    """Search papers and return source-preserving, extractive evidence."""

    def __init__(self, papers: Sequence[Mapping[str, Any]], *, paragraph_reranker=None, sentence_selector=None) -> None:
        self.paragraph_reranker = paragraph_reranker
        self.sentence_selector = sentence_selector
        self.papers: dict[str, dict[str, Any]] = {}
        self.search_rows: list[tuple[str, list[str]]] = []
        for row in papers:
            paper_id = str(row["id"])
            full_text = row.get("full_text") or {}
            section_names = full_text.get("section_name") or []
            section_paragraphs = full_text.get("paragraphs") or []
            paragraphs: list[dict[str, Any]] = []
            for section_index, values in enumerate(section_paragraphs):
                section = str(section_names[section_index]) if section_index < len(section_names) else f"Section {section_index + 1}"
                for paragraph_index, text in enumerate(values or []):
                    text = re.sub(r"\s+", " ", str(text or "")).strip()
                    if text:
                        paragraphs.append({
                            "id": f"s{section_index + 1}:p{paragraph_index + 1}",
                            "section_id": section,
                            "section_name": section,
                            "paragraph_index": paragraph_index + 1,
                            "text": text,
                        })
            paper = {
                "paper_id": paper_id,
                "title": str(row.get("title") or "Untitled paper"),
                "abstract": str(row.get("abstract") or ""),
                "paragraphs": paragraphs,
                "source": dict(row.get("source") or {"kind": "qasper_official_corpus"}),
            }
            self.papers[paper_id] = paper
            self.search_rows.append((paper_id, _tokens(f"{paper['title']} {paper['abstract']}")))
        if not self.papers:
            raise ValueError("Qasper paper collection is empty")
        self.document_frequency = Counter(term for _, tokens in self.search_rows for term in set(tokens))

    def add_paper(self, row: Mapping[str, Any]) -> dict[str, Any]:
        parsed = QasperEvidenceWorkbench([row])
        paper_id = str(row['id'])
        if paper_id in self.papers:
            return self.get_paper(paper_id)
        self.papers.update(parsed.papers)
        self.search_rows.extend(parsed.search_rows)
        self.document_frequency.update(set(parsed.search_rows[0][1]))
        return self.get_paper(paper_id)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "QasperEvidenceWorkbench":
        import pyarrow.parquet as pq

        return cls(pq.read_table(path, columns=["id", "title", "abstract", "full_text"]).to_pylist())

    def load_pubmedqa(self, path: str | Path) -> int:
        records = json.loads(Path(path).read_text(encoding='utf-8'))
        count = 0
        for pmid, record in records.items():
            contexts = record.get('CONTEXTS') or []
            labels = record.get('LABELS') or []
            if not contexts:
                continue
            # Decisions, long answers and model predictions are never indexed.
            self.add_paper({'id': 'pubmed-' + str(pmid), 'title': 'PubMedQA question: ' + str(record.get('QUESTION', pmid)),
                            'abstract': ' '.join(contexts),
                            'full_text': {'section_name': [str(labels[i]) if i < len(labels) else 'Abstract' for i in range(len(contexts))],
                                          'paragraphs': [[text] for text in contexts]},
                            'source': {'kind': 'pubmedqa_context', 'pmid': str(pmid), 'title_is_dataset_question': True}})
            count += 1
        return count

    def search_papers(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        terms = Counter(_tokens(query))
        document_frequency = self.document_frequency
        ranked = []
        for paper_id, tokens in self.search_rows:
            counts = Counter(tokens)
            score = 0.0
            for term, query_frequency in terms.items():
                frequency = counts[term]
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.search_rows) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2) * min(query_frequency, 2)
            if score > 0:
                ranked.append((score, paper_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {key: self.papers[paper_id][key] for key in ("paper_id", "title", "abstract")} | {
                "paragraph_count": len(self.papers[paper_id]["paragraphs"]), "score": score,
            }
            for score, paper_id in ranked[:limit]
        ]

    def get_paper(self, paper_id: str) -> dict[str, Any]:
        paper = self.papers.get(paper_id)
        if paper is None:
            raise KeyError(paper_id)
        return {key: paper[key] for key in ("paper_id", "title", "abstract", "source")} | {
            "paragraph_count": len(paper["paragraphs"]),
        }

    def get_paragraph(self, paper_id: str, paragraph_id: str) -> dict[str, Any]:
        paper = self.papers.get(paper_id)
        if paper is None:
            raise KeyError(paper_id)
        return next((dict(row) for row in paper["paragraphs"] if row["id"] == paragraph_id), None) or self._missing(paragraph_id)

    @staticmethod
    def _missing(key: str):
        raise KeyError(key)

    @lru_cache(maxsize=128)
    def _retriever(self, paper_id: str) -> HierarchicalParagraphRetriever:
        return HierarchicalParagraphRetriever(self.papers[paper_id]["paragraphs"])

    def ask(self, paper_id: str, question: str, *, top_k: int = 6) -> dict[str, Any]:
        started = time.perf_counter()
        if not question.strip():
            raise ValueError("question must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        paper = self.papers.get(paper_id)
        if paper is None:
            raise KeyError(paper_id)
        paragraphs = paper["paragraphs"]
        if not paragraphs:
            return {
                "paper_id": paper_id, "question": question, "answer": None,
                "abstained": True, "abstention_reason": "paper_has_no_full_text",
                "evidence": [], "retrieval_mode": "section_first_bm25_mmr_extractive",
            }
        retriever = self._retriever(paper_id)
        if self.paragraph_reranker is not None:
            candidates = retriever.retrieve_global(question, top_k=max(24, top_k))
            hits = self.paragraph_reranker.rerank(question, candidates)[:top_k]
            retrieval_mode = 'global_bm25_cross_encoder_extractive'
        else:
            hits = [row for row in retriever.retrieve(question, top_k=top_k) if float(row["score"]) > 0]
            retrieval_mode = 'section_first_bm25_mmr_extractive'
        evidence = []
        query_terms = set(_tokens(question))
        selected_by_paragraph = {}
        if self.sentence_selector is not None:
            selected = self.sentence_selector.select(question, hits, top_k=max(2, top_k * 2), per_paragraph=2)
            for item in selected:
                selected_by_paragraph.setdefault(item['paragraph_id'], []).append(item)
        for row in hits:
            if self.sentence_selector is not None:
                best = [(float(item['score']), int(item['sentence_index']), item['text'])
                        for item in selected_by_paragraph.get(row['id'], [])]
            else:
                query_terms = set(_tokens(question))
                sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(row["text"])) if part.strip()]
                ranked_sentences = sorted(
                    ((len(query_terms & set(_tokens(sentence))) / max(len(query_terms), 1), index, sentence)
                     for index, sentence in enumerate(sentences, start=1)),
                    key=lambda item: (-item[0], item[1]),
                )
                best = [(score, index, sentence) for score, index, sentence in ranked_sentences[:2] if score > 0]
            if not best:
                continue
            evidence.append({
                "evidence_id": f"E_{row['id'].replace(':', '_')}",
                "paper_id": paper_id,
                "title": paper["title"],
                "section": row["section_name"],
                "paragraph_id": row["id"],
                "paragraph_index": row["paragraph_index"],
                "retrieval_score": float(row["score"]),
                "rerank_score": row.get('rerank_score'),
                "sentences": [
                    {"sentence_index": index, "text": sentence,
                     "query_coverage": len(query_terms & set(_tokens(sentence))) / max(len(query_terms), 1),
                     "selection_score": score}
                    for score, index, sentence in best
                ],
            })
        answer = " ".join(
            f"{sentence['text']} [{item['evidence_id']}]"
            for item in evidence for sentence in item["sentences"]
        ) or None
        return {
            "paper_id": paper_id,
            "title": paper["title"],
            "question": question,
            "answer": answer,
            "answer_mode": "extractive_evidence_only",
            "abstained": not evidence,
            "abstention_reason": "no_matching_passage_in_selected_paper" if not evidence else None,
            "evidence": evidence,
            "retrieval_mode": retrieval_mode + ("_sentence_cross_encoder" if self.sentence_selector is not None else ""),
            "sentence_selector": getattr(self.sentence_selector, 'checkpoint', None),
            "paragraph_reranker": getattr(self.paragraph_reranker, 'checkpoint', None),
            "timings_ms": {"total_ms": (time.perf_counter() - started) * 1000},
            "source": paper["source"],
        }


class PaperQAStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    question_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id TEXT NOT NULL REFERENCES questions(question_id),
                    helpful INTEGER NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS feedback_question_idx ON feedback(question_id, feedback_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def save(self, paper_id: str, question: str, result: Mapping[str, Any]) -> dict[str, Any]:
        question_id = uuid.uuid4().hex
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO questions VALUES (?, ?, ?, ?, ?)",
                (question_id, paper_id, question, json.dumps(dict(result), ensure_ascii=False), _utc_now()),
            )
        return self.get(question_id)

    def get(self, question_id: str) -> dict[str, Any]:
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM questions WHERE question_id = ?", (question_id,)).fetchone()
            if row is None:
                raise KeyError(question_id)
            feedback = db.execute(
                "SELECT helpful, note, created_at FROM feedback WHERE question_id = ? ORDER BY feedback_id",
                (question_id,),
            ).fetchall()
        fields = dict(row)
        fields.pop("result_json")
        return {**fields, "result": json.loads(row["result_json"]), "feedback": [dict(item) for item in feedback]}

    def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with closing(self._connect()) as db, db:
            ids = db.execute("SELECT question_id FROM questions ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [self.get(row["question_id"]) for row in ids]

    def add_feedback(self, question_id: str, helpful: bool, note: str) -> dict[str, Any]:
        with closing(self._connect()) as db, db:
            if db.execute("SELECT 1 FROM questions WHERE question_id = ?", (question_id,)).fetchone() is None:
                raise KeyError(question_id)
            db.execute(
                "INSERT INTO feedback(question_id, helpful, note, created_at) VALUES (?, ?, ?, ?)",
                (question_id, int(helpful), note, _utc_now()),
            )
        return self.get(question_id)
