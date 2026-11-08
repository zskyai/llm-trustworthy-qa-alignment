"""Section-to-paragraph retrieval with diversity-aware passage selection."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Mapping, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: object) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _section_id(paragraph: Mapping[str, object]) -> str:
    value = paragraph.get("section_id", paragraph.get("section_name"))
    if value is not None:
        return str(value)
    return str(paragraph.get("id", "document")).split(":", 1)[0]


class HierarchicalParagraphRetriever:
    """Rank parent sections, then select diverse child paragraphs."""

    def __init__(self, paragraphs: Sequence[Mapping[str, object]], *, k1: float = 1.2, b: float = 0.75) -> None:
        if not paragraphs or k1 <= 0 or not 0.0 <= b <= 1.0:
            raise ValueError("paragraphs must be non-empty and BM25 parameters valid")
        self.paragraphs = [dict(row) for row in paragraphs]
        self.tokens = [_tokens(row.get("text", "")) for row in self.paragraphs]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.k1, self.b = k1, b
        self.term_df: Counter[str] = Counter()
        self.sections: dict[str, list[int]] = defaultdict(list)
        for index, (row, tokens) in enumerate(zip(self.paragraphs, self.tokens)):
            self.sections[_section_id(row)].append(index)
            self.term_df.update(set(tokens))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        section_top_k: int | None = None,
        mmr_lambda: float = 0.8,
    ) -> list[dict[str, object]]:
        if top_k <= 0 or (section_top_k is not None and section_top_k <= 0) or not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("top_k must be positive and mmr_lambda in [0, 1]")
        terms = Counter(_tokens(query))
        if not terms:
            return []
        scores = [self._bm25(terms, tokens, length) for tokens, length in zip(self.tokens, self.lengths)]
        section_scores = {}
        for section, indices in self.sections.items():
            best = sorted((scores[index] for index in indices), reverse=True)
            section_scores[section] = best[0] + 0.25 * (best[1] if len(best) > 1 else 0.0)
        parent_limit = section_top_k or max(1, min(len(self.sections), math.ceil(top_k / 2)))
        parents = {
            section
            for section, _ in sorted(section_scores.items(), key=lambda item: (-item[1], item[0]))[:parent_limit]
        }
        candidates = [index for section in parents for index in self.sections[section]]
        if not candidates:
            return []
        low = min(scores[index] for index in candidates)
        high = max(scores[index] for index in candidates)
        relevance = {
            index: (scores[index] - low) / (high - low) if high > low else (1.0 if high > 0 else 0.0)
            for index in candidates
        }
        query_terms = set(terms)
        selected: list[int] = []
        remaining = set(candidates)
        while remaining and len(selected) < top_k:
            def objective(index: int) -> tuple[float, int]:
                novelty = 1.0 - max((self._jaccard(self.tokens[index], self.tokens[other]) for other in selected), default=0.0)
                query_coverage = len(query_terms & set(self.tokens[index])) / len(query_terms)
                score = mmr_lambda * relevance[index] + (1.0 - mmr_lambda) * novelty
                return score + 0.05 * query_coverage, -index

            best = max(remaining, key=objective)
            selected.append(best)
            remaining.remove(best)
        return [self.paragraphs[index] | {"score": float(scores[index]), "section_id": _section_id(self.paragraphs[index])} for index in selected]

    def retrieve_global(self, query: str, *, top_k: int = 24) -> list[dict[str, object]]:
        """Global BM25 candidates avoid irrecoverable parent-section pruning."""
        if top_k <= 0:
            raise ValueError('top_k must be positive')
        terms = Counter(_tokens(query))
        if not terms:
            return []
        scores = [self._bm25(terms, tokens, length) for tokens, length in zip(self.tokens, self.lengths)]
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.paragraphs[i] | {'score': float(scores[i]), 'section_id': _section_id(self.paragraphs[i])}
                for i in indices[:top_k] if scores[i] > 0]

    def _bm25(self, query: Counter[str], document: Sequence[str], length: int) -> float:
        counts = Counter(document)
        score = 0.0
        for term, qtf in query.items():
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = self.term_df[term]
            idf = math.log(1.0 + (len(self.paragraphs) - df + 0.5) / (df + 0.5))
            denominator = tf + self.k1 * (1.0 - self.b + self.b * length / max(self.avg_length, 1.0))
            score += idf * tf * (self.k1 + 1.0) / denominator * min(qtf, 2)
        return score

    @staticmethod
    def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
        a, b = set(left), set(right)
        return len(a & b) / len(a | b) if a or b else 0.0
