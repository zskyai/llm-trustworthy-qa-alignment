"""Neural sentence relevance selection preserving original paragraph positions."""
from __future__ import annotations
import math
import re
from .paragraph_reranker import ParagraphCrossEncoder


class SentenceEvidenceSelector:
    def __init__(self, checkpoint=None, *, scorer=None, max_length=384, batch_size=16, device='cpu'):
        if scorer is None:
            if checkpoint is None:
                raise ValueError('A relevance checkpoint is required')
            scorer = ParagraphCrossEncoder(checkpoint, max_length=max_length, batch_size=batch_size, device=device)
        self.scorer = scorer
        self.checkpoint = str(checkpoint) if checkpoint is not None else getattr(scorer, 'checkpoint', None)

    def score(self, question, paragraphs):
        candidates = []
        paragraph_ids = [str(row['id']) for row in paragraphs]
        if len(set(paragraph_ids)) != len(paragraph_ids):
            raise ValueError('Duplicate source paragraph IDs')
        for paragraph in paragraphs:
            sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', str(paragraph['text'])) if part.strip()]
            for index, sentence in enumerate(sentences, start=1):
                candidates.append({'id': str(paragraph['id']) + '::sentence' + str(index),
                                   'paragraph_id': paragraph['id'], 'sentence_index': index,
                                   'section_name': paragraph.get('section_name', ''), 'text': sentence})
        if not candidates:
            return []
        ranked = self.scorer.rerank(question, candidates)
        expected = {row['id'] for row in candidates}
        if len(ranked) != len(candidates) or {row['id'] for row in ranked} != expected:
            raise ValueError('Relevance scorer changed candidate membership')
        source = {row['id']: row for row in candidates}
        result = []
        for row in ranked:
            score = float(row['rerank_score'])
            if not math.isfinite(score):
                raise ValueError('Non-finite sentence relevance score')
            result.append(dict(source[row['id']], score=score))
        return sorted(result, key=lambda row: (-row['score'], str(row['paragraph_id']), row['sentence_index']))

    def select(self, question, paragraphs, *, top_k=12, per_paragraph=2):
        if type(top_k) is not int or top_k < 1 or type(per_paragraph) is not int or per_paragraph < 1:
            raise ValueError('Sentence budgets must be positive integers')
        counts = {}; selected = []
        for row in self.score(question, paragraphs):
            pid = row['paragraph_id']
            if counts.get(pid, 0) >= per_paragraph:
                continue
            selected.append(row); counts[pid] = counts.get(pid, 0) + 1
            if len(selected) == top_k:
                break
        return selected

    def select_coverage_aware(self, question, paragraphs, *, top_k=3, per_paragraph=2, selector=None):
        """Listwise selection that trades answer-term coverage against redundancy."""
        from .evidence_set_selector import CoverageAwareEvidenceSelector
        selector = selector or CoverageAwareEvidenceSelector()
        return selector.select(question, self.score(question, paragraphs), top_k=top_k, max_per_paragraph=per_paragraph)
