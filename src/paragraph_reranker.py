"""Source-preserving, pretrained neural reranking for paper paragraphs."""
from __future__ import annotations
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ParagraphCrossEncoder:
    def __init__(self, checkpoint, *, max_length=384, batch_size=16, device='cpu'):
        if max_length <= 0 or batch_size <= 0:
            raise ValueError('Token and batch budgets must be positive')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.checkpoint = str(checkpoint)

    @torch.inference_mode()
    def rerank(self, question, paragraphs):
        scores = []
        for start in range(0, len(paragraphs), self.batch_size):
            batch = paragraphs[start:start+self.batch_size]
            texts = [str(p.get('section_name', '')) + '\n' + p['text'] for p in batch]
            encoded = self.tokenizer([question]*len(batch), texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            logits = self.model(**{k:v.to(self.device) for k,v in encoded.items()}).logits
            if logits.ndim != 2 or logits.shape[1] not in (1,2):
                raise ValueError('Expected a relevance head, not a stance classifier')
            values = logits[:,0] if logits.shape[1] == 1 else logits[:,1]-logits[:,0]
            scores.extend(values.cpu().tolist())
        if len(scores) != len(paragraphs) or any(not math.isfinite(x) for x in scores):
            raise ValueError('Invalid relevance scores')
        ranked = [dict(p, rerank_score=float(s)) for p,s in zip(paragraphs,scores)]
        return sorted(ranked, key=lambda p:(-p['rerank_score'], p['id']))
