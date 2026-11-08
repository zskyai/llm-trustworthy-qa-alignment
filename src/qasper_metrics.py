"""Qasper answer/evidence scoring following the released evaluator semantics.

Reference: allenai/qasper-led-baseline, scripts/evaluator.py. Evidence is
paragraph text, not highlighted substrings. Alternative annotations are scored
independently; their best evidence F1 is reported, as in the official code.
"""
from __future__ import annotations
import re
import string
from collections import Counter


def normalize_answer(text):
    text = ''.join(c for c in str(text).lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_f1(prediction, reference):
    pred, gold = normalize_answer(prediction).split(), normalize_answer(reference).split()
    overlap = sum((Counter(pred) & Counter(gold)).values())
    return 2 * overlap / (len(pred) + len(gold)) if overlap else 0.0


def references(group, *, text_only=False):
    result = []
    for annotation in group.get('answer') or []:
        if annotation.get('unanswerable'):
            result.append({'answer': 'Unanswerable', 'evidence': [], 'type': 'none'})
            continue
        if annotation.get('extractive_spans'):
            answer, kind = ', '.join(annotation['extractive_spans']), 'extractive'
        elif annotation.get('free_form_answer'):
            answer, kind = annotation['free_form_answer'], 'abstractive'
        elif annotation.get('yes_no') is not None:
            answer, kind = ('Yes' if annotation['yes_no'] else 'No'), 'boolean'
        else:
            raise ValueError('Answerable Qasper annotation has no answer')
        evidence = list(annotation.get('evidence') or [])
        if text_only:
            evidence = [text for text in evidence if 'FLOAT SELECTED' not in text]
        result.append({'answer': answer, 'evidence': evidence, 'type': kind})
    if not result:
        raise ValueError('Question has no reference annotation')
    return result


def evidence_prf(prediction, gold):
    if not prediction and not gold:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    same = len(set(prediction) & set(gold))
    precision = same / len(prediction) if prediction else 0.0
    recall = same / len(gold) if gold else 0.0
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if same else 0.0}


def score_evidence(prediction, refs):
    scores = [evidence_prf(prediction, ref['evidence']) for ref in refs]
    return max(scores, key=lambda x: x['f1'])


def match_evidence_positions(paragraphs, evidence):
    """Audit source availability with whitespace normalization only.

    This mapping is diagnostic; the official-compatible score above retains
    exact text comparison. Figure/table evidence cannot be mapped to paragraphs.
    """
    canonical = lambda text: ' '.join(str(text).split())
    lookup = {}
    for paragraph in paragraphs:
        lookup.setdefault(canonical(paragraph['text']), []).append(paragraph['id'])
    mapped, missing, floats = [], [], []
    for text in evidence:
        if 'FLOAT SELECTED' in text:
            floats.append(text)
        elif canonical(text) in lookup:
            mapped.append({'text': text, 'paragraph_ids': lookup[canonical(text)]})
        else:
            missing.append(text)
    return {'mapped': mapped, 'missing': missing, 'figure_table': floats}
