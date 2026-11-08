"""Full held-out paper evaluation using source text, never oracle evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import token_f1
from src.research_workbench import QasperEvidenceWorkbench
from run_qasper_protocol import choose, gold_spans


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def references(group):
    refs = []
    for answer in group.get('answer') or []:
        if answer.get('unanswerable'): refs.append('')
        elif answer.get('yes_no') is not None: refs.append('yes' if answer['yes_no'] else 'no')
        elif answer.get('free_form_answer'): refs.append(answer['free_form_answer'])
        else: refs.append(' '.join(answer.get('extractive_spans') or []))
    return refs or ['']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qasper', type=Path, default=Path('data/qasper_train.parquet'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--model', type=Path)
    p.add_argument('--nli-model', type=Path)
    p.add_argument('--min-support', type=float, default=.5)
    a = p.parse_args()
    if bool(a.model) != bool(a.nli_model): p.error('--model and --nli-model must be supplied together')
    if a.output.exists() and any(a.output.iterdir()): raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True)
    # The inference object reads only id/title/abstract/full_text columns.
    wb = QasperEvidenceWorkbench.from_parquet(a.qasper)
    model = None
    if a.model:
        from src.local_grounded_qa import LocalGroundedQA
        model = LocalGroundedQA(a.model, a.nli_model, min_support=a.min_support)
    raw = pq.read_table(a.qasper, columns=['id', 'qas']).to_pylist()
    heldout = [paper for paper in raw if int(hashlib.sha1(str(paper['id']).encode()).hexdigest()[:8], 16) / 2**32 < .2]
    rows = []; expected = sum(len(p['qas'].get('question') or []) for p in heldout); failures = []
    with (a.output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for paper in heldout:
            paper_id = str(paper['id']); qas = paper['qas']
            for i, question in enumerate(qas.get('question') or []):
                tick = time.perf_counter()
                try:
                    result = model.ask(wb, paper_id, question, top_k=a.top_k) if model else wb.ask(paper_id, question, top_k=a.top_k)
                    elapsed = (time.perf_counter() - tick) * 1000
                    # Annotations become visible only after prediction has finished.
                    group = qas['answers'][i]; annotation = choose(group); gold = gold_spans(annotation)
                    sources = [wb.get_paragraph(paper_id, e['paragraph_id'])['text'] for e in result['evidence']]
                    recalled = sum(max((token_f1(span, source) for source in sources), default=0) >= .5 for span in gold)
                    matched = sum(max((token_f1(source, span) for span in gold), default=0) >= .5 for source in sources) if gold else 0
                    valid = total = 0
                    for evidence in result['evidence']:
                        paragraph = wb.get_paragraph(paper_id, evidence['paragraph_id'])
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph['text'])
                        for sentence in evidence['sentences']:
                            total += 1; index = sentence['sentence_index'] - 1
                            valid += int(0 <= index < len(sentences) and sentences[index].strip() == sentence['text'])
                    refused = bool(result['abstained']); answerable = not bool(annotation.get('unanswerable'))
                    refs = references(group); prediction = '' if refused else result['answer'] or ''
                    answer_f1 = max(token_f1(prediction, ref) for ref in refs)
                    row = {'paper_id': paper_id, 'question': question, 'question_index': i, 'result': result,
                           'answerable': answerable, 'answer_token_f1_proxy': answer_f1,
                           'evidence_recall_proxy': recalled / len(gold) if gold else None,
                           'evidence_precision_proxy': matched / len(sources) if gold and sources else (0.0 if gold else None),
                           'source_sentences': total, 'valid_source_sentences': valid, 'latency_ms': elapsed,
                           'document_words': sum(len(p['text'].split()) for p in wb.papers[paper_id]['paragraphs'])}
                    rows.append(row); stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
                except Exception as exc:
                    failed = {'paper_id': paper_id, 'question_index': i, 'error': str(exc)}
                    failures.append(failed); stream.write(json.dumps({'failure': failed}) + '\n'); stream.flush()
            print(json.dumps({'completed': len(rows), 'expected': expected, 'failures': len(failures)}), flush=True)
    def mean(key, items):
        vals = [r[key] for r in items if r[key] is not None]
        return sum(vals) / len(vals) if vals else None
    answerable = [r for r in rows if r['answerable']]; unanswerable = [r for r in rows if not r['answerable']]
    refusals = [r for r in rows if r['result']['abstained']]; true_refusals = sum(not r['answerable'] for r in refusals)
    total_source = sum(r['source_sentences'] for r in rows)
    summary = {'split': 'paper_sha1_first8_lt_0.2_from_official_train', 'papers': len(heldout), 'expected_questions': expected,
               'completed_questions': len(rows), 'failures': failures, 'complete': len(rows) == expected and not failures,
               'official_test': False, 'test_used': False, 'gold_evidence_in_input': False,
               'source_sha256': hashlib.sha256(a.qasper.read_bytes()).hexdigest(),
               'mode': 'local_model_nli_verified' if model else 'extractive_evidence_only', 'top_k': a.top_k,
               'mean_answer_token_f1_proxy': mean('answer_token_f1_proxy', rows),
               'evidence_recall_proxy': mean('evidence_recall_proxy', rows), 'evidence_precision_proxy': mean('evidence_precision_proxy', rows),
               'source_location_validity': sum(r['valid_source_sentences'] for r in rows) / total_source if total_source else None,
               'citation_support_rate': None, 'refusal_rate': len(refusals) / max(len(rows), 1),
               'refusal_precision': true_refusals / len(refusals) if refusals else None,
               'refusal_recall': true_refusals / len(unanswerable) if unanswerable else None,
               'over_refusal_rate': sum(r['result']['abstained'] for r in answerable) / max(len(answerable), 1),
               'latency_ms': {'p50': percentile([r['latency_ms'] for r in rows], .5), 'p95': percentile([r['latency_ms'] for r in rows], .95)},
               'length_buckets': {name: {'count': len(group), 'answer_f1_proxy': mean('answer_token_f1_proxy', group), 'evidence_recall_proxy': mean('evidence_recall_proxy', group)}
                                  for name, group in [('under_2048_words', [r for r in rows if r['document_words'] < 2048]),
                                                      ('2048_to_4095_words', [r for r in rows if 2048 <= r['document_words'] < 4096]),
                                                      ('4096_plus_words', [r for r in rows if r['document_words'] >= 4096])]},
               'scorer': 'token overlap proxy, not official Qasper evaluator; evidence uses paragraph/span F1 >= 0.5',
               'human_reviews': 0}
    (a.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary['complete']: raise SystemExit(1)


if __name__ == '__main__': main()
