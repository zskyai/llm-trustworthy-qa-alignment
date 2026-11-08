"""Check every cached retrieval prediction against an inspected upstream scorer."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_paragraph_retrieval import load_papers
from src.research_workbench import QasperEvidenceWorkbench
from src.qasper_metrics import references, score_evidence, answer_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qasper', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--official-evaluator', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('official_qasper_evaluator', a.official_evaluator)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    papers = load_papers(a.qasper); wb = QasperEvidenceWorkbench(papers)
    groups = {qid: paper['qas']['answers'][i] for paper in papers for i, qid in enumerate(paper['qas']['question_id'])}
    rows = [json.loads(s) for s in a.predictions.read_text(encoding='utf8').splitlines() if s.strip()]
    if len(rows) != len({r['question_id'] for r in rows}):
        raise ValueError('Duplicate prediction IDs')
    total = {}; maximum = 0.0; checks = 0
    for row in rows:
        refs = references(groups[row['question_id']])
        positions = {r['id']:r['text'] for r in wb.papers[row['paper_id']]['paragraphs']}
        for method, ids in row['predicted_positions'].items():
            texts = [positions[i] for i in ids]
            ours = score_evidence(texts, refs)['f1']
            theirs = max(official.paragraph_f1_score(texts, ref['evidence']) for ref in refs)
            maximum = max(maximum, abs(ours-theirs)); checks += 1
            if abs(ours-theirs) > 1e-12 or abs(row['scores'][method]['f1']-theirs) > 1e-12:
                raise AssertionError((row['question_id'], method, ours, theirs))
            total.setdefault(method, []).append(theirs)
        for ref in refs:
            for prediction in (ref['answer'], 'Unanswerable', '', 'Yes', 'No', 'The experimental method'):
                if abs(answer_f1(prediction, ref['answer'])-official.token_f1_score(prediction, ref['answer'])) > 1e-12:
                    raise AssertionError('Answer F1 mismatch')
    report = {'passed': True, 'questions': len(rows), 'evidence_checks': checks, 'max_abs_difference': maximum,
              'official_script_sha256': hashlib.sha256(a.official_evaluator.read_bytes()).hexdigest(),
              'predictions_sha256': hashlib.sha256(a.predictions.read_bytes()).hexdigest(),
              'evidence_f1': {key:sum(values)/len(values) for key,values in total.items()},
              'answer_generation_evaluated': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__': main()
