"""Import full application predictions for source review, never as human labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_workbench import PaperQAStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--db', type=Path, default=Path('results/paper_qa_history.sqlite3'))
    a = p.parse_args()
    summary = json.loads(a.predictions.with_name('summary.json').read_text(encoding='utf-8'))
    if not summary.get('complete') or summary.get('split') != 'paper_sha1_first8_lt_0.2_from_official_train':
        raise ValueError('Requires the complete held-out paper evaluation')
    digest = hashlib.sha256(a.predictions.read_bytes()).hexdigest()
    store = PaperQAStore(a.db)
    existing = {item['result'].get('review_context', {}).get('origin_key') for item in store.list(100000)}
    imported = 0
    rows = [json.loads(line) for line in a.predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
    # Lowest answer scores appear first in the latest-first review list.
    for row in sorted(rows, key=lambda r:r['answer_token_f1_proxy'], reverse=True):
        origin = digest + ':' + row['paper_id'] + ':' + str(row['question_index'])
        if origin in existing: continue
        result = dict(row['result'])
        result['review_context'] = {'origin_key':origin, 'source_run_sha256':digest, 'dataset':'Qasper',
                                    'split':summary['split'], 'question_index':row['question_index'],
                                    'answer_token_f1_proxy':row['answer_token_f1_proxy'],
                                    'evidence_recall_proxy':row['evidence_recall_proxy'], 'training_eligible':False}
        store.save(row['paper_id'], row['question'], result)
        existing.add(origin); imported += 1
    print(json.dumps({'imported_for_review':imported, 'human_feedback_created':0, 'training_eligible':False}))


if __name__ == '__main__': main()
