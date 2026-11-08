"""Export human feedback as audit candidates, not automatically accepted DPO."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', type=Path, default=Path('results/paper_qa_history.sqlite3'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(a.db.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        records = db.execute('SELECT q.*, f.helpful, f.note, f.created_at AS feedback_at FROM questions q JOIN feedback f ON q.question_id = f.question_id WHERE f.feedback_id = (SELECT MAX(f2.feedback_id) FROM feedback f2 WHERE f2.question_id=q.question_id) ORDER BY q.created_at').fetchall()
    with a.output.open('x', encoding='utf-8') as out:
        for record in records:
            row = dict(record); row['result'] = json.loads(row.pop('result_json'))
            row.update({'purpose':'human_feedback_audit_candidate', 'training_eligible':False,
                        'required_checks':['split_membership', 'source_span_match', 'answer_consistency', 'nli_support', 'human_correction']})
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'exported':len(records), 'training_eligible':False, 'output':str(a.output)}))


if __name__ == '__main__': main()
