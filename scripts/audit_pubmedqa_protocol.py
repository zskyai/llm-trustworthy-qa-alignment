"""Audit prior artifact overlap with reconstructed released PQA-L test IDs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pubmedqa_protocol import reconstruct_pqal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('data/pubmedqa.json'))
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    data = json.loads(a.source.read_text(encoding='utf8'))
    manifest = reconstruct_pqal(data)
    manifest['source_sha256'] = hashlib.sha256(a.source.read_bytes()).hexdigest()
    paths = sorted(Path('results').glob('*/train.jsonl'))
    reports = []
    for path in paths:
        rows = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
        ids = {str(r['document_id']) for r in rows if r.get('source') == 'pubmedqa'}
        overlaps = sorted(ids & set(manifest['test_ids']))
        reports.append({'artifact':str(path),'pubmedqa_unique_training_ids':len(ids),
                        'reconstructed_official_test_overlap_count':len(overlaps),'overlap_ids':overlaps,
                        'eligible_for_reconstructed_official_test_evaluation':not bool(overlaps)})
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    report = {'source_sha256':manifest['source_sha256'],'split_counts':{key:len(manifest[key]) for key in ('train_ids','validation_ids','test_ids')},
              'membership_provenance':manifest['protocol'],'original_official_file_hash_verified':False,
              'artifacts':reports,'test_questions_or_answers_used_for_model_selection':False}
    (a.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({**report,'artifacts':[{k:v for k,v in r.items() if k!='overlap_ids'} for r in reports]}))


if __name__=='__main__':main()
