"""Keep only frozen train records before model-negative filtering/DPO."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.manifest.read_text(encoding='utf8'))
    train_ids = {str(x) for x in protocol['train_ids']}
    dev_ids = {str(x) for x in protocol['validation_ids']}
    test_ids = {str(x) for x in protocol['test_ids']}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise ValueError('overlapping protocol partitions')
    rows = [json.loads(x) for x in a.input.read_text(encoding='utf8').splitlines() if x.strip()]
    kept, dropped = [], {'test': 0, 'dev': 0, 'unknown': 0}
    for row in rows:
        if row.get('source') != 'pubmedqa':
            if row.get('split') == 'test':
                raise ValueError('non-PubMed test record in preference input')
            kept.append(row); continue
        rid = str(row['document_id'])
        if rid in test_ids: dropped['test'] += 1
        elif rid in dev_ids: dropped['dev'] += 1
        elif rid in train_ids: kept.append(row)
        else: dropped['unknown'] += 1
    if not kept or any(str(x.get('document_id')) in test_ids for x in kept if x.get('source') == 'pubmedqa'):
        raise AssertionError('filtered preference input still contains test')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in kept) + '\n', encoding='utf8')
    report = {'complete': True, 'input_records': len(rows), 'kept_records': len(kept), 'dropped': dropped,
              'train_ids': len(train_ids), 'validation_ids': len(dev_ids), 'test_ids': len(test_ids),
              'test_used': False, 'manifest': str(a.manifest)}
    report_path = a.output.with_suffix('.manifest.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
