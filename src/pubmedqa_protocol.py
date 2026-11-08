"""Reconstruct the released PQA-L 500-test / ten-fold CV algorithm.

Reference: pubmedqa/pubmedqa, preprocess/split_dataset.py. Original JSON key
order is significant; the manifest records its hash and is a reconstruction.
"""
from __future__ import annotations
import math
import random


def _partition(data, folds, rng):
    labels = {label: [] for label in ('yes', 'no', 'maybe')}
    for pmid, row in data.items():
        labels[row['final_decision']].append(pmid)
    pieces = {}
    for label, ids in labels.items():
        rng.shuffle(ids); size = math.ceil(len(ids)/folds)
        pieces[label] = [ids[i*size:(i+1)*size] if i < folds-1 else ids[i*size:] for i in range(folds)]
    result = [{pid:data[pid] for label in labels for pid in pieces[label][i]} for i in range(folds)]
    if len(result[-1]) != len(result[0]):
        for i in range(folds-1):
            chosen = rng.choice(list(result[i]))
            result[-1][chosen] = result[i].pop(chosen)
    return result


def reconstruct_pqal(data, *, fold=0):
    if len(data) != 1000 or not 0 <= fold < 10:
        raise ValueError('PQA-L protocol requires the complete 1000 labeled records and fold 0..9')
    rng = random.Random(0)
    cv, test = _partition(data, 2, rng)
    cv_folds = _partition(cv, 10, rng)
    train = {pid:row for i, group in enumerate(cv_folds) if i != fold for pid,row in group.items()}
    dev = cv_folds[fold]
    groups = [set(train),set(dev),set(test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('PubMedQA split overlap')
    return {'train_ids': list(train), 'validation_ids': list(dev), 'test_ids': list(test),
            'fold': fold, 'seed': 0, 'protocol': 'released PQA-L split algorithm reconstructed from local original JSON ordering'}


def split_records(records, manifest):
    keys = ('train_ids','validation_ids','test_ids')
    groups = [set(map(str,manifest[k])) for k in keys]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('PubMedQA train/dev/test overlap')
    by_id = {str(r['document_id']):r for r in records}
    if len(by_id) != len(records) or set(by_id) != set.union(*groups):
        raise ValueError('PubMedQA source records do not match split manifest')
    return ([dict(by_id[str(pid)], split='train') for pid in manifest['train_ids']],
            [dict(by_id[str(pid)], split='validation') for pid in manifest['validation_ids']])
