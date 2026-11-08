"""Frozen PQA-L protocol and complete, source-positioned supervised records."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from .pubmedqa_protocol import split_records

LABELS = ('yes', 'no', 'maybe')

def read_jsonl(path):
    """Read LF-delimited records without splitting Unicode inside JSON strings."""
    with Path(path).open('r', encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_clean_records(source, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf8'))
    data = json.loads(Path(source).read_text(encoding='utf8'))
    if manifest.get('source_sha256') != sha256(source):
        raise ValueError('Frozen manifest source hash mismatch')
    keys = ('train_ids', 'validation_ids', 'test_ids')
    if [len(manifest[k]) for k in keys] != [450, 50, 500]:
        raise ValueError('PQA-L requires exactly 450/50/500 records')
    if any(len(set(map(str, manifest[k]))) != len(manifest[k]) for k in keys):
        raise ValueError('Duplicate IDs in frozen manifest')
    # Only train/dev records are materialized; split_records validates the full
    # source/manifest membership without returning test labels to the caller.
    train, dev = split_records([{'document_id': str(pid)} for pid in data], manifest)
    def convert(row):
        pid = row['document_id']; item = data[pid]
        contexts = item['CONTEXTS']; label = item['final_decision'].lower()
        if not contexts or label not in LABELS or any(not c.strip() for c in contexts):
            raise ValueError('Invalid official source record')
        evidence = [{'id': 'E'+str(i+1), 'text': text, 'position': {'document_id': pid, 'context_index': i, 'char_start': 0, 'char_end': len(text)}} for i, text in enumerate(contexts)]
        target = {'label': label, 'final_answer': item['LONG_ANSWER'], 'evidence': evidence, 'confidence': None, 'refusal_reason': None}
        # PQA-L does not annotate supporting context spans or confidence. All
        # contexts remain available; these fields are explicitly unannotated.
        target['evidence'] = []
        question = item['QUESTION']
        instruction = ('Answer the biomedical question from the numbered contexts. Return JSON with label (yes/no/maybe), final_answer, evidence (id, quote, context_index), confidence (0 to 1), and refusal_reason (null or reason). Cite only text you use. Maybe is a task label, not refusal. If the evidence cannot justify an answer, set label to null and provide a refusal_reason.')
        prompt = instruction+'\nQuestion: '+question+'\nContexts:\n'+'\n'.join('['+e['id']+'] '+e['text'] for e in evidence)+'\nAnswer JSON:'
        return {'source': 'pubmedqa', 'document_id': pid, 'record_id': pid, 'split': row['split'], 'question': question, 'evidence': evidence, 'prompt': prompt, 'gold_label': label, 'gold_answer': item['LONG_ANSWER'], 'sft_target': json.dumps(target, ensure_ascii=False, separators=(',', ':')), 'evidence_annotation': 'unavailable; empty target does not mean no supporting evidence', 'confidence_annotation': 'unavailable; use model label likelihood for classification calibration'}
    return [convert(r) for r in train], [convert(r) for r in dev], manifest

def adapter_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if name.rsplit('.', 1)[-1] in {'a', 'b'} and parameter.requires_grad}

def load_adapter_strict(model, path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = set(adapter_state(model))
    if set(state) != expected:
        raise ValueError('Adapter keys differ from the exact LoRA a/b parameter set')
    model.load_state_dict(state, strict=False)

def validate_adapter_provenance(directory, base, source, manifest):
    directory = Path(directory)
    report = json.loads((directory/'provenance.json').read_text(encoding='utf8'))
    if not report.get('complete') or report.get('test_used') is not False:
        raise ValueError('Adapter training is incomplete or has no clean test exclusion record')
    if report.get('manifest_sha256') != sha256(manifest) or report.get('source_sha256') != sha256(source):
        raise ValueError('Adapter trained against a different dataset manifest')
    if report.get('base_fingerprint') != checkpoint_fingerprint(base):
        raise ValueError('Adapter trained from a different base checkpoint')
    if report.get('adapter_sha256') != sha256(directory/'adapter_model.pt'):
        raise ValueError('Adapter hash mismatch')
    frozen = json.loads(Path(manifest).read_text(encoding='utf8'))
    train_ids = set(map(str, report['train_ids']))
    if train_ids != set(map(str, frozen['train_ids'])) or train_ids & set(map(str, frozen['test_ids']+frozen['validation_ids'])):
        raise ValueError('Adapter training IDs violate frozen split')
    return report

def checkpoint_fingerprint(directory):
    directory = Path(directory)
    files = sorted(set(directory.glob('*.safetensors')) | set(directory.glob('pytorch_model*.bin')) | {directory/'config.json'})
    if len(files) < 2: raise ValueError('Checkpoint must contain config and real weights')
    return {p.name: sha256(p) for p in files}
