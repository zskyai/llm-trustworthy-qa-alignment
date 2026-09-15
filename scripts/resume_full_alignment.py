"""Resume the full Qasper/PubMedQA SFT+DPO run after base evaluation.

The train/validation JSONL files are the immutable paper-level split emitted by
real_pipeline.py. This script never reads a test file and evaluates every
validation record.
"""
from __future__ import annotations

import argparse, hashlib, json, random, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_pipeline import (evaluate, inject_lora, logps, save_adapter,
                           train_dpo, train_sft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--validation', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--sft-epochs', type=int, default=1)
    ap.add_argument('--dpo-epochs', type=int, default=1)
    ap.add_argument('--max-length', type=int, default=384)
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--sft-lr', type=float, default=2e-4)
    ap.add_argument('--dpo-lr', type=float, default=1e-4)
    ap.add_argument('--beta', type=float, default=.1)
    ap.add_argument('--stage', choices=['sft','dpo','all'], default='all')
    a = ap.parse_args()
    torch.manual_seed(42); random.seed(42); a.output_dir.mkdir(parents=True, exist_ok=True)
    train = [json.loads(x) for x in a.train.read_text(encoding='utf-8').splitlines() if x.strip()]
    valid = [json.loads(x) for x in a.validation.read_text(encoding='utf-8').splitlines() if x.strip()]
    if not train or not valid: raise ValueError('train/validation files must be non-empty')
    if any(x.get('split') == 'test' for x in train + valid): raise ValueError('test records are forbidden')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, local_files_only=True, attn_implementation='eager')
    model.config.use_cache = False; model = inject_lora(model)
    sft_seconds = 0.0
    adapter_path = a.output_dir / 'sft_adapter' / 'adapter_model.pt'
    if a.stage == 'dpo' and adapter_path.exists():
        adapter_state = torch.load(adapter_path, map_location='cpu')
        model.load_state_dict(adapter_state, strict=False)
    else:
        started = time.time(); train_sft(model, tok, train, a.sft_epochs, a.sft_lr, a.max_length); sft_seconds = time.time() - started
        save_adapter(model, tok, a.output_dir / 'sft_adapter')
    if a.stage == 'sft':
        print(json.dumps({'status':'sft_complete','train_records':len(train),'validation_records':len(valid),'sft_seconds':sft_seconds,'test_used':False}, indent=2))
        return
    sft_metrics = evaluate(model, tok, valid, 'sft', a.output_dir, a.max_length, 0, a.max_new_tokens)
    reference = []
    for row in train:
        scores, counts = logps(model, tok, row['prompt'], [row['chosen'], row['rejected']], a.max_length)
        reference.append(float((scores / counts)[0] - (scores / counts)[1]))
    started = time.time(); train_dpo(model, tok, train, reference, a.dpo_epochs, a.dpo_lr, a.beta, a.max_length); dpo_seconds = time.time() - started
    save_adapter(model, tok, a.output_dir / 'dpo_adapter')
    dpo_metrics = evaluate(model, tok, valid, 'sft_dpo', a.output_dir, a.max_length, 0, a.max_new_tokens)
    payload = {'status':'real_model_full_training_resumed', 'train_records':len(train), 'validation_records':len(valid),
               'train_documents':len({x.get('document_id') for x in train}), 'validation_documents':len({x.get('document_id') for x in valid}),
               'sft_seconds':sft_seconds, 'dpo_seconds':dpo_seconds, 'sft_epochs':a.sft_epochs, 'dpo_epochs':a.dpo_epochs,
               'max_length':a.max_length, 'max_new_tokens':a.max_new_tokens, 'test_used':False,
               'dataset':'official Qasper train parquet + PubMedQA ori_pqal.json; deterministic paper-level split',
               'metrics':[sft_metrics,dpo_metrics]}
    (a.output_dir/'resume_run_config.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    (a.output_dir/'metrics_summary_resumed.json').write_text(json.dumps([sft_metrics,dpo_metrics], ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
