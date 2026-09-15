"""Run a leakage-safe DPO pilot on NLI-filtered, model-generated negatives."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_pipeline import encode, inject_lora, save_adapter


def split_by_document(rows, validation_ratio=.25):
    train, valid = [], []
    for row in rows:
        key = f"{row.get('source')}:{row.get('document_id')}".encode()
        bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) / 2**32
        (valid if bucket < validation_ratio else train).append(row)
    if not valid and len({x.get('document_id') for x in rows}) > 1:
        held_out = sorted({x.get('document_id') for x in rows})[-1]
        train = [x for x in rows if x.get('document_id') != held_out]
        valid = [x for x in rows if x.get('document_id') == held_out]
    return train, valid


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def batch_logps(model, tokenizer, rows, answers, max_length, grad=False):
    """Return token-mean completion log-probabilities for one completion per row."""
    encoded = [encode(tokenizer, row["prompt"], answer, max_length) for row, answer in zip(rows, answers)]
    width = max(ids.numel() for ids, _ in encoded)
    ids, labels, masks = [], [], []
    for input_ids, target in encoded:
        pad = width - input_ids.numel()
        ids.append(F.pad(input_ids, (0, pad), value=tokenizer.pad_token_id))
        labels.append(F.pad(target, (0, pad), value=-100))
        masks.append(F.pad(torch.ones_like(input_ids), (0, pad), value=0))
    ids, labels, masks = torch.stack(ids), torch.stack(labels), torch.stack(masks)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(input_ids=ids, attention_mask=masks, use_cache=False).logits[:, :-1]
        target = labels[:, 1:]
        valid = target.ne(-100)
        token_logps = F.log_softmax(logits, -1).gather(-1, target.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
        return (token_logps * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def pair_margins(model, tokenizer, rows, max_length, batch_size, grad=False):
    chosen = batch_logps(model, tokenizer, rows, [x["chosen"] for x in rows], max_length, grad=grad)
    rejected = batch_logps(model, tokenizer, rows, [x["rejected"] for x in rows], max_length, grad=grad)
    return chosen - rejected


def margins(model, tokenizer, rows, max_length, batch_size):
    values = []
    model.eval()
    for batch in batched(rows, batch_size):
        values.extend(float(x) for x in pair_margins(model, tokenizer, batch, max_length, batch_size))
    return {
        "pairs": len(values),
        "preference_accuracy": sum(x > 0 for x in values) / max(len(values), 1),
        "mean_margin": sum(values) / max(len(values), 1),
    }


def train_sft(model, tokenizer, rows, max_length, lr, batch_size):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    losses = []
    model.train()
    for batch in batched(rows, batch_size):
        encoded = [encode(tokenizer, row["prompt"], row["chosen"], max_length) for row in batch]
        width = max(ids.numel() for ids, _ in encoded)
        ids = torch.stack([F.pad(x, (0, width - x.numel()), value=tokenizer.pad_token_id) for x, _ in encoded])
        labels = torch.stack([F.pad(x, (0, width - x.numel()), value=-100) for _, x in encoded])
        mask = ids.ne(tokenizer.pad_token_id)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return losses


def train_dpo(model, tokenizer, rows, references, max_length, lr, beta, batch_size):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    losses = []
    model.train()
    for start, batch in enumerate(batched(rows, batch_size)):
        references_batch = torch.tensor(references[start * batch_size : start * batch_size + len(batch)])
        score_margin = pair_margins(model, tokenizer, batch, max_length, batch_size, grad=True)
        loss = -F.logsigmoid(beta * (score_margin - references_batch)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, nargs='+', required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--dpo-lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    torch.manual_seed(42); random.seed(42)
    rows = [json.loads(x) for path in args.pairs for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows or not all(x.get("rejected_source") == "base_model_generation" for x in rows):
        raise ValueError("All preference pairs must contain real base_model_generation negatives")
    if not all((x.get("filter_checks") or {}).get("chosen_nli_entails", not x.get("answerable", True)) and (x.get("filter_checks") or {}).get("rejected_nli_not_supported", not x.get("answerable", True)) for x in rows):
        raise ValueError("All pairs must pass the configured NLI filter")
    train, valid = split_by_document(rows)
    if not train or not valid:
        raise ValueError("Document-level train/held-out split is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
    model.config.use_cache = False
    model = inject_lora(model)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    stages = {"base_lora": {"train": margins(model, tokenizer, train, args.max_length, args.batch_size), "held_out": margins(model, tokenizer, valid, args.max_length, args.batch_size)}}

    started = time.perf_counter(); sft_losses = train_sft(model, tokenizer, train, args.max_length, args.sft_lr, args.batch_size); sft_seconds = time.perf_counter() - started
    stages["sft"] = {"train": margins(model, tokenizer, train, args.max_length, args.batch_size), "held_out": margins(model, tokenizer, valid, args.max_length, args.batch_size)}
    references = []
    for batch in batched(train, args.batch_size):
        references.extend(float(x) for x in pair_margins(model, tokenizer, batch, args.max_length, args.batch_size))
    started = time.perf_counter(); dpo_losses = train_dpo(model, tokenizer, train, references, args.max_length, args.dpo_lr, args.beta, args.batch_size); dpo_seconds = time.perf_counter() - started
    stages["sft_dpo"] = {"train": margins(model, tokenizer, train, args.max_length, args.batch_size), "held_out": margins(model, tokenizer, valid, args.max_length, args.batch_size)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_adapter(model, tokenizer, args.output_dir / "adapter")
    payload = {
        "status": "real_model_full_filtered_training",
        "model_path": str(args.model), "pair_files": [str(x) for x in args.pairs],
        "preference_source": "base_model_generation_then_real_NLI_filter",
        "input_pairs": len(rows), "train_pairs": len(train), "held_out_pairs": len(valid),
        "train_documents": len({x["document_id"] for x in train}), "held_out_documents": len({x["document_id"] for x in valid}),
        "split": "SHA1(source:document_id), with deterministic document fallback",
        "test_used": False, "seed": 42, "max_length": args.max_length, "batch_size": args.batch_size, "beta": args.beta,
        "dpo_learning_rate": args.dpo_lr, "completion_logprob_reduction": "mean_per_non_prompt_token",
        "sft_seconds": sft_seconds, "dpo_seconds": dpo_seconds,
        "sft_loss": {"first": sft_losses[0], "last": sft_losses[-1], "mean": sum(sft_losses)/len(sft_losses)},
        "dpo_loss": {"first": dpo_losses[0], "last": dpo_losses[-1], "mean": sum(dpo_losses)/len(dpo_losses)},
        "stages": stages,
        "limitations": ["Only NLI-filtered generated pairs are used", "Preference metrics do not replace answer/citation evaluation", "The official Qasper test split is not locally available and remains unused"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "train.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in train)+"\n", encoding="utf-8")
    (args.output_dir / "held_out.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in valid)+"\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
