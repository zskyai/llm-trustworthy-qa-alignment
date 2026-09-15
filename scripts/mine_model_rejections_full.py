"""Generate model negatives for every available training preference record.

The script is resumable at batch boundaries and writes a manifest containing
the exact denominator, model, seed and source split. It never reads validation
or test records.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-input-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    rows = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError("input is empty")
    if any(x.get("source") not in {"qasper", "pubmedqa"} for x in rows):
        raise ValueError("input must contain official Qasper/PubMedQA records")
    if any(x.get("split") == "test" for x in rows):
        raise ValueError("test records are forbidden in training negative generation")
    start = 0
    if args.resume and args.output.exists():
        start = sum(1 for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
        if start > len(rows):
            raise ValueError("existing output has more rows than input")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start else "w"
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8") as handle:
        for offset in range(start, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            inputs = tokenizer([x["prompt"] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=.8, top_p=.9, pad_token_id=tokenizer.pad_token_id)
            prompt_width = inputs["input_ids"].shape[1]
            for row, sequence in zip(batch, generated):
                rejected = tokenizer.decode(sequence[int(prompt_width):], skip_special_tokens=True).strip()
                handle.write(json.dumps({**row, "rejected": rejected, "rejected_source": "base_model_generation", "rejected_model_path": str(args.model), "rejected_generation_seed": args.seed, "rejected_generation_offset": offset, "rejected_citations": re.findall(r"\[E(\d+)\]", rejected)}, ensure_ascii=False) + "\n")
            handle.flush()
            done = min(offset + len(batch), len(rows))
            if done % max(args.batch_size * 10, 40) == 0 or done == len(rows):
                print(json.dumps({"generated": done, "total": len(rows), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    manifest = {"status": "complete", "input": str(args.input), "output": str(args.output), "model_path": str(args.model), "seed": args.seed, "records": len(rows), "generated_records": len(rows), "split": "training records only; paper/document-level validation excluded", "test_used": False, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "elapsed_seconds": time.perf_counter() - started}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
