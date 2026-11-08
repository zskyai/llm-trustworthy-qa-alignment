"""Compare saved Base, full-data SFT, and filtered-pair SFT+DPO on one split."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_pipeline import evaluate, inject_lora


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("validation file is empty")
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("test records are not allowed in this evaluation")
    return rows


def load_stage(base_model: Path, adapter_path: Path | None, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = True
    if adapter_path is not None:
        model = inject_lora(model)
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        expected = set(model.state_dict())
        if not state or not set(state).issubset(expected):
            raise ValueError(f"adapter keys do not match the injected model: {adapter_path}")
        model.load_state_dict(state, strict=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--dpo-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all validation records")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    rows = load_rows(args.validation)
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    stages = [
        ("base", None),
        ("full_sft", args.sft_adapter),
        ("filtered_pair_sft_dpo", args.dpo_adapter),
    ]
    results = []
    for stage, adapter in stages:
        model = load_stage(args.base_model, adapter, tokenizer)
        metrics = evaluate(
            model,
            tokenizer,
            rows,
            stage,
            args.output_dir,
            args.max_length,
            0,
            args.max_new_tokens,
            include_pair_metrics=False,
            generation_batch_size=args.batch_size,
        )
        results.append(metrics)
        del model
        gc.collect()

    report = {
        "dataset": "Qasper/PubMedQA paper-level validation emitted by real_pipeline.py",
        "evaluation_records": len(rows),
        "validation_context": "Qasper gold evidence spans; oracle-evidence generation, not retrieval end-to-end",
        "run_config": {
            "base_model": str(args.base_model.resolve()),
            "validation_file": str(args.validation.resolve()),
            "sft_adapter": str(args.sft_adapter.resolve()),
            "dpo_adapter": str(args.dpo_adapter.resolve()),
            "max_input_length_for_generation": 512,
            "max_length_for_scoring": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "decoding": "greedy (do_sample=false)",
            "dtype": "float32",
            "device": "cpu",
            "generation_latency_recorded": False,
        },
        "stages": {
            "base": "base checkpoint",
            "full_sft": "2,862-record LoRA SFT adapter",
            "filtered_pair_sft_dpo": "712 filtered-pair SFT+DPO adapter; independently initialized training run",
        },
        "test_used": False,
        "metrics": results,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
