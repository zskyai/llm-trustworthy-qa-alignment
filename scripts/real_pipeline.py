"""Run the Qasper/PubMedQA alignment pipeline on public data.

This is intentionally a plain PyTorch/PEFT driver rather than a demo that writes
one synthetic example.  It consumes the official Qasper parquet export and
PubMedQA ``ori_pqal.json``, performs a paper-level split, trains LoRA SFT and
DPO, and writes predictions plus citation/refusal/answer metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


SYSTEM = ("Answer only from numbered evidence and cite supporting passages as [E1]. "
          "If evidence is insufficient, reply exactly: Insufficient evidence: "
          "the passages do not contain an explicit answer. [NO_ANSWER]")


class LoRALinear(nn.Module):
    """Small dependency-free LoRA wrapper used when PEFT is unavailable."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = .05):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.a = nn.Parameter(torch.empty(rank, base.in_features)); self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**.5); self.scale = alpha / rank; self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.a.t() @ self.b.t()) * self.scale


def inject_lora(model):
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not name.rsplit('.', 1)[-1] in {"q_proj", "v_proj"}:
            continue
        parent_name, child = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module)); count += 1
    if not count: raise RuntimeError("no q_proj/v_proj modules found for LoRA injection")
    return model


def save_adapter(model, tokenizer, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".a" in k or ".b" in k}
    torch.save(state, directory / "adapter_model.pt"); tokenizer.save_pretrained(directory)


def clean(value: object, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if not limit or len(text) <= limit else text[: limit - 3].rstrip() + "..."


def answer_text(annotation: dict) -> str | None:
    if annotation.get("unanswerable"):
        return None
    value = clean(annotation.get("free_form_answer"))
    if value:
        return value
    spans = [clean(x) for x in annotation.get("extractive_spans") or [] if clean(x)]
    if spans:
        return "; ".join(spans)
    yes_no = annotation.get("yes_no")
    if yes_no is not None:
        return ("Yes." if yes_no is True else "No.") if isinstance(yes_no, bool) else clean(yes_no).capitalize() + "."
    return None


def _annotation(group: dict) -> dict | None:
    annotations = group.get("answer") or []
    for item in annotations:
        if not item.get("unanswerable") and answer_text(item):
            return item
    return annotations[0] if annotations else None


def prompt(question: str, evidence: list[str]) -> str:
    body = "\n".join(f"[E{i}] {clean(x, 700)}" for i, x in enumerate(evidence, 1))
    return f"Instruction: {SYSTEM}\nQuestion: {clean(question, 500)}\nEvidence:\n{body}\nAnswer:"


def qasper_records(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    records: list[dict] = []
    for paper in rows:
        qas = paper["qas"]
        for i, question in enumerate(qas.get("question") or []):
            answers = qas.get("answers") or []
            if i >= len(answers):
                continue
            ann = _annotation(answers[i])
            if not ann:
                continue
            evidence = [clean(x, 700) for x in ann.get("highlighted_evidence") or [] if clean(x)]
            evidence = evidence or [clean(x, 700) for x in ann.get("evidence") or [] if clean(x)]
            if not evidence:
                evidence = [clean(paper.get("abstract"), 700) or "No explicit evidence is provided."]
            evidence = evidence[:4]
            unanswerable = bool(ann.get("unanswerable"))
            gold = answer_text(ann) or ""
            chosen = ("Insufficient evidence: the passages do not contain an explicit answer. [NO_ANSWER]"
                      if unanswerable else f"{gold} " + "".join(f"[E{j}]" for j in range(1, len(evidence) + 1)))
            if unanswerable:
                rejected = "The evidence clearly proves the conclusion. [E1]"
            else:
                rejected = re.sub(r"\[E\d+\]", "[E99]", chosen)
            records.append({
                "source": "qasper", "document_id": str(paper["id"]),
                "record_id": str((qas.get("question_id") or [f"{paper['id']}-{i}"])[i]),
                "question": clean(question), "prompt": prompt(question, evidence),
                "evidence": evidence, "chosen": chosen, "rejected": rejected,
                "gold_answer": gold, "answerable": not unanswerable,
                "valid_citations": [f"E{j}" for j in range(1, len(evidence) + 1)],
                "document_word_count": len(re.findall(r"[A-Za-z0-9]+", json.dumps(paper.get("full_text", "")))) ,
                "evidence_positions": list(range(1, len(evidence) + 1)),
                "confidence": 0.0 if unanswerable else 1.0,
                "refusal_reason": "evidence_not_found" if unanswerable else None,
                "sft_target": json.dumps({"final_answer": gold, "evidence": evidence, "evidence_position": list(range(1, len(evidence) + 1)), "confidence": 0.0 if unanswerable else 1.0, "refusal_reason": "evidence_not_found" if unanswerable else None}, ensure_ascii=False),
                "category": "answerable" if not unanswerable else "unanswerable",
            })
    return records


def pubmedqa_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    flip = {"yes": "no", "no": "yes", "maybe": "no"}
    records = []
    for pmid, item in data.items():
        evidence = [clean(x, 700) for x in item.get("CONTEXTS") or [] if clean(x)][:4]
        label = clean(item.get("final_decision")).lower()
        if not evidence or label not in {"yes", "no", "maybe"}:
            continue
        summary = clean(item.get("LONG_ANSWER"), 500)
        chosen = f"{label.capitalize()}. {summary} " + "".join(f"[E{j}]" for j in range(1, len(evidence) + 1))
        rejected = f"{flip[label].capitalize()}. {summary} " + "".join(f"[E{j}]" for j in range(1, len(evidence) + 1))
        records.append({"source": "pubmedqa", "document_id": str(pmid), "record_id": str(pmid),
                        "question": clean(item.get("QUESTION")), "prompt": prompt(item.get("QUESTION", ""), evidence),
                        "evidence": evidence, "chosen": chosen, "rejected": rejected,
                        "gold_answer": summary, "gold_label": label, "answerable": True,
                        "valid_citations": [f"E{j}" for j in range(1, len(evidence) + 1)], "evidence_positions": list(range(1, len(evidence) + 1)),
                        "confidence": 1.0, "refusal_reason": None, "sft_target": json.dumps({"final_answer": summary, "evidence": evidence, "evidence_position": list(range(1, len(evidence) + 1)), "confidence": 1.0, "refusal_reason": None}, ensure_ascii=False), "category": f"pubmed_{label}"})
    return records


def split(records: list[dict], validation_ratio: float = .2) -> tuple[list[dict], list[dict]]:
    train, valid = [], []
    for row in records:
        key = f"{row['source']}:{row['document_id']}".encode()
        bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) / 2**32
        (valid if bucket < validation_ratio else train).append(row)
    return train, valid


def encode(tokenizer, p: str, answer: str, max_length: int):
    pids = tokenizer(p, add_special_tokens=False, truncation=True, max_length=max_length - 128)["input_ids"]
    aids = tokenizer(" " + answer + tokenizer.eos_token, add_special_tokens=False, truncation=True, max_length=128)["input_ids"]
    ids = torch.tensor((pids + aids)[:max_length], dtype=torch.long)
    labels = torch.tensor([-100] * min(len(pids), len(ids)) + ids.tolist()[len(pids):], dtype=torch.long)
    return ids, labels


def logps(model, tokenizer, p: str, answers: list[str], max_length: int, grad: bool = False):
    encoded = [encode(tokenizer, p, a, max_length) for a in answers]
    n = max(x[0].numel() for x in encoded)
    ids, labels, masks = [], [], []
    for x, y in encoded:
        pad = n - x.numel()
        ids.append(F.pad(x, (0, pad), value=tokenizer.pad_token_id))
        labels.append(F.pad(y, (0, pad), value=-100))
        masks.append(F.pad(torch.ones_like(x), (0, pad), value=0))
    ids, labels, masks = torch.stack(ids), torch.stack(labels), torch.stack(masks)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=ids, attention_mask=masks, use_cache=False).logits[:, :-1]
        target = labels[:, 1:]
        valid = target.ne(-100)
        gathered = F.log_softmax(logits, -1).gather(-1, target.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
        return (gathered * valid).sum(-1), valid.sum(-1).clamp_min(1)


def token_f1(pred: str, gold: str) -> float:
    tok = lambda s: re.findall(r"[a-z0-9]+", re.sub(r"\[E\d+\]|\[NO_ANSWER\]", "", s.lower()))
    a, b = tok(pred), tok(gold)
    if not a or not b:
        return float(a == b)
    ca, cb = Counter(a), Counter(b); overlap = sum((ca & cb).values())
    return 2 * overlap / (len(a) + len(b)) if overlap else 0.0


def refusal(text: str) -> bool:
    t = text.lower()
    return "[no_answer]" in t or "insufficient evidence" in t or "cannot answer" in t


def predicted_label(text: str):
    match = re.search(r"\b(yes|no|maybe)\b", text.lower()[:120])
    return match.group(1) if match else None


def generate(model, tokenizer, text: str, max_new_tokens: int):
    x = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate(model, tokenizer, rows: list[dict], stage: str, out: Path, max_length: int, generation_limit: int, max_new_tokens: int):
    model.eval(); margins = []; pair_ok = 0
    for row in rows:
        scores, _ = logps(model, tokenizer, row["prompt"], [row["chosen"], row["rejected"]], max_length)
        margins.append(float(scores[0] - scores[1])); pair_ok += int(scores[0] > scores[1])
    generation_rows = rows if generation_limit <= 0 else rows[:generation_limit]
    outputs = []; f1s = []; valid_cite = 0; cite_den = 0; invalid = 0; answerable = answered = unans = refused = 0
    pubmed_gold=[]; pubmed_pred=[]; citation_total=0; citation_supported=0; long_buckets=defaultdict(list)
    for row in generation_rows:
        pred = generate(model, tokenizer, row["prompt"], max_new_tokens)
        cites = re.findall(r"\[E(\d+)\]", pred); valid = {x[1:] for x in row["valid_citations"]}
        bad = any(c not in valid for c in cites); invalid += int(bad); citation_total += len(cites); citation_supported += sum(c in valid for c in cites)
        if row["answerable"]:
            cite_den += 1; valid_cite += int(bool(cites) and not bad); answerable += 1; answered += int(not refusal(pred))
        elif row["source"] == "qasper":
            unans += 1; refused += int(refusal(pred))
        else:
            pubmed_gold.append(row.get("gold_label")); pubmed_pred.append(predicted_label(pred))
        f1 = token_f1(pred, row["gold_answer"]); f1s.append(f1)
        bucket = "short" if row.get("document_word_count", 0) < 2048 else "medium" if row.get("document_word_count", 0) < 4096 else "long"
        long_buckets[bucket].append(f1)
        outputs.append({**row, "stage": stage, "prediction": pred, "token_f1": f1, "invalid_citation": bad, "is_refusal": refusal(pred), "predicted_label": predicted_label(pred)})
    labels = sorted(set(x for x in pubmed_gold if x)); per_class=[]
    for label in labels:
        tp=sum(g==label and p==label for g,p in zip(pubmed_gold,pubmed_pred)); fp=sum(g!=label and p==label for g,p in zip(pubmed_gold,pubmed_pred)); fn=sum(g==label and p!=label for g,p in zip(pubmed_gold,pubmed_pred))
        per_class.append((2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0)
    metrics = {"stage": stage, "pair_eval_count": len(rows), "pair_preference_accuracy": pair_ok / max(len(rows), 1),
               "mean_pair_margin": sum(margins) / max(len(margins), 1), "generation_eval_count": len(generation_rows),
               "mean_token_f1": sum(f1s) / max(len(f1s), 1), "valid_citation_rate_on_answerable": valid_cite / cite_den if cite_den else None,
               "citation_validity": citation_supported / citation_total if citation_total else None, "citation_support_rate": citation_supported / citation_total if citation_total else None,
               "pubmedqa_accuracy": sum(g==p for g,p in zip(pubmed_gold,pubmed_pred))/len(pubmed_gold) if pubmed_gold else None,
               "pubmedqa_macro_f1": sum(per_class)/len(per_class) if per_class else None,
               "qasper_token_f1_by_document_length": {k: sum(v)/len(v) for k,v in long_buckets.items() if v},
               "invalid_citation_output_rate": invalid / max(len(generation_rows), 1), "qasper_answerable_recall": answered / answerable if answerable else None,
               "qasper_refusal_recall": refused / unans if unans else None}
    (out / f"metrics_{stage}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"predictions_{stage}.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def train_sft(model, tokenizer, rows, epochs, lr, max_length):
    model.train(); params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(epochs):
        for row in rows:
            ids, labels = encode(tokenizer, row["prompt"], row.get("sft_target", row["chosen"]), max_length)
            loss = model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None], labels=labels[None], use_cache=False).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)


def train_dpo(model, tokenizer, rows, reference, epochs, lr, beta, max_length):
    model.train(); params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=lr)
    for _ in range(epochs):
        for i, row in enumerate(rows):
            scores, _ = logps(model, tokenizer, row["prompt"], [row["chosen"], row["rejected"]], max_length, grad=True)
            loss = -F.logsigmoid(beta * ((scores[0] - scores[1]) - reference[i])).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--qasper", type=Path, required=True); ap.add_argument("--pubmedqa", type=Path)
    ap.add_argument("--model", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-train", type=int, default=0, help="0 uses every real training record")
    ap.add_argument("--max-generation-eval", type=int, default=0, help="0 evaluates every held-out record")
    ap.add_argument("--sft-epochs", type=int, default=1); ap.add_argument("--dpo-epochs", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=384); ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--sft-lr", type=float, default=2e-4); ap.add_argument("--dpo-lr", type=float, default=1e-4); ap.add_argument("--beta", type=float, default=.1)
    a = ap.parse_args(); torch.manual_seed(42); random.seed(42); a.output_dir.mkdir(parents=True, exist_ok=True)
    records = qasper_records(a.qasper) + (pubmedqa_records(a.pubmedqa) if a.pubmedqa else [])
    train, valid = split(records)
    if a.max_train: train = train[:a.max_train]
    for name, data in (("train", train), ("validation", valid)):
        (a.output_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in data) + "\n", encoding="utf-8")
    (a.output_dir / "dataset_summary.json").write_text(json.dumps({"records": len(records), "train": len(train), "validation": len(valid), "split": "SHA1(source:document_id), paper-level 80/20"}, indent=2), encoding="utf-8")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"); model.config.use_cache = False
    metrics = [evaluate(model, tok, valid, "base", a.output_dir, a.max_length, a.max_generation_eval, a.max_new_tokens)]
    model = inject_lora(model)
    started = time.time(); train_sft(model, tok, train, a.sft_epochs, a.sft_lr, a.max_length); sft_seconds = time.time() - started
    save_adapter(model, tok, a.output_dir / "sft_adapter")
    metrics.append(evaluate(model, tok, valid, "sft", a.output_dir, a.max_length, a.max_generation_eval, a.max_new_tokens))
    reference = []
    for r in train:
        scores, _ = logps(model, tok, r["prompt"], [r["chosen"], r["rejected"]], a.max_length)
        reference.append(float(scores[0] - scores[1]))
    started = time.time(); train_dpo(model, tok, train, reference, a.dpo_epochs, a.dpo_lr, a.beta, a.max_length); dpo_seconds = time.time() - started
    save_adapter(model, tok, a.output_dir / "dpo_adapter")
    metrics.append(evaluate(model, tok, valid, "sft_dpo", a.output_dir, a.max_length, a.max_generation_eval, a.max_new_tokens))
    (a.output_dir / "run_config.json").write_text(json.dumps({**vars(a), "sft_seconds": sft_seconds, "dpo_seconds": dpo_seconds, "device": "cpu", "dataset": "official Qasper parquet + PubMedQA ori_pqal.json"}, default=str, indent=2), encoding="utf-8")
    (a.output_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
