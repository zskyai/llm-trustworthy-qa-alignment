"""Model-derived coverage-risk calibration on leakage-safe Qasper contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_qasper_protocol import choose, paragraphs, select


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def answer_text(annotation):
    if annotation.get("unanswerable"):
        return ""
    if clean(annotation.get("free_form_answer")):
        return clean(annotation["free_form_answer"])
    spans = [clean(x) for x in annotation.get("extractive_spans") or [] if clean(x)]
    if spans:
        return "; ".join(spans)
    value = annotation.get("yes_no")
    if value is not None:
        return "Yes." if value is True else "No." if value is False else clean(value)
    return ""


def token_f1(pred, gold):
    words = lambda x: re.findall(r"[a-z0-9]+", re.sub(r"\[E\d+\]|\[NO_ANSWER\]", "", x.lower()))
    a, b = Counter(words(pred)), Counter(words(gold)); overlap = sum((a & b).values())
    return 2 * overlap / max(sum(a.values()) + sum(b.values()), 1)


def is_refusal(text):
    value = text.lower()
    return "[no_answer]" in value or "insufficient evidence" in value or "cannot answer" in value


def load_validation(path):
    import pyarrow.parquet as pq
    rows = []
    for paper in pq.read_table(path).to_pylist():
        bucket = int(hashlib.sha1(str(paper["id"]).encode()).hexdigest()[:8], 16) / 2**32
        if bucket >= .2:
            continue
        ps = paragraphs(paper); qas = paper.get("qas") or {}
        for i, question in enumerate(qas.get("question") or []):
            groups = qas.get("answers") or []
            if i >= len(groups): continue
            ann = choose(groups[i]); gold = answer_text(ann)
            rows.append({"paper_id": str(paper["id"]), "question_id": str((qas.get("question_id") or [i])[i]), "question": question, "paragraphs": ps, "answerable": not bool(ann.get("unanswerable")), "gold_answer": gold})
    return rows


def prompt(row, top_k):
    chosen = select(row, "bm25", top_k)
    evidence = "\n".join(f"[E{i}] {clean(x['text'])[:900]}" for i, x in enumerate(chosen, 1))
    text = ("Answer only from the numbered evidence. Cite evidence as [E1]. If it is insufficient, reply "
            "exactly: Insufficient evidence: the passages do not contain an explicit answer. [NO_ANSWER]\n"
            f"Question: {clean(row['question'])}\nEvidence:\n{evidence}\nAnswer:")
    return text, chosen


def generate_with_confidence(model, tokenizer, text, max_new_tokens):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    generated = output.sequences[0, inputs["input_ids"].shape[1]:]
    logps = []
    for token, logits in zip(generated, output.scores):
        logps.append(float(torch.log_softmax(logits[0], -1)[token]))
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return decoded, sum(logps) / max(len(logps), 1), len(logps)


def sigmoid(x):
    return 1 / (1 + math.exp(-max(min(x, 30), -30)))


def fit_platt(xs, ys):
    a = torch.tensor(1.0, requires_grad=True); b = torch.tensor(0.0, requires_grad=True)
    x = torch.tensor(xs); y = torch.tensor(ys)
    opt = torch.optim.LBFGS([a, b], max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(a * x + b, y); loss.backward(); return loss
    opt.step(closure)
    return float(a.detach()), float(b.detach())


def metrics(rows, a, b):
    items = [{**x, "confidence": sigmoid(a*x["mean_token_logprob"]+b)} for x in rows]
    curve = []
    for threshold in [i/20 for i in range(21)]:
        kept = [x for x in items if x["confidence"] >= threshold]
        curve.append({"threshold": threshold, "coverage": len(kept)/max(len(items),1), "risk": 1-sum(x["correct"] for x in kept)/len(kept) if kept else None, "over_refusal_rate": sum(x["answerable"] for x in items if x["confidence"] < threshold)/max(sum(x["answerable"] for x in items),1)})
    ece = 0.0; bins = []
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        group = [x for x in items if lo <= x["confidence"] < hi or i == 9 and x["confidence"] == 1]
        if group:
            conf = sum(x["confidence"] for x in group)/len(group); acc = sum(x["correct"] for x in group)/len(group)
            ece += len(group)/max(len(items),1)*abs(conf-acc); bins.append({"lo":lo,"hi":hi,"count":len(group),"confidence":conf,"accuracy":acc})
    return {"count":len(items),"accuracy":sum(x["correct"] for x in items)/max(len(items),1),"ece":ece,"bins":bins,"coverage_risk":curve}, items


def main():
    p=argparse.ArgumentParser(); p.add_argument('--qasper',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--limit',type=int,default=40); p.add_argument('--top-k',type=int,default=6); p.add_argument('--max-new-tokens',type=int,default=48); args=p.parse_args()
    torch.manual_seed(42); rows=load_validation(args.qasper); rows=rows[:args.limit] if args.limit else rows
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True); tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    generated=[]; started=time.perf_counter()
    for row in rows:
        text,evidence=prompt(row,args.top_k); pred,logp,tokens=generate_with_confidence(model,tokenizer,text,args.max_new_tokens)
        correct=float((not row['answerable'] and is_refusal(pred)) or (row['answerable'] and token_f1(pred,row['gold_answer'])>=.5))
        generated.append({"paper_id":row['paper_id'],"question_id":row['question_id'],"question":row['question'],"answerable":row['answerable'],"gold_answer":row['gold_answer'],"prediction":pred,"mean_token_logprob":logp,"generated_tokens":tokens,"correct":correct,"retrieved_paragraph_ids":[x['id'] for x in evidence]})
    calibration=[x for x in generated if int(hashlib.sha1(x['paper_id'].encode()).hexdigest()[8:16],16)%2==0]
    evaluation=[x for x in generated if x not in calibration]
    if not calibration or not evaluation or len({x['correct'] for x in calibration})<2: raise RuntimeError('Calibration/evaluation split needs both correctness classes; increase --limit')
    pa,pb=fit_platt([x['mean_token_logprob'] for x in calibration],[x['correct'] for x in calibration])
    cal_metrics,_=metrics(calibration,pa,pb); eval_metrics,scored=metrics(evaluation,pa,pb)
    payload={'status':'real_model_pilot' if args.limit else 'real_model_full_validation','model_path':str(args.model),'split':'official Qasper train -> paper-level validation -> disjoint calibration/evaluation papers','test_used':False,'seed':42,'context':'lexical retrieval from paper full text; gold evidence not used as input','top_k':args.top_k,'generated':len(generated),'calibration_papers':len({x['paper_id'] for x in calibration}),'evaluation_papers':len({x['paper_id'] for x in evaluation}),'platt':{'a':pa,'b':pb},'calibration':cal_metrics,'evaluation':eval_metrics,'latency_seconds':time.perf_counter()-started,'limitations':['Correctness uses token-F1>=0.5 or correct refusal; manual review remains required','Pilot sample is not a formal full-validation estimate'] if args.limit else []}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps({'metrics':payload,'predictions':scored},indent=2,ensure_ascii=False),encoding='utf8'); print(json.dumps(payload,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
