"""Materialize all 450/50 train/dev records with frozen source provenance."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clean_pubmedqa import load_clean_records, sha256, checkpoint_fingerprint

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--nli-model',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--min-evidence-entailment',type=float,default=.05)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    train,dev,manifest=load_clean_records(a.source,a.manifest)
    a.output.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.nli_model,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(a.nli_model,local_files_only=True).eval()
    entail=next(int(i) for i,label in model.config.id2label.items() if str(label).lower()=='entailment')
    counts={}
    for split,rows in [('train',train),('validation',dev)]:
        supported=0
        with (a.output/(split+'.jsonl')).open('x',encoding='utf8') as stream:
            for i,row in enumerate(rows):
                statements=[x.strip() for x in re.split(r'(?<=[.!?])\s+',row['gold_answer']) if x.strip()]
                pairs=[(e,s) for s in statements for e in row['evidence']]
                encoded=tok([e['text'] for e,s in pairs],[s for e,s in pairs],padding=True,truncation=True,max_length=512,return_tensors='pt')
                with torch.inference_mode(): probabilities=model(**encoded).logits.softmax(-1)[:,entail].tolist()
                n=len(row['evidence']); selected={}; best=[]
                for j in range(len(statements)):
                    candidates=probabilities[j*n:(j+1)*n]; k=max(range(n),key=candidates.__getitem__); best.append(candidates[k])
                    # Always retain the best real context for a sentence.
                    # Low NLI confidence is recorded as silver uncertainty;
                    # dropping the citation teaches the generator to omit
                    # evidence even when the source contains a usable span.
                    evidence=row['evidence'][k]
                    selected[evidence['id']]={'id':evidence['id'],'quote':evidence['text'],'context_index':k}
                confidence=min(best) if best else 0.; supported+=int(confidence>=a.min_evidence_entailment)
                target={'label':row['gold_label'],'final_answer':row['gold_answer'],'evidence':list(selected.values()),'confidence':round(confidence,6),'refusal_reason':None}
                row.update(sft_target=json.dumps(target,ensure_ascii=False,separators=(',',':')),evidence_annotation='silver NLI context-to-answer-sentence matching; not official rationale gold',confidence_annotation='minimum over answer sentences of maximum context entailment probability; not human confidence',teacher_entailment=best,teacher_complete_support=confidence>=a.min_evidence_entailment)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                if (i+1)%25==0: print(json.dumps({'split':split,'completed':i+1,'total':len(rows),'teacher_complete_support':supported}),flush=True)
        counts[split]={'records':len(rows),'teacher_complete_support':supported}
    report={'complete':True,'source_sha256':sha256(a.source),'manifest_sha256':sha256(a.manifest),'teacher_checkpoint':checkpoint_fingerprint(a.nli_model),'threshold':a.min_evidence_entailment,'threshold_selection':'fixed before this run; no validation/test tuning','splits':counts,'train_ids':manifest['train_ids'],'validation_ids':manifest['validation_ids'],'test_used':False,'test_records_materialized':0,'confidence_is_teacher_probability':True,'gold_confidence_or_rationale_annotations':False}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report),flush=True)
if __name__=='__main__':main()
