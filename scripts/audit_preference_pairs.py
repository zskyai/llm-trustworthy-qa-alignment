"""Audit and filter preference pairs built from real annotations/results."""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,nargs='+',required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--sample-size',type=int,default=100); p.add_argument('--nli-model',type=Path); p.add_argument('--nli-threshold',type=float,default=.5); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--limit',type=int,default=0); a=p.parse_args()
    rows=[json.loads(x) for path in a.input for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
    rows=rows[:a.limit] if a.limit else rows
    nli=None
    if a.nli_model:
        from transformers import pipeline
        nli=pipeline('text-classification',model=str(a.nli_model),tokenizer=str(a.nli_model),local_files_only=True,device=-1)
    def label(x):
        t=str(x).lower()
        return 'entailment' if 'entail' in t else 'contradiction' if 'contrad' in t else 'neutral'
    def answer_text(x):
        return re.sub(r'\[E\d+\]|\[NO_ANSWER\]',' ',str(x or '')).strip()
    nli_pairs=[]; pair_keys=[]
    if nli:
        for i,r in enumerate(rows):
            evidence=[str(x) for x in (r.get('evidence') or []) if str(x).strip()]
            for variant in ('chosen','rejected'):
                for j,span in enumerate(evidence):
                    nli_pairs.append({'text':span,'text_pair':answer_text(r.get(variant))}); pair_keys.append((i,variant,j))
        nli_out=nli(nli_pairs,batch_size=a.batch_size,truncation=True) if nli_pairs else []
        nli_by_row={}
        for key,pred in zip(pair_keys,nli_out):
            nli_by_row.setdefault(key[0],{}).setdefault(key[1],[]).append({'label':label(pred['label']),'score':float(pred['score'])})
    else:
        nli_by_row={}
    kept=[]; reasons={}
    nli_pass=0
    for i,r in enumerate(rows):
        chosen=r.get('chosen',''); rejected=r.get('rejected',''); valid=set(r.get('valid_citations') or []); cc=re.findall(r'\[E(\d+)\]',chosen); rc=re.findall(r'\[E(\d+)\]',rejected)
        checks={'chosen_citation_valid': all('E'+x in valid for x in cc) if cc else not r.get('answerable',True),'rejected_has_invalid_or_refusal_error': bool(any('E'+x not in valid for x in rc) or (r.get('answerable') and '[NO_ANSWER]' not in rejected and rejected!=chosen) or (not r.get('answerable') and '[NO_ANSWER]' in chosen)),'answer_consistency': (not r.get('answerable')) or bool(r.get('gold_answer') and any(tok in chosen.lower() for tok in re.findall(r'[a-z0-9]{4,}',r.get('gold_answer','').lower())[:3]))}
        if nli:
            chosen_scores=nli_by_row.get(i,{}).get('chosen',[]); rejected_scores=nli_by_row.get(i,{}).get('rejected',[])
            chosen_entail=max((x['score'] for x in chosen_scores if x['label']=='entailment'),default=0.0)
            rejected_entail=max((x['score'] for x in rejected_scores if x['label']=='entailment'),default=0.0)
            checks['chosen_nli_entails']=(not r.get('answerable',True)) or chosen_entail>=a.nli_threshold
            checks['rejected_nli_not_supported']=(not r.get('answerable',True)) or rejected_entail<a.nli_threshold
            checks['nli_chosen_max_entailment']=chosen_entail
            checks['nli_rejected_max_entailment']=rejected_entail
            if checks['chosen_nli_entails'] and checks['rejected_nli_not_supported']: nli_pass+=1
        ok=all(v for k,v in checks.items() if not k.startswith('nli_') or k.endswith(('entails','supported')))
        reasons['kept' if ok else 'filtered']=reasons.get('kept' if ok else 'filtered',0)+1
        if ok: kept.append({**r,'filter_checks':checks})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'filtered.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in kept)+'\n',encoding='utf8')
    rng=random.Random(42); sample=rng.sample(rows,min(a.sample_size,len(rows)))
    (a.output_dir/'human_sample.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in sample)+'\n',encoding='utf8')
    report={'input_files':[str(x) for x in a.input],'input_pairs':len(rows),'kept_pairs':len(kept),'filtered_pairs':len(rows)-len(kept),'filter_pass_rate':len(kept)/max(len(rows),1),'counts':reasons,'human_sample_size':len(sample),'nli':{'status':'real_model' if nli else 'not_run','model_path':str(a.nli_model) if a.nli_model else None,'threshold':a.nli_threshold if nli else None,'pairs_scored':len(nli_pairs),'pair_pass_rate':nli_pass/max(len(rows),1) if nli else None},'manual_review_required':True,'test_used':False}
    (a.output_dir/'filter_report.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
