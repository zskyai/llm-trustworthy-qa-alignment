"""Evaluate Base/SFT/DPO on the reconstructed PubMedQA dev split only."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_pipeline import inject_lora
from src.clean_pubmedqa import load_clean_records, load_adapter_strict, validate_adapter_provenance, sha256, checkpoint_fingerprint, LABELS
from train_clean_pubmedqa import chat_prompt
from src.selective_calibration import expected_calibration_error

def load_stage(base, adapter):
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
    if adapter:
        model = inject_lora(model)
        load_adapter_strict(model, adapter / 'adapter_model.pt')
    return model.eval()

def decode_json(text):
    value=text.strip()
    if value.startswith('```'):
        value=value.split('\n',1)[-1].rsplit('```',1)[0].strip()
    try:
        obj=json.loads(value)
        return obj if isinstance(obj,dict) else None
    except json.JSONDecodeError:
        return None

@torch.inference_mode()
def classify(model, tokenizer, text):
    prefix=text+'{"label":"'; scores=[]
    pids=tokenizer(prefix,add_special_tokens=False)['input_ids']
    for label in LABELS:
        aids=tokenizer(label,add_special_tokens=False)['input_ids'];ids=torch.tensor([pids+aids])
        logits=model(input_ids=ids,use_cache=False).logits[0]
        token_logits=logits[len(pids)-1:len(pids)+len(aids)-1].log_softmax(-1)
        scores.append(token_logits.gather(-1,torch.tensor(aids).unsqueeze(-1)).sum())
    probs=torch.stack(scores).softmax(-1).tolist();i=max(range(3),key=probs.__getitem__)
    return LABELS[i],probs

def summarize(rows):
    def classification(key):
        per_class={}
        for label in LABELS:
            tp=sum(r[key]==r['gold_label']==label for r in rows);fp=sum(r[key]==label and r['gold_label']!=label for r in rows);fn=sum(r[key]!=label and r['gold_label']==label for r in rows)
            per_class[label]={'gold':tp+fn,'predicted':tp+fp,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
        return {'accuracy':sum(r[key]==r['gold_label'] for r in rows)/len(rows),'macro_f1':sum(p['f1'] for p in per_class.values())/3,'per_class':per_class}
    citations=sum(r['citation_count'] for r in rows);valid=sum(r['valid_citation_count'] for r in rows)
    return {'questions':len(rows),'generated_answer':classification('predicted_label'),'label_likelihood':classification('classification_label'),'json_validity':sum(r['json_valid'] for r in rows)/len(rows),'citation_validity':valid/citations if citations else None,'citation_support_rate':None,'refusal_rate':sum(r['is_refusal'] for r in rows)/len(rows),'over_refusal_rate':sum(r['is_refusal'] for r in rows)/len(rows),'refusal_precision':None,'refusal_recall':None,'refusal_definition':'PQA-L provides no unanswerable gold; maybe is a class, not refusal. Refusal P/R are undefined.','classification_ece':expected_calibration_error([r['classification_confidence'] for r in rows],[r['classification_label']==r['gold_label'] for r in rows]),'mean_generation_ms':sum(r['generation_ms'] for r in rows)/len(rows),'generated_tokens':sum(r['generated_tokens'] for r in rows),'generation_budget_exhaustion_rate':sum(r['generation_budget_exhausted'] for r in rows)/len(rows)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pubmedqa',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--base-model',type=Path,required=True); p.add_argument('--sft-adapter',type=Path); p.add_argument('--dpo-adapter',type=Path)
    p.add_argument('--stages',nargs='+',choices=['base','sft','sft_dpo'],default=['base','sft','sft_dpo'])
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--max-length',type=int,default=4096); p.add_argument('--max-new-tokens',type=int,default=1024);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    torch.set_num_threads(a.threads)
    train,dev,manifest=load_clean_records(a.pubmedqa,a.manifest)
    adapters={'base':None,'sft':a.sft_adapter,'sft_dpo':a.dpo_adapter}
    for name in a.stages:
        if name!='base':
            if adapters[name] is None:raise ValueError('Requested stage needs its clean adapter: '+name)
            validate_adapter_provenance(adapters[name],a.base_model,a.pubmedqa,a.manifest)
    a.output_dir.mkdir(parents=True); tokenizer=AutoTokenizer.from_pretrained(a.base_model,local_files_only=True); tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    results=[]
    import time
    for name in a.stages:
        model=load_stage(a.base_model,adapters[name]);rows=[]
        with (a.output_dir/('predictions_'+name+'.jsonl')).open('x',encoding='utf8') as stream:
            for row in dev:
                text=chat_prompt(tokenizer,row['prompt']);encoded=tokenizer(text,return_tensors='pt',add_special_tokens=False)
                if encoded['input_ids'].shape[1]+a.max_new_tokens>a.max_length:raise ValueError('Complete generation exceeds context budget')
                label,probs=classify(model,tokenizer,text);started=time.perf_counter()
                with torch.inference_mode():out=model.generate(**encoded,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                new=out[0,encoded['input_ids'].shape[1]:];pred=tokenizer.decode(new,skip_special_tokens=True);obj=decode_json(pred);cites=obj.get('evidence',[]) if obj else []
                cites=cites if isinstance(cites,list) else [];valid=0;seen=set()
                for citation in cites:
                    if not isinstance(citation,dict):continue
                    position=citation.get('context_index');quote=citation.get('quote');cid=citation.get('id')
                    if type(position)is int and 0<=position<len(row['evidence']) and cid==row['evidence'][position]['id'] and isinstance(quote,str) and quote.strip() and quote in row['evidence'][position]['text'] and (cid,quote) not in seen:valid+=1;seen.add((cid,quote))
                result={'document_id':row['document_id'],'split':'validation','stage':name,'gold_label':row['gold_label'],'gold_answer':row['gold_answer'],'prediction':pred,'predicted_label':obj.get('label') if obj else None,'classification_label':label,'classification_probabilities':dict(zip(LABELS,probs)),'classification_confidence':max(probs),'classification_confidence_definition':'Normalized likelihood among yes/no/maybe completions; not citation support','json_valid':obj is not None,'citation_count':len(cites),'valid_citation_count':valid,'is_refusal':bool(obj and (obj.get('refusal_reason') or obj.get('label') is None)),'generation_ms':(time.perf_counter()-started)*1000,'generated_tokens':len(new),'generation_budget_exhausted':len(new)>=a.max_new_tokens}
                rows.append(result);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({'stage':name,'completed':len(rows),'expected':50}),flush=True)
        results.append({'stage':name,**summarize(rows)});del model
    report={'complete':True,'paired_comparison_complete':set(a.stages)=={'base','sft','sft_dpo'},'protocol':'PQA-L reconstructed frozen split','base_fingerprint':checkpoint_fingerprint(a.base_model),'source_sha256':sha256(a.pubmedqa),'train_records':len(train),'dev_records':len(dev),'test_records':len(manifest['test_ids']),'test_used':False,'manifest_sha256':sha256(a.manifest),'prompt_truncation':False,'max_new_tokens':a.max_new_tokens,'stages':results}
    (a.output_dir/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
