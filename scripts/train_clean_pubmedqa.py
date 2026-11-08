"""Train full-manifest PQA-L LoRA SFT, retaining full prompts and targets."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from real_pipeline import inject_lora
from src.clean_pubmedqa import load_clean_records, sha256, checkpoint_fingerprint, adapter_state

def chat_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)

def encode_complete(tokenizer, row, max_length):
    prompt=chat_prompt(tokenizer,row['prompt'])
    pids=tokenizer(prompt,add_special_tokens=False)['input_ids']
    target=tokenizer(row['sft_target']+tokenizer.eos_token,add_special_tokens=False)['input_ids']
    if len(pids)+len(target)>max_length: raise ValueError('Full prompt/target exceeds token budget for '+row['record_id'])
    ids=torch.tensor([pids+target]); labels=torch.tensor([[-100]*len(pids)+target])
    return ids,labels

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--prepared-data',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--max-length',type=int,default=4096);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(42)
    train,dev,manifest=load_clean_records(a.source,a.manifest)
    prepared=json.loads((a.prepared_data/'manifest.json').read_text(encoding='utf8'))
    if not prepared.get('complete') or prepared['manifest_sha256']!=sha256(a.manifest) or prepared['source_sha256']!=sha256(a.source): raise ValueError('Prepared data provenance mismatch')
    # JSON strings may contain literal Unicode paragraph separators. JSONL
    # records are delimited by LF; str.splitlines() also splits those valid
    # in-string characters and corrupts otherwise valid biomedical records.
    rows=[json.loads(x) for x in (a.prepared_data/'train.jsonl').read_text(encoding='utf8').split('\n') if x.strip()]
    if len(rows)!=450 or {r['record_id'] for r in rows}!=set(manifest['train_ids']) or any(r['split']!='train' for r in rows): raise ValueError('SFT requires exactly the frozen 450 train records')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume: raise FileExistsError(a.output)
    if a.epochs<1:raise ValueError('epochs must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.base_model,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token
    encoded=[encode_complete(tok,r,a.max_length) for r in rows]
    config={'base_fingerprint':checkpoint_fingerprint(a.base_model),'source_sha256':sha256(a.source),'manifest_sha256':sha256(a.manifest),'prepared_data_sha256':sha256(a.prepared_data/'train.jsonl'),'epochs':a.epochs,'learning_rate':a.learning_rate,'max_length':a.max_length,'train_ids':manifest['train_ids'],'test_used':False,'seed':42,'prompt_truncation':False,'target_truncation':False,'max_observed_sequence_tokens':max(ids.shape[1] for ids,_ in encoded)}
    if a.resume and json.loads((a.output/'config.json').read_text(encoding='utf8'))!=config: raise ValueError('Resume configuration mismatch')
    (a.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    model=AutoModelForCausalLM.from_pretrained(a.base_model,local_files_only=True,dtype=torch.float32,attn_implementation='eager')
    model=inject_lora(model);params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.learning_rate);resume=None
    if a.resume and (a.output/'checkpoint.pt').exists():
        resume=torch.load(a.output/'checkpoint.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(resume['adapter'],strict=False);opt.load_state_dict(resume['optimizer'])
    print(json.dumps({'phase':'training','train_records':len(rows),'trainable_parameters':sum(p.numel() for p in params),'total_parameters':sum(p.numel() for p in model.parameters()),'max_sequence_tokens':config['max_observed_sequence_tokens']}),flush=True)
    started=time.perf_counter();model.train();losses=[]
    for epoch in range(a.epochs):
        order=list(range(len(rows)));random.Random(42+epoch).shuffle(order)
        if resume and epoch<resume['epoch']:continue
        torch.set_rng_state(resume['torch_rng']) if resume and epoch==resume['epoch'] else torch.manual_seed(42+epoch)
        for step,index in enumerate(order,1):
            if resume and epoch==resume['epoch'] and step<=resume['step']:continue
            ids,labels=encoded[index];opt.zero_grad(set_to_none=True)
            loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();losses.append(float(loss.detach()))
            if step%10==0 or step==len(order):
                log={'epoch':epoch+1,'step':step,'total':len(order),'loss':losses[-1],'mean_loss':sum(losses)/len(losses),'seconds':time.perf_counter()-started}
                print(json.dumps(log),flush=True)
                with (a.output/'progress.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(log)+'\n')
            if step%25==0 or step==len(order):
                temp=a.output/'checkpoint.pending.pt';torch.save({'adapter':adapter_state(model),'optimizer':opt.state_dict(),'epoch':epoch,'step':step,'torch_rng':torch.get_rng_state()},temp);temp.replace(a.output/'checkpoint.pt')
    directory=a.output/'sft_adapter';directory.mkdir(exist_ok=True);torch.save(adapter_state(model),directory/'adapter_model.pt')
    provenance={**config,'complete':True,'stage':'sft','adapter_sha256':sha256(directory/'adapter_model.pt'),'seconds_this_invocation':time.perf_counter()-started,'confidence_supervision':'NLI silver probability, not calibrated correctness','official_rationale_annotations':False}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8');print(json.dumps(provenance),flush=True)
if __name__=='__main__':main()
