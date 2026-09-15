"""Generate rejected answers from an actual local model, not templates."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--offset',type=int,default=0);p.add_argument('--seed',type=int,default=42);a=p.parse_args();torch.manual_seed(a.seed);rows=[json.loads(x) for x in a.input.read_text(encoding='utf8').splitlines() if x.strip()];rows=rows[a.offset:a.offset+a.limit] if a.limit else rows[a.offset:]
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.float32,attn_implementation='eager');model.eval();out=[]
 for r in rows:
  x=tok(r['prompt'],return_tensors='pt',truncation=True,max_length=384)
  with torch.no_grad(): y=model.generate(**x,max_new_tokens=64,do_sample=True,temperature=.8,top_p=.9,pad_token_id=tok.pad_token_id)
  rejected=tok.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip();out.append({**r,'rejected':rejected,'rejected_source':'base_model_generation','rejected_model_path':str(a.model),'rejected_generation_seed':a.seed,'rejected_generation_offset':a.offset,'rejected_citations':re.findall(r'\[E(\d+)\]',rejected)})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n',encoding='utf8');print({'generated':len(out),'output':str(a.output)})
if __name__=='__main__':main()
