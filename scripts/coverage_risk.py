"""Compute coverage-risk and ECE from held-out prediction confidence."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=json.loads(a.predictions.read_text(encoding='utf8'));items=[]
 for r in rows:
  confidence=1.0 if not r.get('is_refusal') and not r.get('invalid_citation') else .25; correct=float(r.get('token_f1',0)>=.5);items.append((confidence,correct))
 curve=[]
 for t in [i/20 for i in range(21)]:
  kept=[x for x in items if x[0]>=t];curve.append({'threshold':t,'coverage':len(kept)/max(len(items),1),'risk':1-sum(x[1] for x in kept)/len(kept) if kept else None})
 bins=[];ece=0
 for lo in [i/10 for i in range(10)]:
  b=[x for x in items if lo<=x[0]<(lo+.1) or lo==.9 and x[0]==1]
  if b:
   conf=sum(x[0] for x in b)/len(b);acc=sum(x[1] for x in b)/len(b);ece+=len(b)/len(items)*abs(conf-acc);bins.append({'lo':lo,'count':len(b),'confidence':conf,'accuracy':acc})
 out={'count':len(items),'ece':ece,'coverage_risk':curve,'bins':bins,'confidence_source':'heuristic citation/refusal proxy, not model probability','status':'diagnostic_only'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
