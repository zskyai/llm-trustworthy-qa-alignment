"""Select a refusal threshold on validation predictions only."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

def is_refusal(s):
    s=s.lower(); return '[no_answer]' in s or 'insufficient evidence' in s or 'cannot answer' in s
def score(row):
    pred=row.get('prediction',''); cites=re.findall(r'\[E(\d+)\]',pred); valid={x[1:] for x in row.get('valid_citations',[])}
    if is_refusal(pred): return 0.0
    return min(1.0, 0.5 + 0.25*bool(cites) + 0.25*all(c in valid for c in cites))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); rows=json.loads(a.predictions.read_text(encoding='utf8'))
    best=None; curve=[]
    for t in [i/20 for i in range(21)]:
        correct=0; fp=fn=0
        for r in rows:
            gold=not r.get('answerable',True); pred=score(r)<t
            correct+=int(pred==gold); fp+=int(pred and not gold); fn+=int(not pred and gold)
        item={'threshold':t,'balanced_accuracy':correct/max(len(rows),1),'over_refusal_rate':fp/max(sum(r.get('answerable',False) for r in rows),1),'missed_refusal_rate':fn/max(sum(not r.get('answerable',True) for r in rows),1)}; curve.append(item)
        if best is None or item['balanced_accuracy']>best['balanced_accuracy']: best=item
    report={'selected_on':'validation predictions only','best':best,'curve':curve,'test_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
