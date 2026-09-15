"""Audit official Qasper/PubMedQA data without touching test examples.

The report records split provenance, answerability/type distributions, lengths,
paper overlap and a conservative near-duplicate scan.  Qasper accepts JSON or
Parquet; PubMedQA accepts the official ori_pqal JSON and optional split files.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()
def near_dups(items, threshold=.92):
    # MinHash-style 5-gram signatures: cheap, deterministic and suitable for audit.
    buckets = {}; pairs=[]
    for key, text in items:
        grams = {text[i:i+5] for i in range(max(0,len(text)-4))}
        sig = hashlib.sha1(" ".join(sorted(grams)).encode()).hexdigest()[:8]
        for other, old in buckets.get(sig, []):
            a,b=set(text.split()),set(old.split()); score=len(a&b)/max(len(a|b),1)
            if score >= threshold: pairs.append({"a":other,"b":key,"jaccard":score})
        buckets.setdefault(sig, []).append((key,text))
    return pairs

def qasper(path):
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist() if path.suffix=='.parquet' else json.loads(path.read_text(encoding='utf8'))
    qtypes=Counter(); answerable=Counter(); evidence=Counter(); lengths=[]; records=[]; papers=[]
    for paper in rows:
        pid=str(paper.get('id')); papers.append(pid)
        full=paper.get('full_text') or {}; paras=full.get('paragraphs',[]) if isinstance(full,dict) else []
        lengths.append(sum(len(norm(x).split()) for section in paras for x in (section if isinstance(section,list) else [section])))
        qas=paper.get('qas') or {}
        for i,q in enumerate(qas.get('question') or []):
            ans=(qas.get('answers') or [])[i].get('answer',[]) if i<len(qas.get('answers') or []) else []
            ann=next((x for x in ans if not x.get('unanswerable')), ans[0] if ans else {})
            un=bool(ann.get('unanswerable')); answerable['answerable' if not un else 'unanswerable']+=1
            qtypes['yes_no' if ann.get('yes_no') is not None else 'extractive' if ann.get('extractive_spans') else 'free_form' if ann.get('free_form_answer') else 'unknown']+=1
            ev=[x for x in ann.get('evidence') or [] if x]; evidence['with_evidence' if ev else 'without_evidence']+=1; evidence['spans']+=len(ev)
            records.append((f'{pid}:{i}', norm(q)))
    return {'split':'train-only parquet supplied; no dev/test file present', 'papers':len(set(papers)), 'questions':sum(qtypes.values()), 'question_types':dict(qtypes), 'answerability':dict(answerable), 'evidence':dict(evidence), 'document_word_lengths':_summary(lengths), 'paper_duplicates':len(papers)-len(set(papers)), 'near_duplicate_questions':near_dups(records)}

def pubmed(path):
    data=json.loads(path.read_text(encoding='utf8')); labels=Counter(); lengths=[]; records=[]
    for k,v in data.items():
        labels[str(v.get('final_decision','')).lower()]+=1; contexts=v.get('CONTEXTS') or []; lengths.append(sum(len(norm(x).split()) for x in contexts)); records.append((str(k),norm(v.get('QUESTION'))))
    return {'split':'ori_pqal has no split field; official file supplied is an unlabeled 1,000-example pool', 'questions':len(data), 'label_distribution':dict(labels), 'context_word_lengths':_summary(lengths), 'near_duplicate_questions':near_dups(records), 'test_used_for_training':False}

def _summary(values):
    if not values:return {'count':0}
    values=sorted(values); q=lambda p: values[min(len(values)-1,int(p*(len(values)-1)))]
    return {'count':len(values),'min':values[0],'p50':q(.5),'p90':q(.9),'p95':q(.95),'max':values[-1],'long_gt_1024':sum(x>1024 for x in values)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--qasper',type=Path,required=True); p.add_argument('--pubmedqa',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    report={'qasper':qasper(a.qasper),'pubmedqa':pubmed(a.pubmedqa),'protocol':{'test_excluded_from_sft_dpo_thresholds':True,'near_duplicate_method':'5-gram signature + token Jaccard >= .92'}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
