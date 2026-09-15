"""Leakage-safe Qasper protocol evaluation.

The script evaluates context construction independently of generation.  Gold
evidence is used only for scoring; full-context, lexical BM25-like, and
extractive contexts are built from the paper text.  This makes retrieval and
long-document results auditable even when a local model cannot finish a full
generation run.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from collections import Counter

def norm(s): return re.findall(r"[a-z0-9]+", str(s or "").lower())
def f1(a,b):
    aa,bb=Counter(norm(a)),Counter(norm(b)); overlap=sum((aa&bb).values())
    return 2*overlap/(sum(aa.values())+sum(bb.values())) if overlap else 0.0
def paragraphs(paper):
    full=paper.get('full_text') or {}; out=[]
    if isinstance(full,dict):
        for section, values in zip(full.get('section_name',[]), full.get('paragraphs',[])):
            for i,text in enumerate(values if isinstance(values,list) else [values]):
                text=str(text or '').strip()
                if text: out.append({'id':f'{section}:{i}','text':text})
    return out
def gold_spans(annotation):
    spans=[]
    for key in ('highlighted_evidence','evidence'):
        spans.extend(str(x).strip() for x in annotation.get(key) or [] if str(x).strip())
        if spans: break
    return spans
def choose(group):
    anns=group.get('answer') or []
    return next((x for x in anns if not x.get('unanswerable') and (x.get('free_form_answer') or x.get('extractive_spans') or x.get('yes_no') is not None)), anns[0] if anns else {})
def rows(path):
    import pyarrow.parquet as pq
    out=[]
    for paper in pq.read_table(path).to_pylist():
        ps=paragraphs(paper); qas=paper.get('qas') or {}
        for i,q in enumerate(qas.get('question') or []):
            groups=qas.get('answers') or []
            if i>=len(groups): continue
            ann=choose(groups[i]); spans=gold_spans(ann)
            out.append({'paper_id':str(paper['id']),'question_id':str((qas.get('question_id') or [f'{paper["id"]}-{i}'])[i]),'question':q,'paragraphs':ps,'gold_spans':spans,'answerable':not bool(ann.get('unanswerable'))})
    return out
def split(items):
    import hashlib
    tr,va=[],[]
    for x in items:
        v=int(hashlib.sha1(x['paper_id'].encode()).hexdigest()[:8],16)/2**32
        (va if v<.2 else tr).append(x)
    return tr,va
def select(item, mode, k, encoder=None):
    ps=item['paragraphs']; q=set(norm(item['question']))
    if mode=='full': return ps
    if mode=='dense' and encoder is not None:
        import numpy as np
        vecs=encoder.encode([item['question']]+[p['text'] for p in ps], normalize_embeddings=True, show_progress_bar=False)
        scores=(vecs[1:] @ vecs[0]).tolist(); ranked=sorted(zip(scores,ps),key=lambda x:(x[0],x[1]['id']),reverse=True)
        return [p for _,p in ranked[:k]]
    scored=[]
    for p in ps:
        toks=set(norm(p['text'])); overlap=len(q&toks)/max(len(q),1); scored.append((overlap,p))
    scored.sort(key=lambda x:(x[0],x[1]['id']), reverse=True)
    return [p for score,p in scored[:k]]
def score(item, selected):
    selected_text=[p['text'] for p in selected]; gold=item['gold_spans']
    if not gold: return {'evidence_recall':None,'evidence_precision':None,'evidence_f1':None,'selected':len(selected),'gold':0}
    hit=sum(max((f1(s,g) for s in selected_text),default=0)>=.5 for g in gold)
    precision=sum(max((f1(s,g) for g in gold),default=0)>=.5 for s in selected_text)/max(len(selected_text),1)
    recall=hit/len(gold)
    return {'evidence_recall':recall,'evidence_precision':precision,'evidence_f1':2*precision*recall/(precision+recall) if precision+recall else 0.0,'selected':len(selected),'gold':len(gold)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--qasper',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--top-k',type=int,default=8); p.add_argument('--dense-model',type=Path); a=p.parse_args()
    all_rows=rows(a.qasper); train,valid=split(all_rows); report={'dataset':{'papers':len(set(x['paper_id'] for x in all_rows)),'questions':len(all_rows),'train_questions':len(train),'validation_questions':len(valid),'test_used':False},'baselines':{}}
    encoder=None
    if a.dense_model:
        from sentence_transformers import SentenceTransformer
        encoder=SentenceTransformer(str(a.dense_model), device='cpu')
    for mode in ('full','bm25','dense','extractive'):
        if mode == 'dense' and encoder is None:
            report['baselines'][mode] = {'status': 'not_run', 'reason': 'pass --dense-model with a local sentence-transformers checkpoint'}
            continue
        k=len(valid[0]['paragraphs']) if mode=='full' and valid else a.top_k
        vals=[score(x,select(x,'full' if mode=='full' else mode,k,encoder)) for x in valid]
        numeric=[v for v in vals if v['evidence_recall'] is not None]
        report['baselines'][mode]={'queries':len(valid),'answerable_queries':len(numeric),'evidence_recall':sum(v['evidence_recall'] for v in numeric)/max(len(numeric),1),'evidence_precision':sum(v['evidence_precision'] for v in numeric)/max(len(numeric),1),'evidence_f1':sum(v['evidence_f1'] for v in numeric)/max(len(numeric),1),'long_document':{'count':sum(len(x['paragraphs'])>=32 for x in valid),'evidence_recall':sum(v['evidence_recall'] for x,v in zip(valid,vals) if len(x['paragraphs'])>=32 and v['evidence_recall'] is not None)/max(sum(len(x['paragraphs'])>=32 and v['evidence_recall'] is not None for x,v in zip(valid,vals)),1)},'implementation':'sentence-transformers cosine' if mode=='dense' else 'lexical paragraph overlap' if mode!='full' else 'all paragraphs'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
