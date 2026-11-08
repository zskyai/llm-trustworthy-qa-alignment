"""Evaluate neural sentence selection on exactly the cached paragraph run.

Qasper labels paragraphs, not individual sentences. Official evidence scoring
therefore uses the source paragraphs of selected sentences; no sentence-gold
precision is claimed. Gold annotations never enter ranking.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_paragraph_retrieval import load_papers
from src.sentence_evidence_selector import SentenceEvidenceSelector
from src.research_workbench import QasperEvidenceWorkbench
from src.qasper_metrics import references, score_evidence

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--qasper', type=Path, required=True)
    p.add_argument('--cached-predictions', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    metadata=json.loads(a.cached_predictions.with_name('summary.json').read_text(encoding='utf8'))
    if not metadata['complete'] or metadata['source_sha256'] != hashlib.sha256(a.qasper.read_bytes()).hexdigest():
        raise ValueError('Incomplete cache or source hash mismatch')
    if metadata['split'] != 'train_paper_holdout': raise ValueError('This run requires the frozen train-paper holdout')
    papers=load_papers(a.qasper)
    papers=[r for r in papers if int(hashlib.sha1(str(r['id']).encode()).hexdigest()[:8],16)/2**32 < .2]
    wb=QasperEvidenceWorkbench(papers); paper_by_id={str(p['id']):p for p in papers}
    saved=[json.loads(s) for s in a.cached_predictions.read_text(encoding='utf8').splitlines() if s.strip()]
    expected={qid for p in papers for qid in p['qas']['question_id']}
    if len(saved)!=len(expected) or {r['question_id'] for r in saved}!=expected: raise ValueError('Incomplete or duplicate questions')
    a.output.mkdir(parents=True,exist_ok=False)
    selector=SentenceEvidenceSelector(a.checkpoint); records=[]
    with (a.output/'predictions.jsonl').open('x',encoding='utf8') as stream:
        for row in saved:
            paper=paper_by_id[row['paper_id']]; qi=row['question_index']
            if paper['qas']['question_id'][qi]!=row['question_id'] or paper['qas']['question'][qi]!=row['question']: raise ValueError('Question mismatch')
            paragraphs=wb.papers[row['paper_id']]['paragraphs']; lookup={p['id']:p for p in paragraphs}
            candidates=row['reranked_candidates'][:6]
            for candidate in candidates:
                if candidate['text']!=lookup[candidate['id']]['text']: raise ValueError('Source paragraph changed')
            start=time.perf_counter(); ranked=selector.score(row['question'],candidates); elapsed=time.perf_counter()-start
            refs=references(paper['qas']['answers'][qi]); scores={}; selections={}
            for k in (1,3,6,12):
                selected=ranked[:k]; ids=list(dict.fromkeys(r['paragraph_id'] for r in selected))
                scores['sentence_parent@'+str(k)]=score_evidence([lookup[pid]['text'] for pid in ids],refs)
                selections[str(k)]={'sentences':selected,'paragraph_ids':ids,'context_words':sum(len(s['text'].split()) for s in selected)}
            record={'paper_id':row['paper_id'],'question_id':row['question_id'],'document_words':row['document_words'],'scores':scores,'selections':selections,'neural_ms':elapsed*1000}
            records.append(record); stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush()
            if len(records)%25==0: print(json.dumps({'completed':len(records),'expected':len(saved)}),flush=True)
    def aggregate(rows):
        return {method:{m:sum(r['scores'][method][m] for r in rows)/len(rows) if rows else None for m in ('precision','recall','f1')} for method in records[0]['scores']}
    report={'complete':len(records)==len(expected),'questions':len(records),'papers':len(papers),'split':metadata['split'],'source_sha256':metadata['source_sha256'],'cache_sha256':hashlib.sha256(a.cached_predictions.read_bytes()).hexdigest(),'checkpoint':str(a.checkpoint),'test_used':False,'gold_used_for_selection':False,'hyperparameters_selected_on_this_split':False,'paragraph_candidates':6,'metrics':aggregate(records),'length_buckets':{name:{'count':len(rows),'metrics':aggregate(rows)} for name,rows in [('under_2048_words',[r for r in records if r['document_words']<2048]),('2048_to_4095_words',[r for r in records if 2048<=r['document_words']<4096]),('4096_plus_words',[r for r in records if r['document_words']>=4096])]},'sentence_gold_available':False,'scorer':'Qasper official-compatible evidence P/R/F1 on unique parent paragraphs; not sentence-level annotation','answer_generation_evaluated':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report),flush=True)
if __name__=='__main__': main()
