"""Full paper QA retrieval comparison with released Qasper scoring semantics."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_workbench import QasperEvidenceWorkbench
from src.hierarchical_retrieval import HierarchicalParagraphRetriever
from src.qasper_metrics import references, score_evidence, match_evidence_positions


def load_papers(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    data = json.loads(path.read_text(encoding='utf8'))
    result = []
    for pid, paper in data.items():
        qas = paper['qas']; full = paper['full_text']
        result.append({'id': pid, 'title':paper.get('title',''), 'abstract':paper.get('abstract',''),
                       'full_text':{'section_name':[s['section_name'] for s in full], 'paragraphs':[s['paragraphs'] for s in full]},
                       'qas':{'question':[q['question'] for q in qas], 'question_id':[q['question_id'] for q in qas],
                              'answers':[{'answer':[a['answer'] for a in q['answers']]} for q in qas]}})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qasper', type=Path, required=True)
    p.add_argument('--split', choices=['train_paper_holdout','official_dev'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reranker', type=Path)
    p.add_argument('--candidate-k', type=int, default=24)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cached-predictions', type=Path, help='Reuse complete neural scores while recomputing metrics and lexical baselines')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    papers = load_papers(a.qasper)
    if a.split == 'train_paper_holdout':
        papers = [r for r in papers if int(hashlib.sha1(str(r['id']).encode()).hexdigest()[:8],16)/2**32 < .2]
    elif a.qasper.suffix != '.json' or 'dev' not in a.qasper.name:
        raise ValueError('Official dev requires the downloaded official dev JSON')
    wb = QasperEvidenceWorkbench(papers)
    cached = None
    cached_summary = None
    if a.cached_predictions:
        cached_summary = json.loads(a.cached_predictions.with_name('summary.json').read_text(encoding='utf8'))
        if not cached_summary.get('complete') or cached_summary['source_sha256'] != hashlib.sha256(a.qasper.read_bytes()).hexdigest():
            raise ValueError('Cached run is incomplete or uses a different source')
        saved = [json.loads(s) for s in a.cached_predictions.read_text(encoding='utf8').splitlines() if s.strip()]
        cached = {r['question_id']:r for r in saved}
        expected_ids = {qid for paper in papers for qid in paper['qas']['question_id']}
        if len(cached) != len(saved) or set(cached) != expected_ids:
            raise ValueError('Cached question membership mismatch')
    neural = None
    if a.reranker and cached is None:
        import torch
        from src.paragraph_reranker import ParagraphCrossEncoder
        torch.set_num_threads(a.threads)
        neural = ParagraphCrossEncoder(a.reranker)
    expected = sum(len(p['qas']['question']) for p in papers)
    records = []; source_mapping = {'evidence_items':0,'mapped_items':0,'missing_items':0,'figure_table_items':0}
    with (a.output/'predictions.jsonl').open('x',encoding='utf8') as stream:
        for paper in papers:
            pid = str(paper['id']); paragraphs = wb.papers[pid]['paragraphs']
            index = HierarchicalParagraphRetriever(paragraphs)
            for qi, question in enumerate(paper['qas']['question']):
                start = time.perf_counter()
                # Only question and document text enter retrieval / neural scoring.
                hierarchy = {k:index.retrieve(question, top_k=k) for k in (6,12)}
                saved = cached[paper['qas']['question_id'][qi]] if cached is not None else None
                if saved and (saved['paper_id'] != pid or saved['question'] != question):
                    raise ValueError('Cached paper/question text mismatch')
                global_hits = saved['global_candidates'] if saved else index.retrieve_global(question, top_k=a.candidate_k)
                global_ms = (time.perf_counter()-start)*1000
                tick = time.perf_counter()
                reranked = saved['reranked_candidates'] if saved else (neural.rerank(question,global_hits) if neural else None)
                rerank_ms = (time.perf_counter()-tick)*1000
                refs = references(paper['qas']['answers'][qi])
                mapped = [match_evidence_positions(paragraphs,ref['evidence']) for ref in refs]
                for m in mapped:
                    source_mapping['evidence_items'] += len(m['mapped'])+len(m['missing'])+len(m['figure_table'])
                    source_mapping['mapped_items'] += len(m['mapped'])
                    source_mapping['missing_items'] += len(m['missing'])
                    source_mapping['figure_table_items'] += len(m['figure_table'])
                predictions = {}; scoring = {}
                methods = {'hierarchical':hierarchy, 'global_bm25':global_hits}
                if reranked is not None: methods['global_bm25_cross_encoder'] = reranked
                for name,hits in methods.items():
                    for k in (6,12):
                        key = name+'@'+str(k); selected = hits[k] if name == 'hierarchical' else hits[:k]
                        predictions[key] = [r['id'] for r in selected]
                        scoring[key] = score_evidence([r['text'] for r in selected],refs)
                scoring['global_candidates@'+str(a.candidate_k)] = score_evidence([r['text'] for r in global_hits],refs)
                record = {'paper_id':pid,'question_id':paper['qas']['question_id'][qi],'question_index':qi,'question':question,
                          'document_words':sum(len(r['text'].split()) for r in paragraphs),
                          'answerability_disagreement':len({r['type']=='none' for r in refs})>1,
                          'all_unanswerable':all(r['type']=='none' for r in refs),
                          'has_text_evidence':any(m['mapped'] for m in mapped),'predicted_positions':predictions,'scores':scoring,
                          'global_candidates':global_hits, 'reranked_candidates':reranked,
                          'timings_ms':{'retrieval_ms':global_ms,'rerank_ms':rerank_ms}}
                records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                if len(records)%50 == 0:print(json.dumps({'completed':len(records),'expected':expected}),flush=True)
    def aggregate(rows):
        return {key:{metric:sum(r['scores'][key][metric] for r in rows)/len(rows) if rows else None
                     for metric in ('precision','recall','f1')} for key in records[0]['scores']}
    report = {'complete':len(records)==expected,'questions':len(records),'papers':len(papers),'split':a.split,
              'source_sha256':hashlib.sha256(a.qasper.read_bytes()).hexdigest(),'test_used':False,'gold_used_for_retrieval':False,
              'scorer':'released Qasper exact paragraph evidence matching; max evidence F1 over alternative annotations',
              'answer_generation_evaluated':False,'source_mapping':source_mapping,'all_questions':aggregate(records),
              'questions_with_mappable_text_evidence':{'count':sum(r['has_text_evidence'] for r in records),'metrics':aggregate([r for r in records if r['has_text_evidence']])},
              'length_buckets':{name:{'count':len(group),'metrics':aggregate(group)} for name,group in [
                  ('under_2048_words',[r for r in records if r['document_words']<2048]),
                  ('2048_to_4095_words',[r for r in records if 2048<=r['document_words']<4096]),
                  ('4096_plus_words',[r for r in records if r['document_words']>=4096])]},
              'reranker':cached_summary['reranker'] if cached_summary else (str(a.reranker) if a.reranker else None),'candidate_k':a.candidate_k,
              'rescored_from':str(a.cached_predictions) if cached is not None else None,
              'human_reviews':0,'hyperparameters_selected_on_this_split':False}
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
