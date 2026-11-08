import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from src.research_api import build_app
from src.research_workbench import QasperEvidenceWorkbench
from src.local_grounded_qa import citation_errors
from src.structured_answer import parse_structured_answer


class ResearchAPITests(unittest.TestCase):
    def test_source_report_feedback_and_disabled_model(self):
        paper = {'id': 'p', 'title': 'Treatment outcome', 'full_text': {'section_name': ['Results'], 'paragraphs': [['Treatment reduced risk.']]}}
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(build_app(QasperEvidenceWorkbench([paper]), history_db_path=Path(directory) / 'db.sqlite3')) as client:
                item = client.post('/v1/questions', json={'paper_id': 'p', 'question': 'Treatment outcome?'}).raise_for_status().json()
                qid = item['question_id']
                self.assertEqual(client.get('/v1/papers/p/paragraphs/s1:p1').json()['text'], 'Treatment reduced risk.')
                self.assertEqual(client.get('/v1/papers/p/paragraphs/missing').status_code, 404)
                client.post(f'/v1/questions/{qid}/feedback', json={'helpful': False, 'note': 'Review candidate'}).raise_for_status()
                report = client.get(f'/v1/questions/{qid}/report').raise_for_status()
                self.assertFalse(report.json()['training_eligible'])
                self.assertEqual(report.json()['feedback'][0]['note'], 'Review candidate')
                self.assertEqual(client.get('/v1/questions/missing/report').status_code, 404)
                self.assertEqual(client.post('/v1/questions', json={'paper_id':'p', 'question':'Treatment?', 'mode':'verified_model'}).status_code, 503)
                self.assertEqual(client.post('/v1/questions', json={'paper_id':'p', 'question':'   '}).status_code, 422)

    def test_citation_must_match_sentence_and_quote(self):
        evidence = [{'evidence_id':'E_s1_p1', 'sentences':[{'sentence_index':1, 'text':'Treatment reduced risk.'}]}]
        def answer(sentence='1', quote='reduced risk'):
            return parse_structured_answer({'final_answer':'Risk decreased.', 'citations':[{'evidence_id':'E_s1_p1', 'sentence_id':sentence, 'quote':quote}]})
        self.assertEqual(citation_errors(answer(), evidence), [])
        self.assertIn('unknown_sentence_id', citation_errors(answer(sentence='2'), evidence))
        self.assertIn('quote_does_not_match_source', citation_errors(answer(quote='cured everyone'), evidence))
        with self.assertRaises(ValueError):
            parse_structured_answer({'final_answer':'Answer', 'abstained':'false'})


if __name__ == '__main__': unittest.main()
