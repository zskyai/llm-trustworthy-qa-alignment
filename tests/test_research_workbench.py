import tempfile
import unittest
import json
from pathlib import Path

from src.research_workbench import PaperQAStore, QasperEvidenceWorkbench


def paper_row():
    return {
        "id": "paper-1",
        "title": "A study of treatment response",
        "abstract": "Treatment reduced the measured outcome in the trial.",
        "full_text": {
            "section_name": ["Results", "Limitations"],
            "paragraphs": [
                ["The treatment group showed a lower measured outcome after twelve weeks. The control group did not change."],
                ["The study enrolled a small cohort at a single site."],
            ],
        },
    }


class QasperEvidenceWorkbenchTests(unittest.TestCase):
    def test_optional_neural_reranker_preserves_original_source_positions(self):
        class Reranker:
            checkpoint = 'test-relevance-head'
            def rerank(self, question, candidates):
                return [dict(row, rerank_score=1.) for row in reversed(candidates)]
        workbench = QasperEvidenceWorkbench([paper_row()])
        workbench.paragraph_reranker = Reranker()
        result = workbench.ask('paper-1', 'What did the study treatment measure?')
        self.assertEqual(result['retrieval_mode'], 'global_bm25_cross_encoder_extractive')
        for evidence in result['evidence']:
            source = workbench.get_paragraph('paper-1', evidence['paragraph_id'])
            for sentence in evidence['sentences']:
                self.assertIn(sentence['text'], source['text'])

    def test_optional_sentence_selector_replaces_lexical_sentence_ranking(self):
        class Selector:
            checkpoint = 'test-sentence-selector'
            def select(self, question, paragraphs, **kwargs):
                row = paragraphs[0]
                return [{'paragraph': row, 'paragraph_id': row['id'], 'sentence_index': 1,
                         'text': row['text'].split('. ')[0] + '.', 'score': 4.0}]
        workbench = QasperEvidenceWorkbench([paper_row()], sentence_selector=Selector())
        result = workbench.ask('paper-1', 'What did the study treatment measure?')
        self.assertIn('sentence_cross_encoder', result['retrieval_mode'])
        self.assertEqual(result['sentence_selector'], 'test-sentence-selector')
        self.assertTrue(result['evidence'])
        self.assertIn(result['evidence'][0]['sentences'][0]['text'], result['evidence'][0]['title'] + ' ' + paper_row()['full_text']['paragraphs'][0][0])

    def setUp(self):
        self.workbench = QasperEvidenceWorkbench([paper_row()])

    def test_paper_search_and_question_return_source_location(self):
        hits = self.workbench.search_papers("treatment response")
        self.assertEqual(hits[0]["paper_id"], "paper-1")

        result = self.workbench.ask("paper-1", "What happened to the measured outcome?")
        self.assertFalse(result["abstained"])
        self.assertIn("lower measured outcome", result["answer"])
        self.assertEqual(result["evidence"][0]["section"], "Results")
        self.assertEqual(result["evidence"][0]["paragraph_id"], "s1:p1")

    def test_no_matching_passage_abstains(self):
        result = self.workbench.ask("paper-1", "What was the telescope aperture?")
        self.assertTrue(result["abstained"])
        self.assertEqual(result["abstention_reason"], "no_matching_passage_in_selected_paper")
        self.assertIsNone(result["answer"])

    def test_question_feedback_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PaperQAStore(Path(directory) / "history.sqlite3")
            result = self.workbench.ask("paper-1", "What happened to the outcome?")
            question = store.save("paper-1", "What happened to the outcome?", result)
            reopened = PaperQAStore(Path(directory) / "history.sqlite3")
            updated = reopened.add_feedback(question["question_id"], False, "Wrong paragraph")
            self.assertEqual(updated["feedback"][0]["helpful"], 0)
            self.assertEqual(reopened.list(1)[0]["question_id"], question["question_id"])

    def test_pubmedqa_indexes_contexts_without_label_or_long_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'pubmedqa.json'
            source.write_text(json.dumps({'123': {'QUESTION':'Treatment outcome?', 'CONTEXTS':['Treatment reduced risk.'], 'LABELS':['RESULTS'], 'LONG_ANSWER':'SECRET_GOLD_ANSWER', 'final_decision':'SECRET_GOLD_LABEL'}}), encoding='utf-8')
            self.assertEqual(self.workbench.load_pubmedqa(source), 1)
            stored = json.dumps(self.workbench.papers['pubmed-123'])
            self.assertNotIn('SECRET_GOLD', stored)
            self.assertFalse(self.workbench.ask('pubmed-123', 'Treatment risk?')['abstained'])


if __name__ == "__main__":
    unittest.main()
