import unittest
from src.sentence_evidence_selector import SentenceEvidenceSelector


class SentenceSelectorTests(unittest.TestCase):
    class Scorer:
        checkpoint = 'test-relevance-scorer'
        def rerank(self, question, candidates):
            return [dict(row, rerank_score=float(row['sentence_index'])) for row in candidates]

    def paragraphs(self):
        return [{'id': 's1:p1', 'section_name': 'Results', 'text': 'First result. Second finding. Third detail.'},
                {'id': 's2:p1', 'section_name': 'Methods', 'text': 'First method. Second procedure.'}]

    def test_neural_order_and_original_sentence_positions(self):
        selector = SentenceEvidenceSelector(scorer=self.Scorer())
        selected = selector.select('question', self.paragraphs(), top_k=3, per_paragraph=2)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0]['paragraph_id'], 's1:p1')
        self.assertEqual(selected[0]['sentence_index'], 3)
        self.assertEqual(selected[0]['text'], 'Third detail.')
        self.assertEqual(sum(row['paragraph_id'] == 's1:p1' for row in selected), 2)

    def test_no_lexical_overlap_gate_and_empty_sources(self):
        selector = SentenceEvidenceSelector(scorer=self.Scorer())
        self.assertTrue(selector.select('unrelated query terms', self.paragraphs(), top_k=1))
        self.assertEqual(selector.select('question', []), [])
        with self.assertRaises(ValueError):
            selector.select('question', self.paragraphs(), top_k=0)

    def test_candidate_membership_and_finite_scores_are_required(self):
        class Broken:
            def rerank(self, question, candidates):
                return [dict(row, rerank_score=float('nan')) for row in candidates]
        with self.assertRaises(ValueError):
            SentenceEvidenceSelector(scorer=Broken()).score('question', self.paragraphs())
        class Missing:
            def rerank(self, question, candidates):
                return []
        with self.assertRaises(ValueError):
            SentenceEvidenceSelector(scorer=Missing()).score('question', self.paragraphs())


if __name__ == '__main__': unittest.main()
