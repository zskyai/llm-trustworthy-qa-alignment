import unittest
from src.qasper_metrics import references, score_evidence, answer_f1, match_evidence_positions


class QasperMetricTests(unittest.TestCase):
    def test_full_evidence_not_highlight_is_the_reference(self):
        text = 'The experiment uses a fully described dataset and method.'
        refs = references({'answer': [{'extractive_spans': ['dataset'], 'evidence': [text], 'highlighted_evidence': ['dataset']}]})
        self.assertEqual(score_evidence([text], refs)['f1'], 1.0)
        self.assertEqual(score_evidence(['dataset'], refs)['f1'], 0.0)

    def test_alternative_annotations_are_not_merged(self):
        refs = [{'evidence': ['first']}, {'evidence': ['second', 'third']}]
        self.assertEqual(score_evidence(['first'], refs)['f1'], 1.0)

    def test_no_and_unanswerable_remain_distinct(self):
        self.assertEqual(references({'answer': [{'yes_no': False}]})[0]['answer'], 'No')
        self.assertEqual(references({'answer': [{'unanswerable': True}]})[0]['answer'], 'Unanswerable')
        self.assertEqual(answer_f1('Unanswerable', 'Unanswerable'), 1.0)
        self.assertEqual(answer_f1('', ''), 0.0)

    def test_float_evidence_is_counted_separately(self):
        mapping = match_evidence_positions([{'id': 's1:p2', 'text': 'Actual  evidence'}], ['Actual evidence', 'FLOAT SELECTED: Figure 1', 'missing'])
        self.assertEqual(mapping['mapped'][0]['paragraph_ids'], ['s1:p2'])
        self.assertEqual(len(mapping['figure_table']), 1)
        self.assertEqual(mapping['missing'], ['missing'])


if __name__ == '__main__': unittest.main()
