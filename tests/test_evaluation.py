import unittest

from src.evaluation import evaluate_predictions


class EvaluationMetricTests(unittest.TestCase):
    def test_refusal_precision_counts_only_true_positive_refusals(self):
        metrics = evaluate_predictions([
            {"prediction": "I cannot answer. [E9]", "gold_answer": "Supported answer", "answerable": True, "valid_citations": ["E1"]},
            {"prediction": "Insufficient evidence. [NO_ANSWER]", "gold_answer": "", "answerable": False, "valid_citations": []},
        ])
        self.assertEqual(metrics["refusal_precision"], 0.5)
        self.assertEqual(metrics["refusal_recall"], 1.0)
        self.assertIsNone(metrics["citation_support_rate"])

    def test_semantic_support_is_reported_only_when_labeled(self):
        metrics = evaluate_predictions([
            {"prediction": "Supported answer [E1]", "gold_answer": "Supported answer", "answerable": True,
             "valid_citations": ["E1"], "supported_citation_ids": ["E1"]},
        ])
        self.assertEqual(metrics["citation_id_validity"], 1.0)
        self.assertEqual(metrics["citation_support_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
