import unittest

from src.evidence_set_selector import CoverageAwareEvidenceSelector


class EvidenceSetSelectorTests(unittest.TestCase):
    def test_selection_adds_coverage_and_preserves_ids(self):
        candidates = [
            {"id": "p1::s1", "paragraph_id": "p1", "section_name": "Results", "text": "The treatment improved survival", "rerank_score": 0.8},
            {"id": "p1::s2", "paragraph_id": "p1", "section_name": "Results", "text": "The treatment improved outcome", "rerank_score": 0.79},
            {"id": "p2::s1", "paragraph_id": "p2", "section_name": "Methods", "text": "The study measured survival", "rerank_score": 0.7},
        ]
        selected = CoverageAwareEvidenceSelector().select("Did treatment improve survival outcome?", candidates, top_k=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["id"] for row in selected}), 2)
        self.assertGreater(selected[-1]["coverage_after"], 0.0)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            CoverageAwareEvidenceSelector().select("question", [{"id": "x", "text": "a"}, {"id": "x", "text": "b"}])


if __name__ == "__main__":
    unittest.main()
