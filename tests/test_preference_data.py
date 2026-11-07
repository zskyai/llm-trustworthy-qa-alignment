import unittest

from src.preference_data import build_hard_negative, validate_preference_pair


class PreferenceDataTests(unittest.TestCase):
    def setUp(self):
        self.row = {"prompt": "Question", "chosen": "Yes [E1]", "record_id": "q1"}

    def test_invalid_citation_is_explicit(self):
        pair = build_hard_negative(self.row, "invalid_citation")
        report = validate_preference_pair(pair, {"E1"})
        self.assertTrue(report["chosen_citations_valid"])
        self.assertTrue(report["rejected_contains_invalid_citation"])

    def test_unknown_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            build_hard_negative(self.row, "unknown")


if __name__ == "__main__":
    unittest.main()
