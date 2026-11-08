import unittest

from src.preference_data import audit_model_preference_candidate, build_hard_negative, validate_preference_pair


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

    def test_model_pair_requires_nli_and_answer_consistency(self):
        row = {"prompt": "Question", "chosen": "Yes [E1]", "rejected": "Yes [E2]", "record_id": "q1", "answerable": True, "error_type": "invalid_citation"}
        report = audit_model_preference_candidate(row, valid_citations={"E1"}, chosen_entailment=.9, rejected_entailment=.2, chosen_answer_match=True)
        self.assertTrue(report["accepted"])
        self.assertTrue(report["checks"]["chosen_entails_evidence"])

    def test_model_pair_filters_unsupported_chosen_answer(self):
        row = {"prompt": "Question", "chosen": "Yes [E1]", "rejected": "No [E2]", "record_id": "q1", "answerable": True, "error_type": "polarity_flip"}
        report = audit_model_preference_candidate(row, valid_citations={"E1"}, chosen_entailment=.2, rejected_entailment=.1, chosen_answer_match=False)
        self.assertFalse(report["accepted"])


if __name__ == "__main__":
    unittest.main()
