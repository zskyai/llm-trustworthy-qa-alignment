import unittest

from src.structured_answer import extract_inline_citations, parse_structured_answer


class StructuredAnswerTests(unittest.TestCase):
    def test_parse_fenced_json_and_validate_citation(self):
        answer = parse_structured_answer('```json {"final_answer":"Yes", "citations":["[E1]"], "confidence":0.8}```')
        self.assertEqual(answer.final_answer, "Yes")
        self.assertEqual(answer.validate_citations({"E1"}), ())

    def test_abstention_requires_reason(self):
        with self.assertRaises(ValueError):
            parse_structured_answer({"final_answer": "I cannot tell", "abstained": True})

    def test_inline_citations_are_unique(self):
        self.assertEqual(extract_inline_citations("a [E1] b [E1] c [E2]"), ("E1", "E2"))


if __name__ == "__main__":
    unittest.main()
