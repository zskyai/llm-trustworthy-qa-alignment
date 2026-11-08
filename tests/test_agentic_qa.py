import unittest

from src.agentic_qa import EvidenceQAAgent


class EvidenceQAAgentTests(unittest.TestCase):
    def test_invalid_citation_retrieves_more_and_repairs(self):
        requested = []
        generated = []

        def retrieve(_question, top_k):
            requested.append(top_k)
            return [{"evidence_id": "E1", "text": "The trial reduced risk."}]

        def generate(_question, _evidence, _feedback):
            generated.append(True)
            citation = "[E9]" if len(generated) == 1 else "[E1]"
            return {"final_answer": "The trial reduced risk.", "citations": [citation], "confidence": 0.8}

        agent = EvidenceQAAgent(retrieve, generate, lambda _answer, _evidence: 0.9, top_k=2)
        result = agent.run("What did the trial find?")
        self.assertFalse(result.answer.abstained)
        self.assertEqual(requested, [2, 4])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.answer.citations[0].evidence_id, "E1")

    def test_unsupported_answer_fails_closed_after_retry(self):
        agent = EvidenceQAAgent(
            lambda _question, _k: [{"evidence_id": "E1", "text": "Unrelated passage."}],
            lambda *_args: {"final_answer": "An unsupported claim.", "citations": ["E1"], "confidence": 0.9},
            lambda _answer, _evidence: 0.1,
            max_retries=1,
        )
        result = agent.run("What happened?")
        self.assertTrue(result.answer.abstained)
        self.assertEqual(result.answer.refusal_reason, "evidence_validation_failed")
        self.assertEqual(result.attempts, 2)

    def test_answer_without_citation_is_rejected(self):
        agent = EvidenceQAAgent(
            lambda _question, _k: [{"evidence_id": "E1", "text": "Relevant evidence."}],
            lambda *_args: {"final_answer": "A claim.", "citations": [], "confidence": 0.9},
            lambda _answer, _evidence: 0.99,
            max_retries=0,
        )
        result = agent.run("Question?")
        self.assertTrue(result.answer.abstained)
        self.assertIn("missing_citation", result.trace[-2]["invalid_citations"])


if __name__ == "__main__":
    unittest.main()
