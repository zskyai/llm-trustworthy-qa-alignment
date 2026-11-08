import unittest

from src.hierarchical_retrieval import HierarchicalParagraphRetriever


class HierarchicalRetrievalTests(unittest.TestCase):
    def test_section_first_retrieval_preserves_paragraph_provenance(self):
        rows = [
            {"id": "methods:0", "section_name": "methods", "text": "The protocol uses a randomized treatment."},
            {"id": "methods:1", "section_name": "methods", "text": "Randomized treatment follows the same protocol."},
            {"id": "results:0", "section_name": "results", "text": "The randomized treatment reduced the outcome."},
            {"id": "discussion:0", "section_name": "discussion", "text": "Limitations are discussed by the authors."},
        ]
        result = HierarchicalParagraphRetriever(rows).retrieve(
            "randomized treatment outcome", top_k=2, section_top_k=2
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["section_id"], "results")
        self.assertTrue(all("id" in item and "score" in item for item in result))

    def test_mmr_discourages_redundant_paragraphs(self):
        rows = [
            {"id": "s1:0", "section_name": "s1", "text": "alpha beta gamma shared evidence"},
            {"id": "s1:1", "section_name": "s1", "text": "alpha beta gamma shared evidence"},
            {"id": "s2:0", "section_name": "s2", "text": "alpha beta gamma different result"},
        ]
        result = HierarchicalParagraphRetriever(rows).retrieve(
            "alpha beta gamma", top_k=2, section_top_k=2, mmr_lambda=0.2
        )
        self.assertNotEqual(result[0]["id"], result[1]["id"])
        self.assertIn("different", " ".join(item["text"] for item in result))


if __name__ == "__main__":
    unittest.main()
