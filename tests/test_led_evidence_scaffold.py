import unittest
from src.led_evidence_scaffold import LEDEvidenceScaffold

class FakeTokenizer:
    def __call__(self,text,**kwargs): return {'input_ids':[1]*len(text.split())}

class LEDScaffoldTests(unittest.TestCase):
    def test_build_preserves_paragraph_and_sentence_provenance(self):
        scaffold=object.__new__(LEDEvidenceScaffold);scaffold.tokenizer=FakeTokenizer();scaffold.max_input_tokens=50
        prompt,items=scaffold.build('What changed?', [{'paragraph_id':'s1:p2','text':'A result. B result.','sentence_ids':[2,3]}])
        self.assertIn('[E1] A result. B result.',prompt);self.assertEqual(items[0].evidence_id,'E1');self.assertEqual(items[0].sentence_ids,(2,3))
    def test_budget_is_rejected_before_generation(self):
        scaffold=object.__new__(LEDEvidenceScaffold);scaffold.tokenizer=FakeTokenizer();scaffold.max_input_tokens=3
        with self.assertRaises(ValueError):scaffold.build('question', [{'paragraph_id':'p','text':'one two three four'}])

if __name__=='__main__':unittest.main()
