import unittest
from src.pubmedqa_protocol import reconstruct_pqal, split_records


class PubmedProtocolTests(unittest.TestCase):
    def data(self):
        return {str(i):{'final_decision':'yes' if i<552 else 'no' if i<890 else 'maybe'} for i in range(1000)}

    def test_reconstruction_is_deterministic_and_disjoint(self):
        manifest = reconstruct_pqal(self.data())
        self.assertEqual(manifest,reconstruct_pqal(self.data()))
        self.assertEqual([len(manifest[k]) for k in ('train_ids','validation_ids','test_ids')],[450,50,500])
        self.assertEqual(len(set(manifest['train_ids']+manifest['validation_ids']+manifest['test_ids'])),1000)

    def test_test_records_cannot_enter_training(self):
        manifest = reconstruct_pqal(self.data())
        rows = [{'document_id':str(i)} for i in range(1000)]
        train,dev = split_records(rows,manifest)
        self.assertFalse({r['document_id'] for r in train+dev} & set(manifest['test_ids']))
        manifest['train_ids'].append(manifest['test_ids'][0])
        with self.assertRaises(ValueError):split_records(rows,manifest)


if __name__=='__main__':unittest.main()
