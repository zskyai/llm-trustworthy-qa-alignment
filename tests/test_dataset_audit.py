import importlib.util
import unittest
from pathlib import Path
spec = importlib.util.spec_from_file_location('dataset_audit', Path(__file__).resolve().parents[1]/'scripts/audit_datasets.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class DuplicateAuditTests(unittest.TestCase):
    def test_near_duplicates_do_not_require_identical_character_ngrams(self):
        base = ' '.join('token'+str(i) for i in range(30))
        pairs = module.near_dups([('a', base), ('b', base+' addition'), ('c', 'entirely unrelated')])
        self.assertEqual([(p['a'],p['b']) for p in pairs], [('a','b')])
        self.assertAlmostEqual(pairs[0]['jaccard'], 30/31)

    def test_reordered_duplicate_and_empty_input(self):
        self.assertEqual(len(module.near_dups([('a','alpha beta'),('b','beta alpha'),('c','')])), 1)
        self.assertEqual(module.near_dups([]), [])


if __name__ == '__main__': unittest.main()
