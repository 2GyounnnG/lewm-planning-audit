import unittest
import numpy as np
from statistics import case_indices,family_weights,paired_summary,menu_metrics

class ScientificContract(unittest.TestCase):
    def test_signed_term_negative(self):
        r=menu_metrics([1,0,2],[0,1,2],[2,1,0],['a','b','c'])
        self.assertEqual((r['D_total'],r['D_goal'],r['D_model_signed']),(1,2,-1))
    def test_aliases_do_not_change_logical_weights(self):
        r=menu_metrics([0,0,2],[1,1,0],[1,1,0],['b','a','c'])
        self.assertEqual(r['model_choice'],'a');self.assertEqual(r['logical_candidates'],3)
    def test_case_pairing_and_missing(self):
        x=np.array([[0,1],[1,0],[1,np.nan]])
        r=paired_summary('reacher',['a','b','c'],x)[1]
        self.assertEqual(r['paired_cases'],2);self.assertEqual(r['difference'],0);self.assertEqual(r['missing_cases'],['c'])
    def test_family_episode_ratio(self):
        w=family_weights('pusht',['a','b','c'],['large','large','small'])
        self.assertTrue(np.array_equal(w[:,0],w[:,1]));self.assertEqual(w.shape,(5000,3))
        x=np.array([[0,1],[0,1],[0,0.]])
        r=paired_summary('pusht',['a','b','c'],x,families=['large','large','small'])[1]
        self.assertEqual(r['difference'],2/3)
    def test_nonfinite_not_ranked(self):
        with self.assertRaises(ValueError):menu_metrics([0,np.nan],[0,1],[0,1],['a','b'])
    def test_shared_indices(self):
        self.assertTrue(np.array_equal(case_indices('pusht',['a','b']),case_indices('pusht',['a','b'])))

if __name__=='__main__':unittest.main()
