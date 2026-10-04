import json,tempfile,unittest
from pathlib import Path
import numpy as np
from aggregate import ARMS,summarize,s1,write,read

class PairedAggregation(unittest.TestCase):
    def test_fixed_refit_mean_requires_all_three(self):
        values={('a','H0'):[1],('a',ARMS[1]):[2],('a',ARMS[2]):[3],('a',ARMS[3]):[4],
                ('b','H0'):[7],('b',ARMS[1]):[0],('b',ARMS[2]):[0]}
        rows=summarize('reacher',['a','b'],{'a':None,'b':None},values,ARMS,'value',{})
        row=rows[-1]
        self.assertEqual(row['arm'],'FIXED3_REFIT_MEAN_NOT_ENSEMBLE')
        self.assertEqual(row['cases'],1);self.assertEqual(row['estimate'],3)
        self.assertEqual(row['difference_vs_reference'],2)
    def test_histories_pair_before_case_mean(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'manifests').mkdir()
            (root/'manifests/reacher_data_roles.json').write_text(json.dumps({'cases':{'EVAL':[{'case_id':'a','family_id':None}]}}))
            rows=[]
            for anchor,histories in [(0,['H_POLICY']),(5,['H_POLICY','H_REAL3'])]:
                for history in histories:
                    for j,arm in enumerate(ARMS):
                        rows.append({'case_id':'a','source_policy':'OFFLINE_RECORDED','anchor_raw':anchor,'history_kind':history,'horizon_raw':5,
                            'model':arm,'valid':True,'primary_anchor':True,'missing_reason':'','latent_mse':100 if anchor==0 else j+(2 if history=='H_REAL3' else 0),
                            'optimism':None,'source_relation':'OFFLINE'})
            write(root/'input.csv',rows);s1(root/'input.csv',root,'reacher',root/'output')
            paired=read(root/'output/S1_HISTORY_PAIRED_RAW.csv')
            self.assertEqual(len(paired),4)
            self.assertTrue(all(r['anchor_raw']=='5' and float(r['difference_REAL3_minus_POLICY'])==2 for r in paired))
            means=read(root/'output/S1_HISTORY_PAIRED_TABLE.csv')
            self.assertEqual(len(means),5);self.assertTrue(all(float(r['estimate'])==2 for r in means))

if __name__=='__main__':unittest.main()
