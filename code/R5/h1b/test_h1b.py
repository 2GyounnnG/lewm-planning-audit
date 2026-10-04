import tempfile,unittest
from pathlib import Path
import numpy as np
from common import atomic,fold
from dataset import Dataset
from fit import moments,ridge_path,sse

class ScientificContract(unittest.TestCase):
    def test_covariance_solver_matches_explicit_design_and_holdout(self):
        rng=np.random.default_rng(710);x=rng.normal(size=(100,9));x[:,8]=3;y=rng.normal(size=(100,3));m={'n':len(x),'sx':x.sum(0),'sy':y.sum(0),'xx':x.T@x,'xy':x.T@y,'yy':(y*y).sum(0)}
        alpha=.03;(w,b),info=ridge_path(m,[alpha])[0][0],ridge_path(m,[alpha])[1]
        xx=x-x.mean(0);sc=x.std(0);sc[sc==0]=1;xx/=sc
        expected=np.linalg.solve(xx.T@xx+len(x)*alpha*np.eye(9),xx.T@(y-y.mean(0)))/sc[:,None]
        np.testing.assert_allclose(w,expected,atol=2e-14);np.testing.assert_allclose(sse(m,w,b),((x@w+b-y)**2).sum(0),atol=1e-12)
    def test_anchor_prefix_future_and_finite_actions(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);z=np.repeat(np.arange(60,dtype=np.float32)[:,None],192,1);actions=np.zeros((60,2),np.float32);actions[20]=np.nan
            np.save(p/'z.npy',z);np.save(p/'actions.npy',actions);np.save(p/'state.npy',np.arange(60)[:,None]);atomic(p/'manifest.json',{'episodes':[{'offset':0,'length':30,'fold':0,'holdout':False},{'offset':30,'length':30,'fold':1,'holdout':True}],'state_groups':{'position':[0]}})
            d=Dataset(p,'latent_1','three');x,y=d.xy(np.arange(len(d.anchors)));ix=d.anchors
            self.assertTrue(all(10<=v<25 or 40<=v<55 for v in ix));self.assertFalse(any(16<=v<=20 for v in ix));np.testing.assert_array_equal(x[:,0],ix-10);np.testing.assert_array_equal(x[:,192],ix-5);np.testing.assert_array_equal(x[:,384],ix);np.testing.assert_array_equal(y[:,0],ix+5)
            s=Dataset(p,'latent_1','single');np.testing.assert_array_equal(s.anchors,d.anchors)
    def test_episode_group_stability(self):
        e={'episode_id':'ep1','split_group':{'kind':'family','id':'foo'}};self.assertEqual(fold(e,'pusht'),fold({**e,'split_group':{'kind':'family','id':'bar'}},'pusht'))
    def test_latent_reporting_is_s1_float32_with_raw_prediction_retained(self):
        import types,json
        from unittest.mock import patch
        from fit import save_eval
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic(p/'manifest.json',{'test':True});np.savez(p/'fit.npz',dummy=np.array(0));x=np.zeros((2,192),np.float32);y=x.copy();z=x.copy();raw=np.full_like(x,.123456789,dtype=np.float64)
            a=types.SimpleNamespace(task='tworoom',target='latent_1',history='single',seed=0,model='RIDGE');ds=types.SimpleNamespace(path=p,h=1);cases=[{'case_id':str(i)} for i in range(2)]
            with patch('fit.evaluation',return_value=(x,y,z,cases,[])):save_eval(a,ds,lambda x,z:raw,p,p/'fit.npz')
            arr=np.load(p/'evaluation.npz');self.assertEqual(arr['pred'].dtype,np.float32);self.assertEqual(arr['pred_model_raw'].dtype,np.float64);self.assertEqual(json.load(open(p/'case_values.json'))[0]['mse'],float(np.mean(raw[0].astype(np.float32)**2)))

if __name__=='__main__':unittest.main()
