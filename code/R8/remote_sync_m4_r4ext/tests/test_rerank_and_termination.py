import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from r4.s3 import Reranker
from r4.replay import run_branch

class Scale:
    def inverse_transform(self,x):return x

class MutatingCost(torch.nn.Module):
    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1))
    def get_cost(self,info,candidates):
        assert set(info)=={'pixels','goal','action'}
        info.pop('action');info['pixels']=torch.zeros_like(info['pixels'])
        return candidates.square().sum((-1,-2))

class TinyTerminatingBranch:
    step_calls=0
    def __init__(self,*args):self.n=0;self.true_terminated=False;self.physics_steps=0
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def observe(self):return {'state':np.array([self.n],dtype=float),'pixels':np.zeros((2,2,3),np.uint8)}
    def replay(self,actions):
        for action in actions:self.step(action)
    def step(self,action):
        if self.true_terminated:raise AssertionError('Would auto-reset')
        self.n+=1;type(self).step_calls+=1;self.physics_steps+=1;self.true_terminated=self.n==3
        return {'margin':1/(self.n+1),'success':self.true_terminated}

class Tests(unittest.TestCase):
    def test_h0_menu_never_uses_simulator_and_input_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            rerank=Reranker('pusht',{'case_id':'test'},'R3_ORIGINAL','H0_MENU_RERANK',MutatingCost(),Scale(),None,lambda:None,tmp,'cpu')
            info={'pixels':torch.ones(1,1,3,2,2),'goal':torch.ones(1,1,3,2,2),'action':torch.zeros(1,1,2)}
            original={k:v.clone() for k,v in info.items()};rng=torch.get_rng_state().clone()
            with patch('r4.replay.run_branch',side_effect=AssertionError('Learning control cannot branch')):
                selected,record=rerank({'actions':torch.ones(1,5,10)},info,np.zeros((300,5,10),np.float32),np.empty((0,2)),0)
            self.assertEqual(record['branch_count'],0);self.assertEqual(record['selected_score'],0)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertTrue(torch.all(selected['actions']==0))
            for key in original:self.assertTrue(torch.equal(info[key],original[key]))
            self.assertTrue((Path(tmp)/'replan_00/MENU_LOCK.json').exists())
    def test_real_last_preserves_success_and_missing_horizon(self):
        TinyTerminatingBranch.step_calls=0
        with patch('r4.replay.ReplayBranch',TinyTerminatingBranch):
            result=run_branch('reacher',{},lambda:None,np.empty((0,2)),np.zeros((25,2)))
        self.assertEqual(TinyTerminatingBranch.step_calls,3);self.assertFalse(result['valid_fixed_horizon'])
        self.assertTrue(result['any_success']);self.assertEqual(result['candidate_raw_steps'],3)
        self.assertEqual(result['candidate_requested_raw_steps'],25)
        self.assertEqual(result['missing_reason'],'TRUE_DMC_LAST_BEFORE_FIXED_HORIZON')

if __name__=='__main__':unittest.main()
