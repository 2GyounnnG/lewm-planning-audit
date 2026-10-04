"""CPU semantic checks: only raw25 changes; actual action prefix precedes future."""
import unittest
import numpy as np
import torch
from .context import history_indices,real_history_observation,candidates_with_prefix

class Processor:
    def transform(self,x):return (x-10)/2

class ContextTests(unittest.TestCase):
    def test_only_second_replan_changes_input(self):
        observations=[{'pixels':np.full((2,2,3),i,dtype=np.uint8)} for i in range(26)]
        info={'pixels':observations[0]['pixels'][None,None],'goal':np.array([9]),'action':np.zeros((1,1,2))}
        for i in range(25):self.assertIs(real_history_observation(info,observations[:i+1],i),info)
        changed=real_history_observation(info,observations,25)
        np.testing.assert_array_equal(changed['pixels'][0,:,0,0,0],[15,20,25])
        self.assertIs(changed['goal'],info['goal']);self.assertIs(changed['action'],info['action'])
        self.assertEqual(history_indices(0),[0]);self.assertEqual(history_indices(25),[15,20,25])
    def test_action_alignment_and_no_cem_horizon_change(self):
        c=torch.arange(1*300*5*10,dtype=torch.float32).reshape(1,300,5,10)
        self.assertIs(candidates_with_prefix({'pixels':torch.zeros(1,300,1,1)},c,[],Processor()),c)
        rows=[{'action':np.array([i*2,i*2+1],dtype=np.float32)} for i in range(25)]
        out=candidates_with_prefix({'pixels':torch.zeros(1,300,3,1)},c,rows,Processor())
        self.assertEqual(out.shape,(1,300,7,10));self.assertTrue(torch.equal(out[:,:,2:],c))
        wanted=(np.arange(30,50,dtype=np.float32)-10)/2
        np.testing.assert_array_equal(out[0,0,:2].numpy().reshape(-1),wanted)
        self.assertTrue(torch.equal(out[:,0,:2],out[:,299,:2]))
    def test_official_rollout_has_five_future_predictions(self):
        import importlib.util,os
        from pathlib import Path
        source=Path(os.environ.get('R3_ROOT','/Volumes/MyProj/r3_official_lewm_predictor_refit/recovery'))/'source/lewm/jepa.py'
        spec=importlib.util.spec_from_file_location('h2_test_official_jepa',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class Dummy:
            action_encoder=staticmethod(lambda a:a)
            encode=staticmethod(lambda info:dict(info,emb=info['pixels'][...,0]))
            def __init__(self):self.calls=0
            def predict(self,emb,acts):self.calls+=1;return emb+5
        model=Dummy();pixels=torch.tensor([15.,20.,25.]).reshape(1,1,3,1,1).expand(1,300,3,1,1)
        out=module.JEPA.rollout(model,{'pixels':pixels},torch.zeros(1,300,7,10))
        self.assertEqual(model.calls,5);self.assertEqual(out['predicted_emb'].shape,(1,300,8,1))
        self.assertTrue(torch.all(out['predicted_emb'][...,-1,:]==50))

if __name__=='__main__':unittest.main()
