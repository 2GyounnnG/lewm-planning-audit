import hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from r4.common import replan_seed,atomic_json,atomic_npz,digest,file_record,verify_complete,fixed_subset
from r4.candidates import build_menu,hash_indices
from r4.replay import task_margin

class Scale:
    def inverse_transform(self,x):return x*2+3

class Tests(unittest.TestCase):
    def test_exact_original_stream_and_schedule_independence(self):
        for task in ('pusht','reacher'):
            for index in (0,1,2):
                expected=int.from_bytes(hashlib.sha256(f'R3_CEM_CASE_20261002/{task}/case/{index}'.encode()).digest()[:8],'big')
                self.assertEqual(replan_seed(task,'case',index),expected)
                self.assertNotEqual(replan_seed(task,'case',index,'R4_ALT_CEM_1'),expected)
                self.assertNotEqual(replan_seed(task,'case',index,'R4_ALT_CEM_1'),replan_seed(task,'case',index,'R4_ALT_CEM_2'))
    def test_candidate_return_is_present_alias_keeps_weight(self):
        proposals={arm:{'returned':np.ones((5,10),np.float32),'last_generation':np.zeros((300,5,10),np.float32)} for arm in ('H0','REFIT_103201','REFIT_103202','REFIT_103203')}
        menu=build_menu('pusht','case','initial',proposals,Scale())
        self.assertEqual(menu['raw'].shape,(64,25,2));self.assertEqual(len(menu['metadata']),64)
        self.assertTrue(np.all(menu['normalized'][0]==1));self.assertEqual(menu['metadata'][12]['alias_of'],0)
        second=build_menu('pusht','case','initial',proposals,Scale());self.assertEqual(menu['menu_sha256'],second['menu_sha256'])
        self.assertEqual(sum(x['source']=='INITIAL_DISTRIBUTION' for x in menu['metadata']),16)
        self.assertEqual(len(set(hash_indices('pusht','case','initial','H0',11))),11)
    def test_task_semantics_positions_agent_and_wrapped_angle(self):
        goal=np.zeros(7);s=np.zeros(7);s[0]=21
        self.assertGreater(task_margin('pusht',s,goal),1)
        s[:]=0;s[4]=2*np.pi-.01;s[5]=1e6
        self.assertLess(task_margin('pusht',s,goal),1)
        self.assertGreater(task_margin('reacher',np.array([2*np.pi-.01,0]),np.zeros(2)),1)
    def test_atomic_resume_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);atomic_json(p/'result.json',{'status':'COMPLETE'});atomic_npz(p/'trajectory.npz',x=np.ones(2))
            atomic_json(p/'COMPLETE.json',{'identity_sha256':'x','files':{n:file_record(p/n) for n in ('result.json','trajectory.npz')}})
            self.assertEqual(verify_complete(p,'x')['status'],'COMPLETE')
            (p/'trajectory.npz').write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):verify_complete(p,'x')
    def test_metadata_subset_has_family_round_robin(self):
        cases=[{'case_id':str(i),'family_id':str(i%4)} for i in range(40)]
        selected=fixed_subset('pusht',cases,20)
        self.assertEqual(len(selected),20);self.assertEqual(len({c['family_id'] for c in selected[:4]}),4)
        self.assertEqual(selected,fixed_subset('pusht',list(reversed(cases)),20))
    def test_official_rollout_history_grows_and_matches_cached_path(self):
        root=Path(__file__).resolve().parents[3]/'r3_official_lewm_predictor_refit';sys.path.insert(0,str(root))
        spec=importlib.util.spec_from_file_location('official_jepa_test',root/'source/lewm/jepa.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class Encoder(torch.nn.Module):
            def forward(self,x,**kw):return type('Out',(),{'last_hidden_state':x.mean((-1,-2))[:,None]})()
        class Predictor(torch.nn.Module):
            def __init__(self):super().__init__();self.seen=[]
            def forward(self,z,a):self.seen.append(z.shape[1]);return z+.1*a+z.mean(1,keepdim=True)*.01
        pred=Predictor();model=module.JEPA(Encoder(),pred,torch.nn.Identity());model.r3_contract={'history_size':3}
        from r3.model import cached_rollout
        for hist in (1,3):
            for horizon in (1,2,5):
                pixels=torch.arange(hist*3,dtype=torch.float32).reshape(1,hist,3,1,1);actions=torch.arange((hist+horizon-1)*3,dtype=torch.float32).reshape(1,hist+horizon-1,3)*.01
                official=model.rollout({'pixels':pixels[:,None]},actions[:,None])['predicted_emb'][0,0,-horizon:]
                cached=cached_rollout(model,pixels[...,0,0],actions,horizon)[0]
                torch.testing.assert_close(official,cached,rtol=0,atol=0)
        self.assertTrue({1,2,3}.issubset(pred.seen))

if __name__=='__main__':unittest.main()
