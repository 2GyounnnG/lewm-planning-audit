"""Synthetic/metadata-only checks. No real data, model assets, or GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from r3 import open_loop as o


def example(phase=3):
    n=80; z=np.arange(n*2,dtype=np.float32).reshape(n,2)
    a=(np.arange(n*2,dtype=np.float32).reshape(n,2)+.25)
    cache={'ep':{'z':z,'actions':a,'stride':5,'raw_indices':np.arange(n),'legal_starts':np.arange(n-19)}}
    case={'episode_id':'ep','case_id':'c','length':n,'open_loop_window_start_raw':phase,'open_loop_anchor_raw':phase+10}
    return cache,[case],{'history_size':3,'latent_dim':2,'macro_action_dim':10}


def routes():
    return {'version':'R3_OPEN_LOOP_ROUTING_V1','tasks':{'pusht':{'arms':[
        {'arm':'H0','step':0,'seed':None}]+[
        {'arm':f'REFIT_{s}_{n}','step':n,'seed':s,'checkpoint':{'path':f'artifacts/train/s{s}/checkpoint_{n}.pt','sha256':'a'*64},
         'run_identity':f'artifacts/train/s{s}/RUN_IDENTITY.json','result':f'artifacts/train/s{s}/result.json'}
        for s in (103201,103202,103203) for n in (3000,10000,30000)]}}}


class OpenLoopTests(unittest.TestCase):
    def test_all_raw_phases_full_action_blocks_and_targets(self):
        for phase in range(5):
            cache,cases,contract=example(phase);x=o.build_inputs(cache,cases,contract);t=o.scoring_targets(cache,cases)
            np.testing.assert_array_equal(x['initial_z'][0],cache['ep']['z'][phase+np.array([0,5,10])])
            np.testing.assert_array_equal(x['macro_actions'][0].reshape(35,2),cache['ep']['actions'][phase:phase+35])
            np.testing.assert_array_equal(t['target_z'][0],cache['ep']['z'][phase+np.array([15,20,25,30,35])])
            self.assertEqual(x['macro_actions'].shape,(1,7,10))

    def test_future_targets_are_not_forecast_inputs(self):
        cache,cases,contract=example();before=o.build_inputs(cache,cases,contract)
        cache['ep']['z'][14:]=np.nan
        after=o.build_inputs(cache,cases,contract)
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        with self.assertRaisesRegex(RuntimeError,'future target'):o.scoring_targets(cache,cases)

    def test_illegal_action_and_clip_refused_not_replaced(self):
        cache,cases,contract=example();cache['ep']['actions'][37,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):o.build_inputs(cache,cases,contract)
        cache,cases,contract=example();cases[0]['open_loop_window_start_raw']=41;cases[0]['open_loop_anchor_raw']=51
        with self.assertRaisesRegex(ValueError,'span'):o.build_inputs(cache,cases,contract)
        cache,cases,contract=example();cache['ep']['state']=np.ones((80,2))
        with self.assertRaisesRegex(ValueError,'Input-only'):o.build_inputs(cache,cases,contract)

    def test_raw_index_identity_and_wrong_membership_refused(self):
        cache,cases,contract=example();cache['ep']['raw_indices']+=1
        with self.assertRaisesRegex(ValueError,'identities'):o.build_inputs(cache,cases,contract)
        cache,cases,contract=example();cache['other']=cache['ep']
        with self.assertRaisesRegex(ValueError,'membership'):o.build_inputs(cache,cases,contract)

    def test_variance_is_pooled_per_coordinate_not_global_or_episode_mean(self):
        a=np.array([[0.,100.],[2.,102.]],np.float32);b=np.array([[4.,104.]],np.float32)
        result=o.train_variance({'a':{'z':a},'b':{'z':b}});allz=np.concatenate([a,b]).astype(np.float64)
        self.assertAlmostEqual(result['scalar'],allz.var(axis=0).mean())
        self.assertNotAlmostEqual(result['scalar'],allz.var())
        self.assertEqual(result['frames'],3);self.assertEqual(result['episodes'],2)
        np.testing.assert_allclose(result['coordinate_mean'],allz.mean(0))

    def test_metrics_raw_norms_fixed_scalar_and_nonfinite_retained(self):
        target=np.ones((5,2));pred=target*3;history=np.zeros((3,2))
        rows=o.score_case(pred,target,2.,history)
        self.assertEqual([r['horizon_macro'] for r in rows],[1,2,5])
        self.assertTrue(all(r['latent_raw_MSE']==4 and r['latent_TRAIN_variance_normalized_MSE']==2 for r in rows))
        self.assertEqual(rows[0]['prediction_norm2_over_D'],9.)
        pred[4,0]=np.nan;rows=o.score_case(pred,target,2.,history)
        self.assertEqual(len(rows),3);self.assertEqual(rows[-1]['latent_raw_MSE'],'Infinity')
        self.assertFalse(rows[-1]['prediction_finite']);self.assertEqual(rows[-1]['failure_category'],'METHOD_NONFINITE')
        json.dumps(rows,allow_nan=False)

    def test_exact_ten_routes_and_refuse_extra_or_h0_replacement(self):
        self.assertEqual(len(o.validate_routes(routes(),'pusht')),10)
        r=routes();r['tasks']['pusht']['arms'].pop()
        with self.assertRaisesRegex(ValueError,'Exactly'):o.validate_routes(r,'pusht')
        r=routes();r['tasks']['pusht']['arms'][0]['checkpoint']={'path':'wrong'}
        with self.assertRaisesRegex(ValueError,'H0'):o.validate_routes(r,'pusht')
        r=routes();r['tasks']['pusht']['arms'][1]['step']=True
        with self.assertRaises(ValueError):o.validate_routes(r,'pusht')

    def test_frozen_roles_cases_match_and_tech_cannot_enter(self):
        from r3.roles import build_roles
        records=[{'episode_id':f'e{i}','source_episode_idx':i,'source_asset_sha256':'a'*64,
                  'episode_sha256':f'{i:064x}','length':80,'family_id':f'f{i//20}',
                  'planning_starts':[0,13],'open_loop_starts':[10,13],'source_seed':42} for i in range(200)]
        roles=build_roles('pusht',records,history_size=3,frameskip=5)
        cases=o.validate_cases(roles,'pusht');self.assertEqual(len(cases),36)
        roles['cases']['EVAL'][0]['role']='TECH'
        with self.assertRaisesRegex(ValueError,'TECH'):o.validate_cases(roles,'pusht')

    def test_model_lock_required_and_changed_hash_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaisesRegex(RuntimeError,'BLOCKED'):o.verify_lock(root=root)
            (root/'manifests').mkdir();(root/'code.py').write_text('original')
            lock={'status':'MODELS_AND_SELECTION_LOCKED','files':{'code.py':o.file_record(root/'code.py')}}
            (root/o.LOCK).write_text(json.dumps(lock));o.verify_lock(['code.py'],root=root)
            with self.assertRaisesRegex(RuntimeError,'omits'):o.verify_lock(['missing'],root=root)
            (root/'code.py').write_text('different')
            with self.assertRaisesRegex(RuntimeError,'differs'):o.verify_lock(root=root)

    def test_path_traversal_and_symlink_escape_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'root';root.mkdir();(root/'outside').symlink_to(Path(td),target_is_directory=True)
            for rel in ('../x','/tmp/x','outside/x'):
                with self.assertRaises(ValueError):o.local_path(rel,root)

    def test_npz_exact_resume_with_nonfinite_and_ids(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.npz';values={'z':np.array([1.,np.nan]),'ids':np.array(['a','b'])}
            o.write_or_check_npz(p,values);o.write_or_check_npz(p,values)
            values['z'][0]=2
            with self.assertRaisesRegex(RuntimeError,'differs'):o.write_or_check_npz(p,values)

    def test_checkpoint_task_seed_step_identity_and_freeze(self):
        model=SimpleNamespace(r3_identity={'task':'pusht'},r3_contract={'history_size':3})
        route=o.validate_routes(routes(),'pusht')[1]
        identity={'job':{'task':'pusht','refit_seed':103201,'updates':30000,'job_id':'j'},'technical':False,
                  'role':'REFIT_TRAIN','endpoint':30000,'base_identity':model.r3_identity,'contract':model.r3_contract}
        sha=o.digest(identity);cp={'step':3000,'identity_sha256':sha,'delta':{'base_identity':model.r3_identity,'contract':model.r3_contract,'frozen_sha256':'freeze'}}
        result={'actual_updates':30000,'technical':False,'identity_sha256':sha,'job_id':'j',
                'status':'REFIT_TRAINING_COMPLETE_UNSCORED','frozen_before':'freeze','frozen_after':'freeze'}
        o.validate_checkpoint(route,cp,identity,result,model)
        for key,value in [('step',10000),('identity_sha256','wrong')]:
            other=copy.deepcopy(cp);other[key]=value
            with self.assertRaises(RuntimeError):o.validate_checkpoint(route,other,identity,result,model)
        result['actual_updates']=29999
        with self.assertRaisesRegex(RuntimeError,'finish'):o.validate_checkpoint(route,cp,identity,result,model)

    def test_actual_shared_cached_rollout_recursive_no_teacher_forcing(self):
        import torch
        from r3.model import cached_rollout
        class Toy:
            r3_contract={'history_size':3}
            action_encoder=staticmethod(lambda a:a)
            def predict(self,z,a):return z+a[:,:,:2]
        z=torch.tensor([[[0.,1.],[10.,11.],[20.,21.]]]);a=torch.arange(70,dtype=torch.float32).reshape(1,7,10)
        pred=cached_rollout(Toy(),z,a,5)
        expected=z[:,-1]+a[:,2:,:2].cumsum(1)
        torch.testing.assert_close(pred,expected,rtol=0,atol=0)
        for h in (1,2):torch.testing.assert_close(cached_rollout(Toy(),z,a[:,:3+h-1],h),pred[:,:h],rtol=0,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)
