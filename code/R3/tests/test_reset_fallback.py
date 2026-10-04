"""Synthetic CPU/reset scaffolds only; no real simulator, data or model."""
import ast
import copy
import json
from pathlib import Path
import types
import tempfile
import unittest
import numpy as np
import torch
from scripts import validate_reset_fallback as v
from r3.env_compat import normalize_reset_seed, SEED_COMPATIBILITY


def case(i=0):
    return {'case_id': 'fixed_case_'+str(i), 'episode_id': 'ep'+str(i), 'source_episode_idx': i,
            'start_raw_index': 4, 'goal_raw_index': 29, 'role': 'TECH', 'task': 'pusht',
            'source_asset_sha256': 'a'*64, 'episode_sha256': str(i)*64, 'length': 80,
            'source_seed': None, 'reset_seed': None}


def real_helpers():
    source = Path(v.ROOT)/'source/swm_compat/stable_worldmodel/world/world.py'
    namespace = {'np': np, 'torch': torch, 'deepcopy': copy.deepcopy, 'defaultdict': __import__('collections').defaultdict,
                 'Path': Path, 'save_panel_videos': lambda *a, **k:None}
    tree = ast.parse(source.read_text())
    for name in ('_extract_init_goal', '_apply_callables'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'World')
    node = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '_evaluate_from_dataset')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class FakeEnv:
    dt=.01; control_hz=10; k_p=100; k_v=20; relative=True; action_scale=100
    def __init__(self, cross_seed=False, variation_drift=False):
        self.space=types.SimpleNamespace(damping=0., gravity=(0., 0.)); self.goal_pose=np.zeros(3)
        self.cross_seed=cross_seed; self.variation_drift=variation_drift
        def body(): return types.SimpleNamespace(position=(0.,0.), angle=0., velocity=(0.,0.), angular_velocity=0.,
            force=(0.,0.), torque=0., body_type=0, mass=float('inf'), moment=1., center_of_gravity=(0.,0.))
        self.agent=body();self.block=body();self.goal_state=np.zeros(7);self.state=np.zeros(7)
    def reset(self, seed):
        if type(seed) is not int: raise TypeError('Gymnasium requires a python integer')
        self.seed=int(seed); self.state=np.full(7, float(seed))
        self.variation_space=types.SimpleNamespace(value={'agent': {'start_position':np.array([seed,seed]), 'scale':40+int(self.variation_drift)*seed},
            'block': {'start_position':np.array([seed,seed]), 'angle':float(seed)}, 'background':{'color':[255,255,255]}})
    def _set_state(self, state):
        self.state=np.asarray(state,dtype=np.float64).copy();self.state[0]+=.01 # legitimate official-style dt bias
        self.state[1]+=self.seed if self.cross_seed else 0
        self.agent.position=tuple(self.state[:2]);self.agent.velocity=tuple(self.state[-2:])
        self.block.position=tuple(self.state[2:4]);self.block.angle=self.state[4]
    def _set_goal_state(self, goal_state): self.goal_state=np.asarray(goal_state).copy()
    def _get_obs(self): return self.state.copy()
    def render(self): return np.full((4,4,3),int(self.state[0])%255,dtype=np.uint8)
    def step(self): self.state[0]+=.1; self.agent.position=tuple(self.state[:2])


class FakeWorld:
    def __init__(self, helpers, cross_seed=False, variation_drift=False):
        self.environment=FakeEnv(cross_seed,variation_drift);self.infos={};self.closed=False;self.policy=None
        self.envs=types.SimpleNamespace(envs=[types.SimpleNamespace(unwrapped=self.environment)],step=self.step)
        self.num_envs=1; self.terminateds=np.array([False]); self.reference=helpers['_evaluate_from_dataset']
    def set_policy(self, policy): self.policy=policy;policy.set_env(self.envs)
    def reset(self, seed):
        self.environment.reset(seed[0]);self.infos={'pixels':self.environment.render()[None,None],
            'state':self.environment.state[None,None].copy(),'goal':np.zeros((1,1,4,4,3),np.uint8)}
    def step(self, actions, **kwargs):
        self.environment.step();self.infos['pixels']=self.environment.render()[None,None]
        self.infos['state']=self.environment.state[None,None].copy()
        return None,np.array([-.1]),np.array([False]),np.array([False]),self.infos
    def _run(self, max_steps, mode, on_step):
        assert max_steps==1 and mode=='wait'
        self.envs.step(self.policy.get_action(self.infos));on_step(self)
    def evaluate(self, **kwargs):
        return self.reference(self, kwargs['dataset'], kwargs['episodes_idx'],kwargs['start_steps'],kwargs['goal_offset'],
                              kwargs['eval_budget'],kwargs['callables'],kwargs['video'],'wait')
    def close(self): self.closed=True


def synthetic_runner(cross_seed=False, variation_drift=False):
    official=real_helpers()
    helpers={'extract':official['_extract_init_goal'],'apply':official['_apply_callables'],
             'callables':[{'method':'_set_state','args':{'state':{'value':'state'}}},
                          {'method':'_set_goal_state','args':{'goal_state':{'value':'goal_state'}}}]}
    pixels=np.full((26,4,4,3),9,dtype=np.uint8);states=np.tile(np.array([1.,2,3,4,.1,0,0]),(26,1));states[-1,0]=4
    def run(c,seed,repeat,path):
        return v.one_trial(lambda:FakeWorld(official,cross_seed,variation_drift),helpers,c,pixels,states,seed,repeat,path)
    return run


class ResetFallbackTests(unittest.TestCase):
    def test_complete_fixed_matrix_original_world_method_and_wrapper(self):
        result, arrays=v.evaluate_matrix([case(0),case(1)],synthetic_runner())
        self.assertEqual(result['status'],'PASS_TECH_RESET_FALLBACK');self.assertTrue(result['routing_eligible'])
        self.assertEqual(result['counts']['resets'],16);self.assertEqual(result['counts']['raw_steps'],16)
        self.assertEqual(result['counts']['CEM_calls'],0);self.assertEqual(len(result['comparisons']),14)
        self.assertEqual(len(arrays),16)
        self.assertEqual(result['task'],'pusht')
        self.assertTrue(all(x['reset_seed_type_compatibility']['identity']==SEED_COMPATIBILITY for x in result['trials']))
        self.assertTrue(all(x['reset_seed_type_compatibility']['values_unchanged'] for x in result['trials']))
        self.assertTrue(all(x['reset_seed_type_compatibility']['input_type']=='ndarray' for x in result['trials']))
        self.assertTrue(all(x['source_state_bias']['max_abs_difference']>0 for x in result['trials']))
        self.assertTrue(all(x['source_pixel_bias']['max_abs_difference']>0 for x in result['trials']))

    def test_seed_compatibility_is_lossless_flat_and_preserves_missing(self):
        for seed in (0, 2**32-1, np.int64(0), np.uint64(2**32-1), np.array(7, dtype=np.int64)):
            actual=normalize_reset_seed(seed)
            self.assertIs(type(actual),int);self.assertEqual(actual,int(seed))
        self.assertIsNone(normalize_reset_seed(None))
        self.assertEqual(normalize_reset_seed((np.int64(0),None,np.uint32(1))),[0,None,1])
        self.assertTrue(all(type(x) is int for x in normalize_reset_seed(np.array([0,1],dtype=np.int64))))
        for seed in (True,np.bool_(False),0.,np.float64(1),[1.],[[1]],np.array([[1]]),'0'):
            with self.subTest(seed=repr(seed)),self.assertRaises(TypeError):normalize_reset_seed(seed)
        for seed in (-1,2**32,np.int64(-1),[2**32]):
            with self.subTest(seed=repr(seed)),self.assertRaises(ValueError):normalize_reset_seed(seed)

    def test_seed_dependent_physics_blocks_without_relaxing_tolerance(self):
        result,_=v.evaluate_matrix([case(0),case(1)],synthetic_runner(cross_seed=True))
        self.assertEqual(result['status'],'BLOCKED_RESET_FALLBACK');self.assertFalse(result['routing_eligible'])
        self.assertIsNone(result['candidate_reset_seed'])
        self.assertTrue(any(x['status']=='BLOCKED' for x in result['comparisons']))

    def test_non_overridden_variation_drift_blocks_even_matching_states(self):
        result,_=v.evaluate_matrix([case(0),case(1)],synthetic_runner(variation_drift=True))
        self.assertEqual(result['status'],'BLOCKED_RESET_FALLBACK')
        self.assertTrue(any(not x['non_overridden_variations_and_physics_exact'] for x in result['comparisons']))

    def test_no_selection_by_failure_and_incomplete_matrix_blocks(self):
        calls=[];actual=synthetic_runner()
        def run(c,seed,repeat,path):
            calls.append((c['case_id'],seed,repeat,path));row,values=actual(c,seed,repeat,path)
            if len(calls)==1:row['status']='BLOCKED_EXCEPTION'
            return row,values
        result,_=v.evaluate_matrix([case(0),case(1)],run)
        self.assertEqual(len(calls),16);self.assertEqual(result['status'],'BLOCKED_RESET_FALLBACK')

    def test_missing_seed_contract_and_first_two_tech(self):
        cases=[case(0),case(1),case(2)]
        rows=[{**c,'planning_starts':[4]} for c in cases]
        roles={'task':'pusht','status':'METADATA_ROLES_FROZEN','cases':{'TECH':cases},'episodes':rows}
        self.assertEqual(v.selected_cases(roles),cases[:2])
        roles['cases']['TECH'][0]['reset_seed']=0
        with self.assertRaises(ValueError):v.selected_cases(roles)
        roles['cases']['TECH'][0]['reset_seed']=None;roles['episodes'][0]['role']='EVAL'
        with self.assertRaises(PermissionError):v.selected_cases(roles)

    def test_dataset_cannot_read_other_case_or_reuse(self):
        c=case();dataset=v.FixedCaseDataset(c,np.zeros((26,4,4,3),np.uint8),np.zeros((26,7)),0)
        with self.assertRaises(PermissionError):dataset.load_chunk([0],[5],[30])
        result=dataset.load_chunk([0],[4],[30]);self.assertEqual(set(result[0]),{'pixels','state','seed'})
        with self.assertRaises(RuntimeError):dataset.load_chunk([0],[4],[30])

    def test_verifier_checks_identity_files_and_never_runs_environment(self):
        result,arrays=v.evaluate_matrix([case(0),case(1)],synthetic_runner())
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            for key,values in arrays.items():np.savez(folder/('trial_'+key+'.npz'),**values)
            for row in result['trials']:(folder/('trial_'+row['trial_id']+'.json')).write_text(json.dumps(v.plain(row)))
            for c in (case(0),case(1)):np.savez(folder/(c['case_id']+'_source.npz'),state=np.zeros((26,7)))
            (folder/'PROGRESS.json').write_text('{}')
            result.update(roles_sha256='b'*64,source_files={'a'*64:{'path':'absent_real_source_not_required'}},
                          cases=[case(0),case(1)],files={p.name:{'bytes':p.stat().st_size,'sha256':v.common.sha256(p)} for p in folder.iterdir()})
            path=folder/'RESET_VALIDATION_RECEIPT.json';path.write_text(json.dumps(v.plain(result)))
            self.assertTrue(v.verify_receipt(path,roles_sha256='b'*64,source_asset_sha256='a'*64)['routing_eligible'])
            with self.assertRaises(RuntimeError):v.verify_receipt(path,roles_sha256='c'*64,source_asset_sha256='a'*64)
            (folder/'PROGRESS.json').write_text('different')
            with self.assertRaises(RuntimeError):v.verify_receipt(path,roles_sha256='b'*64,source_asset_sha256='a'*64)


if __name__=='__main__':unittest.main(verbosity=2)
