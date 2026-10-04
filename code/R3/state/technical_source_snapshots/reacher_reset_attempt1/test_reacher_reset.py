"""Synthetic only: pinned World method AST with fake physics, never a simulator."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
from scripts import validate_reacher_reset as v
from tests.test_reset_fallback import real_helpers


def case(i=0):
    return {'case_id':'reacher_tech_'+str(i),'episode_id':'ep'+str(i),'source_episode_idx':i,
            'start_raw_index':4,'goal_raw_index':29,'role':'TECH','task':'reacher',
            'source_asset_sha256':'a'*64,'episode_sha256':str(i)*64,'length':80,
            'source_seed':None,'reset_seed':None}


class FakeEnv:
    _task_name='qpos_match';action_repeat=2;camera_id=0
    def __init__(self, drift=None):
        self.drift=drift;self._cumulative_reward=0.
        self.action_space=types.SimpleNamespace(low=np.full(2,-1.),high=np.ones(2))
        model=types.SimpleNamespace(**{k:np.zeros(2) for k in v.MODEL_FIELDS})
        model.ngeom=2;model.name2id=lambda name,kind:1
        model.geom_matid=np.array([0,1]);model.mat_rgba=np.array([[1.,1.,1.,1.],[1.,0.,0.,0.]])
        model.geom_pos=np.zeros((2,3))
        model.opt=types.SimpleNamespace(timestep=.01,gravity=np.zeros(3),integrator=0,solver=0,iterations=100,tolerance=1e-8)
        data=types.SimpleNamespace(**{k:np.zeros(2) for k in v.DATA_FIELDS})
        data.time=np.asarray(0.);data.act=np.empty(0);data.mocap_pos=np.empty((0,3));data.mocap_quat=np.empty((0,4));data.userdata=np.empty(0)
        data.geom_xpos=np.zeros((2,3))
        physics=types.SimpleNamespace(model=model,data=data)
        physics.get_state=lambda:np.concatenate([data.qpos,data.qvel,data.act])
        task=types.SimpleNamespace(target_qpos=np.zeros(2),qpos_threshold=.05)
        task.get_observation=lambda p:{'position':p.data.qpos.copy(),'velocity':p.data.qvel.copy(),'to_target':p.model.geom_pos[1,:2]-p.data.qpos}
        task.get_termination=lambda p:0. if np.all(np.abs(p.data.qpos-task.target_qpos)<task.qpos_threshold) else None
        self.env=types.SimpleNamespace(physics=physics,task=task)
        self.variation_space=types.SimpleNamespace(value={'rendering':{'render_target':0},'agent':{'density':1000}})
    def reset(self, seed):
        if type(seed) is not int:raise TypeError('Python int required')
        self.seed=seed;self.env.physics.model.geom_pos[1]=[seed+.2,seed+.3,0.]
        self.env.physics.data.geom_xpos[:]=self.env.physics.model.geom_pos
        if self.drift=='hidden':self.env.physics.data.ctrl[0]=seed
        if self.drift=='visible':self.env.physics.model.mat_rgba[1,3]=1.
        if self.drift=='collision':self.env.physics.model.geom_contype[1]=1
    def set_state(self,qpos,qvel):
        self.env.physics.data.qpos[:]=qpos;self.env.physics.data.qvel[:]=qvel
        if self.drift=='qvel':self.env.physics.data.qvel[0]+=self.seed
    def set_target_qpos(self,target_qpos):self.env.task.target_qpos=np.asarray(target_qpos).copy()
    def render(self):return np.full((4,4,3),int(self.env.physics.data.qpos[0]*10)%255,np.uint8)
    def step(self):
        self.env.physics.data.qpos+=.001;self.env.physics.data.time+=.02
        reward=float(self.seed)+.25;self._cumulative_reward+=reward
        return reward


class FakeWorld:
    def __init__(self, official, drift=None):
        self.environment=FakeEnv(drift);self.infos={};self.policy=None;self.closed=False;self.num_envs=1
        self.envs=types.SimpleNamespace(envs=[types.SimpleNamespace(unwrapped=self.environment)],step=self.step)
        self.reference=official['_evaluate_from_dataset'];self.terminateds=np.array([False])
    def set_policy(self,p):self.policy=p;p.set_env(self.envs)
    def reset(self,seed):
        self.environment.reset(seed[0]);d=self.environment.env.physics.data
        self.infos={'pixels':self.environment.render()[None,None],'qpos':d.qpos[None,None].copy(),'qvel':d.qvel[None,None].copy()}
    def step(self,action,**kwargs):
        reward=self.environment.step();d=self.environment.env.physics.data
        self.infos.update(pixels=self.environment.render()[None,None],qpos=d.qpos[None,None].copy(),qvel=d.qvel[None,None].copy())
        self.terminateds=np.array([self.environment.env.task.get_termination(self.environment.env.physics) is not None])
        return None,np.array([reward]),self.terminateds,np.array([False]),self.infos
    def _run(self,max_steps,mode,on_step):
        assert max_steps==1 and mode=='wait';self.envs.step(self.policy.get_action(self.infos));on_step(self)
    def evaluate(self,**kw):
        return self.reference(self,kw['dataset'],kw['episodes_idx'],kw['start_steps'],kw['goal_offset'],kw['eval_budget'],kw['callables'],kw['video'],'wait')
    def close(self):self.closed=True


def runner(drift=None):
    official=real_helpers()
    helpers={'extract':official['_extract_init_goal'],'apply':official['_apply_callables'],
             'callables':[{'method':'set_state','args':{'qpos':{'value':'qpos'},'qvel':{'value':'qvel'}}},
                          {'method':'set_target_qpos','args':{'target_qpos':{'value':'goal_qpos'}}}]}
    pixels=np.full((26,4,4,3),9,np.uint8);qpos=np.tile([.1,.2],(26,1));qvel=np.zeros((26,2));qpos[-1]=[.5,.6]
    return lambda c,s,r,p:v.one_trial(lambda:FakeWorld(official,drift),helpers,c,pixels,qpos,qvel,s,r,p)


class ReacherResetTests(unittest.TestCase):
    def test_fixed_matrix_all_required_effective_fields_exact_but_point_reward_differs(self):
        result,arrays=v.evaluate_matrix([case(0),case(1)],runner())
        self.assertEqual(result['status'],'PASS_TECH_RESET_FALLBACK',str(result['trials'][0]))
        self.assertEqual(result['comparison_class_counts'],{'SAME_SEED_ALL_FIELDS_EXACT':12,'CROSS_SEED_TASK_FIELDS_EXACT':2})
        self.assertEqual(result['counts']['resets'],16);self.assertEqual(result['counts']['raw_steps'],16)
        self.assertEqual(result['counts']['CEM_calls'],0);self.assertEqual(result['counts']['optimizer_updates'],0)
        self.assertTrue(all(t['close_called'] for t in result['trials']))
        self.assertNotEqual(arrays['00']['point_task_diagnostic__step_reward'].tolist(),arrays['04']['point_task_diagnostic__step_reward'].tolist())
        self.assertTrue(all(t['source_pixel_bias']['max_abs_difference']>0 for t in result['trials']))
        for comp in result['comparisons']:
            d=comp['array_differences']['point_task_diagnostic__step_reward']
            self.assertEqual(d['hard_gate'],comp['comparison_class']=='SAME_SEED_ALL_FIELDS_EXACT')

    def test_same_seed_reward_difference_is_hard_failure_including_seed1(self):
        actual=runner()
        def changed(c,s,r,p):
            row,arrays=actual(c,s,r,p)
            if s==1 and r==1 and p==v.PATHS[1]:arrays['point_task_diagnostic__step_reward']+=.1
            return row,arrays
        result,_=v.evaluate_matrix([case(0),case(1)],changed)
        self.assertEqual(result['status'],'BLOCKED_RESET_FALLBACK')
        self.assertTrue(any(c['comparison_class']=='SAME_SEED_ALL_FIELDS_EXACT' and c['status']=='BLOCKED' for c in result['comparisons']))

    def test_cross_seed_joint_or_hidden_control_drift_blocked(self):
        for drift in ('qvel','hidden'):
            with self.subTest(drift=drift):
                result,_=v.evaluate_matrix([case(0),case(1)],runner(drift))
                self.assertEqual(result['status'],'BLOCKED_RESET_FALLBACK')
                self.assertTrue(any(c['comparison_class']=='CROSS_SEED_TASK_FIELDS_EXACT' and c['status']=='BLOCKED' for c in result['comparisons']))

    def test_visible_or_colliding_point_target_is_blocked_before_zero_step(self):
        for drift in ('visible','collision'):
            with self.subTest(drift=drift):
                result,_=v.evaluate_matrix([case(0),case(1)],runner(drift))
                self.assertEqual(result['status'],'BLOCKED_RESET_FALLBACK');self.assertEqual(result['counts']['raw_steps'],0)
                self.assertEqual(len(result['trials']),16)

    def test_nonfinite_point_diagnostic_is_not_silently_ignored(self):
        actual=runner()
        def changed(c,s,r,p):
            row,arrays=actual(c,s,r,p)
            if s==1:arrays['point_task_diagnostic__step_reward'][:]=np.nan
            return row,arrays
        result,_=v.evaluate_matrix([case(0),case(1)],changed)
        self.assertEqual(result['status'],'BLOCKED_RESET_FALLBACK')

    def test_fixed_first_two_tech_and_missing_seed_only(self):
        cases=[case(i) for i in range(3)];roles={'task':'reacher','status':'METADATA_ROLES_FROZEN','cases':{'TECH':cases},'episodes':[{**c,'planning_starts':[4]} for c in cases]}
        self.assertEqual(v.selected_cases(roles),cases[:2]);roles['cases']['TECH'][0]['reset_seed']=0
        with self.assertRaises(ValueError):v.selected_cases(roles)

    def test_dataset_no_actions_and_exact_window_read_once(self):
        d=v.FixedCaseDataset(case(),np.zeros((26,4,4,3),np.uint8),np.zeros((26,2)),np.zeros((26,2)),0)
        with self.assertRaises(PermissionError):d.load_chunk([0],[5],[30])
        self.assertEqual(set(d.load_chunk([0],[4],[30])[0]),{'pixels','qpos','qvel','seed'})
        with self.assertRaises(RuntimeError):d.load_chunk([0],[4],[30])

    def test_receipt_is_recoverable_and_bound_to_classified_matrix(self):
        result,arrays=v.evaluate_matrix([case(0),case(1)],runner())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for key,values in arrays.items():np.savez(root/('trial_'+key+'.npz'),**values)
            for row in result['trials']:(root/('trial_'+row['trial_id']+'.json')).write_text(json.dumps(v.plain(row)))
            for c in (case(0),case(1)):np.savez(root/(c['case_id']+'_source.npz'),qpos=np.zeros((26,2)))
            (root/'PROGRESS.json').write_text('{}')
            result.update(cases=[case(0),case(1)],roles_sha256='b'*64,source_files={'a'*64:{}},files={p.name:{'bytes':p.stat().st_size,'sha256':v.common.sha256(p)} for p in root.iterdir()})
            path=root/'RESET_VALIDATION_RECEIPT.json';path.write_text(json.dumps(v.plain(result)))
            self.assertTrue(v.verify_receipt(path,roles_sha256='b'*64,source_asset_sha256='a'*64)['routing_eligible'])
            original=copy.deepcopy(result)
            result['comparisons'][-1]['label']['reference_trial']='09';path.write_text(json.dumps(v.plain(result)))
            with self.assertRaises(RuntimeError):v.verify_receipt(path,roles_sha256='b'*64,source_asset_sha256='a'*64)
            result=original
            result['comparisons'][-1]['comparison_class']='SAME_SEED_ALL_FIELDS_EXACT';path.write_text(json.dumps(v.plain(result)))
            with self.assertRaises(RuntimeError):v.verify_receipt(path,roles_sha256='b'*64,source_asset_sha256='a'*64)


if __name__=='__main__':unittest.main(verbosity=2)
