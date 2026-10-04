"""CPU semantic tests; synthetic arrays test adapters, never scientific evidence."""
from __future__ import annotations
import ast, json, tempfile, types, unittest
import numpy as np
import torch
from . import core,data,evaluate

class Tiny(torch.nn.Module):
    def __init__(self,raw_dim):
        super().__init__();self.encoder=torch.nn.Linear(4,4);self.projector=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4))
        self.action_encoder=torch.nn.Linear(raw_dim*5,4);self.predictor=torch.nn.Linear(8,4)
        self.pred_proj=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4))
        self.r3_contract={'history_size':3,'latent_dim':4,'macro_action_dim':raw_dim*5}
    def predict(self,z,a):
        shape=z.shape;v=self.predictor(torch.cat((z,a),-1));return self.pred_proj(v.flatten(0,1)).reshape(shape)

def source_function(path,cls,name,namespace):
    tree=ast.parse(path.read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    fn=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace);return namespace[name]

class Semantics(unittest.TestCase):
    def test_source_contract(self):
        for task in core.TASKS:
            c=core.task_contract(task);self.assertEqual(c['executed_raw_per_replan'],25);self.assertEqual(c['cem']['num_samples'],300)
    def test_two_and_five_dimension_windows(self):
        w=core.r3('train_worker')
        for raw_dim in (2,5):
            z=np.arange(50*4,dtype=np.float32).reshape(50,4);a=np.arange(50*raw_dim,dtype=np.float32).reshape(50,raw_dim)
            item={'z':z,'actions':a,'raw_indices':np.arange(50),'stride':5,'legal_starts':data.legal_windows(a,50)}
            adapter=w.WindowAdapter({'ep':item},{'history_size':3,'latent_dim':4,'macro_action_dim':5*raw_dim},allowed_ids=['ep'],role='REFIT_TRAIN')
            zz,aa=adapter.batch([['ep',3]],'cpu',5)
            np.testing.assert_array_equal(zz[0].numpy(),z[3+5*np.arange(8)])
            np.testing.assert_array_equal(aa[0].numpy().reshape(35,raw_dim),a[3:38])
            sampler=w.WindowSampler(adapter,103201);first=sampler.next(advance=False)[1];self.assertEqual(first,sampler.next()[1])
            with self.assertRaises(PermissionError):w.WindowAdapter({'ep':item},adapter.__dict__,allowed_ids=['ep'],role='EVAL')
    def test_future_is_not_teacher_forced(self):
        m=core.r3('model');model=Tiny(5);m.refit_mode(model,False);z=torch.randn(2,3,4);a=torch.randn(2,7,25)
        pred=m.cached_rollout(model,z,a,5)
        self.assertEqual(tuple(pred.shape),(2,5,4));changed=a.clone();changed[:,3:]+=100
        pred2=m.cached_rollout(model,z,changed,5);torch.testing.assert_close(pred[:,0],pred2[:,0]);self.assertFalse(torch.equal(pred[:,-1],pred2[:,-1]))
    def test_train_whitelist_and_bn_buffers(self):
        m=core.r3('model');w=core.r3('train_worker');model=Tiny(5);before=m.frozen_hashes(model);m.refit_mode(model,True)
        old={n:p.clone() for n,p in model.named_parameters() if n.startswith(m.TRAINABLE_PREFIXES)}
        optimizer=torch.optim.AdamW(m.trainable_parameters(model),lr=w.lr_at(500),betas=(.9,.999),eps=1e-8,weight_decay=.001,foreach=False)
        z=torch.randn(128,3,4);a=torch.randn(128,3,25);target=torch.randn(128,3,4)
        for _ in range(3):
            optimizer.zero_grad();loss=(m.cached_predict(model,z,a)-target).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.trainable_parameters(model),1.);optimizer.step()
        self.assertEqual(before,m.assert_frozen(model,before));self.assertTrue(any(not torch.equal(old[n],p) for n,p in model.named_parameters() if n in old))
        self.assertAlmostEqual(w.lr_at(1),1e-7,places=15);self.assertEqual(w.lr_at(500),5e-5);self.assertEqual(w.lr_at(30000),5e-6)
    def test_source_success_boundaries(self):
        path=core.R3_ROOT/'source/swm_compat/stable_worldmodel/envs/two_room/env.py'
        step=source_function(path,'TwoRoomEnv','step',{'torch':torch,'np':np})
        env=types.SimpleNamespace(agent_position=torch.tensor([0.,0.]),target_position=torch.tensor([16.,0.]),
            variation_space={'agent':{'speed':types.SimpleNamespace(value=torch.tensor(1.))}},
            _apply_collisions=lambda old,new:new,_get_obs=lambda:None,_get_info=lambda:{})
        self.assertFalse(step(env,np.array([0.,0.]))[2]);self.assertTrue(step(env,np.array([.01,0.]))[2])
        path=core.R3_ROOT/'source/swm_compat/stable_worldmodel/envs/ogbench/cube_env.py'
        fn=source_function(path,'CubeEnv','_compute_successes',{'np':np});position=np.array([.04,0.,0.])
        cube=types.SimpleNamespace(_num_cubes=1,_cube_target_mocap_ids=[0],_data=types.SimpleNamespace(
            joint=lambda _:types.SimpleNamespace(qpos=position),mocap_pos=np.zeros((1,3))))
        self.assertEqual(fn(cube),[True]);position[0]=.040001;self.assertEqual(fn(cube),[False])
    def test_case_stream_preserves_r3(self):
        import hashlib
        for task in core.TASKS:
            expected=int.from_bytes(hashlib.sha256(f'R3_CEM_CASE_20261002/{task}/case/0'.encode()).digest()[:8],'big')
            self.assertEqual(evaluate.case_seed(task,'case',0),expected)
            self.assertNotEqual(expected,evaluate.case_seed(task,'case',0,'R4_ALT_CEM_1'))
    def test_roles_match_r3_rules(self):
        roles=core.r3('roles');old=roles.TASKS
        records=[{'episode_id':str(i),'source_episode_idx':i,'source_asset_sha256':'a'*64,'episode_sha256':format(i,'064x'),
            'length':100,'family_id':None,'planning_starts':list(range(75)),'open_loop_starts':list(range(10,71)),'source_seed':i} for i in range(600)]
        try:
            roles.TASKS=core.TASKS;r=roles.build_roles('cube',records,history_size=3,frameskip=5)
        finally:roles.TASKS=old
        self.assertEqual(len(r['cases']['TECH']),4);self.assertEqual(len(r['cases']['EVAL']),100)
        for c in r['cases']['EVAL']:self.assertEqual(c['goal_raw_index'],c['start_raw_index']+25)
        groups={role:{e['episode_id'] for e in r['episodes'] if e['role']==role} for role in ('REFIT_TRAIN','MONITOR','EVAL','TECH')}
        for a in groups:
            for b in groups:
                if a!=b:self.assertFalse(groups[a]&groups[b])
    def test_two_gpu_cache_merge_exact_roster_and_moments(self):
        from pathlib import Path
        old=core.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                core.ROOT=Path(tmp);identity={'shards':2};rows=[]
                for i,values in enumerate(([0.,2.],[4.])):
                    path=core.ROOT/f'{i}.npz';z=np.asarray(values,dtype=np.float32)[:,None];data.save_npz(path,z=z)
                    eid=f'ep{i}';rows.append({'episode_id':eid,'role':'REFIT_TRAIN'})
                    core.atomic(core.ROOT/'manifests'/f'tworoom_cache_part_{i}.json',{'status':'FROZEN_OBSERVED_CACHE_SHARD_COMPLETE',
                        'shard':i,'identity':identity,'frozen':{'sha256':'fixed'},'episodes':{eid:{**core.file_record(path),'length':len(z),'role':'REFIT_TRAIN'}},
                        'train_moments':[float(z.sum())],'train_squared_moments':[float((z*z).sum())],'train_variance_frames':len(z),'seconds':i+1})
                core.atomic(core.ROOT/'manifests/tworoom_data_roles.json',{'episodes':rows})
                out=data.merge_cache('tworoom');self.assertEqual(out['frames'],3)
                cm=core.read(core.ROOT/'manifests/tworoom_cache.json');self.assertAlmostEqual(cm['train_scalar_variance'],8/3)
                rows.append({'episode_id':'missing','role':'TECH'});core.atomic(core.ROOT/'manifests/tworoom_data_roles.json',{'episodes':rows})
                with self.assertRaises(RuntimeError):data.merge_cache('tworoom')
        finally:core.ROOT=old

if __name__=='__main__':unittest.main()
