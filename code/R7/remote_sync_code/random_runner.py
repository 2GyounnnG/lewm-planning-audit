from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

SEEDS=(7001,7002,7003)

class R3Raw:
    def __init__(self, path, task):
        import h5py, hdf5plugin
        import torch
        self.file=h5py.File(path,'r',swmr=True,rdcc_nbytes=512*1024**2); self.task=task
        self.lengths=np.asarray(self.file['ep_len'][:],dtype=np.int64); self.offsets=np.asarray(self.file['ep_offset'][:],dtype=np.int64)
        self.column_names=['pixels','state'] if task=='pusht' else ['pixels','qpos','qvel']
    def load_chunk(self, episodes, starts, ends):
        import torch
        out=[]
        for ep,s,e in zip(episodes,starts,ends):
            o=int(self.offsets[int(ep)]); d={}
            for k in self.column_names:
                a=np.asarray(self.file[k][o+int(s):o+int(e)])
                if k=='pixels' and a.shape[-1]==3: a=a.transpose(0,3,1,2)
                d[k]=torch.from_numpy(a.copy())
            out.append(d)
        return out
    def close(self): self.file.close()

def record_row(task, case, seed, success, steps, error=None):
    return {'task':task,'case_id':case['case_id'],'seed':int(seed),'success':int(success),'executed_raw_steps':int(steps),'error':error}

def run_r3(task, root, out):
    os.environ['R3_ROOT']=str(root);sys.path.insert(0,str(root))
    import torch
    from r3 import planning as p
    from r3.env_compat import normalize_reset_seed
    from r3.data import image_transform
    from r4.export_cases import CaseWindows
    from r3.model import load_official
    p_api = p.load_official_api(); import stable_worldmodel.policy as swmp
    roles=json.loads((root/'manifests'/f'{task}_data_roles.json').read_text()); source=json.loads((root/'manifests'/f'{task}_source_map.json').read_text())
    asset=next(iter(source['assets'].values())); raw=R3Raw(root/asset['path'],task); rows=[]
    tmp=Path(out)/'assets'; tmp.mkdir(parents=True,exist_ok=True)
    # Export the exact R3 EVAL 26-row windows once, with no model use.
    import h5py
    for case in roles['cases']['EVAL']:
        a=source['assets'][case['source_asset_sha256']]; h=raw.file; o=int(raw.offsets[case['source_episode_idx']]); s=int(case['start_raw_index']); e=int(case['goal_raw_index'])+1
        keys=['pixels','action','state'] if task=='pusht' else ['pixels','action','qpos','qvel']; arr={k:np.asarray(h[k][o+s:o+e]) for k in keys}
        if arr['pixels'].shape[-1]==3: arr['pixels']=arr['pixels'].transpose(0,3,1,2)
        np.savez_compressed(tmp/(case['case_id']+'.npz'),**arr,source_start_raw=np.int64(s),source_episode_idx=np.int64(case['source_episode_idx']))
    class View:
        def __init__(self, path, case):
            with np.load(path,allow_pickle=False) as f:self.arrays={k:f[k].copy() for k in f.files}
            self.column_names=[k for k in self.arrays if not k.startswith('source_')]+['seed'];self.case=case;self.goal_state=None;self.reads=0
        def load_chunk(self,episodes,starts,ends):
            import torch
            if list(episodes)!=[self.case['source_episode_idx']] or list(starts)!=[self.case['start_raw_index']] or list(ends)!=[self.case['goal_raw_index']+1]:raise RuntimeError('random view escaped frozen slice')
            lo=int(starts[0])-int(self.arrays['source_start_raw']);hi=int(ends[0])-int(self.arrays['source_start_raw'])
            d={k:torch.from_numpy(v[lo:hi].copy()) for k,v in self.arrays.items() if k in self.column_names}; reset_seed=self.case.get('reset_seed')
            if reset_seed is None: reset_seed=0
            d['seed']=torch.full((hi-lo,),int(reset_seed),dtype=torch.int64); self.goal_state=np.asarray(d['state' if task=='pusht' else 'qpos'][-1]).copy();self.reads+=1;return [d]
    for case in roles['cases']['EVAL']:
        for seed in SEEDS:
            world=None
            try:
                kwargs={'task':'qpos_match'} if task=='reacher' else {}
                world=p_api[0].World(env_name='swm/PushT-v1' if task=='pusht' else 'swm/ReacherDMControl-v0',num_envs=1,max_episode_steps=100,image_shape=(224,224),**kwargs)
                original=world.reset;world.reset=lambda seed=None,options=None:original(seed=normalize_reset_seed(seed),options=options)
                pol=swmp.RandomPolicy();world.set_policy(pol);pol.set_seed(seed)
                view=View(tmp/(case['case_id']+'.npz'),case)
                metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=25,eval_budget=50,callables=p.CALLABLES[task],video=None)
                rows.append(record_row(task,case,seed,bool(metrics['episode_successes'][0]),50))
            except BaseException as exc: rows.append(record_row(task,case,seed,0,0,f'{type(exc).__name__}:{exc}'))
            finally:
                if world is not None: world.close()
    raw=Path(out)/f'{task}_RANDOM_RAW_VALUES.csv'; write_csv(raw,rows)
    return rows

def run_x1(task, root, out):
    os.environ['X1_ROOT']=str(root);os.environ['X1_THREADS']='1';sys.path[:0]=['/workspace/r4_v23_execution','/workspace/r5/code']
    if task=='cube': sys.path.insert(0,'/workspace/x1_cube/ogbench_only')
    import x1.core as core, x1.data as data
    import x1.evaluate as ev
    core.r3('planning').load_official_api(); import stable_worldmodel.policy as swmp
    roles=core.read(root/'manifests'/f'{task}_data_roles.json'); rows=[]
    if task=='cube':
        c1=core.read(Path('/workspace/r5/C1/C1_CONTRACT.json')); cases=c1['cases']['EVAL']; raw=data.RawH5(c1['source']['path']);
        from c1.runner import CaseView
        from c0.reset import clear_internal
        for case in cases:
            for seed in SEEDS:
                world=None
                try:
                    planning=core.r3('planning'); swm=planning.load_official_api()[0]; cfg=core.CONFIG['cube']
                    world=swm.World(**cfg['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224));orig=world.reset
                    def reset(seed=None,options=None):
                        out=orig(seed=None,options=options); clear_internal(world.envs.envs[0].unwrapped,__import__('mujoco')); return out
                    world.reset=reset
                    pol=swmp.RandomPolicy(); world.set_policy(pol); pol.set_seed(int(seed)); view=CaseView(raw,case)
                    metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=25,eval_budget=50,callables=cfg['callables'],video=None)
                    rows.append(record_row(task,case,seed,bool(metrics['episode_successes'][0]),50))
                except BaseException as exc: rows.append(record_row(task,case,seed,0,0,f'{type(exc).__name__}:{exc}'))
                finally:
                    if world is not None:
                        r=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
                        if r is not None:r.close()
                        world.close()
        raw.close(); return write_csv(Path(out)/f'{task}_RANDOM_RAW_VALUES.csv',rows)
    cases=roles['cases']['EVAL']; raw=data.RawH5(core.verify(roles['source']));
    from x1.reset import effective_case
    for case0 in cases:
        case=dict(effective_case(task,case0))
        if case.get('reset_seed') is not None: case['reset_seed']=int(case['reset_seed'])
        for seed in SEEDS:
            world=None
            try:
                planning=core.r3('planning'); swm=planning.load_official_api()[0]; cfg=core.CONFIG[task]
                world=swm.World(**cfg['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224)); original=world.reset
                world.reset=lambda seed=None,options=None: original(seed=None if seed is None else int(seed),options=options)
                pol=swmp.RandomPolicy();world.set_policy(pol);pol.set_seed(int(seed));view=ev.CaseView(raw,task,case)
                metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=25,eval_budget=50,callables=cfg['callables'],video=None)
                rows.append(record_row(task,case,seed,bool(metrics['episode_successes'][0]),50))
            except BaseException as exc: rows.append(record_row(task,case,seed,0,0,f'{type(exc).__name__}:{exc}'))
            finally:
                if world is not None: world.close()
    raw.close(); return write_csv(Path(out)/f'{task}_RANDOM_RAW_VALUES.csv',rows)

def write_csv(path, rows):
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['reacher','pusht','cube','tworoom'],required=True);parser.add_argument('--out',required=True);parser.add_argument('--r3-root',default='/workspace/shared_data/r3');parser.add_argument('--x1-root',default='/workspace/x1_tworoom');args=parser.parse_args();
    rows=run_r3(args.task,Path(args.r3_root),Path(args.out)) if args.task in ('reacher','pusht') else run_x1(args.task,Path('/workspace/x1_cube' if args.task=='cube' else '/workspace/x1_tworoom'),Path(args.out));print(json.dumps({'task':args.task,'rows':len(rows),'seeds':list(SEEDS)}))
