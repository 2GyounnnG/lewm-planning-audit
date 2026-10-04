from __future__ import annotations
import argparse, datetime, hashlib, os, socket, time, traceback, shutil
from pathlib import Path
import numpy as np
from .common import *
from c0.reset import clear_internal,snapshot

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def model_digest(model):
    from x1 import core
    tensor_sha=core.r3('model').tensor_sha256
    return digest({k:tensor_sha(v) for k,v in model.state_dict().items()})

def run_case(contract,model,raw,case,arm,phase,device='cuda'):
    import fcntl
    folder=ROOT/'closed_loop'/phase/'R3_ORIGINAL'/arm/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
    with (folder/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_case(contract,model,raw,case,arm,phase,device)

class CaseView:
    def __init__(self,raw,case):
        from x1 import core
        self.raw,self.case=raw,case;self.reads=0
        self.column_names=['pixels']+core.CONFIG['cube']['reset_keys'];raw.column_names=self.column_names
        self.goal_state=None
    def load_chunk(self,episodes,starts,ends):
        c=self.case
        if list(episodes)!=[c['source_episode_idx']] or list(starts)!=[c['start_raw_index']] or list(ends)!=[c['goal_raw_index']+1]:raise RuntimeError('Escaped frozen source slice')
        out=self.raw.load_chunk(episodes,starts,ends)[0]
        if set(out)!=set(self.column_names) or any(len(v)!=26 for v in out.values()):raise RuntimeError('Source schema differs')
        self.reads+=1
        if self.reads!=1:raise RuntimeError('Unexpected reread')
        self.goal_state=np.asarray(out['privileged_block_0_pos'][-1],dtype=np.float64).copy()
        return [out]

def _run_case(contract,model,raw,case,arm,phase,device='cuda'):
    import torch,mujoco as mj
    from x1 import core,data
    from x1.evaluate import case_seed
    from r4.runner import encode_pixels
    planning=core.r3('planning');swm,solver_type,policy_type,source=planning.load_official_api()
    folder=ROOT/'closed_loop'/phase/'R3_ORIGINAL'/arm/case['case_id']
    identity={'contract':sha(ROOT/'C1_CONTRACT.json'),'case':case,'arm':arm,'phase':phase,'stream':'R3_ORIGINAL'};bind=digest(identity)
    complete=folder/'COMPLETE.json'
    if complete.exists():
        receipt=read(complete)
        if receipt['identity_sha256']!=bind:raise RuntimeError('Completed identity differs')
        for r in receipt['files'].values():verify(r)
        return read(folder/'result.json')
    if (folder/'STARTED.json').exists():
        old=read(folder/'STARTED.json')
        if old['identity_sha256']!=bind:raise RuntimeError('Partial identity differs')
        attempts=folder/'attempts';attempts.mkdir(exist_ok=True);dest=attempts/str(len(list(attempts.iterdir())))
        dest.mkdir();partial=[]
        for p in list(folder.iterdir()):
            if p.is_file() and p.name!='RUN.lock':partial.append(record(p));shutil.move(str(p),dest/p.name)
        atomic(dest/'INTERRUPTION.json',{'status':'UNSCORED_INFRASTRUCTURE_ATTEMPT','files_before_move':partial,'reason':'Previous process ended before atomic COMPLETE; same frozen identity retried','utc':utc()})
    atomic(folder/'STARTED.json',{'identity_sha256':bind,'identity':identity,'pid':os.getpid(),'utc':utc()})
    timer=planning._Timer(device);cost=planning.AuditedCost(model,timer);replans=[];planarrays={};rows=[];states={};pixels=[];world=None;metrics=None;failure=None;infra=None
    solver=solver_type(model=cost,device=device,seed=case_seed('cube',case['case_id'],0),**core.CEM)
    official_solve=solver.solve;transform=core.r3('data').image_transform();view=CaseView(raw,case)
    initial_comparison=[];counts=[];simphase='construction';original_mj_step=mj.mj_step;began=time.monotonic()
    def counted(m,d,*args,**kwargs):
        counts.append({'phase':simphase,'substeps':int(kwargs.get('nstep',args[0] if args else 1))});return original_mj_step(m,d,*args,**kwargs)
    mj.mj_step=counted
    def capture(index):
        ss=snapshot(world.envs.envs[0].unwrapped,mj)
        for k,v in ss.items():states[f'{index:03d}:{k}']=v
        return ss
    def solve(info,init_action=None):
        nonlocal simphase,initial_comparison
        i=len(replans);seed=case_seed('cube',case['case_id'],i);solver.torch_gen.manual_seed(seed)
        if i==0:
            first=capture(0);pixels.append(np.asarray(world.infos['pixels'][0,-1]).copy())
            if phase=='TECH':
                src=C0/'trials'/case['case_id']/'0/states.npz'
                with np.load(src) as f:expected={k[4:]:f[k].copy() for k in f.files if k.startswith('000:')}
                initial_comparison=comparisons(first,expected)
                csv_write(folder/'C0_INITIAL_EQUIVALENCE_RAW.csv',initial_comparison)
                if not all(r['bitwise_equal'] for r in initial_comparison):raise RuntimeError('C1 initial snapshot differs from sealed C0')
        planarrays[f'{i}:rng_before']=solver.torch_gen.get_state().cpu().numpy()
        for k in ('pixels','goal','action'):
            if k in info:
                v=info[k];planarrays[f'{i}:input:{k}']=v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v).copy()
        if init_action is not None:planarrays[f'{i}:init_action']=init_action.detach().cpu().numpy()
        timer.synchronize();start=time.monotonic();result=official_solve(info,init_action=init_action);timer.synchronize()
        if not torch.isfinite(result['actions']).all():raise planning.PlanningMethodFailure('NONFINITE_PLAN')
        actions=result['actions'].detach().cpu().numpy();planarrays[f'{i}:returned_plan']=actions
        planarrays[f'{i}:rng_after']=solver.torch_gen.get_state().cpu().numpy()
        replans.append({'replan_index':i,'seed_uint64':seed,'seconds':time.monotonic()-start,'observed_history_frames':int(info['pixels'].shape[1]),'returned_plan':actions.tolist(),'raw_index':len(rows)})
        atomic(folder/'PROGRESS.json',{'completed_replans':len(replans),'completed_raw_steps':len(rows),'utc':utc()})
        simphase='executed_raw'
        return result
    solver.solve=solve
    policy=policy_type(solver=solver,config=swm.PlanConfig(**core.PLAN),process={'action':data.processor('cube')},transform={'pixels':transform,'goal':transform})
    model_before=model_digest(model)
    try:
        world=swm.World(**core.CONFIG['cube']['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224))
        install_reset(world,mj,clear_internal);world.set_policy(planning.IsolatedPolicy(policy));original_step=world.envs.step
        if tuple(world.envs.single_action_space.shape)!=(5,):raise RuntimeError('Unexpected action dimension')
        def step(actions,*args,**kwargs):
            env=world.envs.envs[0].unwrapped
            if env._reset_next_step:raise RuntimeError('Automatic reset forbidden')
            start=time.monotonic();out=original_step(actions,*args,**kwargs);_,reward,terminated,truncated,infos=out
            physical=np.asarray(env._data.joint('object_joint_0').qpos[:3]).copy()
            rows.append({'action':np.asarray(actions[0]).copy(),'physical':physical,'terminated':bool(terminated[0]),'truncated':bool(truncated[0]),'reward':float(reward[0]),'error':float(np.linalg.norm(physical-view.goal_state)),'seconds':time.monotonic()-start})
            capture(len(rows));pixels.append(np.asarray(infos['pixels'][0,-1]).copy());return out
        world.envs.step=step;simphase='official_reset'
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=25,eval_budget=50,callables=core.CONFIG['cube']['callables'],video=None)
        if not 1<=len(rows)<=50 or not 1<=len(replans)<=2:raise RuntimeError('Official budget violated')
        if any(r['terminated'] or r['truncated'] for r in rows[:-1]):raise RuntimeError('Stepped after terminal')
        if bool(metrics['episode_successes'][0])!=any(r['terminated'] for r in rows):raise RuntimeError('Success events differ')
        if model_before!=model_digest(model):raise RuntimeError('Fixed model mutated')
    except planning.PlanningMethodFailure as e:failure={'category':'METHOD_FAILURE','type':type(e).__name__,'message':str(e)}
    except BaseException as e:infra=e;failure={'category':'INFRASTRUCTURE_FAILURE','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    finally:
        mj.mj_step=original_mj_step
        if world is not None:
            renderer=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
            if renderer is not None:renderer.close()
            world.close()
    # Read-only encoding after all control actions; does not change planner state or policy.
    if pixels:
        pixelarray=np.stack(pixels);latent=encode_pixels(model,pixelarray,transform)
    else:pixelarray=np.empty((0,224,224,3),np.uint8);latent=np.empty((0,192),np.float32)
    with np.load(ROOT/'assets'/(case['case_id']+'.npz')) as f:goalpixels=f['goal:pixels'].copy()
    if goalpixels.shape[0]==3:goalpixels=np.moveaxis(goalpixels,0,-1)
    goal_latent=encode_pixels(model,goalpixels[None],transform)[0]
    save_npz(folder/'states.npz',**states)
    save_npz(folder/'plans.npz',**planarrays)
    save_npz(folder/'trajectory.npz',raw_actions=np.stack([r['action'] for r in rows]) if rows else np.empty((0,5)),physical_state=np.stack([r['physical'] for r in rows]) if rows else np.empty((0,3)),goal_state=view.goal_state if view.goal_state is not None else np.empty(0),step_success=np.array([r['terminated'] for r in rows]),step_truncated=np.array([r['truncated'] for r in rows]),reward=np.array([r['reward'] for r in rows]),goal_error=np.array([r['error'] for r in rows]),raw_pixels=pixelarray,raw_latent=latent,goal_pixels=goalpixels,goal_latent=goal_latent)
    success=bool(metrics['episode_successes'][0]) if metrics is not None else False
    result={'status':'COMPLETE' if infra is None else 'INFRASTRUCTURE_FAILURE','task':'cube','evaluation_kind':'CLOSED_LOOP_CEM','case_id':case['case_id'],'family_id':case['family_id'],'arm':arm,'phase':phase,'stream':'R3_ORIGINAL','reset_label':LABEL,'evidence_label':EVIDENCE,'success':int(success) if infra is None else None,'any_step_success':int(success) if infra is None else None,'terminal_success':int(bool(rows and rows[-1]['terminated'])) if infra is None else None,'final_goal_error':rows[-1]['error'] if rows else None,'executed_raw_steps':len(rows),'replan_calls':len(replans),'replans':replans,'failure':failure,'trajectory_wall_seconds':time.monotonic()-began,'planning_synchronized_wall_seconds':sum(r['seconds'] for r in replans),'environment_step_seconds':sum(r['seconds'] for r in rows),'device_timing':timer.totals(),'optimizer_updates':0,'identity_sha256':bind,'model_digest_before':model_before,'model_digest_after':model_digest(model),'C0_initial_fields_compared':len(initial_comparison),'mj_step_ledger':counts,'initialization_physics_substeps':sum(r['substeps'] for r in counts if r['phase']!='executed_raw'),'executed_physics_substeps':sum(r['substeps'] for r in counts if r['phase']=='executed_raw'),'utc':utc()}
    atomic(folder/'result.json',result)
    if infra is not None:raise infra
    files={p.name:record(p) for p in folder.iterdir() if p.is_file() and p.name not in ('COMPLETE.json','RUN.lock')}
    atomic(complete,{'identity_sha256':bind,'files':files})
    print({'case':case['case_id'],'arm':arm,'phase':phase,'status':result['status'],'seconds':result['trajectory_wall_seconds'],'steps':len(rows)},flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['TECH','EVAL'],required=True);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    import fcntl
    (ROOT/'jobs').mkdir(parents=True,exist_ok=True)
    worker_lock=(ROOT/'jobs'/f'{a.phase}_{a.shard}.lock').open('a')
    fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from x1 import core,data
    from x1.evaluate import load_arm
    core.ROOT=Path('/workspace/x1_cube');core.policy()
    c=read(ROOT/'C1_CONTRACT.json')
    if socket.gethostname()!=c['host']:raise RuntimeError('Host changed')
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];expected=c['resources']['GPU'+gpu]
    if list(sorted(os.sched_getaffinity(0)))!=expected:raise RuntimeError('CPU isolation changed')
    for r in c['code'].values():verify(r)
    verify(c['c0_reset']);verify(c['statistics_source'])
    if a.phase=='EVAL' and read(ROOT/'TECH_GATE.json')['status']!='PASS':raise RuntimeError('TECH gate not passed')
    if a.phase=='TECH':jobs=list(zip(c['cases']['TECH'],ARMS))
    else:jobs=[(case,arm) for j,case in enumerate(c['cases']['EVAL']) if j%2==a.shard for arm in ARMS]
    atomic(ROOT/'jobs'/f'{a.phase}_{a.shard}_STARTED.json',{'pid':os.getpid(),'utc':utc(),'jobs':len(jobs),'gpu':gpu,'affinity':expected})
    models={arm:load_arm('cube',arm,'cuda') for arm in ARMS};results=[]
    with data.RawH5(c['source']['path']) as raw:
        for case,arm in jobs:results.append(run_case(c,models[arm],raw,case,arm,a.phase))
    if a.phase=='TECH':
        if len(results)!=4 or any(r['failure'] or not r['C0_initial_fields_compared'] for r in results):raise RuntimeError('TECH execution failed; no formal launch')
        rawtable=[{k:r[k] for k in ('case_id','arm','status','executed_raw_steps','replan_calls','trajectory_wall_seconds','C0_initial_fields_compared')} for r in results];csv_write(ROOT/'TECH_RAW_TABLE.csv',rawtable)
        atomic(ROOT/'TECH_GATE.json',{'status':'PASS','trajectories':4,'C0_initial_fields':sum(r['C0_initial_fields_compared'] for r in results),'p50_seconds':float(np.quantile([r['trajectory_wall_seconds'] for r in results],.5)),'p90_seconds':float(np.quantile([r['trajectory_wall_seconds'] for r in results],.9)),'contract':record(ROOT/'C1_CONTRACT.json'),'raw':record(ROOT/'TECH_RAW_TABLE.csv'),'utc':utc(),'success_not_a_gate':True})
    atomic(ROOT/'jobs'/f'{a.phase}_{a.shard}_COMPLETE.json',{'status':'COMPLETE','trajectories':len(results),'utc':utc()})
if __name__=='__main__':main()
