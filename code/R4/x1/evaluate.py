"""Official closed-loop X1 adapter and paired R3 open-loop metrics."""
from __future__ import annotations
import csv, hashlib, time
from pathlib import Path
import numpy as np
from . import core,data

def case_seed(task,case_id,index,stream='R3_ORIGINAL'):
    if stream=='R3_ORIGINAL': text=f'R3_CEM_CASE_20261002/{task}/{case_id}/{index}'
    elif stream in ('R4_ALT_CEM_1','R4_ALT_CEM_2'):text=f'{stream}/{task}/{case_id}/{index}'
    else:raise ValueError('Unknown planner stream')
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')

class CaseView:
    def __init__(self,raw,task,case):
        self.raw,self.task,self.case=raw,task,case;self.reads=0;self.goal_state=None
        self.column_names=['pixels']+core.CONFIG[task]['reset_keys']+['seed']
        if not set(self.column_names)-{'seed'}<=set(raw.column_names):raise ValueError('Reset schema missing')
        self.raw.column_names=[k for k in self.column_names if k in self.raw.file]
    def load_chunk(self,episodes,starts,ends):
        import torch
        c=self.case
        if list(episodes)!=[c['source_episode_idx']] or list(starts)!=[c['start_raw_index']] or list(ends)!=[c['goal_raw_index']+1]:
            raise ValueError('Case source access escaped frozen endpoints')
        if type(c['reset_seed']) is not int:raise ValueError('Missing source seed requires a validated TECH reset receipt')
        if c.get('source_seed') is None and not c.get('reset_seed_validation_sha256'):raise ValueError('Unverified fallback reset seed')
        chunk=self.raw.load_chunk(episodes,starts,ends)[0];out={k:chunk[k] for k in self.column_names if k!='seed'}
        if any(len(v)!=26 for v in out.values()):raise ValueError('Expected 26 source rows')
        if 'seed' in chunk and not np.all(np.asarray(chunk['seed'])==c['reset_seed']):raise ValueError('Source seed differs')
        out['seed']=np.full(26,c['reset_seed'],dtype=np.int64);self.reads+=1
        if self.reads!=1:raise ValueError('Unexpected repeated source read')
        self.goal_state=np.asarray(out[core.CONFIG[self.task]['goal_key']][-1],dtype=np.float64).copy()
        return [out]

def load_arm(task,arm,device):
    model=core.load_model(task,device)
    if arm!='H0':
        if arm not in [f'REFIT_{s}' for s in core.SEEDS]:raise ValueError('Unknown fixed arm')
        seed=int(arm.split('_')[1]);folder=core.ROOT/'train'/str(seed);result=core.read(folder/'result.json')
        if result['status']!='REFIT_TRAINING_COMPLETE_UNSCORED' or result['actual_updates']!=30000 or result['technical']:
            raise RuntimeError('Only completed fixed30k refits enter EVAL')
        pointer=core.read(folder/'last.json');path=core.verify(pointer)
        if path.name!='checkpoint_30000.pt':raise RuntimeError('Primary checkpoint must be fixed30000')
        core.r3('model').apply_delta(model,path)
    return model.eval().requires_grad_(False)

def closed_loop(task,arm,device='cuda',shard=0,shards=1,phase='EVAL',stream='R3_ORIGINAL'):
    import torch
    core.policy();core.task_contract(task);roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json')
    if phase not in ('TECH','EVAL') or not 0<=shard<shards:raise ValueError('Invalid evaluation phase/shard')
    if phase=='EVAL':
        for s in core.SEEDS:
            r=core.read(core.ROOT/'train'/str(s)/'result.json')
            if r['actual_updates']!=30000 or r['technical']:raise RuntimeError('All fixed refits must finish before EVAL')
    if stream!='R3_ORIGINAL':
        gate=core.read(core.ROOT/'SECOND_BATCH_GATE.json')
        if not gate.get('main_results_delivered') or not gate.get('tech_p90_seconds',float('inf'))<=30:raise RuntimeError('X1 extra streams require completed main results and TECH P90<=30s')
    cases=roles['cases'][phase][shard::shards];model=load_arm(task,arm,device);results=[]
    with data.RawH5(core.verify(roles['source'])) as raw:
        from .reset import effective_case
        for c in cases:results.append(run_case(model,task,effective_case(task,c),raw,arm,phase,stream,device))
    return {'task':task,'arm':arm,'phase':phase,'stream':stream,'cases':len(results),'completed':sum(r['status']=='COMPLETE' for r in results)}

def run_case(model,task,case,raw,arm,phase,stream,device):
    import torch
    planning=core.r3('planning');cfg=core.CONFIG[task];swm,solver_type,policy_type,source=planning.load_official_api()
    folder=core.ROOT/'closed_loop'/phase/stream/arm/case['case_id'];identity={'task':task,'case':case,'arm':arm,'stream':stream,
        'model':model.r3_identity,'source':source,'x1_code':core.sha(__file__),'contract':core.task_contract(task)}
    if arm!='H0':identity['delta']=core.read(core.ROOT/'train'/arm.split('_')[1]/'last.json')
    bind=core.digest(identity);complete=folder/'COMPLETE.json'
    if complete.exists():
        saved=core.read(complete)
        if saved['identity_sha256']!=bind:raise RuntimeError('Completed case identity differs')
        for r in saved['files'].values():core.verify(r)
        return core.read(folder/'result.json')
    if (folder/'STARTED.json').exists():raise RuntimeError('Interrupted case requires explicit technical accounting before restart')
    timer=planning._Timer(device);cost=planning.AuditedCost(model,timer);replans=[]
    solver=solver_type(model=cost,device=device,seed=case_seed(task,case['case_id'],0,stream),**core.CEM)
    official_solve=solver.solve
    def solve(info,init_action=None):
        i=len(replans);seed=case_seed(task,case['case_id'],i,stream);solver.torch_gen.manual_seed(seed)
        timer.synchronize();start=time.monotonic();result=official_solve(info,init_action=init_action);timer.synchronize()
        if not torch.isfinite(result['actions']).all():raise planning.PlanningMethodFailure('NONFINITE_PLAN')
        replans.append({'replan_index':i,'seed_uint64':seed,'seconds':time.monotonic()-start,
            'observed_history_frames':int(info['pixels'].shape[1]),'returned_plan':result['actions'].detach().cpu().numpy().tolist()})
        return result
    solver.solve=solve;transform=core.r3('data').image_transform()
    policy=policy_type(solver=solver,config=swm.PlanConfig(**core.PLAN),process={'action':data.processor(task)},transform={'pixels':transform,'goal':transform})
    view=CaseView(raw,task,case);rows=[];world=None;metrics=None;failure=None;infra=None;began=time.monotonic();hooks=[];stacks={}
    def enter(label):
        def hook(_module,_inputs):stacks.setdefault(label,[]).append(timer.begin(label))
        return hook
    def leave(label):
        def hook(_module,_inputs,_outputs):timer.end(stacks[label].pop())
        return hook
    for label,module in (('visual_encoder_seconds',model.encoder),('observation_projector_seconds',model.projector)):
        hooks.extend([module.register_forward_pre_hook(enter(label)),module.register_forward_hook(leave(label))])
    core.atomic(folder/'STARTED.json',{'identity_sha256':bind,'identity':identity})
    try:
        world=swm.World(**cfg['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224))
        original_reset=world.reset
        world.reset=lambda seed=None,options=None:original_reset(seed=core.r3('env_compat').normalize_reset_seed(seed),options=options)
        raw_dim=model.r3_contract['macro_action_dim']//5
        if tuple(world.envs.single_action_space.shape)!=(raw_dim,):raise RuntimeError('Environment/checkpoint action dimensions differ')
        world.set_policy(planning.IsolatedPolicy(policy));original_step=world.envs.step
        def step(actions,*args,**kwargs):
            t=time.monotonic();out=original_step(actions,*args,**kwargs);_,reward,terminated,truncated,infos=out
            env=world.envs.envs[0].unwrapped
            if task=='tworoom':physical=np.asarray(infos['proprio'][0,-1],dtype=np.float64).copy()
            else:physical=np.asarray(env._data.joint('object_joint_0').qpos[:3],dtype=np.float64).copy()
            rows.append({'action':np.asarray(actions[0]).copy(),'physical_state':physical,'goal_error':float(np.linalg.norm(physical-view.goal_state)),
                'terminated':bool(terminated[0]),'truncated':bool(truncated[0]),'reward':float(reward[0]),'seconds':time.monotonic()-t})
            return out
        world.envs.step=step
        with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],
                goal_offset=25,eval_budget=50,callables=cfg['callables'],video=None)
    except planning.PlanningMethodFailure as e:failure={'category':'METHOD_FAILURE','type':type(e).__name__,'message':str(e)}
    except BaseException as e:infra=e;failure={'category':'INFRASTRUCTURE_FAILURE','type':type(e).__name__,'message':str(e)}
    finally:
        for hook in hooks:hook.remove()
        if world is not None:
            # OGBench 1.2.1 inherits gym.Env.close (a no-op). Release its
            # renderer after the trajectory, before the EGL context vanishes.
            if task=='cube':
                renderer=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
                if renderer is not None:renderer.close()
            world.close()
    success=bool(metrics['episode_successes'][0]) if metrics is not None else False
    if metrics is not None and success!=any(r['terminated'] for r in rows):raise RuntimeError('Official success/log events disagree')
    result={'status':'COMPLETE' if infra is None else 'INFRASTRUCTURE_FAILURE','evaluation_kind':'CLOSED_LOOP_CEM','task':task,
        'case_id':case['case_id'],'family_id':case['family_id'],'arm':arm,'phase':phase,'stream':stream,
        'success':int(success) if infra is None else None,'any_step_success':int(success) if infra is None else None,
        'terminal_success':int(bool(rows and rows[-1]['terminated'])) if infra is None else None,'final_goal_error':rows[-1]['goal_error'] if rows else None,
        'executed_raw_steps':len(rows),'replan_calls':len(replans),'replans':replans,'failure':failure,
        'trajectory_wall_seconds':time.monotonic()-began,'planning_synchronized_wall_seconds':sum(r['seconds'] for r in replans),
        'environment_step_seconds':sum(r['seconds'] for r in rows),'device_timing':timer.totals(),
        'optimizer_updates':0,'identity_sha256':bind}
    data.save_npz(folder/'trajectory.npz',raw_actions=np.stack([r['action'] for r in rows]) if rows else np.empty((0,model.r3_contract['macro_action_dim']//5)),
        physical_state=np.stack([r['physical_state'] for r in rows]) if rows else np.empty((0,0)),
        goal_state=view.goal_state if view.goal_state is not None else np.empty(0),step_success=np.array([r['terminated'] for r in rows]),
        step_truncated=np.array([r['truncated'] for r in rows]),goal_error=np.array([r['goal_error'] for r in rows]))
    core.atomic(folder/'result.json',result)
    if infra is not None:raise infra
    core.atomic(complete,{'identity_sha256':bind,'files':{n:core.file_record(folder/n) for n in ('STARTED.json','trajectory.npz','result.json')}})
    return result

def open_loop(task,device='cpu'):
    import torch
    core.policy();m=core.r3('model');scoring=core.r3('open_loop');roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json')
    cases=roles['cases']['EVAL'];cache=data.load_cache(task,'EVAL');cm=core.read(core.ROOT/'manifests'/f'{task}_cache.json')
    arms=['H0']+[f'REFIT_{s}' for s in core.SEEDS];predictions={};inputs=None;rows=[];folder=core.ROOT/'open_loop';folder.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        model=load_arm(task,arm,device);contract=model.r3_contract;D=contract['latent_dim'];A=contract['macro_action_dim'];L=contract['history_size']
        histories=[];actions=[]
        for c in cases:
            x=cache[c['episode_id']];s=c['open_loop_window_start_raw'];histories.append(x['z'][s+np.arange(L)*5]);actions.append(x['actions'][s:s+35].reshape(7,A))
        zz=np.stack(histories);aa=np.stack(actions);parts=[]
        with torch.inference_mode():
            for start in range(0,len(cases),32):parts.append(m.cached_rollout(model,torch.from_numpy(zz[start:start+32]).to(device),torch.from_numpy(aa[start:start+32]).to(device),5).cpu().numpy())
        predictions[arm]=np.concatenate(parts);data.save_npz(folder/f'{arm}_forecast.npz',prediction=predictions[arm],case_ids=np.asarray([c['case_id'] for c in cases]))
        inputs=zz;del model
    # Materialize future targets only after all four forecasts are persisted.
    for i,c in enumerate(cases):
        target=cache[c['episode_id']]['z'][c['open_loop_anchor_raw']+np.arange(1,6)*5]
        for arm in arms:
            for r in scoring.score_case(predictions[arm][i],target,cm['train_scalar_variance'],inputs[i]):
                rows.append({'task':task,'case_id':c['case_id'],'family_id':c['family_id'],'arm':arm,**r})
    core.atomic(folder/'per_case.json',rows);write_csv(folder/'per_case.csv',rows)
    return {'status':'COMPLETE','task':task,'cases':len(cases),'arms':4,'rows':len(rows),'optimizer_updates':0}

def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)
