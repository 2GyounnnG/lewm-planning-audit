"""Missing source seeds: finite R3-style TECH equality audit, never seed recovery."""
from __future__ import annotations
import copy, importlib, time
import numpy as np
from . import core,data
from .evaluate import CaseView

PROVENANCE='SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED'

def snapshot(env,task):
    result={'render':np.asarray(env.render()).copy()}
    if task=='tworoom':
        result.update(agent=np.asarray(env.agent_position).copy(),target=np.asarray(env.target_position).copy())
    else:
        for k in ('qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat'):
            result[k]=np.asarray(getattr(env._data,k)).copy()
        # The manipulator controller target is task state when present.
        for k,v in vars(env).items():
            if k.startswith(('_target_effector','_target_gripper','_target_arm')) and isinstance(v,(np.ndarray,int,float,bool)):
                result['controller'+k]=np.asarray(v).copy()
    excluded={'agent.position','target.position'} if task=='tworoom' else {'cube.start_position','cube.start_yaw','cube.goal_position','cube.goal_yaw'}
    def walk(space,prefix=''):
        if hasattr(space,'spaces'):
            for key,value in space.spaces.items():walk(value,prefix+'.'+key if prefix else key)
        elif prefix not in excluded and hasattr(space,'value'):
            value=np.asarray(space.value)
            if value.dtype.kind in 'biuf':result['variation:'+prefix]=value.copy()
    walk(env.variation_space)
    return result

def trial(task,case,raw,seed,path):
    planning=core.r3('planning');swm,_,_,_=planning.load_official_api();module=importlib.import_module('stable_worldmodel.world.world')
    cfg=core.CONFIG[task];world=swm.World(**cfg['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224));arrays={};calls=0
    c=copy.deepcopy(case);c.update(reset_seed=seed,reset_seed_validation_sha256='TECH_PROBE_ONLY_NO_FORMAL_ROUTING')
    view=CaseView(raw,task,c);env=world.envs.envs[0].unwrapped;reset=world.reset
    world.reset=lambda seed=None,options=None:reset(seed=core.r3('env_compat').normalize_reset_seed(seed),options=options)
    raw_dim=int(world.envs.single_action_space.shape[0]);step=world.envs.step
    class Probe:
        def set_env(self,env):pass
        def get_action(self,info,**kwargs):
            nonlocal calls
            calls+=1
            if calls!=1:raise RuntimeError('Unexpected extra TECH action')
            arrays['injected_pixels']=np.asarray(info['pixels']).copy();arrays['injected_goal']=np.asarray(info['goal']).copy()
            return np.zeros((1,raw_dim),dtype=np.float32)
    probe=Probe();world.set_policy(probe)
    def observed_step(action,*args,**kwargs):
        for k,v in snapshot(env,task).items():arrays['before:'+k]=v
        out=step(action,*args,**kwargs)
        for k,v in snapshot(env,task).items():arrays['after:'+k]=v
        for k,v in zip(('reward','terminated','truncated'),out[1:4]):arrays['step:'+k]=np.asarray(v).copy()
        arrays['step:pixels']=np.asarray(out[4]['pixels']).copy();return out
    world.envs.step=observed_step
    try:
        if path=='official':
            world.evaluate(dataset=view,episodes_idx=[c['source_episode_idx']],start_steps=[c['start_raw_index']],goal_offset=25,
                eval_budget=1,callables=cfg['callables'],video=None)
        else:
            initial,goal,_=module._extract_init_goal(view,[c['source_episode_idx']],[c['start_raw_index']],25)
            world.reset(seed=initial.get('seed'));module._apply_callables(env,cfg['callables'],{k:v[0] for k,v in {**initial,**goal}.items()})
            prefix=world.infos['pixels'].shape[:2]
            for source in (initial,goal):
                for k,v in source.items():
                    if k in world.infos or k in goal:world.infos[k]=np.broadcast_to(v[:,None,...],prefix+v.shape[1:]).copy()
            world.envs.step(probe.get_action(world.infos))
        if calls!=1 or view.reads!=1:raise RuntimeError('TECH diagnostic counts differ')
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite TECH output')
        return arrays
    finally:world.close()

def audit(task):
    rolespath=core.ROOT/'manifests'/f'{task}_data_roles.json';roles=core.read(rolespath);cases=roles['cases']['TECH'][:2]
    affected=[c for c in roles['cases']['TECH']+roles['cases']['EVAL'] if c['reset_seed'] is None]
    if not affected:return {'status':'SOURCE_SEEDS_PRESENT_NO_FALLBACK','task':task}
    if len(cases)!=2 or any(c['reset_seed'] is not None for c in cases):raise RuntimeError('Fallback audit requires two frozen missing-seed TECH cases')
    folder=core.ROOT/'reset_audit';folder.mkdir(parents=True,exist_ok=True);rows=[];comparisons=[];began=time.monotonic()
    with data.RawH5(core.verify(roles['source'])) as raw:
        for c in cases:
            reference=None
            for seed in (0,918273):
                for repeat in (0,1):
                    for path in ('official','adapter'):
                        arrays=trial(task,c,raw,seed,path);name=f'{len(rows):02d}.npz';data.save_npz(folder/name,**arrays)
                        record={'case_id':c['case_id'],'seed':seed,'repeat':repeat,'path':path,'arrays':core.file_record(folder/name),'raw_steps':1,'CEM_calls':0}
                        rows.append(record)
                        if reference is None:reference=arrays
                        differences={k:bool(k in arrays and reference[k].dtype==arrays[k].dtype and np.array_equal(reference[k],arrays[k])) for k in reference}
                        exact=set(arrays)==set(reference) and all(differences.values())
                        comparisons.append({'trial':name,'exact':exact,'fields':differences})
    good=all(c['exact'] for c in comparisons)
    result={'status':'PASS_TECH_RESET_FALLBACK' if good else 'BLOCKED_RESET_FALLBACK','task':task,
        'provenance':PROVENANCE,'candidate_reset_seed':0 if good else None,'routing_eligible':good,
        'roles':core.file_record(rolespath),'source':roles['source'],'contract':core.task_contract(task),
        'code_sha256':core.sha(__file__),'cases':cases,'affected_case_ids':[c['case_id'] for c in affected],
        'trials':rows,'comparisons':comparisons,'seconds':time.monotonic()-began,'raw_calls':len(rows),'optimizer_updates':0,
        'scope':'Two frozen TECH cases, two candidate reset seeds, two repeats, official/adapter paths; not source-seed recovery or proof over arbitrary seeds',
        'tolerance':'exact state/render/one-step/controller and non-overridden-variation equality; no formal-outcome tuning'}
    core.freeze(folder/'RESET_FALLBACK.json',result);return result

def effective_case(task,case):
    if case['reset_seed'] is not None:return copy.deepcopy(case)
    path=core.ROOT/'reset_audit/RESET_FALLBACK.json';receipt=core.read(path)
    if receipt['task']!=task or receipt['status']!='PASS_TECH_RESET_FALLBACK' or not receipt['routing_eligible'] or receipt['candidate_reset_seed']!=0:
        raise RuntimeError('Missing valid fixed-seed TECH reset audit')
    core.verify(receipt['roles']);roles=core.read(receipt['roles']['path'])
    if receipt['code_sha256']!=core.sha(__file__) or receipt['contract']!=core.task_contract(task):raise RuntimeError('Reset audit code/source differs')
    exact=[c for c in roles['cases']['TECH']+roles['cases']['EVAL'] if c['case_id']==case['case_id']]
    if exact!=[case] or case['case_id'] not in receipt['affected_case_ids']:raise RuntimeError('Case outside fallback audit routing')
    for trial in receipt['trials']:core.verify(trial['arrays'])
    out=copy.deepcopy(case);out.update(reset_seed=0,reset_seed_validation_sha256=core.sha(path),reset_seed_provenance=PROVENANCE)
    return out
