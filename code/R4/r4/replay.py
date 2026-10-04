"""R4 branches always create a fresh environment, reset, then replay actions.

No simulator snapshot or reduced-state reconstruction is used. Physical state
is diagnostic only; learned models receive pixels/actions/goal exclusively.
"""
from __future__ import annotations
import copy,json,time
from pathlib import Path
from .common import atomic_json,atomic_npz,digest

def task_margin(task,state,goal):
    import numpy as np
    state=np.asarray(state);goal=np.asarray(goal)
    if task=='pusht':
        position=np.linalg.norm(state[..., :4]-goal[..., :4],axis=-1)
        angle=np.abs(state[...,4]-goal[...,4]);angle=np.minimum(angle,2*np.pi-angle)
        return np.maximum(position/20,angle/(np.pi/9))
    if task=='reacher':return np.max(np.abs(state-goal)/.05,axis=-1)
    raise ValueError(task)

class ReplayBranch:
    """One fresh pinned World; exact R3 reset+dataset pixel substitution path."""
    def __init__(self,task,case,dataset):
        import numpy as np
        from r3 import planning as p
        from r3.env_compat import normalize_reset_seed
        from .runner import physical_state,dynamical_state
        swm,_,_,_=p.load_official_api()
        from stable_worldmodel.world.world import _extract_init_goal,_apply_callables
        self.task,self.case=task,case;self.steps=0;self.true_terminated=False;self.physics_steps=0
        self.world=swm.World(env_name='swm/PushT-v1' if task=='pusht' else 'swm/ReacherDMControl-v0',num_envs=1,max_episode_steps=100,image_shape=(224,224),**({'task':'qpos_match'} if task=='reacher' else {}))
        view=p.CaseDatasetView(dataset,task,case)
        initial,goal,_=_extract_init_goal(view,[case['source_episode_idx']],[case['start_raw_index']],25)
        self.world.reset(seed=normalize_reset_seed(initial.get('seed')))
        merged={**initial,**goal};env=self.world.envs.envs[0].unwrapped
        _apply_callables(env,p.CALLABLES[task],{k:v[0] for k,v in merged.items()})
        shape=self.world.infos['pixels'].shape[:2]
        for source in (initial,goal):
            for key,value in source.items():
                if key in self.world.infos or key in goal:self.world.infos[key]=np.broadcast_to(value[:,None,...],shape+value.shape[1:]).copy()
        self.goal_snapshot={key:self.world.infos[key].copy() for key in goal};self.goal_state=view.goal_state.copy()
        if task=='reacher':
            original_physics_step=env.env.step
            def observed_physics_step(action):
                if self.true_terminated:raise RuntimeError('DMC_LAST_CANNOT_AUTO_RESET_IN_BRANCH')
                value=original_physics_step(action);self.physics_steps+=1
                if value.last():self.true_terminated=True
                return value
            env.env.step=observed_physics_step
        self.initial=self.observe();self.history=[self.initial]
    def observe(self):
        import numpy as np
        from .runner import physical_state,dynamical_state
        env=self.world.envs.envs[0].unwrapped
        return {'state':physical_state(env,self.task),'dynamical_state':dynamical_state(env,self.task),'pixels':np.asarray(self.world.infos['pixels'][0,-1]).copy(),'render':np.asarray(env.render()).copy()}
    def step(self,action):
        import numpy as np
        # PushT success permits physical continuation. Reacher emits real DMC LAST;
        # guard it before any further physics step can silently auto-reset.
        if self.true_terminated:raise RuntimeError('DMC_LAST_CANNOT_AUTO_RESET_IN_BRANCH')
        _,reward,success,truncated,info=self.world.envs.step(np.asarray(action)[None])
        self.world.infos=info;info.update(copy.deepcopy(self.goal_snapshot));self.steps+=1
        if bool(truncated[0]):raise RuntimeError('TRUE_TERMINATION_OR_TRUNCATION_FIXED_HORIZON_UNEVALUABLE')
        obs=self.observe();obs.update(success=bool(success[0]),reward=float(reward[0]),margin=float(task_margin(self.task,obs['state'],self.goal_state)),true_terminated=self.true_terminated,physics_steps=self.physics_steps)
        self.history.append(obs);return obs
    def replay(self,actions):
        for action in actions:self.step(action)
        return self.history[-1]
    def close(self):self.world.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

def run_branch(task,case,dataset_factory,prefix,candidate,model=None,transform=None):
    import numpy as np
    from .runner import encode_pixels
    start=time.perf_counter()
    if len(candidate)>25:raise ValueError('At most 25 raw branch steps')
    with ReplayBranch(task,case,dataset_factory()) as branch:
        branch.replay(prefix);fork=branch.observe();steps=[]
        for action in candidate:
            if branch.true_terminated:break
            steps.append(branch.step(action))
        terminal=branch.observe()
        valid=len(steps)==len(candidate)
        if not steps:raise RuntimeError('FORK_ALREADY_TRUE_TERMINATED')
        margins=np.asarray([s['margin'] for s in steps]);success=np.asarray([s['success'] for s in steps])
        result={'terminal_state':terminal['state'],'terminal_pixels':terminal['pixels'],'fork_state':fork['state'],'step_margin':margins,'step_success':success,
            'task_terminal':float(margins[-1]),'task_anytime':float(margins.min()),'any_success':bool(success.any()),'prefix_raw_steps':len(prefix),'candidate_raw_steps':len(steps),'candidate_requested_raw_steps':len(candidate),
            'valid_fixed_horizon':valid,'missing_reason':'' if valid else 'TRUE_DMC_LAST_BEFORE_FIXED_HORIZON','observed_candidate_raw_steps':len(steps),'physics_steps':branch.physics_steps,'seconds':time.perf_counter()-start,'scope':'PRIVILEGED_FULL_STATE_BRANCH_REFERENCE'}
        if model is not None:result['terminal_latent']=encode_pixels(model,terminal['pixels'][None],transform)[0]
        return result

def compare_observations(a,b,atol=1e-5,rtol=1e-5):
    import numpy as np
    return {'state_max_abs':float(np.max(np.abs(a['state']-b['state']))),'state_pass':bool(np.allclose(a['state'],b['state'],atol=atol,rtol=rtol)),'dynamical_state_pass':bool(np.allclose(a['dynamical_state'],b['dynamical_state'],atol=atol,rtol=rtol)),
        'pixels_exact':bool(np.array_equal(a['pixels'],b['pixels'])),'render_exact':bool(np.array_equal(a['render'],b['render']))}

def audit_reset(task,entries,assets,output):
    from .export_cases import CaseWindows
    rows=[]
    for entry in entries[:4]:
        if entry['case']['role']!='TECH':raise ValueError('S0 only on original TECH')
        snapshots=[]
        for _ in range(2):
            with ReplayBranch(task,entry['case'],CaseWindows(assets,entry)) as b:snapshots.append(b.initial)
        row={'task':task,'case_id':entry['case']['case_id'],**compare_observations(*snapshots)};rows.append(row)
    result={'status':'PASS' if all(r['state_pass'] and r['dynamical_state_pass'] and r['pixels_exact'] and r['render_exact'] for r in rows) else 'FAIL','check':'S0_A','rows':rows,'reset_calls':2*len(rows),'raw_steps':0,'optimizer_updates':0}
    atomic_json(Path(output)/'S0_A.json',result);return result

def audit_replay(task,entry,assets,trajectory,model,transform,output):
    """b-d: log comparison; A,A and B,A order; pixel/latent end equivalence."""
    import numpy as np
    from .export_cases import CaseWindows
    from .runner import encode_pixels
    if entry['case']['role']!='TECH':raise ValueError('Replay TECH only')
    with np.load(trajectory,allow_pickle=False) as f:log={k:f[k].copy() for k in f.files}
    actions=log['raw_actions'];prefix=actions[:min(10,len(actions))];candidate=actions[len(prefix):len(prefix)+25]
    if len(candidate)==0:return {'status':'UNAVAILABLE_INSUFFICIENT_TECH_SUFFIX','case_id':entry['case']['case_id']}
    factory=lambda:CaseWindows(assets,entry)
    with ReplayBranch(task,entry['case'],factory()) as branch:
        branch.replay(actions);states=np.stack([x['state'] for x in branch.history]);dynamic=np.stack([x['dynamical_state'] for x in branch.history]);pixels=np.stack([x['pixels'] for x in branch.history])
    state_ok=bool(np.allclose(states,log['raw_physical_state'],atol=1e-5,rtol=1e-5));pixel_ok=bool(np.array_equal(pixels,log['raw_pixels']))
    a=run_branch(task,entry['case'],factory,prefix,candidate,model,transform)
    aa=run_branch(task,entry['case'],factory,prefix,candidate,model,transform)
    # B is deterministic metadata-only perturbation, never selected by outcome.
    b=run_branch(task,entry['case'],factory,prefix,np.zeros_like(candidate),model,transform)
    after_b=run_branch(task,entry['case'],factory,prefix,candidate,model,transform)
    end=len(prefix)+len(candidate);baseline_z=log['raw_latent'][end]
    rows={'replay_state_pass':state_ok,'replay_dynamical_state_pass':bool(np.allclose(dynamic,log['raw_dynamical_state'],atol=1e-5,rtol=1e-5)),'replay_pixels_exact':pixel_ok,'aa_state_pass':bool(np.allclose(a['terminal_state'],aa['terminal_state'],atol=1e-5,rtol=1e-5)),
        'order_state_pass':bool(np.allclose(a['terminal_state'],after_b['terminal_state'],atol=1e-5,rtol=1e-5)),
        'order_pixels_exact':bool(np.array_equal(a['terminal_pixels'],after_b['terminal_pixels'])),
        'terminal_pixel_matches_unbranched':bool(np.array_equal(a['terminal_pixels'],log['raw_pixels'][end])),
        'terminal_latent_pass':bool(np.allclose(a['terminal_latent'],baseline_z,atol=1e-5,rtol=1e-5))}
    result={'status':'PASS' if all(rows.values()) else 'FAIL','check':'S0_BCD','task':task,'case_id':entry['case']['case_id'],'checks':rows,'branch_calls':4,'prefix_raw_steps':len(actions)+4*len(prefix),'branch_raw_steps':sum(x['candidate_raw_steps'] for x in (a,aa,b,after_b)),'true_termination_observed':any(not x['valid_fixed_horizon'] for x in (a,aa,b,after_b)),'branch_seconds':[x['seconds'] for x in (a,aa,b,after_b)],'optimizer_updates':0}
    atomic_json(Path(output)/(entry['case']['case_id']+'_S0_BCD.json'),result);return result

def audit_termination(task,entries,assets,output):
    """At most two original TECH cases x three predefined25raw action probes.

    Expert recorded actions are a privileged termination diagnostic only. No
    future observation/action enters a learned planner through this function.
    """
    import numpy as np
    from .export_cases import CaseWindows
    rows=[];steps=[]
    for entry in entries[:2]:
        case=entry['case']
        if case['role']!='TECH':raise ValueError('Termination probes only TECH')
        data=CaseWindows(assets,entry);lo=case['start_raw_index']-int(data.arrays['source_start_raw'])
        recorded=data.arrays['action'][lo:lo+25].copy()
        for name,actions in [('RECORDED_ACTIONS',recorded),('ZERO_RAW_ACTIONS',np.zeros_like(recorded)),('FIXED_PULSE',np.tile(np.array([[.1,-.1]],dtype=np.float32),(25,1)))]:
            with ReplayBranch(task,case,CaseWindows(assets,entry)) as b:
                for index,action in enumerate(actions):
                    if b.true_terminated:break
                    value=b.step(action)
                    steps.append({'task':task,'case_id':case['case_id'],'probe':name,'raw':index+1,'success':value['success'],'true_terminated':value['true_terminated'],'physics_steps_cumulative':value['physics_steps'],'margin':value['margin']})
                rows.append({'task':task,'case_id':case['case_id'],'probe':name,'requested_raw':25,'executed_raw':b.steps,'true_terminated':b.true_terminated,'fixed_horizon_legal':b.steps==25,'physics_steps':b.physics_steps,'prevented_auto_reset':b.true_terminated and b.steps<25})
    result={'status':'COMPLETE_TECHNICAL_OBSERVATION','check':'TRUE_TERMINATION','rows':rows,'steps':steps,'fixed25_unavailable_observed':any(not r['fixed_horizon_legal'] for r in rows),'reset_calls':len(rows),'raw_steps':sum(r['executed_raw'] for r in rows),'optimizer_updates':0,'tech_queries_not_formal_samples':True}
    atomic_json(Path(output)/'TERMINATION_TECH_RAW.json',result);return result
