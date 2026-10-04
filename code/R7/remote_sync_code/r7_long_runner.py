"""Official R3 World/CEM execution with R4 observation logging and stream routing."""
from __future__ import annotations
import argparse,contextlib,fcntl,json,os,sys,time
from pathlib import Path
from r4.common import ARMS,STREAMS,atomic_json,atomic_npz,digest,file_record,replan_seed,sha256,verify_complete

class R7CaseDatasetView:
    """Pinned one-case dataset view with the registered 51-frame source window."""
    def __init__(self, dataset, task, case):
        self.dataset,self.task,self.case=dataset,task,case; self.reads=0; self.goal_state=None
        required=['pixels','state'] if task=='pusht' else ['pixels','qpos','qvel']
        if not set(required).issubset(dataset.column_names): raise ValueError('R7 reset dataset lacks required fields')
        self.column_names=required+['seed']; self.source_seed_present='seed' in dataset.column_names
    def load_chunk(self, episodes, starts, ends):
        import numpy as np
        case=self.case; expected=51
        if list(episodes)!=[case['source_episode_idx']] or list(starts)!=[case['start_raw_index']] or list(ends)!=[case['goal_raw_index']+1]:
            raise ValueError('R7 dataset access escaped frozen endpoints')
        source=self.dataset.load_chunk(episodes,starts,ends)[0]
        result={k:source[k] for k in self.column_names if k!='seed'}
        for k,v in result.items():
            if len(v)!=expected: raise ValueError(f'Expected exact {expected}-row R7 source chunk: {k}')
        if self.source_seed_present:
            seeds=np.asarray(source['seed']).reshape(expected,-1)
            if not np.all(seeds==case['reset_seed']): raise ValueError('Frozen reset seed differs from source')
        elif not case.get('reset_seed_validation_sha256'): raise ValueError('R7 fallback reset receipt missing')
        result['seed']=np.full(expected,case['reset_seed'],dtype=np.int64)
        key='state' if self.task=='pusht' else 'qpos'; self.goal_state=np.asarray(result[key][-1],dtype=np.float64).copy()
        self.reads+=1
        if self.reads!=1: raise RuntimeError('Unexpected repeated R7 source read')
        return [result]

def _numpy(x):
    import numpy as np
    return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()

def encode_pixels(model,pixels,transform,batch=32):
    import numpy as np
    import torch
    device=next(model.parameters()).device;result=[]
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
        for lo in range(0,len(pixels),batch):
            x=torch.stack([transform(torch.from_numpy(p.transpose(2,0,1).copy())) for p in pixels[lo:lo+batch]])
            z=model.encode({'pixels':x[:,None].to(device)})['emb'][:,0]
            result.append(_numpy(z))
    return np.concatenate(result,axis=0)

def physical_state(environment,task):
    import numpy as np
    if task=='pusht':return np.asarray(environment._get_obs(),dtype=np.float64).copy()
    return np.asarray(environment.env.physics.data.qpos,dtype=np.float64).copy()

def dynamical_state(environment,task):
    import numpy as np
    if task=='pusht':
        values=[]
        for body in (environment.agent,environment.block):
            values.extend([*body.position,body.angle,*body.velocity,body.angular_velocity,*body.force,body.torque])
        return np.asarray(values,dtype=np.float64)
    physics=environment.env.physics
    return np.concatenate([physics.get_state().copy(),physics.data.ctrl.copy(),[physics.data.time]])

def run_case(model,task,case,dataset,action_processor,image_transform,*,arm,stream,output,provenance,device='cuda',phase='FORMAL',reranker=None,control_root=None,history_mode='H_REAL3_REPLAN',reference_plan=None):
    """One case; caller process owns one GPU and explicit task CPU allocation.

    Completed cases skip only after all hashes verify. A process interruption
    preserves the entire attempt and starts a separately accounted attempt on
    invocation; method failures are completed outcomes and are never retried.
    """
    import numpy as np
    import torch
    from r3 import planning as p
    from r3.common import fp32_policy
    from r3.env_compat import normalize_reset_seed
    required=('case_id','episode_id','source_episode_idx','start_raw_index','goal_raw_index','reset_seed','family_id')
    if any(k not in case for k in required): raise ValueError('R7 case manifest missing fields')
    if case.get('role') != ('TECH' if phase == 'TECH' else 'EVAL'):
        raise ValueError('R7 TECH/EVAL role mismatch')
    if case['goal_raw_index'] != case['start_raw_index'] + 50:
        raise ValueError('R7 goal offset must be exactly 50 raw steps')
    if arm not in ARMS+('H0_MENU_RERANK','SIM_LAT_RERANK','SIM_TASK_RERANK') or stream not in STREAMS:raise ValueError('Unregistered arm/stream')
    identity={'version':'R7_PREREGISTERED_STRESS_TEST_FRESH_CASES_V1','task':task,'case':case,'arm':arm,'stream':stream,
        'provenance':provenance,'plan':p.PLAN_CONFIG,'cem':p.CEM_CONFIG,'budget_raw':100,'goal_offset_raw':50,'max_replans':3,'phase':phase,
        'code':{n:sha256(Path(__file__).parent/n) for n in ('r7_long_runner.py','context.py')},
        'control_protocol':'R4_S3_learning_policy_same_host_same_case_same_stream','history_mode':history_mode,'history_intervention':'H_POLICY single-frame control; H_REAL3_REPLAN real raw [t-10,t-5,t] at each receding-5 replan; maximum 3 post-initial replans',
        'source':p.verify_official_sources(),'optimizer_updates':0,'reranker':None if reranker is None else {'role':arm,'code':sha256(Path(__file__).parent/'s3.py')}}
    identity_sha=digest(identity);folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'case.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=verify_complete(folder,identity_sha)
        if saved:return saved
        attempts=sorted(folder.glob('attempt_*'));attempt=folder/f'attempt_{len(attempts):03d}';attempt.mkdir()
        atomic_json(attempt/'STARTED.json',{'identity':identity,'identity_sha256':identity_sha,'attempt':len(attempts),'started_unix':time.time()})
        fp32_policy();model.eval();model.requires_grad_(False)
        if any(x.dtype!=torch.float32 for x in model.parameters()):raise ValueError('FP32 model required')
        actual_device=next(model.parameters()).device
        swm,solver_type,policy_type,_=p.load_official_api();timer=p._Timer(actual_device)
        rows=[];observations=[];replans=[];plans=[];generations=[];costs=[]
        class Cost(p.AuditedCost):
            def get_cost(self,info,candidates):
                from context import candidates_with_prefix
                scored=candidates_with_prefix(info,candidates,rows,action_processor,history_mode)
                value=super().get_cost(info,scored)
                self.last_candidates=_numpy(candidates);self.last_costs=_numpy(value)
                return value
        cost=Cost(model,timer)
        solver=solver_type(model=cost,device=actual_device,seed=replan_seed(task,case['case_id'],0,stream),**p.CEM_CONFIG)
        official_solve=solver.solve
        def solve(info,init_action=None):
            index=len(replans);seed=replan_seed(task,case['case_id'],index,stream);solver.torch_gen.manual_seed(seed)
            record={'replan_index':index,'anchor_raw':len(rows),'seed_uint64':seed,'observed_history_frames':int(info['pixels'].shape[1]),'status':'STARTED'}
            from context import history_indices
            hidx = ([0] if len(rows)==0 else [len(rows)]) if history_mode == 'H_POLICY' else history_indices(len(rows))
            record.update(history_raw_indices=hidx,history_action_raw_start=max(0,len(rows)-10),history_action_raw_end=len(rows),future_macro_horizon=5,predicted_terminal_raw=len(rows)+25,cem_candidate_shape=[1,300,5,10])
            if record['observed_history_frames']!=len(record['history_raw_indices']):raise RuntimeError('Real history frame count mismatch')
            replans.append(record);timer.synchronize();start=time.perf_counter()
            try:
                result=official_solve(info,init_action=init_action)
                if not torch.isfinite(result['actions']).all():raise p.PlanningMethodFailure('NONFINITE_CEM_RETURNED_ACTIONS')
                plans.append(_numpy(result['actions'])[0]);generations.append(cost.last_candidates[0]);costs.append(cost.last_costs[0])
                record.update(status='COMPLETE',elite_mean_costs=result.get('costs'),returned_plan_sha256=digest(plans[-1].tolist()))
                if index==0 and reference_plan is not None:
                    if not np.array_equal(reference_plan,plans[-1]): raise RuntimeError('R6 paired first-plan mismatch')
                    record['paired_first_plan_check']={'status':'BITWISE_EQUAL','max_abs_difference':0.0}
                if reranker is not None:
                    result,rerank_record=reranker(result,info,cost.last_candidates[0],np.stack([r['action'] for r in rows]) if rows else np.empty((0,2)),index)
                    record['rerank']=rerank_record
                return result
            finally:
                timer.synchronize();record['synchronized_wall_seconds']=time.perf_counter()-start
        solver.solve=solve
        policy=policy_type(solver=solver,config=swm.PlanConfig(**p.PLAN_CONFIG),process={'action':action_processor},transform={'pixels':image_transform,'goal':image_transform})
        view=R7CaseDatasetView(dataset,task,case);world=None;error=None;method_failure=None;metrics=None;started=time.perf_counter()
        def observe(info):
            key='state' if task=='pusht' else 'qpos'
            observations.append({'pixels':_numpy(info['pixels'][0,-1]),'info_state':_numpy(info[key][0,-1]),
                'physical_state':physical_state(world.envs.envs[0].unwrapped,task),'dynamical_state':dynamical_state(world.envs.envs[0].unwrapped,task)})
        class ObservedPolicy(p.IsolatedPolicy):
            def get_action(self,info,**kwargs):
                observe(info)
                from context import real_history_observation
                prepared=real_history_observation(info,observations,len(rows),history_mode)
                return super().get_action(prepared,**kwargs)
        try:
            kwargs={'task':'qpos_match'} if task=='reacher' else {}
            world=swm.World(env_name='swm/PushT-v1' if task=='pusht' else 'swm/ReacherDMControl-v0',num_envs=1,max_episode_steps=200,image_shape=(224,224),**kwargs)
            original_reset=world.reset
            world.reset=lambda seed=None,options=None:original_reset(seed=normalize_reset_seed(seed),options=options)
            world.set_policy(ObservedPolicy(policy));original_step=world.envs.step
            def step(actions,*args,**kwargs):
                start=time.perf_counter();out=original_step(actions,*args,**kwargs)
                _,reward,terminated,truncated,info=out
                rows.append({'action':_numpy(actions[0]),'reward':float(reward[0]),'terminated':bool(terminated[0]),'truncated':bool(truncated[0]),'seconds':time.perf_counter()-start})
                return out
            world.envs.step=step
            with torch.inference_mode(),torch.autocast(device_type=actual_device.type,enabled=False):
                metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=50,eval_budget=100,callables=p.CALLABLES[task],video=None)
            observe(world.infos)
            pixels=np.stack([r['pixels'] for r in observations]);latent=encode_pixels(model,pixels,image_transform)
            goal_pixels=_numpy(world.infos['goal'][0,-1]);goal_latent=encode_pixels(model,goal_pixels[None],image_transform)[0]
        except p.PlanningMethodFailure as exc:method_failure={'type':type(exc).__name__,'message':str(exc)}
        except BaseException as exc:error=exc
        finally:
            if world is not None:world.close()
        if error is not None:
            atomic_json(attempt/'FAILURE.json',{'status':'INFRASTRUCTURE_FAILURE','error_type':type(error).__name__,'error':str(error),'executed_raw_steps':len(rows),'replans':replans})
            raise error
        success=int(bool(metrics is not None and metrics['episode_successes'][0]))
        result={'status':'COMPLETE','identity_sha256':identity_sha,'task':task,'case_id':case['case_id'],'family_id':case['family_id'],'arm':arm,'stream':stream,'success':success,'method_failure':method_failure,'executed_raw_steps':len(rows),'replan_calls':len(replans),'replans':replans,'attempts':len(attempts)+1,'optimizer_updates':0,'wall_seconds':time.perf_counter()-started,'evidence_scope':'R7_PREREGISTERED_STRESS_TEST_FRESH_CASES','logged_observations':len(observations),'history':history_mode,'entered_replanning':len(replans)>1,'control_source':'R7_PAIRED_ARM','paired_first_plan_checked': reference_plan is not None}
        arrays={'raw_actions':np.stack([r['action'] for r in rows]) if rows else np.empty((0,2)),
            'step_success':np.asarray([r['terminated'] for r in rows]),'step_truncated':np.asarray([r['truncated'] for r in rows]),
            'reward':np.asarray([r['reward'] for r in rows]),'returned_plans_normalized':np.stack(plans) if plans else np.empty((0,5,10)),
            'last_generation_normalized':np.stack(generations) if generations else np.empty((0,300,5,10)),
            'last_generation_cost':np.stack(costs) if costs else np.empty((0,300)),
            'replan_raw':np.asarray([r['anchor_raw'] for r in replans]),'goal_state':view.goal_state}
        if method_failure is None:
            arrays.update(raw_info_state=np.stack([r['info_state'] for r in observations]),
                raw_physical_state=np.stack([r['physical_state'] for r in observations]),raw_dynamical_state=np.stack([r['dynamical_state'] for r in observations]))
            if len(pixels)!=len(rows)+1:raise RuntimeError('Raw observations are not N+1 aligned')
        atomic_npz(folder/'trajectory.npz',**arrays);atomic_json(folder/'result.json',result)
        atomic_json(attempt/'FINISHED.json',result)
        atomic_json(folder/'COMPLETE.json',{'identity_sha256':identity_sha,'files':{n:file_record(folder/n) for n in ('trajectory.npz','result.json')}})
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True)
    p.add_argument('--all-streams',action='store_true');p.add_argument('--stream',choices=STREAMS,default='R3_ORIGINAL');p.add_argument('--device',default='cuda');p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--threads',type=int,default=1);p.add_argument('--case-limit',type=int);p.add_argument('--phase',choices=['TECH','FORMAL'],default='FORMAL');a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    if a.phase=='FORMAL':
        gate=Path(a.selection).with_name(f'R7_{a.task.upper()}_TECH_GATE.json')
        if not gate.exists() or json.loads(gate.read_text()).get('status')!='PASS':
            raise RuntimeError('R7 TECH gate must pass before FORMAL')
    import torch
    import numpy as np
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    from r3.model import load_official,apply_delta,frozen_hashes,assert_frozen
    from r3.data import action_processor,image_transform
    from r4.export_cases import CaseWindows
    manifest=json.loads((Path(a.assets)/f'CASE_WINDOWS_{a.task.upper()}.json').read_text());entries=manifest['cases']
    entries=[e for e in entries if e['case']['role']==('TECH' if a.phase=='TECH' else 'EVAL')]
    if a.case_limit:entries=entries[:a.case_limit]
    entries=entries[a.worker_index::a.workers]
    routing=json.loads((Path(a.r3_root)/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][a.task]['arms']
    streams=STREAMS if a.all_streams else [a.stream]
    arms=ARMS if a.phase=='FORMAL' else ('H0',)
    if a.phase=='TECH': streams=['R3_ORIGINAL']
    for arm in arms:
        model=load_official(a.task,a.device);provenance={'official':model.r3_identity,'case_manifest':sha256(Path(a.assets)/f'CASE_WINDOWS_{a.task.upper()}.json'),'evidence_label':'PREREGISTERED_STRESS_TEST_FRESH_CASES'}
        if arm!='H0':
            route=next(r for r in routing if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]));path=Path(a.r3_root)/route['checkpoint']['path']
            if file_record(path)!={k:route['checkpoint'][k] for k in ('bytes','sha256')}:raise RuntimeError('Fixed30k bytes differ')
            apply_delta(model,path);provenance['checkpoint']=route['checkpoint']
        frozen=frozen_hashes(model)
        for entry in entries:
            case=dict(entry['case']);dataset=CaseWindows(Path(a.assets)/a.task,entry)
            for selected_stream in streams:
                base=Path(a.output)/a.phase/selected_stream/case['case_id']/arm
                policy=run_case(model,a.task,case,dataset,action_processor(a.task),image_transform(),arm=arm,stream=selected_stream,output=base/'H_POLICY',provenance=provenance,device=a.device,phase=a.phase,history_mode='H_POLICY')
                with np.load(base/'H_POLICY'/'trajectory.npz',allow_pickle=False) as f:reference_plan=f['returned_plans_normalized'][0].copy()
                repl=run_case(model,a.task,case,dataset,action_processor(a.task),image_transform(),arm=arm,stream=selected_stream,output=base/'H_REAL3_REPLAN',provenance=provenance,device=a.device,phase=a.phase,history_mode='H_REAL3_REPLAN',reference_plan=reference_plan)
                print(json.dumps({'case_id':case['case_id'],'arm':arm,'stream':selected_stream,'H_POLICY':policy['success'],'H_REAL3_REPLAN':repl['success'],'replan_entered':repl['entered_replanning']},ensure_ascii=False),flush=True)
        assert_frozen(model,frozen);del model
        if a.device.startswith('cuda'):torch.cuda.empty_cache()

if __name__=='__main__':main()
