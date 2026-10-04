"""Official R3 World/CEM execution with R4 observation logging and stream routing."""
from __future__ import annotations
import argparse,contextlib,fcntl,json,os,sys,time
from pathlib import Path
from r4.common import ARMS,STREAMS,atomic_json,atomic_npz,digest,file_record,replan_seed,sha256,verify_complete

class R8CaseDatasetView:
    """Pinned one-case dataset view with the unit's frozen source window."""
    def __init__(self, dataset, task, case):
        self.dataset,self.task,self.case=dataset,task,case; self.reads=0; self.goal_state=None
        required=['pixels','state'] if task=='pusht' else ['pixels','qpos','qvel']
        if not set(required).issubset(dataset.column_names): raise ValueError('R7 reset dataset lacks required fields')
        self.column_names=required+['seed']; self.source_seed_present='seed' in dataset.column_names
    def load_chunk(self, episodes, starts, ends):
        import numpy as np
        case=self.case; expected=int(case['goal_raw_index'])-int(case['start_raw_index'])+1
        if list(episodes)!=[case['source_episode_idx']] or list(starts)!=[case['start_raw_index']] or list(ends)!=[case['goal_raw_index']+1]:
            raise ValueError('R8 dataset access escaped frozen endpoints')
        source=self.dataset.load_chunk(episodes,starts,ends)[0]
        result={k:source[k] for k in self.column_names if k!='seed'}
        for k,v in result.items():
            if len(v)!=expected: raise ValueError(f'Expected exact {expected}-row R8 source chunk: {k}')
        if self.source_seed_present:
            seeds=np.asarray(source['seed']).reshape(expected,-1)
            if not np.all(seeds==case['reset_seed']): raise ValueError('Frozen reset seed differs from source')
        elif not case.get('reset_seed_validation_sha256'): raise ValueError('R7 fallback reset receipt missing')
        result['seed']=np.full(expected,case['reset_seed'],dtype=np.int64)
        key='state' if self.task=='pusht' else 'qpos'; self.goal_state=np.asarray(result[key][-1],dtype=np.float64).copy()
        self.reads+=1
        if self.reads!=1: raise RuntimeError('Unexpected repeated R8 source read')
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

def run_case(model,task,case,dataset,action_processor,image_transform,*,unit,goal_offset,budget,max_replans,arm,stream,output,provenance,device='cuda',phase='FORMAL',reranker=None,control_root=None,history_mode='H_REAL3_REPLAN',reference_plan=None,external_reference_plan=None,external_reference_label=None,hreal_reference_plan=None):
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
    if case.get('role') != 'EVAL': raise ValueError('R8 case role mismatch')
    if case['goal_raw_index'] != case['start_raw_index'] + goal_offset:
        raise ValueError('R8 goal offset differs from frozen unit')
    if arm not in ARMS+('H0_MENU_RERANK','SIM_LAT_RERANK','SIM_TASK_RERANK') or stream not in STREAMS:raise ValueError('Unregistered arm/stream')
    identity={'version':'R8_PREREGISTERED_BUDGET_DECOMPOSITION_V1','unit':unit,'task':task,'case':case,'arm':arm,'stream':stream,
        'provenance':provenance,'plan':p.PLAN_CONFIG,'cem':p.CEM_CONFIG,'budget_raw':budget,'goal_offset_raw':goal_offset,'max_replans':max_replans,'phase':phase,
        'code':{'r8_budget_runner.py':sha256(Path(__file__)),'context.py':sha256(Path(__file__).parent/'context.py')},
        'control_protocol':'R4_S3_learning_policy_same_host_same_case_same_stream','history_mode':history_mode,'history_intervention':'H_POLICY single-frame control; H_REAL3_REPLAN real raw [t-10,t-5,t] at each receding-5 replan; fixed unit maximum post-initial replans',
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
            """Official cost for the first plan, latent-context cost thereafter.

            The old R8 implementation wrote ``info['emb']`` then called the
            official cost.  The official LeWM API discards that key and
            re-encodes ``pixels``.  This implementation follows the official
            rollout algebra while replacing only the initial embedding at the
            PRED_PAST second plan.
            """
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.predicted_history=None
                self.context_used=None
                self._last_goal_emb=None
            def _latent_cost(self,info,candidates):
                import torch
                # ``candidates`` already contains the two executed action
                # blocks followed by the five candidate blocks.  H=3 aligns
                # those prefix blocks with [z_hat15,z_hat20,z25].
                H=int(self.predicted_history.shape[1])
                if H!=3: raise RuntimeError(f'Expected three PRED_PAST context tokens, got {H}')
                B,S,T=candidates.shape[:3]
                # With two executed prefix blocks, H=3 means the first of
                # the five candidate blocks occupies the initial action token;
                # the solver therefore passes T-H=4 future blocks and the
                # final predictor call supplies the fifth prediction.
                if T-H+1!=5: raise RuntimeError(f'Expected five candidate predictions, got T-H={T-H}')
                ctx=self.predicted_history.to(device=candidates.device,dtype=candidates.dtype)
                if ctx.shape[0]!=B: ctx=ctx.expand(B,-1,-1)
                self.context_used=ctx.detach().clone()
                # Encode the goal exactly as the official LeWM get_cost does.
                goal={k:v[:,0] for k,v in info.items() if torch.is_tensor(v)}
                goal['pixels']=goal['goal']
                for k in list(goal):
                    if k.startswith('goal_'):
                        goal[k[len('goal_'):]]=goal.pop(k)
                goal.pop('action',None)
                with torch.no_grad():
                    goal_emb=self.model.encode(goal)['emb']
                self._last_goal_emb=goal_emb.detach().clone()
                # Exact source rollout, except that the encoded initial
                # context is supplied rather than recomputed from pixels.
                emb=ctx[:,None,:,:].expand(B,S,-1,-1).reshape(B*S,H,-1).clone()
                act=candidates[:,:,:H].reshape(B*S,H,-1)
                act_future=candidates[:,:,H:].reshape(B*S,T-H,-1)
                for t in range(T-H):
                    act_emb=self.model.action_encoder(act)
                    emb_trunc=emb[:,-H:]
                    act_trunc=act_emb[:,-H:]
                    pred_emb=self.model.predict(emb_trunc,act_trunc)[:,-1:]
                    emb=torch.cat([emb,pred_emb],dim=1)
                    act=torch.cat([act,act_future[:,t:t+1]],dim=1)
                act_emb=self.model.action_encoder(act)
                pred_emb=self.model.predict(emb[:,-H:],act_emb[:,-H:])[:,-1:]
                emb=torch.cat([emb,pred_emb],dim=1)
                pred_final=emb[:,-1:].reshape(B,S,1,-1)
                target=goal_emb[:,-1:].unsqueeze(1).expand_as(pred_final)
                value=((pred_final-target.detach())**2).sum(dim=(2,3))
                if not torch.isfinite(value).all(): raise p.PlanningMethodFailure('NONFINITE_CEM_MODEL_COST')
                return value
            def get_cost(self,info,candidates):
                from m3_context import candidates_with_prefix
                scored=candidates_with_prefix(info,candidates,rows,action_processor,history_mode)
                if history_mode=='PRED_PAST' and self.predicted_history is not None:
                    value=self._latent_cost(info,scored)
                else:
                    value=super().get_cost(info,scored)
                self.last_candidates=_numpy(candidates);self.last_costs=_numpy(value)
                return value
        cost=Cost(model,timer)
        solver=solver_type(model=cost,device=actual_device,seed=replan_seed(task,case['case_id'],0,stream),**p.CEM_CONFIG)
        official_solve=solver.solve
        def solve(info,init_action=None):
            index=len(replans);seed=replan_seed(task,case['case_id'],index,stream);solver.torch_gen.manual_seed(seed)
            record={'replan_index':index,'anchor_raw':len(rows),'seed_uint64':seed,'observed_history_frames':int(info['pixels'].shape[1]),'status':'STARTED'}
            from m3_context import history_indices
            hidx = [0] if len(rows)==0 else history_indices(len(rows),history_mode)
            record.update(history_raw_indices=hidx,history_action_raw_start=max(0,len(rows)-10),history_action_raw_end=len(rows),future_macro_horizon=5,predicted_terminal_raw=len(rows)+goal_offset,cem_candidate_shape=[1,300,5,10])
            if record['observed_history_frames']!=len(record['history_raw_indices']):raise RuntimeError('Real history frame count mismatch')
            replans.append(record);timer.synchronize();start=time.perf_counter()
            try:
                if history_mode=='PRED_PAST' and index==1:
                    from r3.model import cached_rollout
                    z0=encode_pixels(model,np.asarray([observations[0]['pixels']]),image_transform)[0]
                    zcur=encode_pixels(model,np.asarray([observations[-1]['pixels']]),image_transform)[0]
                    with torch.inference_mode():pred=cached_rollout(model,torch.as_tensor(z0[None,None],device=actual_device),torch.as_tensor(plans[0][None],device=actual_device),5)
                    # cached_rollout returns predictions only (no initial z0):
                    # pred[0,2] and pred[0,3] are raw 15 and raw 20 for the
                    # five 5-raw-step macro actions.
                    zcur_t=torch.as_tensor(zcur,device=actual_device).reshape(1,-1)
                    context=torch.cat([pred[0,2:4],zcur_t],dim=0)[None]
                    cost.predicted_history=context
                    # TECH gate records exact context alignment and verifies
                    # the predicted tokens differ from true frame encodings.
                    true15,true20=encode_pixels(model,np.asarray([observations[15]['pixels'],observations[20]['pixels']]),image_transform)
                    z15,z20=true15,true20
                    l2=((context[0,0].detach().cpu().numpy()-z15)**2).sum()**0.5
                    l2_20=((context[0,1].detach().cpu().numpy()-z20)**2).sum()**0.5
                    record.update(predicted_context_raw_indices=[15,20,25],cached_rollout_output_includes_initial=False,
                                  cached_rollout_pred_slice='pred[0,2:4]',predicted_context_l2_vs_true=[float(l2),float(l2_20)],
                                  predicted_context_shape=list(context.shape))
                    if phase=='TECH' and l2==0.0 and l2_20==0.0:
                        raise RuntimeError('TECH predicted context is bitwise equal to true frame encodings')
                    if phase=='TECH':
                        # A same-seed small latent perturbation must change
                        # the plan.  Restore the unperturbed result afterward.
                        solver.torch_gen.manual_seed(seed)
                        base=official_solve(info,init_action=init_action)
                        base_plan=_numpy(base['actions'])[0].copy()
                        # Fixed TECH perturbation amplitude (pre-registered
                        # before rerun; chosen once, independent of case or
                        # outcome).  1e-2 remains small relative to latent
                        # scale while exercising planner sensitivity.
                        eps=torch.zeros_like(context);eps[...,0]=1e-2
                        cost.predicted_history=context+eps
                        solver.torch_gen.manual_seed(seed)
                        pert=official_solve(info,init_action=init_action)
                        pert_plan=_numpy(pert['actions'])[0]
                        record['perturbation_l2']=float(torch.linalg.vector_norm(eps).detach().cpu())
                        record['perturbation_plan_changed']=bool(not np.array_equal(base_plan,pert_plan))
                        if not record['perturbation_plan_changed']:
                            raise RuntimeError('TECH perturbation did not change plan')
                        cost.predicted_history=context
                        solver.torch_gen.manual_seed(seed)
                        result=official_solve(info,init_action=init_action)
                        if not np.array_equal(base_plan,_numpy(result['actions'])[0]):
                            raise RuntimeError('TECH restore after perturbation changed base plan')
                    else:
                        result=official_solve(info,init_action=init_action)
                else:
                    result=official_solve(info,init_action=init_action)
                if not torch.isfinite(result['actions']).all():raise p.PlanningMethodFailure('NONFINITE_CEM_RETURNED_ACTIONS')
                plans.append(_numpy(result['actions'])[0]);generations.append(cost.last_candidates[0]);costs.append(cost.last_costs[0])
                record.update(status='COMPLETE',elite_mean_costs=result.get('costs'),returned_plan_sha256=digest(plans[-1].tolist()))
                if index==1 and hreal_reference_plan is not None:
                    record['vs_hreal3_second_plan']={'status':'DIFFERENT' if not np.array_equal(hreal_reference_plan,plans[-1]) else 'BITWISE_EQUAL',
                                                       'bitwise_equal':bool(np.array_equal(hreal_reference_plan,plans[-1]))}
                    if phase=='TECH' and record['vs_hreal3_second_plan']['bitwise_equal']:
                        raise RuntimeError('TECH corrected PRED_PAST second plan equals H_REAL3_REPLAN')
                if index==0 and reference_plan is not None:
                    if not np.array_equal(reference_plan,plans[-1]): raise RuntimeError('R6 paired first-plan mismatch')
                    record['paired_first_plan_check']={'status':'BITWISE_EQUAL','max_abs_difference':0.0}
                if index==0 and external_reference_plan is not None:
                    if not np.array_equal(external_reference_plan,plans[-1]): raise RuntimeError(f'R8 cross-unit first-plan mismatch: {external_reference_label}')
                    record['cross_unit_first_plan_check']={'status':'BITWISE_EQUAL','max_abs_difference':0.0,'reference':external_reference_label}
                if reranker is not None:
                    result,rerank_record=reranker(result,info,cost.last_candidates[0],np.stack([r['action'] for r in rows]) if rows else np.empty((0,2)),index)
                    record['rerank']=rerank_record
                return result
            finally:
                timer.synchronize();record['synchronized_wall_seconds']=time.perf_counter()-start
        solver.solve=solve
        policy=policy_type(solver=solver,config=swm.PlanConfig(**p.PLAN_CONFIG),process={'action':action_processor},transform={'pixels':image_transform,'goal':image_transform})
        view=R8CaseDatasetView(dataset,task,case);world=None;error=None;method_failure=None;metrics=None;started=time.perf_counter()
        def observe(info):
            key='state' if task=='pusht' else 'qpos'
            observations.append({'pixels':_numpy(info['pixels'][0,-1]),'info_state':_numpy(info[key][0,-1]),
                'physical_state':physical_state(world.envs.envs[0].unwrapped,task),'dynamical_state':dynamical_state(world.envs.envs[0].unwrapped,task)})
        class ObservedPolicy(p.IsolatedPolicy):
            def get_action(self,info,**kwargs):
                observe(info)
                from m3_context import real_history_observation
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
                metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=goal_offset,eval_budget=budget,callables=p.CALLABLES[task],video=None)
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
        if len(replans)>max_replans+1: raise RuntimeError(f'R8 maximum post-initial replans exceeded: {len(replans)-1}>{max_replans}')
        success=int(bool(metrics is not None and metrics['episode_successes'][0]))
        result={'status':'COMPLETE','identity_sha256':identity_sha,'unit':unit,'task':task,'case_id':case['case_id'],'family_id':case['family_id'],'arm':arm,'stream':stream,'success':success,'method_failure':method_failure,'executed_raw_steps':len(rows),'replan_calls':len(replans),'replans':replans,'attempts':len(attempts)+1,'optimizer_updates':0,'wall_seconds':time.perf_counter()-started,'evidence_scope':'R8_FIX_F2_PRED_PAST_CORRECTED','logged_observations':len(observations),'history':history_mode,'entered_replanning':len(replans)>1,'control_source':'R8_PAIRED_ARM','paired_first_plan_checked': reference_plan is not None,'cross_unit_first_plan_checked': external_reference_plan is not None,'implementation_status':'CORRECTED_PREDICTED_LATENT_CONTEXT' if history_mode=='PRED_PAST' else 'REFERENCE_ONLY'}
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
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--history-mode',choices=['REPEAT3','REAL2','PRED_PAST'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--threads',type=int,default=1);p.add_argument('--case-limit',type=int);p.add_argument('--all-streams',action='store_true');p.add_argument('--phase',choices=['TECH','FORMAL'],default='FORMAL');a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import torch
    import numpy as np
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    from r3.model import load_official,apply_delta,frozen_hashes,assert_frozen
    from r3.data import action_processor,image_transform
    from r4.export_cases import CaseWindows
    manifest=json.loads(Path(a.manifest).read_text());entries=manifest['cases'];entries=[e for e in entries if e['case']['role']=='EVAL']
    if a.case_limit:entries=entries[:a.case_limit]
    entries=entries[a.worker_index::a.workers]
    routing=json.loads((Path(a.r3_root)/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks']['reacher']['arms']
    streams=STREAMS if a.all_streams else ['R3_ORIGINAL']
    arms=ARMS; unit='M3_REACHER'; goal_offset=25; budget=50; max_replans=1; refroot=Path('/workspace/r6/reacher')
    for arm in arms:
        model=load_official('reacher',a.device);provenance={'official':model.r3_identity,'case_manifest':sha256(Path(a.manifest)),'evidence_label':'R8_M3_HISTORY_MECHANISM','reference_root':str(refroot),'reference_label':'R6','history_mode':a.history_mode}
        if arm!='H0':
            route=next(r for r in routing if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]));path=Path(a.r3_root)/route['checkpoint']['path']
            if file_record(path)!={k:route['checkpoint'][k] for k in ('bytes','sha256')}:raise RuntimeError('Fixed30k bytes differ')
            apply_delta(model,path);provenance['checkpoint']=route['checkpoint']
        frozen=frozen_hashes(model)
        for entry in entries:
            case=dict(entry['case']);dataset=CaseWindows(Path(a.assets)/'U1',entry)
            for selected_stream in streams:
                base=Path(a.output)/selected_stream/case['case_id']/arm/a.history_mode
                ref=refroot/'raw/FORMAL'/selected_stream/case['case_id']/arm/'H_POLICY'/'trajectory.npz'
                if not ref.exists():raise RuntimeError(f'Missing frozen R6 reference: {ref}')
                with np.load(ref,allow_pickle=False) as f:external_reference=f['returned_plans_normalized'][0].copy()
                hreal=Path('/workspace/r8/raw/U1/FORMAL')/selected_stream/case['case_id']/arm/'H_REAL3_REPLAN'/'trajectory.npz'
                hreal_plan=None
                if hreal.exists():
                    with np.load(hreal,allow_pickle=False) as f:
                        if f['returned_plans_normalized'].shape[0]>1:hreal_plan=f['returned_plans_normalized'][1].copy()
                result=run_case(model,'reacher',case,dataset,action_processor('reacher'),image_transform(),unit=unit,goal_offset=goal_offset,budget=budget,max_replans=max_replans,arm=arm,stream=selected_stream,output=base,provenance=provenance,device=a.device,phase=a.phase,history_mode=a.history_mode,external_reference_plan=external_reference,external_reference_label='R6',hreal_reference_plan=hreal_plan)
                print(json.dumps({'case_id':case['case_id'],'arm':arm,'stream':selected_stream,'history_mode':a.history_mode,'success':result['success']},ensure_ascii=False),flush=True)
        assert_frozen(model,frozen);del model
        if a.device.startswith('cuda'):torch.cuda.empty_cache()

if __name__=='__main__':main()
