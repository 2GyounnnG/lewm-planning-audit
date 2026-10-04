"""Pinned old SWM inference for R10 B/C/D, with sealed first-plan controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import torch
from r4.common import atomic_json, atomic_npz, digest, file_record, replan_seed
from r4.export_cases import CaseWindows
from r3 import planning, model as model_api, data as r3data
from r3.common import fp32_policy
from r3.env_compat import normalize_reset_seed
import history_context as context

ARMS=('H0','REFIT_103201','REFIT_103202','REFIT_103203')
STREAMS=('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')
ROOT=Path('/workspace/r10')

def arr(x):return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.asarray(x).copy()
def model_hash(model):return digest({k:model_api.tensor_sha256(v) for k,v in model.state_dict().items()})

def verified(record):
    p=Path(record['path']);assert file_record(p)=={k:record[k] for k in ('bytes','sha256')}
    return p

def load(task,arm,device):
    if task=='reacher':
        model=model_api.load_official(task,device)
        if arm!='H0':
            routes=json.loads((ROOT/'shared_data/r3/manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][task]['arms']
            route=next(r for r in routes if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]))
            r=route['checkpoint'];p=ROOT/'shared_data/r3'/r['path'];assert file_record(p)=={k:r[k] for k in ('bytes','sha256')}
            model_api.apply_delta(model,p)
        return model, r3data.action_processor(task)
    from x1 import core
    core.R3_ROOT=ROOT/'shared_data/r3';core.task_contract(task)
    r=json.loads((ROOT/'inputs/X1_MODEL_REGISTRY.json').read_text())[task]
    cfg=json.loads(verified(r['official']['config.json']).read_text())
    model=model_api.construct_official(cfg)
    state=torch.load(verified(r['official']['weights.pt']),map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    assert all(not v.is_floating_point() or torch.isfinite(v).all() for v in state.values())
    model.r3_identity={'task':task,'weights_sha256':r['official']['weights.pt']['sha256'],'config_sha256':r['official']['config.json']['sha256']}
    if arm!='H0':model_api.apply_delta(model,verified(r['refits'][arm]))
    from sklearn.preprocessing import StandardScaler
    n=json.loads((ROOT/f'model_assets/{task}/normalization.json').read_text())['action']
    scale=StandardScaler();scale.mean_=np.asarray(n['mean']);scale.var_=np.asarray(n['variance']);scale.scale_=np.asarray(n['scale']);scale.n_features_in_=len(n['mean']);scale.n_samples_seen_=n['finite_rows']
    return model.to(device=device,dtype=torch.float32).eval().requires_grad_(False),scale

class X1View:
    def __init__(self,raw,task,case):
        from x1.core import CONFIG
        self.raw,self.task,self.case=raw,task,case;self.reads=0
        self.column_names=['pixels']+CONFIG[task]['reset_keys']+([] if task=='cube' else ['seed'])
        assert set(self.column_names)-{'seed'}<=set(raw.column_names)
    def load_chunk(self,episodes,starts,ends):
        c=self.case
        assert list(episodes)==[c['source_episode_idx']] and list(starts)==[c['start_raw_index']] and list(ends)==[c['goal_raw_index']+1]
        chunk=self.raw.load_chunk(episodes,starts,ends)[0];out={k:chunk[k] for k in self.column_names if k!='seed'}
        assert all(len(v)==26 for v in out.values())
        if self.task!='cube':out['seed']=torch.full((26,),int(c['reset_seed']),dtype=torch.int64)
        self.reads+=1;assert self.reads==1
        return [out]

def run(model,scaler,task,entry,assets,folder,arm,stream,mode,reference,baseline,phase,prereg_sha):
    swm,Solver,Policy,source=planning.load_official_api()
    case=entry['case'];macro=model.r3_contract['macro_action_dim'];assert macro%5==0
    identity={'task':task,'case':case,'arm':arm,'stream':stream,'mode':mode,'phase':phase,
              'source':source,'model_sha256':model_hash(model),'asset':{k:entry[k] for k in ('bytes','sha256')},
              'prereg_sha256':prereg_sha,'baseline':baseline,'optimizer_updates':0,
              'code':{n:file_record(Path(__file__).parent/n) for n in ('old_history_runner.py','history_context.py')}}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'STARTED.json').exists():raise RuntimeError('Attempt already exists; no automatic retry')
    atomic_json(folder/'STARTED.json',{'identity':identity,'identity_sha256':digest(identity)})
    actions=[];frames=[];plans=[];probes=[];steps=[];world=None;metrics=None;method_failure=None;start=time.monotonic()
    transform=r3data.image_transform();timer=planning._Timer(torch.device(next(model.parameters()).device))
    class Cost(planning.AuditedCost):
        def get_cost(self,info,candidates):
            scored=context.with_prefix(info,candidates,actions,scaler,mode,macro)
            probes[-1].update(model_pixels_shape=list(info['pixels'].shape),candidate_shape=list(candidates.shape),scored_shape=list(scored.shape))
            assert scored.shape[2]==5+info['pixels'].shape[2]-1
            return super().get_cost(info,scored)
    solver=Solver(model=Cost(model,timer),device=next(model.parameters()).device,seed=replan_seed(task,case['case_id'],0,stream),**planning.CEM_CONFIG)
    original_solve=solver.solve
    def solve(info,init_action=None):
        index=len(probes);seed=replan_seed(task,case['case_id'],index,stream);solver.torch_gen.manual_seed(seed)
        probes.append({'replan_index':index,'anchor_raw':len(actions),'seed_uint64':seed,'frame_indices':context.indices(len(actions),mode),'future_macro_steps':5,'terminal_raw':len(actions)+25})
        result=original_solve(info,init_action=init_action)
        if not torch.isfinite(result['actions']).all():raise planning.PlanningMethodFailure('NONFINITE_PLAN')
        plan=arr(result['actions'])[0];plans.append(plan)
        if index==0:
            equal=plan.dtype==reference.dtype and plan.tobytes()==reference.tobytes()
            probes[-1].update(first_plan_equal=equal,max_abs_difference=float(np.max(np.abs(plan-reference))))
            if not equal:raise RuntimeError('SEALED_FIRST_PLAN_GATE_FAILED')
        return result
    solver.solve=solve
    policy=Policy(solver=solver,config=swm.PlanConfig(**planning.PLAN_CONFIG),process={'action':scaler},transform={'pixels':transform,'goal':transform})
    class Observed(planning.IsolatedPolicy):
        def get_action(self,info,**kwargs):
            frames.append(arr(info['pixels'][0,-1]))
            return super().get_action(context.observation(info,frames,len(actions),mode),**kwargs)
    try:
        raw=CaseWindows(assets,entry)
        if task=='reacher':
            c=dict(case);c['role']='EVAL';view=planning.CaseDatasetView(raw,task,c)
            kwargs={'env_name':'swm/ReacherDMControl-v0','task':'qpos_match'};callables=planning.CALLABLES[task]
        else:
            from x1.core import CONFIG
            view=X1View(raw,task,case);kwargs=dict(CONFIG[task]['world']);callables=CONFIG[task]['callables']
        world=swm.World(**kwargs,num_envs=1,max_episode_steps=100,image_shape=(224,224))
        assert world.envs.single_action_space.shape==(macro//5,)
        original_reset=world.reset
        if task=='cube':
            from cube_reset import clear_internal
            import mujoco
            def reset(seed=None,options=None):
                assert seed is None
                value=original_reset(seed=None,options=options);clear_internal(world.envs.envs[0].unwrapped,mujoco);return value
            world.reset=reset
        else:world.reset=lambda seed=None,options=None:original_reset(seed=normalize_reset_seed(seed),options=options)
        world.set_policy(Observed(policy));original_step=world.envs.step
        def step(value,*args,**kwargs):
            result=original_step(value,*args,**kwargs);actions.append(arr(value[0]));steps.append((float(result[1][0]),bool(result[2][0]),bool(result[3][0])));return result
        world.envs.step=step
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=25,eval_budget=50,callables=callables,video=None)
    except planning.PlanningMethodFailure as e:method_failure=str(e)
    except Exception as e:
        atomic_json(folder/'FAILURE.json',{'status':'INVALID_FIRST_PLAN' if str(e)=='SEALED_FIRST_PLAN_GATE_FAILED' else 'TECHNICAL_MISSING','type':type(e).__name__,'error':str(e),'probes':probes,'traceback':traceback.format_exc()})
        return False
    finally:
        if world is not None:
            if task=='cube':
                renderer=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
                if renderer is not None:renderer.close()
            world.close()
    result={'status':'COMPLETE','identity_sha256':digest(identity),'task':task,'case_id':case['case_id'],'arm':arm,'stream':stream,'mode':mode,'phase':phase,'success':int(bool(metrics is not None and metrics['episode_successes'][0])),'method_failure':method_failure,'entered_replanning':len(probes)>1,'replans':probes,'first_plan_equal':bool(probes and probes[0].get('first_plan_equal')),'wall_seconds':time.monotonic()-start,'executed_raw_steps':len(actions),'optimizer_updates':0}
    if not result['first_plan_equal']:
        atomic_json(folder/'FAILURE.json',{'status':'TECHNICAL_MISSING','error':'First plan unavailable','method_failure':method_failure});return False
    atomic_npz(folder/'trajectory.npz',raw_actions=np.asarray(actions),steps=np.asarray(steps),returned_plans_normalized=np.asarray(plans),raw_pixel_sha256=np.asarray([hashlib.sha256(x.tobytes()).hexdigest() for x in frames]))
    atomic_json(folder/'result.json',result);atomic_json(folder/'COMPLETE.json',{'identity_sha256':digest(identity),'files':{n:file_record(folder/n) for n in ('result.json','trajectory.npz')}})
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['tworoom','cube','reacher']);p.add_argument('--manifest',required=True);p.add_argument('--assets',required=True);p.add_argument('--baseline',required=True);p.add_argument('--plans',required=True);p.add_argument('--output',required=True);p.add_argument('--prereg',required=True);p.add_argument('--phase',choices=['TECH','FORMAL'],required=True);p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);fp32_policy()
    prereg=json.loads(Path(a.prereg).read_text());prereg_sha=file_record(Path(a.prereg))['sha256']
    for name,expected in prereg['execution_code_sha256'].items():assert file_record(Path(__file__).parent/name)['sha256']==expected
    for key,path in [('manifest_sha256',a.manifest),('baseline_sha256',a.baseline),('plans_sha256',a.plans)]:assert file_record(Path(path))['sha256']==prereg[key]
    entries=json.loads(Path(a.manifest).read_text())['cases'];assert [e['case']['case_id'] for e in entries]==prereg['case_order']
    baseline={(r['case_id'],r['arm'],r['stream']):r for r in json.loads(Path(a.baseline).read_text())};plans=np.load(a.plans)['plans']
    if a.phase=='TECH':entries=entries[:4]
    streams=STREAMS[:1] if a.phase=='TECH' else STREAMS
    modes=('REAL3_NULLACT','REPEAT3_REALACT') if a.task=='reacher' else ('H_REAL3_REPLAN',)
    good=bad=0
    for arm in ARMS:
        model,scaler=load(a.task,arm,a.device);before=model_hash(model)
        for entry in entries[a.worker_index::a.workers]:
            for stream in streams:
                ref=baseline[entry['case']['case_id'],arm,stream]
                for mode in modes:
                    folder=Path(a.output)/a.phase/stream/entry['case']['case_id']/arm/mode
                    ok=run(model,scaler,a.task,entry,a.assets,folder,arm,stream,mode,plans[ref['plan_index']],ref,a.phase,prereg_sha)
                    good+=int(ok);bad+=int(not ok);print(json.dumps({'completed':good,'missing':bad,'case':entry['case']['case_id'],'arm':arm,'stream':stream,'mode':mode}),flush=True)
                    if a.task=='cube' and good/(good+bad)<.9:raise RuntimeError('CUBE_DELIVERY_BELOW_90_PERCENT_STOP_NO_RETRY')
        assert model_hash(model)==before
        del model;torch.cuda.empty_cache()
    if bad:raise SystemExit(1)

if __name__=='__main__':main()
