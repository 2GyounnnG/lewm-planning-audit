"""R8 M2 X1 closed loop: four fixed models x three official planner streams."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/r8')
ARMS=('H0','REFIT_103201','REFIT_103202','REFIT_103203')
STREAMS=('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha_file(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_bytes(canonical(obj)+b'\n');os.replace(tmp,p)
def file_record(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha_file(p)}

def setup(task):
 os.environ['X1_ROOT']=str(Path('/workspace/x1_cube' if task=='cube' else '/workspace/x1_tworoom'))
 os.environ['X1_THREADS']=os.environ.get('X1_THREADS','1')
 os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
 sys.path[:0]=['/workspace/r4_v23_execution','/workspace/r5/code']
 if task=='cube': sys.path.insert(0,'/workspace/x1_cube/ogbench_only')
 import x1.core as core
 core.ROOT=Path(os.environ['X1_ROOT']); core.policy()
 return core

class R8CaseView:
 def __init__(self,raw,task,case):
  import torch
  import x1.core as core
  self.raw,self.task,self.case=raw,task,case;self.reads=0;self.goal_state=None
  self.column_names=['pixels']+core.CONFIG[task]['reset_keys']
  self.has_seed=(task!='cube')
  if self.has_seed:self.column_names.append('seed')
  if not set(self.column_names)-{'seed'} <= set(raw.file): raise RuntimeError('R8 source schema missing reset keys')
 def load_chunk(self,episodes,starts,ends):
  import torch
  c=self.case; expected=int(c['goal_raw_index'])-int(c['start_raw_index'])+1
  if list(episodes)!=[c['source_episode_idx']] or list(starts)!=[c['start_raw_index']] or list(ends)!=[c['goal_raw_index']+1]: raise RuntimeError('R8 source access escaped frozen endpoints')
  chunk=self.raw.load_chunk(episodes,starts,ends)[0];out={k:chunk[k] for k in self.column_names if k!='seed'}
  if any(len(v)!=expected for v in out.values()): raise RuntimeError('R8 source chunk length differs')
  if self.has_seed:
   if 'seed' in chunk:
    got=np.asarray(chunk['seed']).reshape(expected,-1)
    if not np.all(got==int(c['reset_seed'])): raise RuntimeError('R8 source seed differs')
   out['seed']=torch.full((expected,),int(c['reset_seed']),dtype=torch.int64)
  self.reads+=1
  if self.reads!=1: raise RuntimeError('R8 repeated source read')
  key='privileged_block_0_pos' if self.task=='cube' else 'proprio';self.goal_state=np.asarray(out[key][-1],dtype=np.float64).copy()
  return [out]

def model_digest(model,core):
 tm=core.r3('model').tensor_sha256
 return digest({k:tm(v) for k,v in model.state_dict().items()})

def run_case(core,task,case,arm,stream,model,raw,outdir,device):
 import torch,mujoco as mj
 from x1 import data
 from x1.evaluate import case_seed
 from r3 import planning
 cfg=core.CONFIG[task];swm,solver_type,policy_type,_=planning.load_official_api();out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
 identity={'version':'R8_M2_FRESH_CASE_REPLICATION_V1','evidence_label':'R8_FRESH_CASE_REPLICATION','task':task,'case':case,'arm':arm,'stream':stream,'official_single_frame_history':True,'optimizer_updates':0,'source':planning.verify_official_sources(),'core_contract':core.task_contract(task),'code_sha256':sha_file(Path(__file__))}
 bind=core.digest(identity) if hasattr(core,'digest') else digest(identity)
 complete=out/'COMPLETE.json'; result_path=out/'result.json'
 if complete.exists() and result_path.exists():
  saved=json.loads(complete.read_text())
  if saved.get('identity_sha256')!=bind: raise RuntimeError('R8 completed identity differs')
  return json.loads(result_path.read_text())
 if (out/'STARTED.json').exists(): raise RuntimeError('Interrupted R8 X1 case requires explicit audit')
 atomic(out/'STARTED.json',{'identity':identity,'identity_sha256':bind,'utc':utc()})
 timer=planning._Timer(device);cost=planning.AuditedCost(model,timer);replans=[];plans=[];rows=[];world=None;metrics=None;failure=None;infra=None;started=time.perf_counter()
 solver=solver_type(model=cost,device=device,seed=case_seed(task,case['case_id'],0,stream),**core.CEM); official_solve=solver.solve
 def solve(info,init_action=None):
  i=len(replans);seed=case_seed(task,case['case_id'],i,stream);solver.torch_gen.manual_seed(seed);t=time.perf_counter();res=official_solve(info,init_action=init_action);timer.synchronize()
  if not torch.isfinite(res['actions']).all(): raise planning.PlanningMethodFailure('NONFINITE_PLAN')
  arr=res['actions'].detach().cpu().numpy(); plans.append(arr.copy());replans.append({'replan_index':i,'seed_uint64':seed,'observed_history_frames':int(info['pixels'].shape[1]),'returned_plan':arr.tolist(),'seconds':time.perf_counter()-t,'returned_plan_sha256':hashlib.sha256(arr.tobytes()).hexdigest()});return res
 solver.solve=solve;transform=core.r3('data').image_transform(); policy=policy_type(solver=solver,config=swm.PlanConfig(**core.PLAN),process={'action':data.processor(task)},transform={'pixels':transform,'goal':transform}); view=R8CaseView(raw,task,case)
 model_before=model_digest(model,core)
 try:
  kwargs=dict(cfg['world']);world=swm.World(**kwargs,num_envs=1,max_episode_steps=100,image_shape=(224,224))
  if task=='cube':
   from c0.reset import clear_internal
   original_reset=world.reset
   def reset(seed=None,options=None):
    if seed is not None: raise ValueError('Cube official reset is seedless')
    outv=original_reset(seed=None,options=options);clear_internal(world.envs.envs[0].unwrapped,mj);return outv
   world.reset=reset
  else:
   original_reset=world.reset; normalize=core.r3('env_compat').normalize_reset_seed
   world.reset=lambda seed=None,options=None: original_reset(seed=normalize(seed),options=options)
  world.set_policy(planning.IsolatedPolicy(policy)); original_step=world.envs.step
  def step(actions,*args,**kwargs):
   outv=original_step(actions,*args,**kwargs);_,reward,terminated,truncated,infos=outv
   env=world.envs.envs[0].unwrapped
   if task=='tworoom': physical=np.asarray(infos['proprio'][0,-1],dtype=np.float64).copy()
   else: physical=np.asarray(env._data.joint('object_joint_0').qpos[:3],dtype=np.float64).copy()
   rows.append({'action':np.asarray(actions[0]).copy(),'physical_state':physical,'terminated':bool(terminated[0]),'truncated':bool(truncated[0]),'reward':float(reward[0])});return outv
  world.envs.step=step
  with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
   metrics=world.evaluate(dataset=view,episodes_idx=[case['source_episode_idx']],start_steps=[case['start_raw_index']],goal_offset=25,eval_budget=50,callables=cfg['callables'],video=None)
  if view.reads!=1: raise RuntimeError('R8 view read count differs')
 except planning.PlanningMethodFailure as e: failure={'category':'METHOD_FAILURE','type':type(e).__name__,'message':str(e)}
 except BaseException as e: infra=e;failure={'category':'INFRASTRUCTURE_FAILURE','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:
  if world is not None:
   if task=='cube':
    r=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
    if r is not None:r.close()
   world.close()
 if infra is not None:
  atomic(out/'FAILURE.json',failure);raise infra
 success=int(bool(metrics['episode_successes'][0])) if metrics is not None else 0
 if metrics is not None and success!=int(any(x['terminated'] for x in rows)):raise RuntimeError('R8 success/event mismatch')
 if model_before!=model_digest(model,core):raise RuntimeError('R8 model mutated')
 action_dim=int(model.r3_contract['macro_action_dim']//5)
 arrs={'raw_actions':np.stack([x['action'] for x in rows]) if rows else np.empty((0,action_dim),np.float32),'physical_state':np.stack([x['physical_state'] for x in rows]) if rows else np.empty((0,0)),'goal_state':view.goal_state if view.goal_state is not None else np.empty((0,)),'step_success':np.asarray([x['terminated'] for x in rows],dtype=np.uint8),'step_truncated':np.asarray([x['truncated'] for x in rows],dtype=np.uint8),'reward':np.asarray([x['reward'] for x in rows]),'returned_plans':np.stack(plans) if plans else np.empty((0,5,action_dim*5),np.float32)}
 tmp=out/'trajectory.npz.tmp';
 with tmp.open('wb') as fh: np.savez_compressed(fh,**arrs)
 os.replace(tmp,out/'trajectory.npz')
 result={'status':'COMPLETE','evidence_label':'R8_FRESH_CASE_REPLICATION','task':task,'case_id':case['case_id'],'episode_id':case['episode_id'],'family_id':case.get('family_id'),'arm':arm,'stream':stream,'success':success,'executed_raw_steps':len(rows),'replan_calls':len(replans),'replans':replans,'entered_replanning':len(replans)>1,'method_failure':failure,'optimizer_updates':0,'identity_sha256':bind,'model_digest_before':model_before,'model_digest_after':model_digest(model,core),'wall_seconds':time.perf_counter()-started,'reset_mode':case.get('reset_mode'),'official_single_frame_history':True,'first_plan_sha256':replans[0]['returned_plan_sha256'] if replans else None,'utc':utc()}
 atomic(result_path,result); atomic(complete,{'identity_sha256':bind,'files':{n:file_record(out/n) for n in ('result.json','trajectory.npz')}});return result

def load_model(core,task,arm,device):
 from x1.evaluate import load_arm
 return load_arm(task,arm,device)

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['cube','tworoom'],required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--gpu',default='0');p.add_argument('--case-limit',type=int);p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu); core=setup(a.task);device='cuda'
 mpath=ROOT/'ops'/f'M2_CASE_WINDOWS_{a.task.upper()}.json';manifest=json.loads(mpath.read_text());entries=manifest['cases'][:a.case_limit] if a.case_limit else manifest['cases'];entries=entries[a.worker_index::a.workers]
 import x1.data as data
 model=load_model(core,a.task,a.arm,device); source=core.read(core.ROOT/'manifests'/f'{a.task}_data_roles.json')['source'];
 with data.RawH5(core.verify(source)) as raw:
  for n,e in enumerate(entries):
   c=e['case']; base=ROOT/'raw'/'M2'/a.task/'FORMAL';
   for stream in STREAMS:
    out=base/stream/a.arm/c['case_id']; print(json.dumps({'task':a.task,'arm':a.arm,'stream':stream,'case_id':c['case_id'],'index':n},ensure_ascii=False),flush=True); run_case(core,a.task,c,a.arm,stream,model,raw,out,device)
 print(json.dumps({'status':'COMPLETE','task':a.task,'arm':a.arm,'cases':len(entries),'streams':len(STREAMS)},ensure_ascii=False),flush=True)

if __name__=='__main__':
 main()
