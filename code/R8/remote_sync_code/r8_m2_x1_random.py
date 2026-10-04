from __future__ import annotations
import argparse,csv,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/r8'); SEEDS=(7001,7002,7003)
def setup(task,gpu):
 os.environ['X1_ROOT']=str(Path('/workspace/x1_cube' if task=='cube' else '/workspace/x1_tworoom'));os.environ['X1_THREADS']='1';os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl');sys.path[:0]=['/workspace/r8/code','/workspace/r4_v23_execution','/workspace/r5/code','/workspace/shared_data/r3/source/swm_compat'];
 if task=='cube':sys.path.insert(0,'/workspace/x1_cube/ogbench_only')
 import x1.core as core; core.ROOT=Path(os.environ['X1_ROOT']);core.policy();return core

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['cube','tworoom'],required=True);p.add_argument('--gpu',default='0');p.add_argument('--case-start',type=int,default=0);p.add_argument('--case-end',type=int,default=100);p.add_argument('--output-suffix',default='');a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);core=setup(a.task,a.gpu)
 import x1.data as data
 from r8_m2_x1_runner import R8CaseView
 planning=core.r3('planning');planning.load_official_api();import stable_worldmodel.policy as swmp;swm=planning.load_official_api()[0];cfg=core.CONFIG[a.task];roles=core.read(core.ROOT/'manifests'/f'{a.task}_data_roles.json');
 with data.RawH5(core.verify(roles['source'])) as raw:
  manifest=json.loads((ROOT/'ops'/f'M2_CASE_WINDOWS_{a.task.upper()}.json').read_text());rows=[]
  for c0 in [e['case'] for e in manifest['cases'][a.case_start:a.case_end]]:
   for seed in SEEDS:
    world=None
    try:
     world=swm.World(**cfg['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224)); original=world.reset
     if a.task=='cube':
      from c0.reset import clear_internal
      import mujoco
      def reset(seed=None,options=None):
       if seed is not None:raise ValueError('Cube random reset must be seedless')
       out=original(seed=None,options=options);clear_internal(world.envs.envs[0].unwrapped,mujoco);return out
      world.reset=reset
     else:
      norm=core.r3('env_compat').normalize_reset_seed;world.reset=lambda seed=None,options=None: original(seed=norm(seed),options=options)
     pol=swmp.RandomPolicy();pol.set_seed(int(seed));world.set_policy(pol);view=R8CaseView(raw,a.task,c0)
     metrics=world.evaluate(dataset=view,episodes_idx=[c0['source_episode_idx']],start_steps=[c0['start_raw_index']],goal_offset=25,eval_budget=50,callables=cfg['callables'],video=None)
     rows.append({'task':a.task,'case_id':c0['case_id'],'episode_id':c0['episode_id'],'seed':int(seed),'success':int(bool(metrics['episode_successes'][0])),'executed_raw_steps':50,'error':''})
    except BaseException as e: rows.append({'task':a.task,'case_id':c0['case_id'],'episode_id':c0['episode_id'],'seed':int(seed),'success':0,'executed_raw_steps':0,'error':f'{type(e).__name__}:{e}'})
    finally:
     if world is not None:
      if a.task=='cube':
       r=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
       if r is not None:r.close()
      world.close()
 out=ROOT/'raw'/'M2'/'random';out.mkdir(parents=True,exist_ok=True);pout=out/f'{a.task}_RANDOM_RAW_VALUES{a.output_suffix}.csv'
 with pout.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/f'{a.task}_RANDOM_STATUS{a.output_suffix}.json').write_text(json.dumps({'status':'COMPLETE','task':a.task,'rows':len(rows),'seeds':list(SEEDS),'errors':sum(bool(r['error']) for r in rows)},indent=2)+'\n');print(json.dumps({'status':'COMPLETE','task':a.task,'rows':len(rows)}),flush=True)
if __name__=='__main__':main()
