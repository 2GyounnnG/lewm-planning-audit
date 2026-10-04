from __future__ import annotations
import argparse,csv,json,os,sys
from pathlib import Path
import numpy as np
SEEDS=(8001,8002,8003)
def write(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['reacher','pusht'],required=True);p.add_argument('--root',default='/workspace/r8');p.add_argument('--r3-root',default='/workspace/shared_data/r3');a=p.parse_args();root=Path(a.root);os.environ['R3_ROOT']=a.r3_root;sys.path.insert(0,a.r3_root)
 import torch
 from r3 import planning as pl
 from r3.env_compat import normalize_reset_seed
 from r4.export_cases import CaseWindows
 pl.load_official_api(); import stable_worldmodel.policy as swmp
 man=json.loads((root/'ops'/f'M2_CASE_WINDOWS_{a.task.upper()}.json').read_text());rows=[]; assets=root/'assets'/'M2'/a.task
 for ent in man['cases']:
  c=ent['case']
  for seed in SEEDS:
   world=None
   try:
    kwargs={'task':'qpos_match'} if a.task=='reacher' else {};world=pl.load_official_api()[0].World(env_name='swm/PushT-v1' if a.task=='pusht' else 'swm/ReacherDMControl-v0',num_envs=1,max_episode_steps=100,image_shape=(224,224),**kwargs)
    orig=world.reset;world.reset=lambda seed=None,options=None:orig(seed=normalize_reset_seed(seed),options=options)
    pol=swmp.RandomPolicy();world.set_policy(pol);pol.set_seed(int(seed));view=CaseWindows(assets,ent)
    # CaseWindows has no seed column; inject a wrapper with the official fixed reset seed.
    base=view
    class V:
     def __init__(self):self.column_names=list(base.column_names)+['seed'];self.goal_state=None;self.reads=0
     def load_chunk(self,episodes,starts,ends):
      out=base.load_chunk(episodes,starts,ends)[0];out['seed']=torch.full((len(next(iter(out.values()))),),int(c['reset_seed']),dtype=torch.int64);self.goal_state=np.asarray(out['qpos' if a.task=='reacher' else 'state'][-1]).copy();self.reads+=1;return [out]
    metrics=world.evaluate(dataset=V(),episodes_idx=[c['source_episode_idx']],start_steps=[c['start_raw_index']],goal_offset=25,eval_budget=50,callables=pl.CALLABLES[a.task],video=None)
    rows.append({'task':a.task,'case_id':c['case_id'],'seed':int(seed),'success':int(bool(metrics['episode_successes'][0])),'executed_raw_steps':50,'error':''})
   except BaseException as e:rows.append({'task':a.task,'case_id':c['case_id'],'seed':int(seed),'success':0,'executed_raw_steps':0,'error':f'{type(e).__name__}:{e}'})
   finally:
    if world is not None:world.close()
 out=root/'raw'/f'M2_RANDOM_{a.task.upper()}_RAW_VALUES.csv';write(out,rows);print(json.dumps({'task':a.task,'rows':len(rows),'seeds':SEEDS}))
if __name__=='__main__':main()
