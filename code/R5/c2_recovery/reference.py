"""Post-control derived GPU reference for first fixed case recovery; no simulation."""
import os
from pathlib import Path
import numpy as np
from c2.common import *

def main():
    import torch
    from x1 import core
    from x1.evaluate import load_arm
    core.ROOT=Path('/workspace/x1_cube');core.policy();c=read(ROOT/'C2_CONTRACT.json');case=c['cases']['EVAL'][0]
    if not (ROOT/'reports/MODULE_STATUS.json').exists():raise RuntimeError('Only after C2 formal work finishes')
    out=ROOT/'recovery/cpu_reference';arrays={};sources=[]
    for stream in ('R4_ALT_CEM_1','R4_ALT_CEM_2'):
      for arm in ARMS:
        model=load_arm('cube',arm,'cuda');folder=ROOT/'closed_loop/EVAL'/stream/arm/case['case_id'];r=read(folder/'result.json')
        with np.load(folder/'trajectory.npz') as f:z=f['raw_latent'];goal=f['goal_latent']
        with np.load(folder/'plans.npz') as f:
            for item in r['replans']:
                i=item['replan_index'];idx=item['raw_index'];actions=f[f'{i}:returned_plan'].copy();history=z[idx:idx+1][None]
                with torch.inference_mode():prediction=core.r3('model').cached_rollout(model,torch.from_numpy(history).cuda(),torch.from_numpy(actions).cuda(),5).cpu().numpy()
                key=f'{arm}:{stream}:{i}';arrays[key+':history']=history;arrays[key+':actions']=actions;arrays[key+':goal']=goal;arrays[key+':prediction']=prediction;arrays[key+':cost']=np.sum((prediction[:,-1]-goal)**2,axis=-1)
        sources.append({'arm':arm,'stream':stream,'result':record(folder/'result.json'),'plans':record(folder/'plans.npz'),'trajectory':record(folder/'trajectory.npz')})
    save_npz(out/'FIRST_CASE_REFERENCE.npz',**arrays)
    atomic(out/'REFERENCE_MANIFEST.json',{'status':'READY','case':case,'source_scope':'metadata-order first EVAL, every fixed model and both ALT streams, all actually executed replans','no_new_simulation':True,'reference':record(out/'FIRST_CASE_REFERENCE.npz'),'sources':sources,'source_code':record(__file__),'tolerance':{'atol':1e-5,'rtol':1e-5,'source':'R3 frozen CPU_replay'},'device':'cuda:0','physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']})
if __name__=='__main__':main()
