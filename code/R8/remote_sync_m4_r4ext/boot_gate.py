"""R4-v2.3 H0/refit open-loop re-evaluation on frozen R3 offline anchors."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from .common import atomic_json,atomic_npz,file_record

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import numpy as np
    import torch
    from r3.model import load_official,apply_delta,cached_rollout
    from r3.common import fp32_policy
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);fp32_policy()
    root=Path(a.r3_root);folder=root/'artifacts/open_loop'/a.task
    with np.load(folder/'inputs.npz') as f:inputs={k:f[k].copy() for k in f.files}
    with np.load(folder/'targets.npz') as f:target=f['target_z'].copy()
    routing=json.loads((root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][a.task]['arms'];rows=[]
    outputs={};tol=json.loads((root/'manifests/NUMERICAL_TOLERANCES.json').read_text())['comparisons']['rollout_wrapper']
    for arm in ('H0','REFIT_103201_30000'):
        model=load_official(a.task,a.device)
        if arm!='H0':
            r=next(r for r in routing if r['arm']==arm);cp=root/r['checkpoint']['path']
            if file_record(cp)!={k:r['checkpoint'][k] for k in ('bytes','sha256')}:raise RuntimeError('Checkpoint bytes differ')
            apply_delta(model,cp)
        with torch.inference_mode():
            values=[]
            for lo in range(0,len(target),32):
                values.append(cached_rollout(model,torch.from_numpy(inputs['initial_z'][lo:lo+32]).to(a.device),torch.from_numpy(inputs['macro_actions'][lo:lo+32]).to(a.device),5).cpu().numpy())
        pred=np.concatenate(values);outputs[arm]=pred
        with np.load(folder/'predictions'/(arm+'.npz')) as f:old=f['prediction'].copy()
        if old.shape!=pred.shape:raise RuntimeError('Prediction shape mismatch')
        old_mse=np.mean((old-target)**2,axis=2);new_mse=np.mean((pred-target)**2,axis=2)
        for i,case_id in enumerate(inputs['case_ids']):
            for h in (1,2,5):
                rel=float(abs(new_mse[i,h-1]-old_mse[i,h-1])/max(abs(old_mse[i,h-1]),1e-30))
                exact_tol=bool(np.allclose(pred[i,h-1],old[i,h-1],**tol))
                rows.append({'task':a.task,'case_id':str(case_id),'arm':arm,'h':h,'old_latent_mse':float(old_mse[i,h-1]),'new_latent_mse':float(new_mse[i,h-1]),'relative_mse_difference':rel,'r3_frozen_tolerance_pass':exact_tol,'v23_exception_pass':rel<1e-3,'pass':exact_tol or rel<1e-3})
        del model
    result={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL_STOP_AFFECTED_R4_TASK','task':a.task,'criterion':'original prediction tolerance OR each anchor latent MSE relative difference <1e-3 (R4-v2.3 section2.2)','rows':rows,'optimizer_updates':0}
    out=Path(a.output);atomic_npz(out/'BOOT_PREDICTIONS.npz',**outputs);atomic_json(out/'BOOT_GATE.json',result)
    import csv
    with (out/'boot_anchor_raw.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'status':result['status'],'rows':len(rows),'max_relative_mse_difference':max(r['relative_mse_difference'] for r in rows)}))
    if result['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
