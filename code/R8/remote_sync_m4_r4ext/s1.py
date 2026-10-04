"""History and visited-trajectory cross scoring in frozen official coordinates."""
from __future__ import annotations
import argparse,csv,json,os,sys
from pathlib import Path
from .common import ARMS,atomic_json,atomic_npz,file_record

def predict(model,initial,actions,horizon):
    import torch
    from r3.model import cached_rollout
    device=next(model.parameters()).device
    with torch.inference_mode():return cached_rollout(model,torch.as_tensor(initial,device=device),torch.as_tensor(actions,device=device),horizon).cpu().numpy()

def offline(model,task,arm,r3_root):
    import numpy as np
    folder=Path(r3_root)/'artifacts/open_loop'/task
    with np.load(folder/'inputs.npz') as f:inputs={k:f[k].copy() for k in f.files}
    with np.load(folder/'targets.npz') as f:target=f['target_z'].copy()
    roles=json.loads((Path(r3_root)/f'manifests/{task}_data_roles.json').read_text());cases={c['case_id']:c for c in roles['cases']['EVAL']};rows=[]
    for history in ('H_POLICY','H_REAL3'):
        initial=inputs['initial_z'][:,-1:] if history=='H_POLICY' else inputs['initial_z']
        actions=inputs['macro_actions'][:,2:] if history=='H_POLICY' else inputs['macro_actions']
        preds=[]
        for lo in range(0,len(initial),32):preds.append(predict(model,initial[lo:lo+32],actions[lo:lo+32],5))
        pred=np.concatenate(preds)
        for i,cid in enumerate(inputs['case_ids']):
            for h in (1,2,5):
                case=cases[str(cid)];rows.append({'task':task,'case_id':str(cid),'family_id':case['family_id'],'model':arm,'source_policy':'OFFLINE_RECORDED','anchor_raw':case['open_loop_anchor_raw'],'history_kind':history,'horizon_raw':5*h,'valid':True,'missing_reason':'','latent_mse':float(np.mean((pred[i,h-1]-target[i,h-1])**2)),'predicted_goal_cost':None,'realized_goal_cost':None,'optimism':None,'source_relation':'OFFLINE','primary_anchor':True})
    return rows

def cross_trajectory(model,task,model_arm,case,source,arrays,action_processor):
    import numpy as np
    rows=[];actions=arrays['raw_actions'];z=arrays['raw_latent'];goal=arrays['goal_latent']
    replans=set(map(int,arrays['replan_raw']));anchors=sorted(replans | set(range(0,len(actions),5)))
    normalized=action_processor.transform(actions).astype(np.float32)
    for anchor in anchors:
        for history in ('H_POLICY','H_REAL3'):
            past=0 if history=='H_POLICY' else 10
            for h in (1,2,5):
                row={'task':task,'case_id':case['case_id'],'family_id':case['family_id'],'model':model_arm,'source_policy':source,'anchor_raw':anchor,'history_kind':history,'horizon_raw':h*5,'valid':False,'missing_reason':'','latent_mse':None,'predicted_goal_cost':None,'realized_goal_cost':None,'optimism':None,'source_relation':'OWN' if model_arm==source else 'OTHER','primary_anchor':anchor==min(replans) if replans else False}
                if anchor<past:row['missing_reason']='INSUFFICIENT_LEGAL_PAST'
                elif anchor+5*h>len(actions):row['missing_reason']='EXECUTED_SUFFIX_TOO_SHORT'
                else:
                    idx=np.arange(anchor-past,anchor+1,5);a=normalized[anchor-past:anchor+5*h].reshape(1,-1,10)
                    terminal=predict(model,z[idx][None],a,h)[0,-1];target=z[anchor+5*h]
                    jp=float(np.sum((terminal-goal)**2));jr=float(np.sum((target-goal)**2))
                    row.update(valid=True,latent_mse=float(np.mean((terminal-target)**2)),predicted_goal_cost=jp,realized_goal_cost=jr,optimism=jp-jr)
                rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--module',choices=['offline','cross'],required=True);p.add_argument('--trajectories');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import torch,numpy as np
    from r3.model import load_official,apply_delta
    from r3.data import action_processor
    from r3.common import fp32_policy
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);fp32_policy();root=Path(a.r3_root);rows=[]
    routes=json.loads((root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][a.task]['arms']
    cases=json.loads((root/f'manifests/{a.task}_data_roles.json').read_text())['cases']['EVAL']
    for arm in ARMS:
        model=load_official(a.task,a.device)
        if arm!='H0':
            route=next(r for r in routes if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]));cp=root/route['checkpoint']['path']
            if file_record(cp)!={k:route['checkpoint'][k] for k in ('bytes','sha256')}:raise RuntimeError('Fixed checkpoint changed')
            apply_delta(model,cp)
        if a.module=='offline':rows.extend(offline(model,a.task,arm,root))
        else:
            if not a.trajectories:raise ValueError('New-host trajectories required')
            for case in cases:
                for source in ARMS:
                    folder=Path(a.trajectories)/'FORMAL/R3_ORIGINAL'/case['case_id']/source
                    if not (folder/'COMPLETE.json').exists():raise RuntimeError('Original rerun incomplete; cross scoring waits for all four sources')
                    with np.load(folder/'trajectory.npz',allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
                    if 'raw_latent' not in arrays:raise RuntimeError('Source trajectory lacks scored observed latents')
                    rows.extend(cross_trajectory(model,a.task,arm,case,source,arrays,action_processor(a.task)))
        del model
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/f'S1_{a.module}_raw.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(out/f'S1_{a.module}_COMPLETE.json',{'status':'COMPLETE','rows':len(rows),'valid':sum(r['valid'] for r in rows),'optimizer_updates':0,'comparison_unit':'case; sources and times are paired repeated measures','history_path':'r3.model.cached_rollout uses official model.predict and source-matched 3-token autoregressive truncation'})
    print(json.dumps({'module':a.module,'rows':len(rows),'valid':sum(r['valid'] for r in rows)}))

if __name__=='__main__':main()
