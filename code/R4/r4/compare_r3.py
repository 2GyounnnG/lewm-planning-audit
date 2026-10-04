"""R3 reproduction is a per-case report, never a new-host scientific gate."""
import argparse,csv,json
from pathlib import Path
from .common import ARMS,atomic_json

def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--trajectories',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.r3_root);cases=json.loads((root/f'manifests/{a.task}_data_roles.json').read_text())['cases']['EVAL'];rows=[]
    for case in cases:
        for arm in ARMS:
            row={'task':a.task,'case_id':case['case_id'],'arm':arm}
            old_folder=root/'artifacts/planning/FORMAL'/a.task/case['case_id']/arm
            options=sorted(old_folder.glob('attempt_*/TRAJECTORY_SHA256.json'))
            new_folder=Path(a.trajectories)/'FORMAL/R3_ORIGINAL'/case['case_id']/arm
            if not options or not (new_folder/'COMPLETE.json').exists():row['status']='MISSING_R3_OR_R4_TRAJECTORY';rows.append(row);continue
            old_folder=options[-1].parent;old=json.loads((old_folder/'result.json').read_text());new=json.loads((new_folder/'result.json').read_text())
            with np.load(old_folder/'trajectory.npz',allow_pickle=False) as f:oldarr={k:f[k].copy() for k in f.files}
            with np.load(new_folder/'trajectory.npz',allow_pickle=False) as f:newarr={k:f[k].copy() for k in f.files}
            n=min(len(oldarr['raw_actions']),len(newarr['raw_actions']));state=newarr.get('raw_info_state')
            row.update(status='REPRODUCTION_REPORT_ONLY',r3_success=old['success'],r4_success=new['success'],r3_raw_steps=old['executed_raw_steps'],r4_raw_steps=new['executed_raw_steps'],r3_replan_calls=old.get('replan_calls',len(old.get('replans',[]))),r4_replan_calls=new['replan_calls'],common_raw_steps=n,
                action_max_abs=float(np.max(np.abs(oldarr['raw_actions'][:n]-newarr['raw_actions'][:n]))) if n else None,
                state_max_abs=float(np.max(np.abs(oldarr['physical_state'][:n]-state[1:n+1]))) if n and state is not None else None)
            rows.append(row)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'R3_REPRODUCTION_RAW.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    atomic_json(out/'R3_REPRODUCTION.json',{'rows':len(rows),'status':'REPORT_ONLY_NONBLOCKING','new_optimizer_updates':0})

if __name__=='__main__':main()
