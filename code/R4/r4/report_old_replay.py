"""Nonblocking new-host replay of original R3 TECH actions, never new outcomes."""
import argparse,csv,json,os,sys
from pathlib import Path
from .common import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import numpy as np
    from .export_cases import CaseWindows
    from .replay import ReplayBranch
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks'][a.task]['cases'];entries=[e for e in entries if e['case']['role']=='TECH'][:2];rows=[]
    for entry in entries:
        case=entry['case'];paths=sorted((Path(a.r3_root)/'artifacts/planning/TECH'/a.task/case['case_id']/'H0').glob('attempt_*/TRAJECTORY_SHA256.json'))
        if not paths:
            rows.append({'task':a.task,'case_id':case['case_id'],'raw':None,'status':'MISSING_ORIGINAL_TECH_LOG'});continue
        with np.load(paths[-1].parent/'trajectory.npz',allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
        with ReplayBranch(a.task,case,CaseWindows(a.assets,entry)) as branch:
            for i,action in enumerate(arrays['raw_actions']):
                if branch.true_terminated:
                    rows.append({'task':a.task,'case_id':case['case_id'],'raw':i+1,'status':'NEW_HOST_TRUE_TERMINATED_BEFORE_R3_SUFFIX'});break
                value=branch.step(action);difference=value['state']-arrays['physical_state'][i]
                rows.append({'task':a.task,'case_id':case['case_id'],'raw':i+1,'status':'NONBLOCKING_R3_REPRODUCTION_REPORT','state_max_abs':float(np.max(np.abs(difference))),'state_l2':float(np.linalg.norm(difference)),'frozen_tolerance_pass':bool(np.allclose(value['state'],arrays['physical_state'][i],atol=1e-5,rtol=1e-5)),'r3_success':bool(arrays['step_success'][i]),'new_host_success':value['success'],'true_terminated':value['true_terminated']})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'R3_TECH_REPLAY_RAW.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    atomic_json(out/'R3_TECH_REPLAY_REPORT.json',{'status':'NONBLOCKING_REPORT_COMPLETE','rows':len(rows),'complete_new_CEM_trajectories':0,'optimizer_updates':0,'interpretation':'Old-host step agreement is reported, not an R4-v2.3 gate'})

if __name__=='__main__':main()
