"""TECH-only evidence for fixed25 legality using already logged H0 returns."""
import argparse,json,os,sys
from pathlib import Path
from .common import atomic_json,atomic_npz,file_record,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--trajectories',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import numpy as np
    from r3.data import action_processor
    from .replay import ReplayBranch
    from .export_cases import CaseWindows
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks']['reacher']['cases'];entries=[e for e in entries if e['case']['role']=='TECH'][:4];rows=[];steps=[]
    for entry in entries:
        case=entry['case'];source=Path(a.trajectories)/'TECH/R3_ORIGINAL'/case['case_id']/'H0';receipt=json.loads((source/'COMPLETE.json').read_text())
        if file_record(source/'trajectory.npz')!=receipt['files']['trajectory.npz']:raise RuntimeError('Sealed TECH trajectory differs')
        with np.load(source/'trajectory.npz') as f:candidate=action_processor('reacher').inverse_transform(f['returned_plans_normalized'][0].reshape(25,2)).astype(np.float32)
        with ReplayBranch('reacher',case,CaseWindows(a.assets,entry)) as branch:
            for i,action in enumerate(candidate):
                if branch.true_terminated:break
                value=branch.step(action);steps.append({'case_id':case['case_id'],'raw':i+1,'success':value['success'],'true_terminated':value['true_terminated'],'physics_steps_cumulative':value['physics_steps'],'margin':value['margin']})
            rows.append({'case_id':case['case_id'],'source_trajectory':receipt['files']['trajectory.npz'],'probe':'FIRST_FROZEN_H0_CEM_RETURNED25','requested_raw':25,'executed_raw':branch.steps,'true_terminated':branch.true_terminated,'fixed_horizon_legal':branch.steps==25,'physics_steps':branch.physics_steps,'prevented_auto_reset':branch.true_terminated and branch.steps<25})
    out=Path(a.output);result={'status':'COMPLETE_TECHNICAL_OBSERVATION','check':'TRUE_TERMINATION_H0_RETURNED_PLAN','task':'reacher','rows':rows,'steps':steps,'fixed25_unavailable_observed':any(not r['fixed_horizon_legal'] for r in rows),'reset_calls':len(rows),'raw_steps':sum(r['executed_raw'] for r in rows),'optimizer_updates':0,'AA_or_order_check_branches':0,'new_complete_TECH_episodes':0,'scope':'TECH_NOT_FORMAL','code_sha256':sha256(__file__)}
    atomic_json(out/'TERMINATION_PLAN_TECH_RAW.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
