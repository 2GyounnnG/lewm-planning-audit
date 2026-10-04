"""Bounded TECH capability probe: first before25 LAST ends diagnostic search."""
import argparse,json,os,sys
from pathlib import Path
from .common import atomic_json,atomic_npz,file_record,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--trajectories',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import numpy as np
    from r3.data import action_processor
    from .replay import run_branch
    from .export_cases import CaseWindows
    from .candidates import build_menu
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks']['reacher']['cases'];entries=[e for e in entries if e['case']['role']=='TECH'][:4];out=Path(a.output);rows=[];found=False
    atomic_json(out/'TERMINATION_MENU_TECH_PROTOCOL.json',{'version':'R4_TECH_FIXED_MENU_TERMINATION_V1','cases':[e['case']['case_id'] for e in entries],'maximum_branches':256,'order':'original frozen TECH order, menu candidate index order','stop':'first actual LAST before25 proves unavailable fixed horizon; remaining probes not needed','menu':'S3 fixed H0 returned+47 hash samples+16 independent initial candidates from already logged TECH first CEM; no new model training or complete episodes','formal_samples':0,'code_sha256':sha256(__file__)})
    for entry in entries:
        case=entry['case'];source=Path(a.trajectories)/'TECH/R3_ORIGINAL'/case['case_id']/'H0';receipt=json.loads((source/'COMPLETE.json').read_text())
        if file_record(source/'trajectory.npz')!=receipt['files']['trajectory.npz']:raise RuntimeError('TECH source changed')
        with np.load(source/'trajectory.npz') as f:proposal={'returned':f['returned_plans_normalized'][0].copy(),'last_generation':f['last_generation_normalized'][0].copy()}
        menu=build_menu('reacher',case['case_id'],'R3_ORIGINAL/replan_00',{'H0':proposal},action_processor('reacher'),kind='S3',device='cpu')
        folder=out/'termination_menu_probes'/case['case_id'];atomic_npz(folder/'menu.npz',normalized_actions=menu['normalized'],raw_actions=menu['raw']);atomic_json(folder/'MENU_LOCK.json',{'menu_sha256':menu['menu_sha256'],'metadata':menu['metadata'],'locked_before_truth':True})
        for i,metadata in enumerate(menu['metadata']):
            result=run_branch('reacher',case,lambda:CaseWindows(a.assets,entry),np.empty((0,2)),menu['raw'][i])
            atomic_npz(folder/f'branch_{i:02d}.npz',**{k:v for k,v in result.items() if k not in ('terminal_pixels','scope')})
            rows.append({'case_id':case['case_id'],'candidate_index':i,'candidate_id':metadata['id'],'source_trajectory':receipt['files']['trajectory.npz'],'menu_sha256':menu['menu_sha256'],'requested_raw':25,'executed_raw':result['candidate_raw_steps'],'fixed_horizon_legal':bool(result['valid_fixed_horizon']),'missing_reason':result['missing_reason'],'success':bool(result['any_success']),'physics_steps':result['physics_steps'],'branch_seconds':result['seconds']})
            if not result['valid_fixed_horizon']:found=True;break
        if found:break
    result={'status':'COMPLETE_TECHNICAL_OBSERVATION','check':'TRUE_TERMINATION_FIXED_MENU','task':'reacher','rows':rows,'fixed25_unavailable_observed':found,'reset_calls':len(rows),'raw_steps':sum(r['executed_raw'] for r in rows),'AA_or_order_check_branches':0,'new_complete_TECH_episodes':0,'optimizer_updates':0,'scope':'TECH_NOT_FORMAL','code_sha256':sha256(__file__)}
    atomic_json(out/'TERMINATION_MENU_TECH_RAW.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
