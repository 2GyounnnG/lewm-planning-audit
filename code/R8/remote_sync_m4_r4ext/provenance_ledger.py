"""Noninferential termination and fixed-middle-fork coverage sidecars."""
import argparse,csv,json
from pathlib import Path
from .common import ARMS,atomic_json,file_record,fixed_subset,sha256

def save_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def flags(folder):
    import numpy as np
    folder=Path(folder);receipt=json.loads((folder/'COMPLETE.json').read_text())
    for name in ('result.json','trajectory.npz'):
        if file_record(folder/name)!=receipt['files'][name]:raise RuntimeError('Original seal differs')
    result=json.loads((folder/'result.json').read_text())
    with np.load(folder/'trajectory.npz',allow_pickle=False) as data:
        success=data['step_success'];truncated=data['step_truncated'];n=len(data['raw_actions'])
        if n!=len(success) or n!=len(truncated) or n!=result['executed_raw_steps']:raise RuntimeError('Step flags misaligned')
        first_success=int(np.flatnonzero(success)[0])+1 if success.any() else None
        first_trunc=int(np.flatnonzero(truncated)[0])+1 if truncated.any() else None
    failure=result.get('method_failure')
    reason='METHOD_FAILURE' if failure else 'WORLD_TRUNCATED' if first_trunc else 'WORLD_SUCCESS_TERMINATED' if first_success else 'RAW_BUDGET_EXHAUSTED' if n==50 else 'OTHER_SOURCE_STOP_REQUIRES_AUDIT'
    return {'task':result['task'],'case_id':result['case_id'],'source_arm':result['arm'],'stream':result['stream'],
        'source_executed_raw_steps':n,'source_logged_observations':result['logged_observations'],
        'world_terminated_any':first_success is not None,'world_truncated_any':first_trunc is not None,
        'first_world_terminated_raw':first_success,'first_world_truncated_raw':first_trunc,
        'early_success_before_raw_budget':first_success is not None and first_success<50,
        'raw_budget_50_reached':n==50,'true_dmc_last_implied_by_pinned_env':result['task']=='reacher' and first_success is not None,
        'success':result['success'],'method_failure':failure,'stop_reason':reason,
        'source_result_sha256':receipt['files']['result.json']['sha256'],'source_trajectory_sha256':receipt['files']['trajectory.npz']['sha256']}

def suffix_ledger(source,output):
    rows=[flags(p.parent) for p in sorted((Path(source)/'FORMAL/R3_ORIGINAL').glob('*/*/COMPLETE.json')) if p.parent.name in ARMS]
    if len(rows)!=400:raise RuntimeError('S1 source ledger requires all four original100 trajectories')
    path=Path(output)/'S1_SOURCE_STOP_FLAGS_RAW.csv';save_rows(path,rows)
    record={'status':'COMPLETE_NONINFERENTIAL_SIDECAR','rows':400,'raw':file_record(path),'code_sha256':sha256(__file__),
        'scoring_and_frozen_statistics_unchanged':True,
        'semantics':'The original step_success stores World terminated, while step_truncated stores World truncated. PushT success does not prohibit physical continuation; Reacher success implies real dm_control LAST under the separately locked and TECH-verified interface. This is a source-log ledger, not a new termination trial. Flags are nonexclusive.',
        'suffix_rule':'S1 EXECUTED_SUFFIX_TOO_SHORT stays unchanged. Join task,case_id,source_arm and original stream to explain the observed source endpoint. No missing future is imputed.'}
    atomic_json(Path(output)/'S1_SOURCE_STOP_FLAGS_COMPLETE.json',record);return record

def middle_ledger(task,roles,source,output):
    cases=fixed_subset(task,json.loads(Path(roles).read_text())['cases']['EVAL']);rows=[]
    for case in cases:
        folder=Path(output)/case['case_id'];path=folder/'result.json';result=json.loads(path.read_text()) if path.exists() else None
        original=Path(source)/'FORMAL/R3_ORIGINAL'/case['case_id']/'H0';logged=flags(original)
        reason=(result or {}).get('missing_reason')
        rows.append({'task':task,'case_id':case['case_id'],'family_id':case['family_id'],'fork':'middle_raw10',
            'fixed_metadata_case_no_replacement':True,'status':result['status'] if result else 'NOT_EXECUTED',
            'fork_available':None if result is None else result['status']!='UNAVAILABLE_MID_FORK',
            'missing_reason':reason,'source_executed_raw_steps':logged['source_executed_raw_steps'],
            'source_world_terminated_raw':logged['first_world_terminated_raw'],
            'source_world_truncated_raw':logged['first_world_truncated_raw'],
            'logical_candidates_observed':(result or {}).get('logical_candidates'),
            'valid_fixed_horizon_candidates':(result or {}).get('valid_fixed_horizon_candidates'),
            'source_trajectory_sha256':logged['source_trajectory_sha256'],
            'middle_result_sha256':file_record(path)['sha256'] if result else None})
    if len(rows)!=20:raise RuntimeError('Exactly20 fixed metadata cases required')
    path=Path(output)/'MIDDLE_FORK_AVAILABILITY_RAW.csv';save_rows(path,rows)
    result={'status':'COMPLETE' if all(r['status']!='NOT_EXECUTED' for r in rows) else 'INCOMPLETE','fixed_cases':20,
        'observed_results':sum(r['status']!='NOT_EXECUTED' for r in rows),'raw':file_record(path),'code_sha256':sha256(__file__),
        'source_role_manifest':file_record(roles),'replacement_cases':0,'optimizer_updates':0,'new_simulator_calls':0}
    atomic_json(Path(output)/'MIDDLE_FORK_AVAILABILITY_COMPLETE.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['s1','middle']);p.add_argument('--trajectories',required=True);p.add_argument('--output',required=True);p.add_argument('--task',choices=['pusht','reacher']);p.add_argument('--roles');a=p.parse_args()
    if a.mode=='middle' and not (a.task and a.roles):p.error('Middle ledger needs fixed task/role manifest')
    print(json.dumps(suffix_ledger(a.trajectories,a.output) if a.mode=='s1' else middle_ledger(a.task,a.roles,a.trajectories,a.output)),flush=True)

if __name__=='__main__':main()
