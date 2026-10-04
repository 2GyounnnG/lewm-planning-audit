"""Final local coverage and source-seal checks for a completed R4 main line."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from deliver import record,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--r3-root',required=True)
    p.add_argument('--task',choices=('pusht','reacher'),required=True);a=p.parse_args()
    root=Path(a.root);task=a.task;folder=root/f'r4/reports/remote_{task}_primary_tables'
    values=root/f'r4/reports/remote_{task}_primary_values/s2';manifest=json.loads((folder/'FINAL_PRIMARY_MANIFEST.json').read_text())
    checked=[]
    for name,expected in manifest['files'].items():
        path=folder/name;actual=record(path)
        if any(actual[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('Final source seal mismatch: '+name)
        checked.append(actual)
    for name,sha in manifest['frozen_analysis'].items():
        if record(root/'analysis'/name)['sha256']!=sha:raise ValueError('Final source analysis is not frozen source')
    gate=json.loads((folder/'MAIN_MODULE_STATUS.json').read_text())
    if gate['status'] not in ('COMPLETE','COMPLETE_WITH_TECHNICAL_LIMITATIONS'):raise ValueError('R4 main line incomplete')
    if gate['new_neural_training']!=0:raise ValueError('Unexpected new R4 neural training')
    for module,expected in [('original400',400),('alternate800',800),('H0_MENU_RERANK',60),('S1_offline',2400)]:
        if gate['modules'][module]['status']!='COMPLETE' or gate['modules'][module]['completed']!=expected:raise ValueError('Main module incomplete: '+module)
    s3=json.loads((root/f'manifests/main_validation/r4_{task}_s3.json').read_text())
    if s3['status']!='COMPLETE':raise ValueError('Independent S3 reconstruction incomplete')
    cases={r['case_id'] for r in json.loads((Path(a.r3_root)/f'manifests/{task}_data_roles.json').read_text())['cases']['EVAL']}
    folders=sorted(p.parent for p in values.glob('*/candidate_raw.csv'))
    if len(folders)!=100 or {p.name for p in folders}!=cases:raise ValueError('S2 fixed case universe incomplete')
    import numpy as np
    total=valid=0;invalid_forks=0;source_receipts=[]
    for case in folders:
        complete=json.loads((case/'COMPLETE.json').read_text())
        for name in ('candidate_raw.csv','MENU_LOCK.json','menu.npz','result.json'):
            actual=record(case/name)
            if any(actual[k]!=complete['files'][name][k] for k in ('bytes','sha256')):raise ValueError('S2 source seal mismatch: '+str(case/name))
        source_receipts.append(record(case/'COMPLETE.json'))
        lock=json.loads((case/'MENU_LOCK.json').read_text());meta=lock['metadata']
        with (case/'candidate_raw.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=64 or {r['id'] for r in rows}!={r['id'] for r in meta}:raise ValueError('S2 logical menu mismatch')
        if not lock['locked_before_any_simulator_query']:raise ValueError('Menu must predate simulator scoring')
        if sum(r['source']=='INITIAL_DISTRIBUTION' for r in meta)!=16:raise ValueError('Initialization menu size')
        for arm in ('H0','REFIT_103201','REFIT_103202','REFIT_103203'):
            if sum(r['source']==arm for r in meta)!=12:raise ValueError('Per-model menu size')
        with np.load(case/'menu.npz',allow_pickle=False) as z:
            actions=z['raw_actions'];normalized=z['normalized_actions']
            if actions.shape!=(64,25,2) or normalized.shape!=(64,5,10):raise ValueError('Locked original action geometry')
            for r in meta:
                if hashlib.sha256(actions[r['logical_index']].tobytes(order='C')).hexdigest()!=r['raw_action_sha256']:raise ValueError('Menu action digest differs')
        n=sum(r['valid_fixed_horizon'].lower()=='true' for r in rows);valid+=n;total+=64;invalid_forks+=n<64
    if total!=6400 or gate['modules']['S2_initial']['completed']!=100:raise ValueError('Initial fork coverage incomplete')
    if task=='pusht' and valid!=6400:raise ValueError('Unexpected PushT fixed-horizon missingness')
    if task=='reacher':
        if gate['status']!='COMPLETE_WITH_TECHNICAL_LIMITATIONS' or not gate['technical_limitations']:raise ValueError('Reacher technical limitation missing')
        for role in ('SIM_LAT_RERANK','SIM_TASK_RERANK'):
            if gate['modules'][role]['status']!='TECHNICALLY_UNEVALUABLE' or gate['modules'][role]['completed'] is not None:raise ValueError('Unevaluable SIM role must not masquerade as completed or zero success')
    result={'status':gate['status'],'reason':json.dumps(gate['technical_limitations'],ensure_ascii=False) if gate['technical_limitations'] else None,'task':task,'new_neural_training':0,'S2_candidates':total,'valid_fixed_horizon_candidates':valid,'forks_with_technical_missing_horizon':invalid_forks,'main_files_and_menu_sources_sha_verified':True,'independent_s3_reconstruction':record(root/f'manifests/main_validation/r4_{task}_s3.json'),'inputs':checked,'S2_source_completion_records':source_receipts,'full_trajectory_recovery_is_separate':True}
    atomic(root/f'manifests/main_validation/r4_{task}_main.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','S2_source_completion_records','independent_s3_reconstruction')}))

if __name__=='__main__':main()
