"""Second-batch-only packaging; no extra simulator or model queries."""
import argparse,csv,json,shutil,tarfile,time
from pathlib import Path
from .common import atomic_json,file_record,fixed_subset,sha256,digest
from .provenance_ledger import middle_ledger

def write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def closeout(line,task,root,primary,wait=False):
    from analysis.aggregate import s2
    line,root=Path(line),Path(root);primary=Path(primary)
    if json.loads(primary.read_text()).get('status')!='MAIN_RESULTS_DELIVERED':raise RuntimeError('Actual primary delivery required')
    roles=root/'manifests'/f'{task}_data_roles.json';cases=fixed_subset(task,json.loads(roles.read_text())['cases']['EVAL'])
    directory=line/'s2_middle';directory.mkdir(exist_ok=True);out=line/'secondary_reports';out.mkdir(exist_ok=True)
    if task=='reacher':
        gate=line/'SIM_TECHNICAL_LIMITATION.json';g=json.loads(gate.read_text())
        if g.get('status')!='TECHNICALLY_UNEVALUABLE' or g.get('reason')!='TRUE_DMC_LAST_PREVENTS_FIXED25_CONFIRMED_ON_FROZEN_TECH_MENU':raise RuntimeError('Missing locked real-LAST gate')
        for case in cases:
            folder=directory/case['case_id'];result={'status':'UNAVAILABLE_MID_FORK','task':task,'case_id':case['case_id'],'family_id':case['family_id'],'fork':'middle_raw10',
                'missing_reason':'TASK_FIXED25_TECHNICALLY_UNEVALUABLE_TRUE_DMC_LAST','replacement_case':False,'technical_gate':file_record(gate),
                'new_simulator_calls':0,'new_CEM_calls':0,'optimizer_updates':0,'primary_delivery':file_record(primary)}
            identity=digest(result);result['identity_sha256']=identity
            if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text())!=result:raise RuntimeError('Existing unavailable record differs')
            atomic_json(folder/'result.json',result);atomic_json(folder/'COMPLETE.json',{'identity_sha256':identity,'files':{'result.json':file_record(folder/'result.json')}})
        shutil.copyfile(gate,out/gate.name)
    else:
        while wait and not all((directory/c['case_id']/'COMPLETE.json').exists() for c in cases):time.sleep(10)
    for case in cases:
        folder=directory/case['case_id'];complete=json.loads((folder/'COMPLETE.json').read_text())
        for name,record in complete['files'].items():
            if file_record(folder/name)!=record:raise RuntimeError('Secondary original seal differs')
    availability=middle_ledger(task,roles,line/'trajectories',directory)
    for name in ('MIDDLE_FORK_AVAILABILITY_RAW.csv','MIDDLE_FORK_AVAILABILITY_COMPLETE.json'):shutil.copyfile(directory/name,out/name)
    s2(directory,root,task,out)
    rows=[]
    for case in cases:
        p=directory/case['case_id']/'candidate_raw.csv'
        if p.exists():rows.extend(csv.DictReader(p.open()))
    write(out/'ALL_CANDIDATE_RAW.csv',rows,fields=None if rows else ['task','case_id','family_id','fork','id','valid_fixed_horizon','missing_reason'])
    estimate=line/'SIM_CEM_ESTIMATE.json';shutil.copyfile(estimate,out/estimate.name);est=json.loads(estimate.read_text())
    skip={'status':'NOT_EXECUTED','task':task,'original_estimate':file_record(estimate),'new_full_CEM_simulator_calls':0}
    if task=='pusht':
        p90=est['p90_end_to_end_seconds'];serial=20*2*300*30*p90
        skip.update(reason='ESTIMATED_FULL_SIM_CEM_TIME_EXCEEDS_TWO_HOURS',p90_seconds_per_complete_candidate_branch=p90,
            branch_definition=est['timing_definition'],cases=20,max_replans_per_case=2,total_replans_per_arm=40,queries_per_replan=9000,
            per_arm_total_branch_queries=360000,serial_work_hours_per_arm=serial/3600,both_arms_serial_work_hours=2*serial/3600,
            two_worker_ideal_wall_hours_per_arm=serial/2/3600,two_worker_ideal_wall_hours_both_arms_total=serial/3600,
            concurrency_scope='Two workers total in this task line; ideal division omits overhead and is not a measured parallel speedup. Forty replans is the total over20cases, not per case.',
            observed_H0_replan_total_reference_only=est['original_stream_H0_observed_total_replans'],qualified=False)
    else:skip.update(reason='TRUE_DMC_LAST_PREVENTS_COMMON_FIXED25_OBJECTIVE',technical_gate=file_record(line/'SIM_TECHNICAL_LIMITATION.json'))
    atomic_json(out/'SIM_CEM_SKIP_EVIDENCE.json',skip)
    status={'status':'COMPLETE' if task=='pusht' and len(rows)==1280 and all(r['valid_fixed_horizon']=='True' for r in rows) else 'COMPLETE_WITH_TECHNICAL_LIMITATIONS',
        'task':task,'batch':'SECONDARY_ONLY_AFTER_PRIMARY_DELIVERY','primary_delivery':file_record(primary),'fixed_case_expected':20,'availability_rows':20,
        'recorded_candidate_rows':len(rows) if rows else None,'candidate_rows_semantics':'Observed records; null is technically unavailable, never a zero scientific score.',
        'fixed25_scope':'SECONDARY_FIXED20_H0_VISIT_AND_SURVIVAL_CONDITIONED','SIM_CEM':'ESTIMATE_ONLY_NOT_RUN','replacement_cases':0,'new_neural_training':0}
    atomic_json(out/'MODULE_STATUS.json',status)
    files=[p for p in out.iterdir() if p.is_file()];manifest={'status':status['status'],'batch':'SECONDARY_ONLY','task':task,'files':{p.name:file_record(p) for p in files},'code_sha256':sha256(__file__)}
    atomic_json(out/'SECONDARY_TABLES_MANIFEST.json',manifest);files.append(out/'SECONDARY_TABLES_MANIFEST.json')
    with tarfile.open(str(line)+'_secondary_tables.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=p.name)
    recovery_files={str(p.relative_to(line)):file_record(p) for p in directory.rglob('*') if p.is_file()}
    recovery_files.update({'secondary_reports/'+p.name:file_record(p) for p in files})
    rec={'status':'SEALED_SECONDARY_ORIGINAL_OUTPUTS','batch':'SECONDARY_ONLY_PRIMARY_ARCHIVES_UNCHANGED','task':task,'files':recovery_files,
        'fixed_cases':20,'source_H0_trajectories':'Referenced by exact source SHA; primary minimum recovery retains raw10 prefixes.',
        'model_weights':'Same frozen R3 H0 and three30krefits; available in primary asset lock.',
        'CPU_scope':'PushT first metadata middle fork will be rescored without simulator; Reacher has no model/branch output because the frozen technical gate prevents execution.'}
    atomic_json(line/'SECONDARY_RECOVERY_MANIFEST.json',rec)
    with tarfile.open(str(line)+'_secondary_recovery.tar.gz','w:gz') as archive:
        archive.add(line/'SECONDARY_RECOVERY_MANIFEST.json',arcname='SECONDARY_RECOVERY_MANIFEST.json')
        for relative in recovery_files:archive.add(line/relative,arcname=relative)
    print(json.dumps(status),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--line',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--r3-root',required=True);p.add_argument('--primary-complete',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();closeout(a.line,a.task,a.r3_root,a.primary_complete,a.wait)

if __name__=='__main__':main()
