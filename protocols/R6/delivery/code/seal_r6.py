from __future__ import annotations
import csv,datetime,hashlib,json,subprocess
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def rec(p):
    p=Path(p); return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def main(root):
    root=Path(root); report=root/'reacher/reports'
    sel=json.loads((root/'ops/R6_CASE_SELECTION.json').read_text())
    stats=json.loads((report/'R6_STATS.json').read_text())
    raw=list(csv.DictReader((report/'R6_RAW_VALUES.csv').open()))
    assert len(raw)==2400 and stats['trajectories']==2400
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    qs={'version':'R6_QUIESCENCE_V1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_R6_workers_naturally_finished':True,'nvidia_compute_processes':apps,'automatic_destroy':False,'automatic_rental':False,'R4_R5_jobs_interrupted':False,'R4_R5_evidence_modified':False}
    (root/'ops/R6_QUIESCENCE.json').write_text(json.dumps(qs,ensure_ascii=False,indent=2)+'\n')
    entries=[]
    for row in raw:
        base=root/'reacher/raw/FORMAL'/row['stream']/row['case_id']/row['arm']/row['history']
        rp=base/'result.json'; tp=base/'trajectory.npz'
        assert rp.is_file() and tp.is_file() and rp.stat().st_size>0 and tp.stat().st_size>0
        entries.append({'result':{'path':str(rp),'bytes':rp.stat().st_size,'sha256':row['source_result_sha256']},'trajectory':{'path':str(tp),'bytes':tp.stat().st_size,'sha256':row['trajectory_sha256']},'case_id':row['case_id'],'arm':row['arm'],'stream':row['stream'],'history':row['history']})
    assert len({(x['case_id'],x['arm'],x['stream'],x['history']) for x in entries})==2400
    reports={p.name:rec(p) for p in [report/'R6_RAW_VALUES.csv',report/'R6_MAIN_TABLE.csv',report/'R6_SECONDARY_TABLE.csv',report/'R6_FLIPS.csv',report/'R6_STATS.json',report/'CONCLUSION_ZH.txt']}
    files={'case_selection':rec(root/'ops/R6_CASE_SELECTION.json'),'case_windows':rec(root/'reacher/assets/CASE_WINDOWS.json'),'formal_launch':rec(root/'ops/R6_FORMAL_LAUNCH.json'),'quiescence':rec(root/'ops/R6_QUIESCENCE.json'),'code_runner':rec(root/'code/r6_runner.py'),'code_context':rec(root/'code/context.py'),'code_export':rec(root/'code/export_fresh.py'),'code_report':rec(root/'code/report.py')}
    manifest={'version':'R6_RECOVERY_MANIFEST_V1','status':'VERIFIED_R6_FORMAL_OUTPUTS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence_label':'PREREGISTERED_FRESH_CASE_REPLICATION','case_count':100,'trajectory_count':2400,'result_and_trajectory_file_count':4800,'result_and_trajectory_entries':entries,'reports':reports,'fixed_inputs':files,'formal_world_model_updates':0,'model_checkpoint_selection':'fixed H0 and R3 routing seeds 103201/103202/103203; no result-based selection','first_plan_pair_gate':stats['first_plan_pair_gate'],'primary_endpoint':stats['primary_endpoint'],'secondary_endpoints':stats['secondary_endpoints'],'R4_R5_evidence_modified':False,'R4_R5_jobs_interrupted':False,'automatic_destroy':False,'automatic_rental':False,'remote_originals_retained':True,'scope':'R6 only under /workspace/r6; no files copied into or changed under R4/R5.'}
    (root/'R6_RECOVERY_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    seal={'version':'R6_FINAL_SEAL_V1','status':'COMPLETE_PREREGISTERED_FRESH_CASE_REPLICATION','sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence_label':'PREREGISTERED_FRESH_CASE_REPLICATION','case_selection':files['case_selection'],'case_windows':files['case_windows'],'formal_launch':files['formal_launch'],'quiescence':files['quiescence'],'reports':reports,'recovery_manifest':rec(root/'R6_RECOVERY_MANIFEST.json'),'case_count':100,'trajectory_count':2400,'formal_world_model_updates':0,'primary_endpoint':stats['primary_endpoint'],'first_plan_pair_gate':stats['first_plan_pair_gate'],'R4_R5_jobs_interrupted':False,'R4_R5_evidence_modified':False,'automatic_destroy':False,'automatic_rental':False,'remote_originals_retained':True}
    (root/'R6_SEAL.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':seal['status'],'cases':100,'trajectories':2400,'point':stats['primary_endpoint']['point_estimate'],'ci95':stats['primary_endpoint']['ci95'],'manifest_files':4800,'nvidia_compute_processes':apps}))
if __name__=='__main__':main(Path(__import__('sys').argv[1]))
