"""Finite post-processing queue; never controls or selects formal model jobs."""
import csv,gzip,json,os,subprocess,time
from pathlib import Path
from . import common
from .evaluate import file

def training_table(root):
    out=root/'reports/training';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    rows=[];sources=[]
    for seed in common.SEEDS:
        folder=root/f'artifacts/train/H3X_reacher_s{seed}';r=common.read_json(folder/'result.json');assert r['actual_updates']==30000 and not r['technical'];sources.append(file(folder/'updates.jsonl'))
        for line in (folder/'updates.jsonl').read_text().splitlines():rows.append({'label':common.LABEL,'seed':seed,**json.loads(line)})
    assert len(rows)==90000
    with gzip.open(out/'TRAIN_UPDATES_RAW.csv.gz','wt',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'label':common.LABEL,'seed':s,'updates':30000,'mean_recorded_batch_MSE':sum(r['loss_raw_MSE'] for r in rows if r['seed']==s)/30000,'last_batch_MSE':next(r['loss_raw_MSE'] for r in reversed(rows) if r['seed']==s),'final_checkpoint_step':30000,'selection':False} for s in common.SEEDS]
    with (out/'TRAIN_SUMMARY.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
    common.atomic_json(out/'COMPLETE.json',{'status':'COMPLETE','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','raw_rows':90000,'formal_updates':90000,'technical_updates':101,'encoder_updates':0,'source_journals':sources,'files':[file(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json']})

def main():
    root=common.ROOT;queue=root/'delivery';queue.mkdir(exist_ok=True);packed=set();report_done=False
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    lock=common.read_json(Path(__file__).with_name('REPORT_CODE_FREEZE.json'))
    for n,h in lock['files'].items():assert common.sha256(Path(__file__).with_name(n))==h
    while len(packed)<3 or not report_done:
        train_ready=all((root/f'artifacts/train/H3X_reacher_s{s}/result.json').exists() for s in common.SEEDS)
        if train_ready:training_table(root)
        # Publish tables before optional larger trajectory recovery packages.
        allclosed=all((root/f'evaluation/closed_loop/CLOSED_COMPLETE_{s}.json').exists() for s in common.SEEDS)
        if allclosed and not report_done:
            with (queue/'report.log').open('ab',buffering=0) as log:rc=subprocess.call(['taskset','-c','112','/workspace/env/bin/python','-B','-m','h3x.report'],env=env,stdout=log,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError('Report failed; retained log; formal closed loop already complete')
            report_done=True
        for s in common.SEEDS:
            if s in packed or not (root/f'evaluation/closed_loop/CLOSED_COMPLETE_{s}.json').exists():continue
            with (queue/f'pack_{s}.log').open('ab',buffering=0) as log:rc=subprocess.call(['taskset','-c','112','/workspace/env/bin/python','-B','-m','h3x.recover','pack','--seed',str(s)],env=env,stdout=log,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError('Recovery packing failed '+str(s))
            packed.add(s)
        common.atomic_json(queue/'STATUS.json',{'label':common.LABEL,'pid':os.getpid(),'unix':time.time(),'training_complete':train_ready,'main_report_complete':report_done,'packed_seeds':sorted(packed)})
        supervisor=root/'SUPERVISOR_COMPLETE.json'
        if supervisor.exists() and common.read_json(supervisor)['status']!='COMPLETE':raise RuntimeError('Formal supervisor failed; no automatic model retry')
        if len(packed)<3 or not report_done:time.sleep(20)
    common.atomic_json(queue/'COMPLETE.json',{'status':'REMOTE_TABLES_AND_RECOVERY_PACKS_COMPLETE','label':common.LABEL,'seeds':sorted(packed),'main_report':file(root/'reports/main/COMPLETE.json')})
if __name__=='__main__':main()
