"""Read-only line telemetry from atomically completed cache receipts and nvidia-smi."""
import argparse, csv, json, os, subprocess, time
from pathlib import Path
import numpy as np
from . import core

def run(task,samples=7,interval=10):
    roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json');episodes=[e for e in roles['episodes'] if e['role'] in ('REFIT_TRAIN','MONITOR','TECH','EVAL')]
    episodes.sort(key=lambda e:(e['role']!='TECH',e['episode_id']));assignment={e['episode_id']:i%2 for i,e in enumerate(episodes)}
    totals={s:sum(e['length'] for i,e in enumerate(episodes) if i%2==s) for s in (0,1)};gpus=(4,5) if task=='tworoom' else (6,7)
    rows=[];completion=[];directory=core.ROOT/'telemetry';directory.mkdir(parents=True,exist_ok=True)
    for sample in range(samples):
        completed={0:0,1:0};counts={0:0,1:0};completion=[]
        for path in (core.ROOT/'cache').glob('*.json'):
            d=core.read(path);e=d['identity']['episode'];s=assignment[e['episode_id']]
            completed[s]+=e['length'];counts[s]+=1;completion.append({'shard':s,'mtime':path.stat().st_mtime,'frames':e['length']})
        values=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,gpus)),'--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        gpu={}
        for line in values.strip().splitlines():
            idx,uuid,util,memory=[x.strip() for x in line.split(',')];gpu[int(idx)]={'uuid':uuid,'utilization_percent':float(util),'memory_MiB':float(memory)}
        now=time.time()
        for s in (0,1):rows.append({'sample':sample,'wall_time':now,'shard':s,'gpu':gpus[s],**gpu[gpus[s]],'completed_episodes':counts[s],
            'completed_frames':completed[s],'total_frames':totals[s]})
        core.atomic(directory/'CACHE_TELEMETRY_PROGRESS.json',rows[-2:])
        if sample+1<samples:time.sleep(interval)
    summary=[]
    for s in (0,1):
        part=[r for r in rows if r['shard']==s];dt=part[-1]['wall_time']-part[0]['wall_time'];frames=part[-1]['completed_frames']-part[0]['completed_frames'];rate=frames/dt if dt>0 else None
        timeline=sorted((r for r in completion if r['shard']==s),key=lambda r:r['mtime']);durations=np.diff([r['mtime'] for r in timeline])
        intervals=[(b['completed_frames']-a['completed_frames'])/(b['wall_time']-a['wall_time']) for a,b in zip(part,part[1:])]
        remaining=totals[s]-part[-1]['completed_frames'];summary.append({'task':task,'shard':s,'gpu':gpus[s],
            'measured_seconds':dt,'measured_frames':frames,'frames_per_second':rate,'remaining_frames':remaining,'eta_seconds':remaining/rate if rate and rate>0 else None,
            'interval_throughput_P50':float(np.quantile(intervals,.5)) if intervals else None,'interval_throughput_P90':float(np.quantile(intervals,.9)) if intervals else None,
            'episode_completion_gap_P50_seconds':float(np.quantile(durations,.5)) if len(durations) else None,
            'episode_completion_gap_P90_seconds':float(np.quantile(durations,.9)) if len(durations) else None,
            'GPU_utilization_mean_percent':float(np.mean([r['utilization_percent'] for r in part])),
            'total_frames':totals[s],'completed_frames':part[-1]['completed_frames']})
    for name,records in [('CACHE_TELEMETRY_RAW.csv',rows),('CACHE_ETA_RAW.csv',summary)]:
        with (directory/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result={'status':'COMPLETE','task':task,'rows':summary,'scope':'End-to-end cache file completion timestamps include encoding and I/O; GPU utilization from nvidia-smi; no scientific outcomes',
        'code_sha256':core.sha(__file__),'formal_optimizer_updates':0};core.atomic(directory/'CACHE_PROFILE.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);p.add_argument('--samples',type=int,default=7);p.add_argument('--interval',type=float,default=10);a=p.parse_args();run(a.task,a.samples,a.interval)
