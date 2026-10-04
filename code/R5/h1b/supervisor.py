"""R5-only resumable scheduling, at most 64 probe CPU threads plus encoders."""
import concurrent.futures, os, subprocess, time
from pathlib import Path
from common import *

CODE=Path(__file__).resolve().parent;OUT=Path('/workspace/r5/H1b');PY='/workspace/env/bin/python'
def invoke(args,log,threads=1):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),CUDA_VISIBLE_DEVICES='')
    with log.open('a') as f:
        p=subprocess.Popen(['taskset','-c','112-191',PY,'-B','-u',*map(str,args)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);rc=p.wait()
    if rc:raise RuntimeError(f'Process failed {rc}: {args}; log={log}')
    return str(log)

def main():
    assert os.uname().nodename=='72531573d994';freeze=read(CODE/'CODE_FREEZE.json')
    for name,digest in freeze['files'].items():assert sha(CODE/name)==digest,('CODE_FREEZE_MISMATCH',name)
    assert read(OUT/'technical/ridge_single_h1/COMPLETE.json')['status']=='TECHNICAL_COMPLETE'
    assert read(OUT/'technical/mlp_three_h1_seed0/COMPLETE.json')['status']=='TECHNICAL_COMPLETE'
    atomic(OUT/'STARTED.json',{'pid':os.getpid(),'utc':now(),'affinity':sorted(os.sched_getaffinity(0)),'code_freeze_sha256':sha(CODE/'CODE_FREEZE.json'),'protocol_sha256':sha(CODE/'FIT_PROTOCOL.json'),'scope':'H1b measurement only; world model updates0; no H2/H3/C1/C2'})
    pending=set(['pusht','reacher','tworoom','cube']);submitted=set();futures={};done=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        while pending or futures:
            audit=read(OUT/'INPUT_AUDIT.json')
            for task in sorted(pending):
                if task not in audit['tasks']:continue
                c=task_config(task)
                if task in ('pusht','reacher') and not c['cache'].exists():
                    cache=c['cache'].parent
                    if not all((cache/f'part_{s}.json').exists() for s in range(2)):continue
                    invoke([CODE/'encode.py',task,'--output',cache,'--merge'],OUT/'logs'/f'{task}_cache_merge.log')
                if not c['cache'].exists():continue
                matrix=OUT/'matrices'/task
                invoke([CODE/'dataset.py',task,'--output',matrix],OUT/'logs'/f'{task}_matrix.log')
                state_targets=[k for k,v in read(CODE/'FIT_PROTOCOL.json')['state_targets'][task].items() if v is not None]
                # All queued fits share the same immutable all-TRAIN matrices.
                for model,targets,seeds,threads in [('RIDGE',state_targets+['latent_1','latent_2','latent_5'],[0],1),('MLP',['state','latent_1','latent_2','latent_5'],[0,1,2],4)]:
                    for target in targets:
                        for hist in ['single','three']:
                            for seed in seeds:
                                job=f'{task}/{model}/{target}/{hist}/seed{seed}';dest=OUT/'fits'/job
                                if (dest/'COMPLETE.json').exists():done.append(job);continue
                                args=[CODE/'fit.py',task,'--matrix',matrix,'--output',dest,'--model',model,'--target',target,'--history',hist,'--seed',seed,'--threads',threads]
                                f=pool.submit(invoke,args,OUT/'logs'/(job.replace('/','_')+'.log'),threads);futures[f]=job
                pending.remove(task);submitted.add(task)
            for f in list(futures):
                if f.done():
                    job=futures.pop(f)
                    try:f.result();done.append(job)
                    except Exception as e:errors.append({'job':job,'reason':str(e)})
            atomic(OUT/'STATUS.json',{'status':'RUNNING','utc':now(),'pid':os.getpid(),'pending_cache_or_matrix':sorted(pending),'submitted_tasks':sorted(submitted),'completed_jobs':len(done),'active_or_queued':len(futures),'errors':errors,'done_jobs':done,'world_model_updates':0})
            if pending or futures:time.sleep(20)
    atomic(OUT/'STATUS.json',{'status':'FAILED' if errors else 'ALL_FITS_COMPLETE','utc':now(),'completed_jobs':len(done),'errors':errors,'done_jobs':done,'world_model_updates':0})

if __name__=='__main__':main()
