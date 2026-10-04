"""Background X1 line coordinator after immutable asset/schema intake.

This process never rents/destroys hosts and never starts second-batch streams.
All subprocesses inherit the outer taskset partition. Workers receive one GPU.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from . import core

def run(task):
    gpus=(4,5) if task=='tworoom' else (6,7);logs=core.ROOT/'logs';logs.mkdir(parents=True,exist_ok=True)
    def start(label,args,gpu=None):
        env=os.environ.copy();env.update(X1_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
            NUMEXPR_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        path=logs/(label+'.log');f=path.open('a')
        proc=subprocess.Popen([sys.executable,'-m','x1.runner',*args],env=env,stdout=f,stderr=subprocess.STDOUT)
        f.close();core.atomic(core.ROOT/'state'/(label+'.json'),{'pid':proc.pid,'args':args,'gpu':gpu,'log':str(path),'started_at':time.time()})
        return proc
    def wait(label,p):
        code=p.wait()
        core.atomic(core.ROOT/'state'/(label+'_exit.json'),{'returncode':code,'ended_at':time.time()})
        if code:raise RuntimeError(label+' failed; other line jobs are not terminated')
    wait('model_check',start('model_check',['check-model',task,'--device','cpu']))
    caching=[]
    for shard,gpu in enumerate(gpus):
        label=f'cache_{shard}';caching.append((label,start(label,['cache',task,'--device','cuda','--microbatch','128','--shard',str(shard),'--shards','2'],gpu)))
    for label,p in caching:wait(label,p)
    wait('cache_merge',start('cache_merge',['cache-merge',task,'--shards','2']))
    wait('reset_audit',start('reset_audit',['reset-audit',task],gpus[0]))
    wait('train_tech',start('train_tech',['train',task,'--device','cuda','--seed','103201','--technical-steps','128'],gpus[0]))
    wait('closed_loop_tech',start('closed_loop_tech',['closed-loop',task,'--device','cuda','--phase','TECH','--arm','H0'],gpus[0]))
    workers=[]
    for seed,gpu,slot in ((103201,gpus[0],0),(103202,gpus[1],0),(103203,gpus[0],1)):
        label=f'train_{seed}';workers.append((label,start(label,['train',task,'--device','cuda','--seed',str(seed),'--slot',str(slot)],gpu)))
    # All workers are started before any join. A failed one leaves others running.
    errors=[]
    for label,p in workers:
        try:wait(label,p)
        except RuntimeError as e:errors.append(str(e))
    if errors:raise RuntimeError('; '.join(errors))
    wait('open_loop',start('open_loop',['open-loop',task,'--device','cuda'],gpus[0]))
    for arm in ['H0']+[f'REFIT_{seed}' for seed in core.SEEDS]:
        pair=[]
        for shard,gpu in enumerate(gpus):
            label=f'closed_{arm}_{shard}';pair.append((label,start(label,['closed-loop',task,'--device','cuda','--arm',arm,'--shard',str(shard),'--shards','2'],gpu)))
        for label,p in pair:wait(label,p)
    wait('report',start('report',['report',task]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);a=p.parse_args();run(a.task)
