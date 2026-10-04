"""Only run after the global main-delivery gate is explicitly installed."""
import fcntl,os,subprocess,sys,time
from . import core

def main():
    if core.ROOT!=core.Path('/workspace/x1_tworoom'):raise RuntimeError('Cube has no eligible closed-loop TECH gate')
    gatepath=core.ROOT/'SECOND_BATCH_GATE.json';gate=core.read(gatepath)
    tech=core.read(core.ROOT/'reports/tworoom_TECH/MODULE_STATUS.json')
    if gate.get('main_results_delivered') is not True or not gate.get('tech_p90_seconds',float('inf'))<=30:
        raise RuntimeError('Root MAIN_BATCH_DELIVERED and recorded TECH P90<=30 required')
    if gate['tech_p90_seconds']!=tech['closed_loop_P90_seconds'] or not tech['second_batch_eligible_after_main_delivery']:
        raise RuntimeError('Gate does not match recorded TECH latency')
    if core.read(core.ROOT/'reports/tworoom/MODULE_STATUS.json')['status']!='COMPLETE':raise RuntimeError('Main incomplete')
    state=core.ROOT/'state';handle=(state/'second_batch.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    core.freeze(state/'SECOND_BATCH_SCOPE.json',{'task':'tworoom','gate':core.file_record(gatepath),'entrypoint':core.file_record(__file__),
        'cases':100,'arms':['H0']+[f'REFIT_{s}' for s in core.SEEDS],'streams':['R4_ALT_CEM_1','R4_ALT_CEM_2'],'new_trajectories':800,'new_optimizer_updates':0})
    for stream in ('R4_ALT_CEM_1','R4_ALT_CEM_2'):
        for arm in ['H0']+[f'REFIT_{s}' for s in core.SEEDS]:
            workers=[]
            for shard,gpu in enumerate((4,5)):
                name=f'alternate_{stream}_{arm}_{shard}';args=['closed-loop','tworoom','--device','cuda','--arm',arm,'--stream',stream,'--shard',str(shard),'--shards','2']
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),X1_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
                with (core.ROOT/'logs'/(name+'.log')).open('a') as f:p=subprocess.Popen([sys.executable,'-m','x1.runner',*args],env=env,stdout=f,stderr=subprocess.STDOUT)
                core.atomic(state/(name+'.json'),{'pid':p.pid,'gpu':gpu,'args':args,'started_at':time.time()});workers.append((name,p))
            errors=[]
            for name,p in workers:
                code=p.wait();core.atomic(state/(name+'_exit.json'),{'returncode':code,'ended_at':time.time()})
                if code:errors.append(name)
            if errors:raise RuntimeError('Failed workers: '+','.join(errors))
    core.atomic(state/'SECOND_BATCH_COMPLETE.json',{'status':'COMPLETE','additional_trajectories':800,'new_optimizer_updates':0,'completed_at':time.time()})
    print('ALTERNATE_800_COMPLETE',flush=True)

if __name__=='__main__':main()
