"""Cube's authorized offline continuation after an unchanged failed reset gate."""
import fcntl,os,subprocess,sys,time
from . import core

def main():
    if core.ROOT!=core.Path('/workspace/x1_cube'):raise RuntimeError('Cube root required')
    state=core.ROOT/'state';state.mkdir(parents=True,exist_ok=True)
    lock=(state/'offline_continuation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=core.read(core.ROOT/'reset_audit/RESET_FALLBACK.json')
    if gate['status']!='BLOCKED_RESET_FALLBACK' or gate['raw_calls']!=16:raise RuntimeError('Requires preserved exact 16-trial failure')
    core.freeze(state/'OFFLINE_SCOPE.json',{'task':'cube','reason':'RESET_FALLBACK_EXACT_CHECK_FAILED','reset_gate':core.file_record(core.ROOT/'reset_audit/RESET_FALLBACK.json'),
        'scope':'finish cache, separate128 TECH, fixed3x30k and open-loop; no further reset trial or closed-loop trajectory',
        'seeds':list(core.SEEDS),'formal_updates_per_seed':30000,'source_sha256':core.sha(__file__)})
    def start(label,module,args,gpu=None):
        env=os.environ.copy();env.update(X1_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        if gpu is not None:env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu))
        log=core.ROOT/'logs'/(label+'.log');f=log.open('a');p=subprocess.Popen([sys.executable,'-m',module,*args],env=env,stdout=f,stderr=subprocess.STDOUT);f.close()
        core.atomic(state/(label+'.json'),{'pid':p.pid,'gpu':gpu,'module':module,'args':args,'started_at':time.time(),'entrypoint_sha256':core.sha(core.Path(__file__).parent/(module.split('.')[-1]+'.py'))})
        return p
    def wait(label,p):
        code=p.wait();core.atomic(state/(label+'_exit.json'),{'returncode':code,'ended_at':time.time()})
        if code:raise RuntimeError(label+' failed; other workers are retained')
    print('WAIT_EXISTING_CACHE_SHARDS',flush=True)
    while not all((core.ROOT/'manifests'/f'cube_cache_part_{s}.json').exists() for s in (0,1)):time.sleep(10)
    if not (core.ROOT/'manifests/cube_cache.json').exists():wait('cache_merge',start('cache_merge','x1.runner',['cache-merge','cube','--shards','2']))
    techpath=core.ROOT/'technical_train/103201/result.json'
    if techpath.exists():
        tech=core.read(techpath)
        if tech['status']!='TECHNICAL_COMPLETE' or tech['actual_updates']!=128 or not tech['technical'] or tech['frozen_before']!=tech['frozen_after']:
            raise RuntimeError('Existing TECH record is not an eligible completed128 result')
        core.freeze(state/'OFFLINE_TECH_REUSE.json',{'result':core.file_record(techpath),'additional_optimizer_updates':0})
    else:
        wait('train_tech',start('train_tech','x1.runner',['train','cube','--device','cuda','--seed','103201','--technical-steps','128'],6))
    workers=[]
    for seed,gpu,slot in ((103201,6,0),(103202,7,0),(103203,6,1)):
        label=f'train_{seed}';workers.append((label,start(label,'x1.runner',['train','cube','--device','cuda','--seed',str(seed),'--slot',str(slot)],gpu)))
    errors=[]
    for label,p in workers:
        try:wait(label,p)
        except RuntimeError as e:errors.append(str(e))
    if errors:raise RuntimeError('; '.join(errors))
    wait('train_report',start('train_report','x1.train_report',['cube']))
    wait('open_loop',start('open_loop','x1.runner',['open-loop','cube','--device','cuda'],6))
    wait('open_loop_report',start('open_loop_report','x1.open_loop_report',['cube']))
    wait('offline_report',start('offline_report','x1.offline_report',['cube']))
    print('COMPLETE_WITH_TECHNICAL_LIMITATIONS',flush=True)

if __name__=='__main__':main()
