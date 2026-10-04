"""Independently dispatch Reacher learning-only dependents after original400."""
import fcntl,json,os,subprocess,time
from pathlib import Path
from .common import atomic_json,file_record,sha256

def main():
    line=Path('/workspace/r4_reacher');lock=(line/'LEARNING_FOLLOWUP.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (line/'LEARNING_FOLLOWUP_LAUNCH.json').exists():raise RuntimeError('Follow-up launch already journaled')
    atomic_json(line/'LEARNING_FOLLOWUP_LAUNCH.json',{'pid':os.getpid(),'code_sha256':sha256(__file__),'started':time.time()})
    while True:
        files=list((line/'trajectories/FORMAL/R3_ORIGINAL').glob('*/*/COMPLETE.json'))
        if len(files)==400:break
        time.sleep(5)
    for path in files:
        for name,record in json.loads(path.read_text())['files'].items():
            if file_record(path.parent/name)!=record:raise RuntimeError('Original400 seal changed')
    atomic_json(line/'ORIGINAL_400_VERIFIED.json',{'status':'PASS','completed':400,'all_completion_hashes_verified':True})
    common=['--r3-root','/workspace/shared_data/r3','--assets','/workspace/shared_data/r4_assets_reacher','--task','reacher'];base=os.environ.copy();base.update(PYTHONPATH='/workspace/r4_v23_execution',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl');jobs=[]
    def launch(name,args,gpu):
        env=base.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);f=(line/(name+'.log')).open('ab');p=subprocess.Popen(['taskset','-c','24-47','/workspace/env/bin/python']+args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append({'name':name,'pid':p.pid,'command':args});atomic_json(line/'LEARNING_FOLLOWUP_JOBS.json',jobs);print(name,p.pid,flush=True);return p
    launch('s1_cross',['-m','r4.s1','--r3-root','/workspace/shared_data/r3','--task','reacher','--module','cross','--trajectories',str(line/'trajectories'),'--output',str(line/'s1')],2)
    for stream in ('R4_ALT_CEM_1','R4_ALT_CEM_2'):
        for i in range(2):launch(stream+'_worker'+str(i),['-m','r4.runner']+common+['--output',str(line/'trajectories'),'--stream',stream,'--worker-index',str(i),'--workers','2'],2+i)
    # The menu control needs no simulator-replay or fixed-horizon capability gate.
    for stream in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2'):
        processes=[]
        for i in range(2):processes.append(launch('H0_MENU_'+stream+'_'+str(i),['-m','r4.s3']+common+['--output',str(line/'trajectories'),'--role','H0_MENU_RERANK','--stream',stream,'--worker-index',str(i),'--workers','2'],2+i))
        for process in processes:
            if process.wait():raise RuntimeError('H0 menu infrastructure failure; other independent jobs retained')

if __name__=='__main__':main()
