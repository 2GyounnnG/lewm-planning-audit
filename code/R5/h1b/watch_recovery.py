"""Local recovery queue; no SSH, no training, exact archived code execution."""
import argparse,os,subprocess,tarfile,time
from pathlib import Path
from common import *

def main(a):
    root=Path(a.root).resolve();archive=root/'archives';code=root/'recovery/code_bundle/code/h1b';tasks=['cube','tworoom','pusht','reacher'];done=[]
    while len(done)<4:
        for task in tasks:
            if task in done:continue
            p=archive/f'{task}_ARCHIVE.json';z=archive/f'{task}_probe_recovery_v1.tar.gz'
            if not p.exists() or not z.exists():continue
            record=read(p);assert z.stat().st_size==record['archive']['bytes'] and sha(z)==record['archive']['sha256']
            out=root/'recovery'/task;out.mkdir(parents=True,exist_ok=True)
            if not (out/'CPU_RECOVERY_CHECK.json').exists():
                with tarfile.open(z) as tf:
                    assert all((out/m.name).resolve().is_relative_to(out.resolve()) and (m.isfile() or m.isdir()) for m in tf.getmembers());tf.extractall(out,filter='data')
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                with (out/'CPU_RECOVERY.log').open('a') as log:subprocess.run([a.python,'-B',str(code/'recover_cpu.py'),'--task',task,'--root',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=read(out/'CPU_RECOVERY_CHECK.json');done.append(task);print(task,result['probes_reevaluated'],result['max_prediction_abs_difference'],result['max_error_abs_difference'],flush=True)
        atomic(root/'reports/LOCAL_RECOVERY_PROGRESS.json',{'status':'COMPLETE' if len(done)==4 else 'WAITING_ARCHIVE_TRANSFER','tasks_complete':done,'tasks_pending':[t for t in tasks if t not in done],'utc':now(),'pid':os.getpid()})
        if len(done)<4:time.sleep(10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--python',required=True);main(p.parse_args())
