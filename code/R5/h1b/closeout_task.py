import argparse,os,subprocess
from pathlib import Path
from common import *

def main(task):
    root=Path('/workspace/r5/H1b');code=Path(__file__).resolve().parent;out=root/'reports'/task;receipt=out/'COMPLETE.json'
    if receipt.exists() and read(receipt)['report_code_sha256']!=sha(code/'report.py'):
        old=read(receipt)['report_code_sha256'][:16];history=root/'reports/history';history.mkdir(parents=True,exist_ok=True);out.rename(history/(task+'_report_'+old))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if not receipt.exists():subprocess.run(['/workspace/env/bin/python','-B',str(code/'report.py'),'--root',str(root),'--lewm',str(root/'HISTORY_CONDITION_4TASK_RAW.csv'),'--output',str(out),'--tasks',task],check=True,env=env)
    subprocess.run(['/workspace/env/bin/python','-B',str(code/'pack.py'),'--root',str(root),'--task',task],check=True,env=env)
    print(task,'CLOSEOUT_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');main(p.parse_args().task)
