"""Produce task reports only after all fixed formal cells exist."""
import json,time,os,subprocess,sys
from pathlib import Path
from r4.common import file_record,atomic_json

def main():
    root=Path('/workspace/r5/H2');launch=json.loads((root/'FORMAL_LAUNCH_V1.json').read_text())
    assert file_record(Path(__file__).parent/'report.py')==launch['code']['report.py']
    pending={'reacher','pusht'}
    while pending:
      for task in sorted(pending):
        path=root/task;out=path/'reports'
        if (out/'MODULE_STATUS.json').exists():pending.remove(task);continue
        count=len(list(path.glob('FORMAL/*/*/*/COMPLETE.json')))
        if count<1200:continue
        if count!=1200:raise RuntimeError('Formal cardinality exceeds fixed1200: '+task)
        cmd=[sys.executable,'-B','-m','h2.report','--task',task,'--root',str(path),'--r3-root','/workspace/shared_data/r3',
            '--statistics','/workspace/r4_v23_execution/analysis/statistics.py','--out',str(out)]
        result=subprocess.run(cmd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
        atomic_json(path/'REPORT_PROCESS_RECEIPT.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'command':cmd,'report_code':file_record(Path(__file__).parent/'report.py')})
        if result.returncode:raise RuntimeError('Report validation failed for '+task+': '+result.stderr)
        print(result.stdout,flush=True);pending.remove(task)
      if pending:time.sleep(5)
    atomic_json(root/'MAIN_REPORTS_COMPLETE.json',{'status':'COMPLETE','tasks':['reacher','pusht'],'formal_new_trajectories':2400,'reused_controls':2400,'world_model_new_updates':0})

if __name__=='__main__':main()
