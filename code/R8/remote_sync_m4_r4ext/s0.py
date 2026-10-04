import argparse,json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--output',required=True);p.add_argument('--check',choices=['a','bcd','termination'],default='a');p.add_argument('--trajectories');p.add_argument('--device',default='cuda');a=p.parse_args()
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from .replay import audit_reset,audit_replay,audit_termination
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks'][a.task]['cases'];entries=[e for e in entries if e['case']['role']=='TECH']
    if a.check=='a':print(json.dumps(audit_reset(a.task,entries,a.assets,a.output)));return
    if a.check=='termination':print(json.dumps(audit_termination(a.task,entries,a.assets,a.output)));return
    if not a.trajectories:raise ValueError('New host TECH rerun logs required for bcd')
    from r3.model import load_official
    from r3.data import image_transform
    from r3.common import fp32_policy
    fp32_policy();model=load_official(a.task,a.device)
    for entry in entries:
        path=Path(a.trajectories)/'TECH/R3_ORIGINAL'/entry['case']['case_id']/'H0/trajectory.npz'
        if path.exists():print(json.dumps(audit_replay(a.task,entry,a.assets,path,model,image_transform(),a.output)))

if __name__=='__main__':main()
