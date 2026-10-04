"""Small per-case closed-loop values for delivery without transferring raw pixels."""
import argparse,csv,json
from pathlib import Path
from .common import atomic_json,file_record
from .augment_metrics import augment

def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectories',required=True);p.add_argument('--out',required=True);p.add_argument('--stream');a=p.parse_args()
    root=Path(a.trajectories);rows=[]
    for path in sorted(root.glob('FORMAL/*/*/*/COMPLETE.json')):
        folder=path.parent;result=json.loads((folder/'result.json').read_text())
        if a.stream and result['stream']!=a.stream:continue
        receipt=json.loads(path.read_text())
        for name,record in receipt['files'].items():
            if file_record(folder/name)!=record:raise RuntimeError('Completed output changed')
        metrics=augment(folder)
        row={k:result.get(k) for k in ('task','case_id','family_id','arm','stream','success','executed_raw_steps','replan_calls','wall_seconds','attempts','optimizer_updates')}
        row.update({k:metrics.get(k) for k in ('initial_margin','anytime_margin','terminal_margin','branch_count','branch_raw_steps','prefix_replay_raw_steps','branch_seconds','physics_steps_including_prefix')})
        row.update(trajectory_sha256=receipt['files']['trajectory.npz']['sha256'],result_sha256=receipt['files']['result.json']['sha256'],method_failure=json.dumps(result.get('method_failure')),CEM_seconds=sum(r['synchronized_wall_seconds'] for r in result['replans']))
        rows.append(row)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'CLOSED_LOOP_RAW.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipt={'status':'COMPLETE_DERIVED','rows':len(rows),'all_original_completion_hashes_verified':True,'source_logs_unchanged':True,'by_arm_stream':{arm+'/'+stream:{'n':sum(r['arm']==arm and r['stream']==stream for r in rows),'successes':sum(r['success'] for r in rows if r['arm']==arm and r['stream']==stream)} for arm,stream in sorted({(r['arm'],r['stream']) for r in rows})}}
    atomic_json(out/'CLOSED_LOOP_COMPLETE.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
