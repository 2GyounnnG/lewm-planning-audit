"""Derive margins and branch accounting without changing sealed trajectory files."""
import argparse,json
from pathlib import Path
from .common import atomic_json,atomic_npz,file_record
from .replay import task_margin

def augment(folder):
    import numpy as np
    folder=Path(folder);result=json.loads((folder/'result.json').read_text())
    with np.load(folder/'trajectory.npz',allow_pickle=False) as f:
        if 'raw_physical_state' not in f:
            data={'status':'MISSING_STATE_AFTER_METHOD_FAILURE','task':result['task'],'case_id':result['case_id'],'arm':result['arm'],'stream':result['stream']}
            atomic_json(folder/'DERIVED_METRICS.json',data);return data
        margins=task_margin(result['task'],f['raw_physical_state'],f['goal_state'])
    records=[r['rerank'] for r in result['replans'] if 'rerank' in r];evaluated=margins[1:]
    data={'status':'COMPLETE_DERIVED','task':result['task'],'case_id':result['case_id'],'arm':result['arm'],'stream':result['stream'],'initial_margin':float(margins[0]),'anytime_margin':float(evaluated.min()) if len(evaluated) else None,'terminal_margin':float(evaluated[-1]) if len(evaluated) else None,'success':result['success'],'executed_raw_steps':result['executed_raw_steps'],'replan_calls':result['replan_calls'],'branch_count':sum(r['branch_count'] for r in records),'branch_raw_steps':sum(r['branch_raw_steps'] for r in records),'prefix_replay_raw_steps':sum(r['prefix_replay_raw_steps'] for r in records),'branch_seconds':sum(r['branch_seconds'] for r in records),'physics_steps_including_prefix':sum(r.get('physics_steps_including_prefix',0) for r in records),'original_trajectory':file_record(folder/'trajectory.npz'),'original_result':file_record(folder/'result.json'),'optimizer_updates':0}
    if result['task']=='pusht':
        data['original_instrumented_dmc_calls']=data['physics_steps_including_prefix']
        data['physics_steps_including_prefix']=10*(data['branch_raw_steps']+data['prefix_replay_raw_steps'])
        data['physics_step_accounting']='Derived exactly from pinned PushT dt=.01 and control_hz=10: 10 space.step calls per raw action. Reset setup calls reported separately; original Reacher-only counter was zero.'
        data['branch_reset_set_state_physics_steps']=data['branch_count']
        data['closed_loop_physics_steps']=10*data['executed_raw_steps']
    atomic_npz(folder/'DERIVED_MARGIN.npz',raw_task_margin=margins);atomic_json(folder/'DERIVED_METRICS.json',data);return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectories',required=True);a=p.parse_args()
    n=0
    for completed in Path(a.trajectories).rglob('COMPLETE.json'):
        if (completed.parent/'trajectory.npz').exists():augment(completed.parent);n+=1
    print(json.dumps({'augmented':n,'original_files_unchanged':True}))

if __name__=='__main__':main()
