"""Strict, predeclared TECH gates. No result-dependent retries or tolerances."""
import argparse,json
from pathlib import Path
import numpy as np
from r4.common import atomic_json,sha256,file_record

def report(task,root,audit_path):
    audit=json.loads(audit_path.read_text());assert audit['status']=='PASS_CONTROL_REUSE_IDENTITY' and audit['control_count']==1200
    model_path=root/'AUDIT_MODEL_IDENTITY.json';model_audit=json.loads(model_path.read_text())
    assert model_audit['status']=='PASS_MODEL_IDENTITIES' and model_audit['task']==task
    assert model_audit['control_audit']['sha256']==sha256(audit_path)
    results=sorted(root.glob('TECH/R3_ORIGINAL/*/H0/result.json'));assert len(results)==4
    rows=[];replans=0
    for p in results:
        d=json.loads(p.read_text());assert d['status']=='COMPLETE' and d['method_failure'] is None and d['optimizer_updates']==0
        assert d['replans'][0]['control_first_plan_check']['status']=='BITWISE_EQUAL'
        assert d['replans'][0]['history_raw_indices']==[0] and d['replans'][0]['observed_history_frames']==1
        with np.load(p.parent/'trajectory.npz',allow_pickle=False) as tr:
            control=Path(d['control_source'])
            base=json.loads((control/'result.json').read_text())
            assert (d['replan_calls']>1)==(base['replan_calls']>1)
            with np.load(control/'trajectory.npz',allow_pickle=False) as old:
                prefix=min(25,d['executed_raw_steps'])
                for name in ['raw_actions','raw_pixels','raw_physical_state','raw_dynamical_state']:
                    upto=prefix if name=='raw_actions' else prefix+1
                    if not np.array_equal(tr[name][:upto],old[name][:upto]):raise RuntimeError('TECH first-segment state/action/render differs: '+name)
            for rp in d['replans'][1:]:
                assert rp['replan_index']==1 and rp['anchor_raw']==25 and rp['history_raw_indices']==[15,20,25]
                assert rp['observed_history_frames']==3 and rp['future_macro_horizon']==5
                assert rp['history_action_raw_start']==15 and rp['history_action_raw_end']==25
                assert rp['predicted_terminal_raw']==50
                assert len(tr['raw_actions'])>=25
                replans+=1
        rows.append({'case_id':d['case_id'],'success':d['success'],'executed_raw_steps':d['executed_raw_steps'],'replan_calls':d['replan_calls'],'wall_seconds':d['wall_seconds'],'first_plan_bitwise_equal':True,'first_segment_state_action_render_bitwise_equal':True,'result':{'path':str(p),**file_record(p)}})
    if replans==0:raise RuntimeError('Fixed four TECH cases did not exercise real-history replan; need same-fixed-TECH offline model-cost receipt, no further trajectories')
    receipt={'status':'PASS','module':'R5_H2','task':task,'TECH_cases':4,'TECH_executed_replans':replans,'rows':rows,
        'code_sha256':{n:sha256(Path(__file__).parent/n) for n in ['run.py','context.py']},'control_audit':{'path':str(audit_path),**file_record(audit_path)},
        'model_identity_audit':{'path':str(model_path),**file_record(model_path)},
        'first_plan_gate':'BITWISE_EQUAL','future_raw_endpoint':50,'history_raw':[15,20,25],'actual_action_prefix_raw':[15,25],
        'no_optimizer_updates':True,'new_TECH_trajectories':4,'technical_scope_amendment':'Full fixed4 TECH metadata list, approved after first2 terminated before replan; no outcome-selected case or changed stream','no_tolerance_relaxation':True,'producer':file_record(__file__)}
    atomic_json(root/'TECH_GATE.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--root',required=True,type=Path);p.add_argument('--audit',required=True,type=Path);a=p.parse_args();report(a.task,a.root,a.audit)
