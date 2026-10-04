"""TECH timing only; no SIM_CEM scientific job is dispatched here."""
import argparse,json,os,sys,time
from pathlib import Path
from .common import atomic_json,file_record,fixed_subset,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--trajectories',required=True);p.add_argument('--s0',required=True);p.add_argument('--task',required=True,choices=['pusht','reacher']);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    gate=json.loads((Path(a.s0)/'TERMINATION_TECH_RAW.json').read_text())
    if gate['fixed25_unavailable_observed']:
        atomic_json(a.output,{'status':'TECHNICALLY_UNEVALUABLE_FIXED_HORIZON','reason':'TECH observed true LAST before25; full SIM_CEM not dispatched','timing_queries':0});return
    receipts=[json.loads(p.read_text()) for p in Path(a.s0).glob('*_S0_BCD.json')]
    if not receipts or any(r['status']!='PASS' for r in receipts):raise RuntimeError('Timing prefix requires S0 b-d')
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import numpy as np,torch
    from r3.model import load_official
    from r3.data import action_processor,image_transform
    from r3.common import fp32_policy
    from .export_cases import CaseWindows
    from .replay import run_branch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);fp32_policy();model=load_official(a.task,a.device);processor=action_processor(a.task);transform=image_transform()
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks'][a.task]['cases'];tech=[e for e in entries if e['case']['role']=='TECH'][:4]
    rows=[]
    for entry in tech:
        path=Path(a.trajectories)/'TECH/R3_ORIGINAL'/entry['case']['case_id']/'H0/trajectory.npz'
        with np.load(path) as log:raw=log['raw_actions'].copy();plan=log['returned_plans_normalized'][0].copy()
        candidate=processor.inverse_transform(plan.reshape(25,2)).astype(np.float32)
        for repeat in range(5):
            prefix=raw[:min(10,len(raw))] if repeat%2 else np.empty((0,2),np.float32)
            torch.cuda.synchronize();start=time.perf_counter()
            result=run_branch(a.task,entry['case'],lambda:CaseWindows(a.assets,entry),prefix,candidate,model,transform)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            rows.append({'task':a.task,'case_id':entry['case']['case_id'],'repeat':repeat,'prefix_raw':len(prefix),'candidate_raw':result['candidate_raw_steps'],'valid_fixed_horizon':bool(result['valid_fixed_horizon']),'end_to_end_branch_render_encode_seconds':elapsed,'source_trajectory':file_record(path),'scope':'TECH_TIMING_NOT_FORMAL'})
    if not all(r['valid_fixed_horizon'] for r in rows):raise RuntimeError('Fixed horizon became unavailable in TECH timing')
    p90=float(np.quantile([r['end_to_end_branch_render_encode_seconds'] for r in rows],.9))
    cases=json.loads((Path(a.r3_root)/f'manifests/{a.task}_data_roles.json').read_text())['cases']['EVAL'];subset=fixed_subset(a.task,cases);replans=[]
    for case in subset:
        path=Path(a.trajectories)/'FORMAL/R3_ORIGINAL'/case['case_id']/'H0/result.json';result=json.loads(path.read_text());replans.append({'case_id':case['case_id'],'observed_H0_replans':result['replan_calls'],'source':file_record(path)})
    total=sum(r['observed_H0_replans'] for r in replans);serial=total*9000*p90
    result={'status':'ESTIMATE_ONLY_NO_SIM_CEM_DISPATCH','task':a.task,'timing_queries':len(rows),'AA_or_order_check_queries':0,'TECH_raw_steps':sum(r['candidate_raw']+r['prefix_raw'] for r in rows),'p90_end_to_end_seconds':p90,'timing_definition':'Synchronized wall time enclosing fresh reset, prefix replay,25candidate rawsteps, terminal rendering, frozen H0 encoding and environment close','queries_per_replan':9000,'cases':20,'original_stream_H0_observed_total_replans':total,'replan_rows':replans,'per_arm_serial_seconds_using_observed_replans':serial,'both_arms_serial_seconds_using_observed_replans':2*serial,'per_arm_serial_seconds_budget_max40replans':40*9000*p90,'parallel_assumptions':[{'workers':n,'ideal_per_arm_seconds':serial/n,'status':'UNMEASURED_IDEAL_BOUND_NOT_AN_EXECUTION_QUALIFICATION'} for n in (2,12,24)],'qualified_for_second_batch_by_measured_serial_time':2*serial<=7200,'second_batch_requires_root_MAIN_RESULTS_DELIVERED':True,'raw_timing_rows':rows,'code_sha256':sha256(__file__),'optimizer_updates':0}
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('raw_timing_rows','replan_rows')}),flush=True)

if __name__=='__main__':main()
