"""CPU-only re-scoring of a metadata-selected recovered R4 trajectory.

This is a recovery check, not additional experimental observations. It does not
reconstruct pixels or claim CPU simulator equivalence to the original host.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
from deliver import record,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True)
    p.add_argument('--package',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.r3_root).resolve();package=Path(a.package).resolve();out=Path(a.out)
    os.environ['R3_ROOT']=str(root);sys.path.insert(0,str(root))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from r3.common import fp32_policy
    from r3.model import load_official,apply_delta,tensor_sha256
    from r3.data import action_processor
    from r4.s1 import cross_trajectory
    torch.set_num_threads(1);torch.set_num_interop_threads(1);fp32_policy()
    derivation=json.loads((package/'DERIVATION.json').read_text())
    sources=[('trajectory.compact.npz','compact_trajectory'),('source_result.json','source_result'),('source_COMPLETE.json','source_completion')]
    for filename,key in sources:
        r=record(package/filename)
        if any(r[k]!=derivation[key][k] for k in ('bytes','sha256')):raise ValueError('Recovery hash mismatch: '+filename)
    result=json.loads((package/'source_result.json').read_text())
    complete=json.loads((package/'source_COMPLETE.json').read_text())
    if result['status']!='COMPLETE' or complete['identity_sha256']!=result['identity_sha256']:raise ValueError('Unsealed original case')
    if complete['files']['trajectory.npz']!=derivation['source_trajectory'] or complete['files']['result.json']!=derivation['source_result']:raise ValueError('Original completion record does not bind derivation sources')
    case=derivation['case_window_entry']['case'];task=case['task']
    with np.load(package/'trajectory.compact.npz',allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    with Path(a.reference).open(newline='') as f:remote=[r for r in csv.DictReader(f) if r['case_id']==case['case_id'] and r['source_policy']==result['arm']]
    if not remote:raise ValueError('No original score rows for selected case')
    def key(r):return tuple(str(r[k]) for k in ('model','anchor_raw','history_kind','horizon_raw'))
    refs={key(r):r for r in remote}
    if len(refs)!=len(remote):raise ValueError('Duplicate source comparison cell')
    route=json.loads((root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][task]['arms']
    values=[];model_checks=[];checkpoint_inputs=[]
    for arm in ('H0','REFIT_103201','REFIT_103202','REFIT_103203'):
        model=load_official(task,'cpu')
        if arm!='H0':
            entry=next(r for r in route if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]))
            cp=root/entry['checkpoint']['path'];receipt=record(cp)
            if any(receipt[k]!=entry['checkpoint'][k] for k in ('bytes','sha256')):raise ValueError('Fixed checkpoint changed')
            checkpoint_inputs.append(receipt);apply_delta(model,cp)
        before={k:tensor_sha256(t) for k,t in model.state_dict().items()}
        scores=cross_trajectory(model,task,arm,case,result['arm'],arrays,action_processor(task))
        after={k:tensor_sha256(t) for k,t in model.state_dict().items()}
        if before!=after:raise ValueError('Parameters or buffers changed during recovery inference')
        model_checks.append({'arm':arm,'parameters_and_buffers_unchanged':True,'state_entries':len(before),'state_hash':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()})
        for r in scores:
            old=refs[key(r)]
            if bool(r['valid'])!=(old['valid'].lower()=='true') or r['missing_reason']!=old['missing_reason']:raise ValueError('Recovered eligibility differs')
            for metric in ('latent_mse','predicted_goal_cost','realized_goal_cost','optimism'):
                v=r[metric];original=float(old[metric]) if old[metric] else None
                values.append({'task':task,'case_id':case['case_id'],'source_policy':result['arm'],**{k:r[k] for k in ('model','anchor_raw','history_kind','horizon_raw','valid','missing_reason')},'metric':metric,'original_newhost_value':original,'recovered_cpu_value':v,'absolute_difference':abs(v-original) if v is not None else None,'relative_difference':abs(v-original)/max(abs(original),1e-30) if v is not None else None,'purpose':'TECHNICAL_RECOVERY_NOT_NEW_EXPERIMENT'})
        del model
    if len(values)!=4*len(remote):raise ValueError('CPU comparison coverage mismatch')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'RECOVERED_S1_CPU_RAW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(values[0]));w.writeheader();w.writerows(values)
    summary={'status':'COMPLETE_RECOVERY_COMPARISON','case_id':case['case_id'],'case_selection':'FIXED_METADATA_SUBSET_ORDER_FIRST_CASE_H0','source_trajectories_checked':1,'models_checked':4,'score_rows':len(remote),'raw_comparison_rows':len(values),'eligibility_exact':True,'all_parameters_and_buffers_unchanged':True,'optimizer_updates':0,'new_simulator_steps':0,'pixel_reconstruction_tested':False,'no_new_scientific_gate_or_tolerance':True,'model_checks':model_checks,'inputs':[record(package/name) for name in ('DERIVATION.json','trajectory.compact.npz','source_result.json','source_COMPLETE.json')]+[record(a.reference)]+checkpoint_inputs,'metrics':{metric:{'maximum_absolute_difference':max(r['absolute_difference'] for r in values if r['metric']==metric and r['valid']),'maximum_relative_difference':max(r['relative_difference'] for r in values if r['metric']==metric and r['valid'])} for metric in ('latent_mse','predicted_goal_cost','realized_goal_cost','optimism')},'outputs':[record(out/'RECOVERED_S1_CPU_RAW.csv')]}
    atomic(out/'RECOVERY_CHECK.json',summary);print(json.dumps({k:summary[k] for k in ('status','case_id','score_rows','metrics')}))

if __name__=='__main__':main()
