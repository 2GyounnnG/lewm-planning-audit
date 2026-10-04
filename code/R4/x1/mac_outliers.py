"""Instrument an identical CPU replay repeat; preserve the original failed receipt."""
import argparse,os,shutil
import numpy as np
from . import core,recovery_cpu
from .evaluate import write_csv

def run(source):
    source=core.Path(source).resolve();parent=source.parent/'MAC_OUTLIER_DIAGNOSTIC';bundle=parent/source.name
    if bundle.exists():raise RuntimeError('Diagnostic repeat already exists; no silent repetition')
    shutil.copytree(source,bundle,copy_function=os.link)
    m=core.r3('model');original=m.cached_rollout;observed=[]
    def capture(*args,**kwargs):
        result=original(*args,**kwargs);observed.append(result.detach().cpu().numpy().copy());return result
    m.cached_rollout=capture
    try:
        try:recovery_cpu.check(bundle)
        except RuntimeError as e:
            if str(e)!='Recovered CPU forecast exceeds original frozen tolerance':raise
    finally:m.cached_rollout=original
    if len(observed)!=16:raise RuntimeError('Expected4 unchanged batches per4 fixed arms')
    task=core.read(bundle/'RECOVERY_MANIFEST.json')['task'];rows=[];tol=core.read(bundle/'r3_reference/manifests/NUMERICAL_TOLERANCES.json')['comparisons']['CPU_replay']
    for j,arm in enumerate(['H0']+[f'REFIT_{s}' for s in core.SEEDS]):
        pred=np.concatenate(observed[4*j:4*j+4])
        with np.load(bundle/'open_loop'/f'{arm}_forecast.npz',allow_pickle=False) as f:reference=f['prediction'];ids=f['case_ids']
        diff=np.abs(pred-reference);threshold=tol['atol']+tol['rtol']*np.abs(reference)
        for case,h,coordinate in np.argwhere(diff>threshold):
            rows.append(dict(task=task,arm=arm,case_id=str(ids[case]),case_array_index=int(case),horizon_macro=int(h)+1,latent_coordinate_zero_based=int(coordinate),reference=float(reference[case,h,coordinate]),CPU_prediction=float(pred[case,h,coordinate]),signed_difference=float(pred[case,h,coordinate]-reference[case,h,coordinate]),absolute_difference=float(diff[case,h,coordinate]),threshold=float(threshold[case,h,coordinate]),tolerance_ratio=float(diff[case,h,coordinate]/threshold[case,h,coordinate])))
    out=parent/(task+'_CPU_RECOVERY');old=source.parent/(task+'_CPU_RECOVERY')
    if core.sha(out/'CPU_RECOVERY_RAW.csv')!=core.sha(old/'CPU_RECOVERY_RAW.csv'):raise RuntimeError('Repeat raw aggregate differs; retain as new diagnostic without substituting first outcome')
    write_csv(out/'OUTLIER_COORDINATES.csv',rows)
    core.atomic(out/'DIAGNOSTIC_RECEIPT.json',{'status':'IDENTICAL_FAILURE_REPRODUCED','outlier_coordinates':len(rows),'original_receipt':core.file_record(old/'CPU_RECOVERY_STATUS.json'),'repeat_receipt':core.file_record(out/'CPU_RECOVERY_STATUS.json'),'outliers':core.file_record(out/'OUTLIER_COORDINATES.csv'),'recovery_cpu_source':core.file_record(recovery_cpu.__file__),'instrumentation_source':core.file_record(__file__),'scope':'Calls the same cached_rollout implementation with identical FP32 inputs/batches/threads, copies outputs after return; no changed tolerance/precision/seed/batch, no new cases','repeat_forward_arm_case_pairs':400,'new_optimizer_updates':0,'new_simulator_trajectories':0})
    print(core.canonical({'status':'IDENTICAL_FAILURE_REPRODUCED','outliers':rows}).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle');a=p.parse_args();run(a.bundle)
