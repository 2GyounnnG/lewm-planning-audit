"""Zero-update all100 cached-input recovery, not simulator/training recovery."""
import argparse
import numpy as np
from . import core
from .evaluate import write_csv
from .report import OPEN_METRICS

def check(bundle):
    import torch
    bundle=core.Path(bundle).resolve();manifest=core.read(bundle/'RECOVERY_MANIFEST.json');task=manifest['task'];core.policy()
    for rel,r in manifest['files'].items():core.verify(dict(r,path=str(bundle/rel)))
    m=core.r3('model');scoring=core.r3('open_loop');roles=core.read(bundle/'manifests'/f'{task}_data_roles.json');cases=roles['cases']['EVAL']
    config=core.read(bundle/'official/config.json');identity=core.read(bundle/'technical_model/MODEL_CHECK.json')['actual_author_model_identity']
    if identity['weights_sha256']!=core.sha(bundle/'official/weights.pt') or identity['config_sha256']!=core.sha(bundle/'official/config.json') or identity['source_contract_sha256']!=core.digest(core.task_contract(task)):
        raise RuntimeError('Recovered official model/source identity mismatch')
    tol=core.read(bundle/'r3_reference/manifests/NUMERICAL_TOLERANCES.json')['comparisons']['CPU_replay']
    with np.load(bundle/'replay/fixed_open_loop_inputs.npz',allow_pickle=False) as f:inputs={k:f[k] for k in f.files}
    ids=[c['case_id'] for c in cases]
    if len(cases)!=100 or ids!=inputs['case_ids'].tolist():raise RuntimeError('Original frozen100 case order differs')
    variance=core.read(bundle/'manifests'/f'{task}_cache.json')['train_scalar_variance'];rows=[];metrics=[];allpass=True
    saved={(r['case_id'],r['arm'],r['horizon_macro']):r for r in core.read(bundle/'open_loop/per_case.json')}
    for arm in ['H0']+[f'REFIT_{s}' for s in core.SEEDS]:
        model=m.construct_official(config);model.load_state_dict(torch.load(bundle/'official/weights.pt',map_location='cpu',weights_only=True),strict=True)
        model.r3_identity=identity;model.eval().requires_grad_(False);before=m.frozen_hashes(model)
        if arm!='H0':m.apply_delta(model,bundle/'train'/arm.split('_')[1]/'checkpoint_30000.pt')
        parts=[]
        with torch.inference_mode():
            for start in range(0,100,32):
                parts.append(m.cached_rollout(model,torch.from_numpy(inputs['initial_z'][start:start+32]),torch.from_numpy(inputs['macro_actions'][start:start+32]),5).numpy())
        predicted=np.concatenate(parts)
        with np.load(bundle/'open_loop'/f'{arm}_forecast.npz',allow_pickle=False) as f:
            expected=f['prediction'];assert ids==f['case_ids'].tolist()
        frozen=m.assert_frozen(model,before);diff=np.abs(predicted-expected);limit=tol['atol']+tol['rtol']*np.abs(expected)
        passed=bool(np.all(diff<=limit));allpass &= passed
        rows.append(dict(task=task,arm=arm,cases=100,latent_values=int(predicted.size),max_abs=float(diff.max()),max_tolerance_ratio=float(np.max(diff/limit)),values_outside_tolerance=int((diff>limit).sum()),passed=passed,frozen_sha256=frozen['sha256'],optimizer_updates=0))
        for i,c in enumerate(cases):
            for score in scoring.score_case(predicted[i],inputs['target_z'][i],variance,inputs['initial_z'][i]):
                h=score['horizon_macro'];old=saved[c['case_id'],arm,h]
                metrics.append(dict(task=task,case_id=c['case_id'],arm=arm,horizon_macro=h,**{'cpu_'+k:score[k] for k in OPEN_METRICS},**{'original_'+k:old[k] for k in OPEN_METRICS}))
        del model
    out=bundle.parent/(task+'_CPU_RECOVERY');write_csv(out/'CPU_RECOVERY_RAW.csv',rows)
    write_csv(out/'OPEN_LOOP_METRICS_CPU_RAW.csv',metrics)
    result={'task':task,'status':'PASS' if allpass else 'FAILED_FROZEN_CPU_TOLERANCE','scope':'All100 fixed cached-input h1..5 forecast recomputation for H0 and three final refits; strict weights/delta/source/file hashes; no simulator or full training/resume environment claim',
        'CPU_tolerance':tol,'cases':100,'arms':4,'new_optimizer_updates':0,'new_case_selection':False,'encoder_reencoding_checked':False,'simulator_replay_checked':False,'manifest':core.file_record(bundle/'RECOVERY_MANIFEST.json'),'report_source':core.file_record(__file__),'raw_table':core.file_record(out/'CPU_RECOVERY_RAW.csv')}
    core.atomic(out/'CPU_RECOVERY_STATUS.json',result);print(core.canonical(result).decode())
    if not allpass:raise RuntimeError('Recovered CPU forecast exceeds original frozen tolerance')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle');a=p.parse_args();check(a.bundle)
