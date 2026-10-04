"""Explicit compact H3X transfer and full100-anchor CPU reevaluation.

Original remote files are retained. Derived trajectory files omit pixels only;
every retained tensor is hashed and linked to the original NPZ SHA. CPU mode
recomputes saved offline forecasts, never claims to replay a simulator.
"""
import argparse,hashlib,json,os,shutil,tarfile,time
from pathlib import Path
import numpy as np
from . import common
from .evaluate import file,state_sha

def tsha(x):
    x=np.ascontiguousarray(x);h=hashlib.sha256(json.dumps({'shape':list(x.shape),'dtype':str(x.dtype)},sort_keys=True).encode());h.update(x.tobytes());return h.hexdigest()

def pack(seed):
    root=common.ROOT;train=root/f'artifacts/train/H3X_reacher_s{seed}';r=common.read_json(train/'result.json');assert r['actual_updates']==30000 and not r['technical'];closedcomplete=root/f'evaluation/closed_loop/CLOSED_COMPLETE_{seed}.json';d=common.read_json(closedcomplete);assert d['trajectories']==300
    out=root/f'recovery/bundles/{seed}';out.mkdir(parents=True,exist_ok=True);mapping=[];sources=[]
    def copy(src,rel):
        src=Path(src);dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert common.sha256(dst)==common.sha256(src);sources.append({'source':file(src),'bundle_path':rel});return dst
    for n in ['checkpoint_30000.pt','result.json','updates.jsonl','monitor.jsonl','freeze_checks.jsonl','RUN_IDENTITY.json','FROZEN_INITIAL.json','PARAMETER_WHITELIST.json','sampling_counts.npz','last.json','resume.json']:copy(train/n,f'train/{seed}/{n}')
    for p in sorted((root/f'evaluation/open_loop/{seed}').glob('*')):
        if p.is_file():copy(p,f'evaluation/open_loop/{seed}/{p.name}')
    copy(closedcomplete,f'evaluation/closed_loop/CLOSED_COMPLETE_{seed}.json')
    from r4.common import STREAMS
    roles=common.read_json('manifests/reacher_data_roles.json');ids=sorted(c['case_id'] for c in roles['cases']['EVAL'])
    for stream in STREAMS:
        for cid in ids:
            folder=root/f'evaluation/closed_loop/trajectories/FORMAL/{stream}/{cid}/H3X_{seed}';rel=f'evaluation/closed_loop/trajectories/FORMAL/{stream}/{cid}/H3X_{seed}';complete=common.read_json(folder/'COMPLETE.json');original=folder/'trajectory.npz';assert common.sha256(original)==complete['files']['trajectory.npz']['sha256']
            for p in sorted(folder.rglob('*.json')):copy(p,rel+'/'+str(p.relative_to(folder)))
            with np.load(original,allow_pickle=False) as f:
                excluded=[k for k in f.files if k in ('raw_pixels','goal_pixels')];retained={k:f[k].copy() for k in f.files if k not in excluded}
            target=out/rel/'trajectory_numeric.npz';np.savez_compressed(target,**retained);mapping.append({'source_original':file(original),'derived':file(target),'bundle_path':str(target.relative_to(out)),'omitted_keys':excluded,'retained_tensor_sha256':{k:tsha(x) for k,x in retained.items()},'scope':'DERIVED_NUMERIC_ALL_RETAINED_ARRAYS_EXACT; original untouched and retained remote'})
    remoteonly=[file(train/n) for n in ('resume.pt','checkpoint_3000.pt','checkpoint_10000.pt')]
    common.atomic_json(out/'RECOVERY_MANIFEST.json',{'label':common.LABEL,'status':'COMPACT_RECOVERY_BUNDLE_COMPLETE','seed':seed,'full_final_predictor_checkpoint':True,'closed_trajectories':300,'derived_trajectories':mapping,'copied_original_files':sources,'remote_only_files':remoteonly,'original_closed_pixel_trajectories_remote_retained':True,'files':[{**file(p),'relative_path':str(p.relative_to(out))} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='RECOVERY_MANIFEST.json'],'scope':'final weights/losses/monitor/sampling/raw result identities/offline forecasts exact; all closed numerical tensors exact, pixels remote-only; no claims of local simulator replay'})
    archive=root/f'recovery/H3X_{seed}.tar.gz';tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:tar.add(out,arcname=str(seed))
    tmp.replace(archive);common.atomic_json(archive.with_suffix('.json'),{'archive':file(archive),'manifest':file(out/'RECOVERY_MANIFEST.json'),'seed':seed});print('PACK_COMPLETE',seed,archive.stat().st_size,flush=True)

def cpu(seed,bundle,r3root):
    import torch
    from r3.model import load_official,apply_delta,cached_rollout
    torch.set_num_threads(4);torch.set_num_interop_threads(1);common.fp32_policy();out=Path(bundle);manifest=common.read_json(out/'RECOVERY_MANIFEST.json')
    for rec in manifest['files']:
        p=out/rec['relative_path'];assert p.stat().st_size==rec['bytes'] and common.sha256(p)==rec['sha256']
    for rec in manifest['derived_trajectories']:
        with np.load(out/rec['bundle_path'],allow_pickle=False) as f:assert {k:tsha(f[k]) for k in f.files}==rec['retained_tensor_sha256']
    cp=out/f'train/{seed}/checkpoint_30000.pt';offline=out/f'evaluation/open_loop/{seed}';receipt=common.read_json(offline/'OPEN_LOOP_COMPLETE.json');ref=Path(r3root)/'artifacts/open_loop/reacher'
    for n in ('inputs.npz','targets.npz'):assert common.sha256(ref/n)==receipt['sources'][n]['sha256']
    with np.load(ref/'inputs.npz',allow_pickle=False) as f:initial=f['initial_z'].copy();actions=f['macro_actions'].copy();ids=f['case_ids'].copy()
    with np.load(ref/'targets.npz',allow_pickle=False) as f:target=f['target_z'].copy()
    with np.load(offline/'OPEN_LOOP_PREDICTIONS.npz',allow_pickle=False) as f:expected={k:f[k].copy() for k in f.files}
    assert np.array_equal(ids,expected['case_ids']) and np.array_equal(target,expected['target_z']);m=load_official('reacher','cpu');apply_delta(m,cp);m.eval().requires_grad_(False);before=state_sha(m);rows=[];arrays={};t=time.monotonic()
    for kind in ('H_POLICY','H_REAL3'):
        z=initial[:,-1:] if kind=='H_POLICY' else initial;a=actions[:,2:] if kind=='H_POLICY' else actions;parts=[]
        with torch.inference_mode():
            for lo in range(0,100,32):parts.append(cached_rollout(m,torch.as_tensor(z[lo:lo+32]),torch.as_tensor(a[lo:lo+32]),5).numpy())
        pred=np.concatenate(parts);arrays[kind]=pred
        for h in (1,2,5):
            actual=np.mean((pred[:,h-1]-target[:,h-1])**2,axis=1);old=np.mean((expected[kind][:,h-1]-target[:,h-1])**2,axis=1)
            rows.append({'history':kind,'horizon_macro':h,'cases':100,'saved_GPU_mean_MSE':float(old.astype(np.float64).mean()),'local_CPU_mean_MSE':float(actual.astype(np.float64).mean()),'max_case_MSE_abs_difference':float(abs(actual-old).max()),'mean_MSE_abs_difference':float(abs(actual.astype(np.float64).mean()-old.astype(np.float64).mean()))})
        rows.append({'history':kind,'scope':'ALL5_FUTURE_LATENT_PREDICTIONS','max_latent_abs_difference':float(abs(pred-expected[kind]).max()),'prediction_values':int(pred.size)})
    after=state_sha(m);assert before==after;np.savez_compressed(out/'LOCAL_CPU_REEVALUATION.npz',case_ids=ids,target_z=target,**arrays)
    common.atomic_json(out/'CPU_RECOVERY_CHECK.json',{'status':'COMPLETE_CPU_REEVALUATION_WITH_RAW_DIFFERENCES','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','seed':seed,'cases':100,'histories':2,'horizons_scored':[1,2,5],'all5_forecast_steps_recomputed':True,'new_numerical_tolerance_gate':False,'comparisons':rows,'checkpoint':file(cp),'model_all_state_before':before,'model_all_state_after':after,'optimizer_updates':0,'new_encoder_inference':0,'new_simulator_steps':0,'seconds':time.monotonic()-t,'verified_numeric_closed_trajectories':len(manifest['derived_trajectories']),'local_predictions':file(out/'LOCAL_CPU_REEVALUATION.npz'),'scope':'Recomputed every saved100-anchor offline forecast for both histories from final predictor in original latent coordinate; derived closed numerical tensors only SHA-verified, not rerun','code':file(__file__)})
    print('CPU_RECOVERY_COMPLETE',seed,time.monotonic()-t,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','cpu']);p.add_argument('--seed',type=int,choices=common.SEEDS,required=True);p.add_argument('--bundle');p.add_argument('--r3-root',default=os.environ.get('R3_ROOT','/workspace/shared_data/r3'));a=p.parse_args()
    if a.mode=='pack':
        import sys;sys.path.insert(0,'/workspace/r4_v23_execution');pack(a.seed)
    else:cpu(a.seed,a.bundle,a.r3_root)
