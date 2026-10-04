"""CPU reevaluation of selected recovered probes on the exact100 saved EVAL anchors."""
import argparse,json,tarfile,time
from pathlib import Path
import numpy as np
from common import *

def run(a):
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(a.root);manifest=read(root/'recovery'/f'{a.task}_RECOVERY_MANIFEST.json');checks=[]
    for r in manifest['members']:
        p=root/r['relative_path'];assert p.resolve().is_relative_to(root.resolve());assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],('RECOVERY_SHA_MISMATCH',r['relative_path'])
    for p in sorted((root/'fits'/a.task).glob('**/COMPLETE.json')):
        complete=read(p);folder=p.parent;d=np.load(folder/'evaluation.npz',allow_pickle=False);x=d['x'];y=d['y'];reference=d['pred'];h=complete['target'].startswith('latent_')
        if complete['model']=='RIDGE':
            f=np.load(folder/'fit.npz');pred=x@f['coefficient']+f['intercept'];modelp=folder/'fit.npz'
        else:
            cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);net=nn.Sequential(nn.Linear(x.shape[1],256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,y.shape[1])).float();net.load_state_dict(cp['net'],strict=True);net.eval().requires_grad_(False)
            with torch.inference_mode():pred=(net((torch.from_numpy(x.astype(np.float32))-cp['mean_x'])/cp['scale_x'])*cp['scale_y']+cp['mean_y']).numpy()
            if h:pred=pred+d['current_z']
            modelp=folder/'best.pt'
        pred=pred.astype(np.float32) if h else pred.astype(np.float64)
        if h:remote_error=np.mean((reference-y)**2,axis=1);local_error=np.mean((pred-y)**2,axis=1)
        else:remote_error=(reference-y)**2;local_error=(pred-y)**2
        assert np.isfinite(pred).all() and np.isfinite(local_error).all();npz(folder/'LOCAL_CPU_REEVALUATION.npz',pred=pred,error=local_error)
        checks.append({'task':a.task,'model':complete['model'],'target':complete['target'],'history':complete['history'],'seed':complete['seed'],'cases':len(x),'parameter_file_sha256':sha(modelp),'eval_input_file_sha256':sha(folder/'evaluation.npz'),'prediction_max_abs_difference':float(np.max(np.abs(pred-reference))),'error_max_abs_difference':float(np.max(np.abs(local_error-remote_error))),'mean_error_difference':float(local_error.mean()-remote_error.mean()),'remote_mean_error':float(remote_error.mean()),'local_mean_error':float(local_error.mean()),'optimizer_updates':0,'world_model_updates':0})
    atomic(root/'CPU_RECOVERY_CHECK.json',{'status':'ALL_RECOVERED_MEMBERS_SHA_VERIFIED_AND_ALL_SELECTED_PROBES_CPU_REEVALUATED','task':a.task,'scope':'Every recovered selected ridge/MLP fit, all original100 EVAL anchors per fit, exact saved latent/action inputs and labels. No encoder rerun, simulator, or training. Cross-platform differences reported without silently changing scientific tables or adding a post-result tolerance.','source_manifest_sha256':sha(root/'recovery'/f'{a.task}_RECOVERY_MANIFEST.json'),'source_members_checked':len(manifest['members']),'probes_reevaluated':len(checks),'case_model_cells':sum(r['cases'] for r in checks),'max_prediction_abs_difference':max(r['prediction_max_abs_difference'] for r in checks),'max_error_abs_difference':max(r['error_max_abs_difference'] for r in checks),'checks':checks,'world_model_updates':0,'probe_optimizer_updates':0,'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':1})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task',required=True);run(p.parse_args())
