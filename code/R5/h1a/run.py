"""R5 H1a: frozen inference, readonly R4 recovery inputs, atomic case records."""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
ARMS=['H0','REFIT_103201','REFIT_103202','REFIT_103203']
LABEL='POST_R4_SUPPLEMENT_ON_KNOWN_EVALUATION_CASES'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def record(p): return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with q.open('w') as f: json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    q.replace(p)
def freeze(p,v):
    if p.exists():
        if read(p)!=v: raise RuntimeError('Frozen contract changed')
    else: atomic(p,v)
def writecsv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    with q.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    q.replace(p)
def state_sha(model,api):
    return hashlib.sha256(json.dumps({n:api.tensor_sha256(t) for n,t in model.state_dict().items()},sort_keys=True).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['tworoom','cube'],required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--r3-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=12);a=p.parse_args()
    import numpy as np, torch
    os.environ['R3_ROOT']=str(a.r3_root.resolve());sys.path.insert(0,str(a.r3_root.resolve()))
    from r3 import model as api
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    b=a.bundle;manifest=read(b/'RECOVERY_MANIFEST.json')
    names=['official/config.json','official/weights.pt','technical_model/MODEL_CHECK.json','replay/fixed_open_loop_inputs.npz',f'manifests/{a.task}_data_roles.json']+[f'train/{s}/checkpoint_30000.pt' for s in (103201,103202,103203)]
    sources={}
    for n in names:
        r=record(b/n);old=manifest['files'][n]
        if any(r[k]!=old[k] for k in ('bytes','sha256')): raise RuntimeError('Frozen recovery source differs: '+n)
        sources[n]=r
    sources['RECOVERY_MANIFEST.json']=record(b/'RECOVERY_MANIFEST.json');sources['r3/model.py']=record(api.__file__)
    for component,rel in [('lewm','jepa.py'),('lewm','module.py'),('spt','stable_pretraining/backbone/utils.py')]:
        sources[component+'/'+rel]=record(api._checked_source(component,rel))
    with np.load(b/'replay/fixed_open_loop_inputs.npz',allow_pickle=False) as f: x={k:f[k].copy() for k in f.files}
    cases=read(b/f'manifests/{a.task}_data_roles.json')['cases']['EVAL'];ids=[c['case_id'] for c in cases]
    if len(ids)!=100 or len(set(ids))!=100 or ids!=x['case_ids'].tolist(): raise RuntimeError('Fixed cases differ')
    anchor=np.array([c['open_loop_anchor_raw'] for c in cases]);rawhist=anchor[:,None]+np.array([-10,-5,0]);rawtarget=anchor[:,None]+np.arange(5,26,5)
    if not np.array_equal(x['history_raw_indices'],rawhist) or not np.array_equal(x['target_raw_indices'],rawtarget) or (rawhist<0).any(): raise RuntimeError('Illegal past/target mapping')
    expected_action=np.stack([np.arange(t-10,t+25) for t in anchor])
    if not np.array_equal(x['action_raw_indices'],expected_action): raise RuntimeError('Action alignment changed')
    identity=read(b/'technical_model/MODEL_CHECK.json')['actual_author_model_identity'];config=read(b/'official/config.json')
    if identity['weights_sha256']!=sources['official/weights.pt']['sha256'] or identity['config_sha256']!=sources['official/config.json']['sha256']: raise RuntimeError('Identity mismatch')
    contract=dict(version='R5_H1A_V1',task=a.task,evidence_label=LABEL,arms=ARMS,cases=ids,sources=sources,code=record(__file__),
        histories={'H_POLICY':'last real frame, macro_actions[:,2:], autoregressive growth/truncation to 3','H_REAL3':'real frames t-10,t-5,t; all 7 macro_actions; same autoregressive rollout'},
        horizons_macro=[1,2,5],metric='numpy float32 mean squared latent error, inherited R4 S1',batch_size=32,precision='FP32_NO_TF32_NO_AMP',optimizer_updates=0,
        statistics='5000 inherited R4 case bootstrap; fixed3 mean within case; PushT family sensitivity; relative improvement=(mean_H0-mean_refit)/mean_H0; ratio of means in every bootstrap draw; unadjusted conditional95',
        technical_test='CPU synthetic seeded0 batch2 native model.predict versus cached_rollout, both histories, exact; no data/seed selection',
        H2_tworoom_trigger='H0 h5 mean REAL3 <= 0.5 * mean POLICY; do not execute H2 in initial-only stage')
    cp=a.out/'CONTRACT.json';freeze(cp,contract);ch=sha(cp);rows=[];receipts=[]
    for arm in ARMS:
        model=api.construct_official(config);model.load_state_dict(torch.load(b/'official/weights.pt',map_location='cpu',weights_only=True),strict=True);model.r3_identity=identity
        if arm!='H0': api.apply_delta(model,b/'train'/arm.split('_')[1]/'checkpoint_30000.pt')
        model.eval().requires_grad_(False);before=state_sha(model,api)
        if arm=='H0':
            g=torch.Generator().manual_seed(0);checks=[]
            for history in (1,3):
                z=torch.randn(2,history,model.r3_contract['latent_dim'],generator=g);acts=torch.randn(2,history+4,model.r3_contract['macro_action_dim'],generator=g)
                with torch.inference_mode():
                    cached=api.cached_rollout(model,z,acts,5);hist=z;parts=[]
                    for j in range(5):
                        lo=max(0,hist.shape[1]-3);pred=model.predict(hist[:,lo:],model.action_encoder(acts[:,lo:history+j]))[:,-1];parts.append(pred);hist=torch.cat([hist,pred[:,None]],1)
                    expected=torch.stack(parts,1)
                if not torch.equal(cached,expected): raise RuntimeError('CPU semantics test failed')
                checks.append(dict(history=history,max_abs=float((cached-expected).abs().max())))
            atomic(a.out/'CPU_TECH.json',dict(status='PASS',checks=checks,contract_sha256=ch,optimizer_updates=0,all_state_unchanged=state_sha(model,api)==before,scope='Synthetic semantics only; does not supersede any R4 recovery gate'))
        model=model.to(a.device);armstart=time.time()
        for lo in range(0,100,32):
            hi=min(100,lo+32);paths=[a.out/'cases'/arm/(cid+'.json') for cid in ids[lo:hi]]
            if all(p.exists() for p in paths):
                for q in paths:
                    old=read(q)
                    if old['contract_sha256']!=ch: raise RuntimeError('Resume contract differs')
                    rows.extend(old['rows'])
                continue
            forecasts={}
            for kind in ('H_POLICY','H_REAL3'):
                z=x['initial_z'][lo:hi,-1:] if kind=='H_POLICY' else x['initial_z'][lo:hi]
                acts=x['macro_actions'][lo:hi,2:] if kind=='H_POLICY' else x['macro_actions'][lo:hi]
                with torch.inference_mode(): forecast=api.cached_rollout(model,torch.as_tensor(z,device=a.device),torch.as_tensor(acts,device=a.device),5).cpu().numpy()
                forecasts[kind]=forecast
            for j,i in enumerate(range(lo,hi)):
                values=[]
                for kind,pred in forecasts.items():
                    for h in (1,2,5):
                        values.append(dict(task=a.task,case_id=ids[i],family_id=cases[i].get('family_id','FAMILY_UNKNOWN'),model=arm,source_policy='OFFLINE_RECORDED',anchor_raw=int(anchor[i]),history_kind=kind,horizon_raw=5*h,valid=True,missing_reason='',latent_mse=float(np.mean((pred[j,h-1]-x['target_z'][i,h-1])**2)),predicted_goal_cost=None,realized_goal_cost=None,optimism=None,source_relation='OFFLINE',primary_anchor=True,contract_sha256=ch,evidence_label=LABEL,source_input_sha256=sources['replay/fixed_open_loop_inputs.npz']['sha256']))
                q=a.out/'cases'/arm/(ids[i]+'.json');atomic(q,dict(contract_sha256=ch,rows=values,forecast={k:v[j].tolist() for k,v in forecasts.items()}));rows.extend(values)
        after=state_sha(model,api)
        if after!=before or any(p.requires_grad or p.grad is not None for p in model.parameters()): raise RuntimeError('Frozen model state changed')
        receipts.append(dict(arm=arm,all_state_sha256=before,all_parameters_buffers_unchanged=True,optimizer_updates=0,seconds=time.time()-armstart));del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        atomic(a.out/'PROGRESS.json',dict(complete_arms=len(receipts),rows=len(rows),contract_sha256=ch))
    for r in sources.values():
        if sha(r['path'])!=r['sha256']: raise RuntimeError('Source changed during inference')
    writecsv(a.out/'S1_offline_raw.csv',rows)
    atomic(a.out/'COMPLETE.json',dict(status='COMPLETE',task=a.task,rows=len(rows),cases=100,arms=4,optimizer_updates=0,contract=record(cp),CPU_TECH=record(a.out/'CPU_TECH.json'),raw=record(a.out/'S1_offline_raw.csv'),model_receipts=receipts,device=a.device,torch=torch.__version__,numpy=np.__version__))
    print(json.dumps(read(a.out/'COMPLETE.json')),flush=True)
if __name__=='__main__': main()
