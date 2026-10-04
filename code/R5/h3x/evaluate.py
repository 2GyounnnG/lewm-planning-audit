"""H3X final-only offline plus mandatory official-history closed loop."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from . import common

def verify_code(r4root):
    lock=common.read_json(Path(__file__).with_name('EVALUATION_CODE_FREEZE.json'))
    for n,h in lock['files'].items():
        if common.sha256(Path(__file__).parent/n)!=h:raise RuntimeError('H3X evaluation code changed '+n)
    for n,h in lock['inherited_r4'].items():
        if common.sha256(Path(r4root)/n)!=h:raise RuntimeError('Inherited R4 source changed '+n)
    return lock

def file(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':common.sha256(p)}

def state_sha(m):
    from r3.model import tensor_sha256
    return hashlib.sha256(json.dumps({n:tensor_sha256(t) for n,t in m.state_dict().items()},sort_keys=True).encode()).hexdigest()

def offline(m,seed,cp,out):
    import numpy as np,torch
    from r3.model import cached_rollout
    root=common.R3_ROOT;folder=root/'artifacts/open_loop/reacher';sources={n:file(folder/n) for n in ('inputs.npz','targets.npz')};sources['roles']=file(root/'manifests/reacher_data_roles.json');sources['final_checkpoint']=file(cp)
    with np.load(folder/'inputs.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
    with np.load(folder/'targets.npz',allow_pickle=False) as f:target=f['target_z'].copy()
    cases=common.read_json('manifests/reacher_data_roles.json')['cases']['EVAL'];lookup={c['case_id']:c for c in cases};ids=x['case_ids'].tolist();assert len(ids)==100 and set(ids)==set(lookup)
    assert x['initial_z'].shape==(100,3,192) and target.shape==(100,5,192) and x['macro_actions'].shape==(100,7,10)
    rows=[];predictions={};device=next(m.parameters()).device;before=state_sha(m)
    for history in ('H_POLICY','H_REAL3'):
        z=x['initial_z'][:,-1:] if history=='H_POLICY' else x['initial_z'];a=x['macro_actions'][:,2:] if history=='H_POLICY' else x['macro_actions'];parts=[]
        with torch.inference_mode():
            for lo in range(0,100,32):parts.append(cached_rollout(m,torch.as_tensor(z[lo:lo+32],device=device),torch.as_tensor(a[lo:lo+32],device=device),5).cpu().numpy())
        pred=np.concatenate(parts);predictions[history]=pred
        assert pred.dtype==target.dtype==np.float32
        for i,cid in enumerate(ids):
            for h in (1,2,5):rows.append({'label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','task':'reacher','case_id':cid,'family_id':lookup[cid]['family_id'],'arm':f'H3X_{seed}','refit_seed':seed,'checkpoint_step':30000,'history_kind':history,'horizon_macro':h,'anchor_raw':lookup[cid]['open_loop_anchor_raw'],'valid':True,'missing_reason':'','latent_mse':float(np.mean((pred[i,h-1]-target[i,h-1])**2)),'checkpoint_sha256':sources['final_checkpoint']['sha256'],'input_sha256':sources['inputs.npz']['sha256'],'target_sha256':sources['targets.npz']['sha256']})
    out.mkdir(parents=True,exist_ok=True)
    with (out/'OPEN_LOOP_RAW.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'OPEN_LOOP_PREDICTIONS.npz',case_ids=np.array(ids),target_z=target,**predictions)
    after=state_sha(m);assert before==after
    common.atomic_json(out/'OPEN_LOOP_COMPLETE.json',{'status':'COMPLETE','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','seed':seed,'rows':len(rows),'sources':sources,'model_all_state_before':before,'model_all_state_after':after,'optimizer_updates':0,'raw':file(out/'OPEN_LOOP_RAW.csv'),'predictions':file(out/'OPEN_LOOP_PREDICTIONS.npz'),'code':file(__file__)})

def closed(m,seed,cp,out,assets):
    import torch
    import r4.runner as runner
    from r4.common import STREAMS,sha256
    from r4.export_cases import CaseWindows
    from r3.data import action_processor,image_transform
    arm=f'H3X_{seed}'
    # Only extend the in-memory arm validation list; original runner/CEM/history code stays exact and read-only.
    runner.ARMS=tuple(runner.ARMS)+(arm,)
    manifest=common.read_json(Path(assets)/'CASE_WINDOWS.json');entries=[e for e in manifest['tasks']['reacher']['cases'] if e['case']['role']=='EVAL'];assert len(entries)==100
    roles=common.read_json('manifests/reacher_data_roles.json');assert {e['case']['case_id'] for e in entries}=={c['case_id'] for c in roles['cases']['EVAL']}
    before=state_sha(m);provenance={'experimental_label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','official':m.r3_identity,'final_checkpoint':file(cp),'case_manifest':sha256(Path(assets)/'CASE_WINDOWS.json'),'wrapper':file(__file__),'history':'EXACT_OFFICIAL_DEFAULT_R4_RUNNER','optimizer_updates':0};done=0
    for stream in STREAMS:
        for entry in entries:
            case=entry['case'];folder=out/'FORMAL'/stream/case['case_id']/arm
            r=runner.run_case(m,'reacher',case,CaseWindows(assets,entry),action_processor('reacher'),image_transform(),arm=arm,stream=stream,output=folder,provenance=provenance,device='cuda',phase='FORMAL')
            done+=1;common.atomic_json(out.parent/f'CLOSED_PROGRESS_{seed}.json',{'label':common.LABEL,'seed':seed,'completed':done,'target':300,'latest_case':case['case_id'],'stream':stream,'pid':os.getpid(),'unix':time.time()})
            if done%10==0:print('CLOSED',seed,done,300,flush=True)
    after=state_sha(m);assert before==after
    common.atomic_json(out.parent/f'CLOSED_COMPLETE_{seed}.json',{'status':'COMPLETE','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','seed':seed,'trajectories':done,'cases':100,'streams':list(STREAMS),'model_all_state_before':before,'model_all_state_after':after,'optimizer_updates':0,'provenance':provenance})

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=common.SEEDS,required=True);p.add_argument('--assets',default='/workspace/shared_data/r4_assets_reacher');p.add_argument('--r4-root',default='/workspace/r4_v23_execution');p.add_argument('--module',choices=['both','offline','closed'],default='both');a=p.parse_args()
    import torch
    sys.path.insert(0,a.r4_root);verify_code(a.r4_root)
    from r3.model import load_official,apply_delta
    common.fp32_policy();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    train=common.ROOT/f'artifacts/train/H3X_reacher_s{a.seed}';result=common.read_json(train/'result.json');assert result['actual_updates']==30000 and not result['technical'];cp=train/'checkpoint_30000.pt';assert common.sha256(cp)==result['last_checkpoint']['sha256']
    m=load_official('reacher','cuda');apply_delta(m,cp);m.eval().requires_grad_(False)
    offline_error=None
    if a.module in ('both','offline'):
        try:offline(m,a.seed,cp,common.ROOT/f'evaluation/open_loop/{a.seed}')
        except Exception as error:
            offline_error=error
            common.atomic_json(common.ROOT/f'evaluation/open_loop/{a.seed}/TECHNICAL_FAILURE.json',{'label':common.LABEL,'error_type':type(error).__name__,'error':str(error),'closed_loop_still_required':True})
    # Independent of every open-loop value and even an open-loop reporting failure.
    if a.module in ('both','closed'):closed(m,a.seed,cp,common.ROOT/'evaluation/closed_loop/trajectories',a.assets)
    if offline_error is not None:raise RuntimeError('Closed loop completed; independent open-loop technical repair remains') from offline_error
if __name__=='__main__':main()
