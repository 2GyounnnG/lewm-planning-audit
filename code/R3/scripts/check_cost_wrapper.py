"""Zero-update real TECH goal/cost equivalence, using fixed synthetic actions."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from r3 import common
from r3.data import RawH5,image_transform,verified_file
from r3.model import load_official,frozen_hashes,assert_frozen
from r3.planning import AuditedCost,_Timer

def run(task,device,output):
    common.require_authorization({},technical=True);common.fp32_policy()
    root=common.ROOT;folder=root/output
    if folder.exists():raise RuntimeError('Preserve previous cost check; explicit unused attempt directory required')
    folder.mkdir(parents=True);common.atomic_json(folder/'STARTED.json',{'task':task,'optimizer_updates':0})
    names=[f'manifests/{task}_data_roles.json',f'manifests/{task}_source_map.json',f'manifests/{task}_normalization.json',f'manifests/{task}_model_assets.json','scripts/check_cost_wrapper.py','r3/planning.py','r3/model.py','r3/data.py','r3/common.py','r3/env_compat.py','manifests/NUMERICAL_TOLERANCES.json','state/lewm_source_manifest.json','state/spt_source_manifest.json','state/swm_compat_source_manifest.json']
    hashes={p:common.sha256(root/p) for p in names};roles=common.read_json(names[0]);sources=common.read_json(names[1]);model=load_official(task,device).eval().requires_grad_(False);before=frozen_hashes(model);transform=image_transform();rows=[];arrays={}
    cases=roles['cases']['TECH'][:2];guards={}
    if roles.get('status')!='METADATA_ROLES_FROZEN' or len(cases)!=2 or len({c['case_id'] for c in cases})!=2 or any(c['role']!='TECH' or c['task']!=task for c in cases):raise RuntimeError('Exactly two actual distinct frozen TECH cases required')
    for i,case in enumerate(cases):
        asset=sources['assets'][case['source_asset_sha256']]
        if asset['path'] not in guards:
            path=verified_file(asset);s=path.stat();guards[asset['path']]=(s.st_size,s.st_mtime_ns,s.st_ino)
        with RawH5(root/asset['path'],keys=['pixels']) as raw:
            pixels=np.concatenate([raw.array(case['source_episode_idx'],'pixels',j,j+1) for j in (case['start_raw_index'],case['goal_raw_index'])])
        x=torch.from_numpy(pixels)
        if pixels.shape[-1]==3:x=x.permute(0,3,1,2)
        if x.ndim!=4 or x.shape[1]!=3:raise RuntimeError('Actual source pixels do not have three channels')
        x=transform(x).to(device)
        info={'pixels':x[None,None,:1],'goal':x[None,None,1:2],'action':torch.zeros(1,1,1,10,device=device)}
        # Synthetic normalized actions only. Never use the source expert future.
        candidates=torch.zeros(1,2,5,10,device=device);candidates[0,1,0,0]=.25
        with torch.no_grad():
            direct_input={k:v.clone() for k,v in info.items()};direct=model.get_cost(direct_input,candidates.clone())
            timed=AuditedCost(model,_Timer(device));wrapped=timed.get_cost({k:v.clone() for k,v in info.items()},candidates.clone())
            goal=model.encode({'pixels':x[None,1:2]})['emb']
            rollout=model.rollout({'pixels':info['pixels'].clone()},candidates.clone())['predicted_emb']
            manual=(rollout[:,:,-1]-goal[:,None,-1]).square().sum(-1)
        exact=bool(torch.equal(direct,wrapped));goal_exact=bool(torch.equal(goal,direct_input['goal_emb']))
        tol=common.read_json('manifests/NUMERICAL_TOLERANCES.json')['comparisons']['rollout_wrapper']
        manual_equal=bool(torch.allclose(direct,manual,**tol));finite=all(bool(torch.isfinite(a).all()) for a in (direct,wrapped,manual,goal,rollout))
        rows.append({'case_id':case['case_id'],'zero_update_wrapped_cost_exact':exact,'official_goal_encoding_exact':goal_exact,'terminal_squared_sum_cost_matches':manual_equal,'manual_comparison_tolerance':tol,'all_finite':finite,'candidate_count':2,'candidate_identity':'all0; second candidate first macro raw coordinate0=0.25; no expert actions','direct_cost':direct.cpu().tolist(),'wrapped_cost':wrapped.cpu().tolist(),'manual_cost':manual.cpu().tolist()})
        for k,v in [('direct_cost',direct),('wrapped_cost',wrapped),('manual_cost',manual),('goal',goal),('rollout',rollout),('candidates',candidates)]:arrays[str(i)+'_'+k]=v.cpu().numpy()
    assert_frozen(model,before)
    for n,h in hashes.items():
        if common.sha256(root/n)!=h:raise RuntimeError('Cost check source changed')
    for name,expected in guards.items():
        s=(root/name).stat()
        if (s.st_size,s.st_mtime_ns,s.st_ino)!=expected:raise RuntimeError('Verified source changed during cost check')
    with (folder/'COST_ARRAYS.npz').open('wb') as f:np.savez(f,**arrays)
    passed=all(all(r[k] for k in ('zero_update_wrapped_cost_exact','official_goal_encoding_exact','terminal_squared_sum_cost_matches','all_finite')) for r in rows)
    result={'status':'PASS' if passed else 'FAIL','task':task,'rows':rows,'input_hashes':hashes,'optimizer_updates':0,'complete_CEM_trajectories':0,'environment_steps':0,'source_future_actions_read':False,'frozen_hashes':before}
    common.atomic_json(folder/'COST_WRAPPER_CHECK.json',result)
    common.atomic_json(folder/'SHA256.json',{'status':result['status'],'files':{p.name:{'sha256':common.sha256(p),'bytes':p.stat().st_size} for p in folder.iterdir() if p.is_file() and p.name!='SHA256.json'}})
    print(json.dumps(result),flush=True)
    if not passed:raise RuntimeError('Cost wrapper hard gate failed; preserve evidence')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=common.TASKS);p.add_argument('--device',default='cuda');p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.task,a.device,a.output_dir)
