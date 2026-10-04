"""Independent local hash/numeric recovery of the derived C1 bundle; no simulation."""
import argparse, hashlib, json, tarfile, time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--recovery',type=Path,required=True);a=p.parse_args();root=a.recovery
    archive=root/'C1_minimal_v1.tar.gz';receipt=json.loads((root/'RECOVERY_ARCHIVE.json').read_text())
    if archive.stat().st_size!=receipt['archive']['bytes'] or sha(archive)!=receipt['archive']['sha256']:raise RuntimeError('Archive hash mismatch')
    out=root/'extracted'
    if not (out/'bundle/RECOVERY_MANIFEST.json').exists():
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if m.issym() or m.islnk() or not (out/m.name).resolve().is_relative_to(out.resolve()):raise RuntimeError('Unsafe archive member')
            tf.extractall(out,filter='data')
    bundle=out/'bundle';manifest=json.loads((bundle/'RECOVERY_MANIFEST.json').read_text());checked=0
    if sha(bundle/'RECOVERY_MANIFEST.json')!=receipt['manifest']['sha256']:raise RuntimeError('Manifest hash mismatch')
    for key,r in manifest['files'].items():
        f=bundle/key
        if f.stat().st_size!=r['bytes'] or sha(f)!=r['sha256']:raise RuntimeError('Member mismatch '+key)
        checked+=1
    c=json.loads((bundle/'C1_CONTRACT.json').read_text());cases=c['cases']['EVAL'];metrics=[];arrays=0;omitted=0;trajectories=0
    for deriv in sorted(bundle.glob('closed_loop/*/*/*/*/DERIVATION.json')):
        d=json.loads(deriv.read_text());folder=deriv.parent;r=json.loads((folder/'result.json').read_text());trajectories+=1
        for original,mapping in d['source_arrays'].items():
            with np.load(folder/original.replace('.npz','_compact.npz')) as f:
                for key,spec in mapping['fields'].items():
                    if spec['representation']=='RECONSTRUCT_RGB_WITH_FROZEN_RESET_ACTIONS_AND_SOURCE_ENDPOINTS':omitted+=1;continue
                    actual=f[spec['reference_key']] if spec['representation']=='IDENTICAL_INITIAL_MODEL_ARRAY' else f[key]
                    if str(actual.dtype)!=spec['dtype'] or list(actual.shape)!=spec['shape'] or ahash(actual)!=spec['array_sha256']:raise RuntimeError('Derived array mismatch '+str(folder)+'/'+key)
                    arrays+=1
        with np.load(folder/'trajectory_compact.npz') as t,np.load(folder/'plans_compact.npz') as plans,np.load(folder/'states_compact.npz') as states:
            n=len(t['raw_actions'])
            if n!=r['executed_raw_steps'] or len(t['raw_latent'])!=n+1:raise RuntimeError('Executed index mismatch')
            if int(t['step_success'].any())!=r['success'] or bool(t['step_success'][:-1].any() or t['step_truncated'][:-1].any()):raise RuntimeError('Success/terminal mismatch')
            errors=np.linalg.norm(t['physical_state']-t['goal_state'],axis=-1)
            if not np.allclose(errors,t['goal_error'],atol=1e-14,rtol=1e-14) or not np.isclose(errors[-1],r['final_goal_error'],atol=1e-14,rtol=1e-14):raise RuntimeError('Goal errors changed')
            for item in r['replans']:
                i=item['replan_index'];expected=int.from_bytes(hashlib.sha256(f'R3_CEM_CASE_20261002/cube/{r["case_id"]}/{i}'.encode()).digest()[:8],'big')
                if item['seed_uint64']!=expected or not np.array_equal(plans[f'{i}:returned_plan'],np.asarray(item['returned_plan'])):raise RuntimeError('Plan seed/output mismatch')
            if r['phase']=='EVAL':metrics.append({'case_id':r['case_id'],'arm':r['arm'],'success':r['success'],'steps':n})
    if len(metrics)!=400 or trajectories!=404:raise RuntimeError('Coverage mismatch')
    symmetry_fields=0
    for case in cases:
        base=bundle/'closed_loop/EVAL/R3_ORIGINAL';cid=case['case_id']
        with np.load(base/'H0'/cid/'states_compact.npz') as f:initial={k:f[k].copy() for k in f.files if k.startswith('000:')}
        h0=json.loads((base/'H0'/cid/'DERIVATION.json').read_text())['source_arrays']['states.npz']['fields']['000:render']['array_sha256']
        for arm in ('REFIT_103201','REFIT_103202','REFIT_103203'):
            with np.load(base/arm/cid/'states_compact.npz') as f:
                for k,v in initial.items():
                    other=f[k]
                    if other.dtype!=v.dtype or other.shape!=v.shape or other.tobytes()!=v.tobytes():raise RuntimeError('Formal arm reset asymmetry '+cid+'/'+arm+'/'+k)
                    symmetry_fields+=1
            info=json.loads((base/arm/cid/'DERIVATION.json').read_text())
            if info['source_arrays']['states.npz']['fields']['000:render']['array_sha256']!=h0:raise RuntimeError('Formal arm initial render hash asymmetry')
            symmetry_fields+=1
    result={'status':'PASS_NUMERIC_DERIVED_RECOVERY','archive_sha256':sha(archive),'manifest_sha256':sha(bundle/'RECOVERY_MANIFEST.json'),'members_verified':checked,'trajectories':trajectories,'formal':len(metrics),'array_identities_verified':arrays,'remote_only_pixel_array_records':omitted,'original_RGB_and_planner_images_local':False,'pixel_scope':'Original source arrays remain remote with exact SHA, dtype/shape and deterministic C0/action/transform reconstruction map; no local MuJoCo or pixel regeneration claimed.','actions_plans_rng_latents_goals_and_complete_nonpixel_dynamic_state_local':True,'success_goal_errors_recomputed':True,'all_100_formal_cases_four_arm_initial_state_and_render_hash_bitwise_equal':True,'formal_symmetry_field_comparisons':symmetry_fields,'code_sha256':sha(__file__)}
    write(root/'LOCAL_RECOVERY_CHECK.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
