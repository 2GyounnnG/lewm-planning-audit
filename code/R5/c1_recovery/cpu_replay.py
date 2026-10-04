"""CPU inference on fixed first case, all four already fixed Cube models."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--cube-root',type=Path,required=True);p.add_argument('--r3-root',type=Path,required=True);a=p.parse_args()
    import torch
    from x1 import core
    from x1.evaluate import load_arm
    core.ROOT=a.cube_root.resolve();core.R3_ROOT=a.r3_root.resolve();core.policy()
    old=core.verify
    def verify(r):
        q=Path(r['path'])
        if str(q).startswith('/workspace/shared_data/x1_assets/') and q.name in ('weights.pt','config.json'):q=core.ROOT/'official'/q.name
        if q.is_relative_to('/workspace/x1_cube'):q=core.ROOT/q.relative_to('/workspace/x1_cube')
        if q.is_relative_to('/workspace/shared_data/r3'):q=core.R3_ROOT/q.relative_to('/workspace/shared_data/r3')
        return old(dict(r,path=str(q)))
    core.verify=verify
    src=json.loads((a.reference/'REFERENCE_MANIFEST.json').read_text());arrays_path=a.reference/'FIRST_CASE_REFERENCE.npz'
    if core.sha(arrays_path)!=src['reference']['sha256']:raise RuntimeError('Reference SHA differs')
    rows=[]
    with np.load(arrays_path) as f:
        for arm in ('H0','REFIT_103201','REFIT_103202','REFIT_103203'):
            model=load_arm('cube',arm,'cpu');before={k:core.r3('model').tensor_sha256(v) for k,v in model.state_dict().items()}
            for key in f.files:
                if not key.startswith(arm+':') or not key.endswith(':history'):continue
                prefix=key[:-len(':history')];history=f[key];actions=f[prefix+':actions'];reference=f[prefix+':prediction'];goal=f[prefix+':goal']
                with torch.inference_mode():actual=core.r3('model').cached_rollout(model,torch.from_numpy(history),torch.from_numpy(actions),5).numpy()
                cost=np.sum((actual[:,-1]-goal)**2,axis=-1);expected=f[prefix+':cost']
                rows.append({'arm':arm,'replan':int(prefix.split(':')[1]),'prediction_max_abs':float(np.max(np.abs(actual-reference))),'prediction_pass':bool(np.allclose(actual,reference,atol=1e-5,rtol=1e-5)),'cost_max_abs':float(np.max(np.abs(cost-expected))),'cost_pass':bool(np.allclose(cost,expected,atol=1e-5,rtol=1e-5))})
            after={k:core.r3('model').tensor_sha256(v) for k,v in model.state_dict().items()}
            if before!=after:raise RuntimeError('Fixed model changed')
    passed=bool(rows) and all(r['prediction_pass'] and r['cost_pass'] for r in rows)
    out={'status':'PASS' if passed else 'FAIL_FROZEN_CPU_TOLERANCE','case_id':src['case']['case_id'],'scope':src['source_scope'],'rows':rows,'fixed_model_states_unchanged':True,'new_simulation':False,'atol':1e-5,'rtol':1e-5,'reference_manifest_sha256':core.sha(a.reference/'REFERENCE_MANIFEST.json'),'source_code_sha256':core.sha(__file__)}
    (a.reference/'CPU_RECOVERY_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
