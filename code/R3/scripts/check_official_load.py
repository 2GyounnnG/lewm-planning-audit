"""Strict public tensor-state loading; this is not a real-data/environment gate."""
import json, os, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from r3.common import atomic_json,fp32_policy
from r3.model import load_official,whitelist_manifest,frozen_hashes,cached_predict

torch.set_num_threads(2);fp32_policy();out={}
for task in ('pusht','reacher'):
    model=load_official(task,'cpu');before=frozen_hashes(model)
    with torch.no_grad():
        z=torch.zeros(2,3,192);a=torch.zeros(2,3,10)
        official=model.predict(z,model.action_encoder(a));wrapped=cached_predict(model,z,a)
    if not torch.equal(official,wrapped):raise RuntimeError('Wrapper changed direct prediction')
    if frozen_hashes(model)!=before:raise RuntimeError('Inference changed frozen state')
    out[task]={'strict_load':model.r3_strict_load,'identity':model.r3_identity,'contract':model.r3_contract,
               'whitelist':whitelist_manifest(model),'frozen_hashes':before,
               'synthetic_wrapper_max_abs_error':float((official-wrapped).abs().max()),
               'real_data_equivalence_still_required':True}
    del model
atomic_json(ROOT/'state/OFFICIAL_STRICT_LOAD_CPU.json',{'status':'STRICT_LOAD_AND_SYNTHETIC_WRAPPER_PASS',
    'tasks':out,'optimizer_updates':0,'environment_trajectories':0,'GPU_calls':0})
print(json.dumps({t:{'strict':r['strict_load'],'trainable_parameters':r['whitelist']['trainable_parameters'],
                    'wrapper_max_abs_error':r['synthetic_wrapper_max_abs_error']} for t,r in out.items()}),flush=True)
