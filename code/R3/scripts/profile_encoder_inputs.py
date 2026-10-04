"""Finite no-update timing of the real frozen TECH input/encoder pipeline."""
from pathlib import Path
import json,os,sys,time,fcntl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from r3 import common
from r3.data import RawH5,image_transform
from r3.model import load_official,frozen_hashes,assert_frozen

def run():
    common.require_authorization({},technical=True);common.fp32_policy()
    root=common.ROOT;output=root/'state/PUSHT_ENCODER_TIMING.json'
    if output.exists():raise RuntimeError('Finite timing already exists; no repeated scan')
    roles=common.read_json('manifests/pusht_data_roles.json');source=common.read_json('manifests/pusht_source_map.json')
    selected=roles['cases']['TECH'][:2];model=load_official('pusht','cuda');model.eval().requires_grad_(False)
    before=frozen_hashes(model);transform=image_transform();rows=[];frames=0
    for case in selected:
        asset=source['assets'][case['source_asset_sha256']]
        with RawH5(root/asset['path'],keys=['pixels','action']) as raw:
            ds=raw.file['pixels'];filters=[ds.id.get_create_plist().get_filter(i) for i in range(ds.id.get_create_plist().get_nfilters())]
            for repeat in (0,1):
                n=min(128,case['length']);t=time.perf_counter();pixels=raw.array(case['source_episode_idx'],'pixels',0,n);read=time.perf_counter()-t
                t=time.perf_counter();x=transform(torch.from_numpy(pixels).permute(0,3,1,2));cpu=time.perf_counter()-t
                torch.cuda.synchronize();t=time.perf_counter();x=x.to('cuda');torch.cuda.synchronize();transfer=time.perf_counter()-t
                t=time.perf_counter()
                with torch.inference_mode():z=model.encode({'pixels':x[:,None]})['emb']
                torch.cuda.synchronize();encode=time.perf_counter()-t
                t=time.perf_counter();values=z.cpu().numpy();back=time.perf_counter()-t
                if not np.isfinite(values).all():raise RuntimeError('Nonfinite frozen output')
                rows.append({'case_id':case['case_id'],'source_episode_idx':case['source_episode_idx'],'repeat':repeat,'frames':n,'read_seconds':read,'CPU_transform_seconds':cpu,'host_to_gpu_seconds':transfer,'encoder_seconds':encode,'gpu_to_host_seconds':back,'pixels_dtype':str(pixels.dtype),'pixels_shape':list(pixels.shape),'filter_metadata':repr(filters),'output_sha256':__import__('hashlib').sha256(values.tobytes()).hexdigest()});frames+=n
    assert_frozen(model,before)
    d={'status':'TECH_ENCODER_TIMING_COMPLETE','rows':rows,'encoded_frames':frames,'optimizer_updates':0,'training_model_outputs_scored':False,'concurrent_PushT_cache_GPU0':True,'GPU_used':os.environ.get('CUDA_VISIBLE_DEVICES'),'scope':'First2 metadata-fixed TECH episodes; two identical input batches each, no planner or predictor; setup/cold first call kept. Does not change active cache implementation.'}
    common.atomic_json(output,d);print(json.dumps(d),flush=True)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('This finite diagnostic is assigned only idle GPU1')
    with (common.ROOT/'state/worker_locks/gpu_1_slot_0.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
