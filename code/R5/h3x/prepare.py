"""Freeze input identities and infer only original MONITOR observation frames."""
import os,sys,time
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,torch
from . import common
from r3.data import RawH5,image_transform,action_processor
from r3.model import load_official,frozen_hashes,assert_frozen

def file(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':common.sha256(p)}

def main():
    common.ROOT.mkdir(parents=True,exist_ok=True);out=common.ROOT/'inputs';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);common.fp32_policy()
    rolespath=common.R3_ROOT/'manifests/reacher_data_roles.json';roles=common.read_json(rolespath)
    source=common.read_json(common.R3_ROOT/'manifests/reacher_source_map.json');asset=next(iter(source['assets'].values()));h5=common.R3_ROOT/asset['path'];assert h5.stat().st_size==asset['bytes']
    previous=common.read_json('/workspace/r5/H1b/INPUT_AUDIT.json');assert previous['tasks']['reacher']['source_h5']['sha256']==asset['sha256']
    cachepath=Path('/workspace/r5/H1b/cache/reacher/manifest.json');cache=common.read_json(cachepath);assert cache['roles_sha256']==common.sha256(rolespath)
    m=load_official('reacher','cuda');frozen=frozen_hashes(m);transform=image_transform();proc=action_processor('reacher')
    # Cache creation provenance independently binds the exact official latent coordinate.
    for rec in cache['parts']:
        part=common.read_json(rec['path']);assert part['identity']['model']==m.r3_identity and part['optimizer_updates']==0
    windows=roles['monitor_windows'];assert len(windows)==256
    byid={e['episode_id']:e for e in roles['episodes']};zs=[];actions=[];t=time.monotonic()
    with RawH5(h5,keys=['pixels','action']) as reader:
        for eid,start in windows:
            e=byid[eid];assert e['role']=='MONITOR';idx=e['source_episode_idx'];raw=np.arange(start,start+40,5)
            px=reader.array(idx,'pixels',int(raw[0]),int(raw[-1])+1)[::5]
            x=torch.from_numpy(px).permute(0,3,1,2)
            with torch.inference_mode():z=m.encode({'pixels':transform(x).to('cuda')[:,None]})['emb'][:,0].cpu().numpy()
            a=proc.transform(reader.array(idx,'action',start,start+35)).astype(np.float32).reshape(7,10)
            assert z.shape==(8,192) and np.isfinite(a).all();zs.append(z);actions.append(a)
    after=assert_frozen(m,frozen)
    p=out/'MONITOR_WINDOWS.npz';np.savez_compressed(p,z=np.asarray(zs,dtype=np.float32),actions=np.asarray(actions,dtype=np.float32),episode_ids=np.array([e for e,s in windows]),starts=np.array([s for e,s in windows]))
    common.atomic_json(out/'CACHE_AUDIT.json',{'label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','h1b_cache_manifest':file(cachepath),'h1b_recovery_seal':file('/workspace/r5/H1b/H1B_RECOVERY_SEAL.json'),'source_h5':asset,'source_h5_prior_full_sha_audit':file('/workspace/r5/H1b/INPUT_AUDIT.json'),'roles':file(rolespath),'monitor_windows':file(p),'monitor_window_cursor_sha256':__import__('hashlib').sha256(__import__('json').dumps(windows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'monitor_scope':'EXACT_256_ORIGINAL_R3_MONITOR_WINDOWS; no sampling, no selection','model_identity':m.r3_identity,'frozen_before':frozen,'frozen_after':after,'optimizer_updates':0,'state_labels_read':0,'seconds':time.monotonic()-t,'pid':os.getpid()})
    print('PREPARE_COMPLETE',time.monotonic()-t,flush=True)
if __name__=='__main__':main()
