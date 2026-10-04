"""One official observed-latent cache, all explicit roles, no optimizer or risk selection."""
from pathlib import Path
import argparse, contextlib, fcntl, hashlib, json, os, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r3 import common
from r3.data import RawH5,image_transform,action_processor,admissible_windows,verified_file,IMAGE_CONTRACT
from r3.model import load_official,frozen_hashes,assert_frozen
import numpy as np
import torch
ROOT=common.ROOT

def save_npz(path,values):
    q=path.with_name(path.name+'.cache.tmp')
    with q.open('wb') as f:np.savez(f,**values);f.flush();os.fsync(f.fileno())
    os.replace(q,path)

def run(task,device,batch_size):
    if batch_size not in (32,64,128):raise ValueError('Unregistered encoder-only microbatch')
    common.require_authorization({},technical=True);common.ensure_space();common.fp32_policy()
    role_file=ROOT/f'manifests/{task}_data_roles.json';roles=common.read_json(role_file)
    if roles.get('status') != 'METADATA_ROLES_FROZEN':raise RuntimeError('Metadata-only case split must already be frozen')
    source_map=common.read_json(f'manifests/{task}_source_map.json')
    normal=common.read_json(f'manifests/{task}_normalization.json')
    processor=action_processor(task);transform=image_transform();model=load_official(task,device)
    model.eval().requires_grad_(False);before=frozen_hashes(model)
    tensors=list(model.parameters())+list(model.buffers());versions=tuple(t._version for t in tensors)
    inputs={str(p.relative_to(ROOT)):common.sha256(p) for p in [role_file,ROOT/f'manifests/{task}_source_map.json',ROOT/f'manifests/{task}_normalization.json',ROOT/f'official/{task}/weights.pt',ROOT/f'official/{task}/config.json',ROOT/'r3/data.py',ROOT/'r3/model.py',ROOT/'r3/common.py',ROOT/'scripts/build_cache.py',ROOT/'state/ENVIRONMENT_READY.json',ROOT/'state/lewm_source_manifest.json',ROOT/'state/spt_source_manifest.json']}
    episodes=[x for x in roles['episodes'] if x['role'] in ('REFIT_TRAIN','MONITOR','TECH','EVAL')]
    identity={'task':task,'input_hashes':inputs,'roles_manifest_sha256':inputs[str(role_file.relative_to(ROOT))],'image':IMAGE_CONTRACT,'encoding_batch_size':batch_size,'fp32_policy':common.fp32_policy(),'encoder_only':True,'optimizer_updates':0,'state_labels_read':0,'freeze_verification':'all parameter/buffer in-place version counters plus module eval/parameter requires_grad every episode; full frozen byte SHA before, every100 new episodes, and after'}
    complete=ROOT/f'manifests/{task}_cache.json'
    if complete.exists():
        saved=common.read_json(complete)
        if any(saved.get(k)!=v for k,v in identity.items()) or saved.get('status')!='FROZEN_OBSERVED_CACHE_COMPLETE':raise RuntimeError('Existing complete cache has different identity')
        if sorted(saved['episodes'])!=sorted(x['episode_id'] for x in episodes):raise RuntimeError('Existing complete cache episode roster differs')
        for record in saved['episodes'].values():verified_file(record)
        print(json.dumps({'status':'REUSED_VERIFIED_COMPLETE_CACHE_UNCHANGED','task':task,'episodes':len(saved['episodes'])}),flush=True)
        return saved
    needed=sum(x['length']*(192*4+2*4+8+8) for x in episodes)
    common.ensure_space(needed)
    directory=ROOT/'data/cache'/task;directory.mkdir(parents=True,exist_ok=True)
    readers={};records={};started=time.perf_counter();s1=np.zeros(192,dtype=np.float64);s2=s1.copy();count=0;completed_frames=0
    try:
        for ep in episodes:
            eid=ep['episode_id'];name=hashlib.sha256(eid.encode()).hexdigest()
            dest=directory/(name+'.npz');side=directory/(name+'.json')
            eidentity={**identity,'episode':ep}
            if side.exists():
                old=common.read_json(side)
                if old['identity']!=eidentity:raise RuntimeError('Existing episode cache has different frozen inputs')
                verified_file(old['file']);record=old['file']
                with np.load(dest,allow_pickle=False) as f:z=f['z']
            else:
                if dest.exists():raise RuntimeError('Unregistered existing cache; retain for explicit audit')
                asset=ep['source_asset_sha256']
                if asset not in readers:
                    source=source_map['assets'][asset];verified_file(source)
                    readers[asset]=RawH5(ROOT/source['path'],keys=['pixels','action'])
                reader=readers[asset];idx=int(ep['source_episode_idx']);n=int(reader.lengths[idx])
                if n!=ep['length']:raise RuntimeError('Frozen episode length differs from source')
                raw_actions=reader.array(idx,'action');actions=processor.transform(raw_actions).astype(np.float32)
                legal=admissible_windows(raw_actions,n)
                # Per-episode raw pixel/action digest discloses exact cache input.
                raw_digest=hashlib.sha256();raw_digest.update(str(raw_actions.dtype).encode());raw_digest.update(raw_actions.tobytes(order='C'))
                zz=[]
                with torch.inference_mode():
                    for start in range(0,n,batch_size):
                        pixels=reader.array(idx,'pixels',start,min(n,start+batch_size))
                        raw_digest.update(pixels.tobytes(order='C'))
                        x=torch.from_numpy(pixels)
                        if pixels.shape[-1]==3:x=x.permute(0,3,1,2)
                        x=transform(x).to(device)
                        zz.append(model.encode({'pixels':x[:,None]})['emb'][:,0].cpu().numpy().astype(np.float32))
                z=np.concatenate(zz)
                if z.shape!=(n,192) or not np.isfinite(z).all():raise RuntimeError('Actual frozen encoder output violates contract')
                if tuple(t._version for t in tensors)!=versions or any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):raise RuntimeError('Frozen encoding tensor/mode changed')
                if (len(records)+1)%100==0:assert_frozen(model,before)
                common.ensure_space(z.nbytes+actions.nbytes+4*1024**2)
                save_npz(dest,{'z':z,'actions':actions,'raw_indices':np.arange(n,dtype=np.int64),'stride':np.array(5,dtype=np.int64),'legal_starts':legal})
                record={'path':str(dest.relative_to(ROOT)),'bytes':dest.stat().st_size,'sha256':common.sha256(dest),'role':ep['role'],'length':n,'legal_windows':len(legal),'raw_pixel_action_sha256':raw_digest.hexdigest()}
                common.atomic_json(side,{'identity':eidentity,'file':record,'source_state_or_reward_read':False})
            records[eid]=record
            completed_frames+=record['length']
            if ep['role']=='REFIT_TRAIN':
                d=z.astype(np.float64);s1+=d.sum(axis=0);s2+=np.square(d).sum(axis=0);count+=len(d)
            common.atomic_json(f'state/{task}_cache_progress.json',{'completed':len(records),'total':len(episodes),'encoded_frames':completed_frames,'seconds':time.perf_counter()-started,'optimizer_updates':0})
        assert_frozen(model,before)
        for p,d in inputs.items():
            if common.sha256(ROOT/p)!=d:raise RuntimeError('Cache input changed during encoding')
        variance=float(np.mean(s2/count-np.square(s1/count)))
        if not np.isfinite(variance) or variance<=0:raise RuntimeError('Invalid TRAIN scalar variance')
        manifest={**identity,'status':'FROZEN_OBSERVED_CACHE_COMPLETE','episodes':records,'train_scalar_variance':variance,'train_variance_formula':'mean_D(E_REFIT_TRAIN_all_raw_frames[z_d^2]-E[z_d]^2), population ddof0, float64 accumulator','train_variance_frames':count,'frozen_hashes':before,'seconds':time.perf_counter()-started}
        common.atomic_json(f'manifests/{task}_cache.json',manifest)
        print(json.dumps({'status':manifest['status'],'task':task,'episodes':len(records),'frames':sum(x['length'] for x in records.values()),'seconds':manifest['seconds']}),flush=True)
    finally:
        for r in readers.values():r.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=common.TASKS);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
    lock=ROOT/f'state/{a.task}_cache.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        paths=[(lock,fcntl.LOCK_EX),(ROOT/'state/execution_phase.lock',fcntl.LOCK_SH)]
        if a.device.startswith('cuda'):
            binding=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if binding not in ('0','1'):raise ValueError('Encoder cache worker must bind one authorized physical GPU')
            paths.append((ROOT/'state/worker_locks'/f'gpu_{binding}_slot_0.lock',fcntl.LOCK_EX))
        for path,kind in paths:
            path.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(path.open('a+'));fcntl.flock(f,kind|fcntl.LOCK_NB)
        run(a.task,a.device,a.batch_size)
