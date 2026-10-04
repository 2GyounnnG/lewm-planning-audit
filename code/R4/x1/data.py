"""Actual HDF5 intake, R3 role assignment and frozen input-only latent caches."""
from __future__ import annotations
import hashlib, os, time, shutil
from pathlib import Path
import numpy as np
from . import core

class RawH5:
    def __init__(self, path, keys=None):
        import h5py
        try: import hdf5plugin  # Registers original compressed filters.
        except ImportError: pass
        self.path = Path(path); self.file = h5py.File(path, 'r', swmr=True, rdcc_nbytes=256*1024**2)
        f = self.file
        if not {'ep_len', 'ep_offset', 'pixels', 'action'} <= set(f): raise ValueError('Unsupported actual SWM HDF5 schema')
        self.lengths = np.asarray(f['ep_len'][:], dtype=np.int64); self.offsets = np.asarray(f['ep_offset'][:], dtype=np.int64)
        if self.lengths.ndim != 1 or self.offsets.shape != self.lengths.shape or np.any(self.lengths<=0): raise ValueError('Invalid episode offsets/lengths')
        if np.any(self.offsets<0) or np.any(self.offsets[1:]<self.offsets[:-1]+self.lengths[:-1]): raise ValueError('Overlapping episodes')
        self.column_names = list(keys or [k for k in f if k not in ('ep_len', 'ep_offset')])
        self._datasets = {}
        p, a = f['pixels'], f['action']
        if p.ndim!=4 or p.dtype!=np.uint8 or (p.shape[-1]!=3 and p.shape[1]!=3): raise ValueError('Unsupported pixels')
        if a.ndim!=2 or len(a)!=len(p) or a.dtype.kind!='f': raise ValueError('Unsupported actual raw actions')
        self.action_dim = int(a.shape[1]); total = int(max(self.offsets+self.lengths))
        if any(k not in f or len(f[k])<total for k in self.column_names): raise ValueError('Source columns missing/misaligned')
    def array(self, episode, key, start=0, end=None):
        n = int(self.lengths[episode]); end = n if end is None else int(end)
        if not 0<=start<=end<=n: raise IndexError('Episode read crossed boundary')
        if key not in self._datasets: self._datasets[key] = self.file[key]
        o = int(self.offsets[episode]); return np.asarray(self._datasets[key][o+start:o+end])
    def load_chunk(self, episodes, starts, ends):
        import torch
        out=[]
        for ep,s,e in zip(episodes,starts,ends,strict=True):
            d={}
            for k in self.column_names:
                a=self.array(int(ep),k,int(s),int(e))
                if a.dtype.kind in 'OSU': d[k]=a[0].decode() if isinstance(a[0],bytes) else a[0]
                else:
                    v=torch.from_numpy(a)
                    if a.ndim==4 and a.shape[-1]==3: v=v.permute(0,3,1,2)
                    d[k]=v
            out.append(d)
        return out
    def close(self): self._datasets.clear(); self.file.close()
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()

def legal_windows(actions, length, history=3, horizon=1, stride=5):
    actions=np.asarray(actions)
    if actions.ndim!=2 or len(actions)!=length: raise ValueError('Misaligned source actions')
    candidates=np.arange(max(0,length-(history+horizon)*stride+1),dtype=np.int64)
    bad=np.r_[0,np.cumsum(~np.isfinite(actions).all(1))]; needed=(history+horizon-1)*stride
    return candidates[(bad[candidates+needed]-bad[candidates])==0]

def processor(task):
    from sklearn.preprocessing import StandardScaler
    normal=core.read(core.ROOT/'manifests'/f'{task}_normalization.json'); a=normal['action']
    p=StandardScaler(); p.mean_=np.asarray(a['mean']);p.var_=np.asarray(a['variance']);p.scale_=np.asarray(a['scale'])
    p.n_features_in_=len(a['mean']); p.n_samples_seen_=a['finite_rows']; return p

def freeze_data(task, path, *, family_column=None, source_receipt=None):
    """No model outcomes enter metadata. Missing reset seeds remain unresolved."""
    from sklearn.preprocessing import StandardScaler
    cfg=core.CONFIG[task]; path=Path(path).resolve(); asset=core.file_record(path)
    if source_receipt is None: raise ValueError('A pinned archive/member receipt is required')
    receipt=core.read(source_receipt)
    if receipt.get('repo')!=cfg['repo'] or receipt.get('revision')!=cfg['data_revision'] or receipt.get('archive',{}).get('sha256')!=cfg['archive_sha256']:
        raise ValueError('Source receipt is not bound to the fixed official X1 dataset revision/archive SHA')
    # Parent-provided receipt must explicitly certify this exact unpacked member.
    candidates=receipt.get('files',[])
    if isinstance(candidates,dict): candidates=list(candidates.values())
    if not any(r.get('sha256')==asset['sha256'] and r.get('bytes')==asset['bytes'] for r in candidates):
        raise ValueError('HDF5 bytes are absent from verified source/member receipt')
    core.freeze(core.ROOT/'manifests'/f'{task}_contract.json',core.task_contract(task))
    with RawH5(path) as raw:
        reset_keys=cfg['reset_keys']; missing=set(reset_keys)-set(raw.file)
        if missing: raise ValueError('Actual reset source lacks: '+str(sorted(missing)))
        if family_column and family_column not in raw.file: raise ValueError('Requested source family column absent')
        config=core.read(core.verify(core.read(core.ROOT/'manifests'/f'{task}_model_assets.json')['files']['config.json']))
        if int(config['action_encoder']['input_dim']) != raw.action_dim*5: raise ValueError('Checkpoint/raw action dimensions mismatch')
        if int(config['predictor']['num_frames'])!=3: raise ValueError('Inherited R3 history requires 3')
        actions=np.asarray(raw.file['action'][:]); finite=np.isfinite(actions).all(1)
        scaler=StandardScaler().fit(actions[finite])
        norm={'task':task,'convention':'official_eval_StandardScaler_population_ddof0','image':core.r3('data').IMAGE_CONTRACT,
            'action':{'mean':scaler.mean_.tolist(),'variance':scaler.var_.tolist(),'scale':scaler.scale_.tolist(),
                'finite_rows':int(finite.sum()),'raw_dim':raw.action_dim,'macro_block':5,'excluded_nonfinite_rows':int((~finite).sum())},
            'population':'all finite actions in pinned official dataset, before refit split; official eval convention',
            'source':asset,'source_receipt':core.file_record(source_receipt)}
        core.freeze(core.ROOT/'manifests'/f'{task}_normalization.json',norm)
        rows=[]
        for i,n in enumerate(raw.lengths):
            n=int(n); o=int(raw.offsets[i]); aa=actions[o:o+n]
            reset={k:raw.array(i,k) for k in reset_keys}
            if any(x.ndim!=2 or len(x)!=n for x in reset.values()): raise ValueError('Reset column schema mismatch')
            valid=np.logical_and.reduce([np.isfinite(x).all(1) for x in reset.values()]); planning=np.arange(max(0,n-25))
            planning=planning[valid[planning]&valid[planning+25]]; offline=legal_windows(aa,n,horizon=5)+10
            seed=None
            if 'seed' in raw.file:
                ss=raw.array(i,'seed').reshape(n,-1)
                if ss.shape[1]!=1 or not np.isfinite(ss).all() or not np.all(ss==ss[0,0]) or ss[0,0]!=int(ss[0,0]) or not 0<=int(ss[0,0])<2**32:
                    raise ValueError('Source seed is not a constant uint32 within episode')
                seed=int(ss[0,0])
            family=None; evidence='FAMILY_UNKNOWN; original demonstration lineage unverified'
            if family_column:
                values=raw.array(i,family_column)
                if not np.all(values==values[0]): raise ValueError('Source family varies within episode')
                value=values[0].tolist()
                if isinstance(value,bytes): value=value.decode()
                family='source_family:'+core.digest(value); evidence='EXPLICIT_SOURCE_COLUMN:'+family_column
            identity={'source_asset_sha256':asset['sha256'],'source_episode_idx':i,'raw_offset':o,'length':n}
            rows.append({'episode_id':f'{task}:{asset["sha256"][:16]}:{i}','source_episode_idx':i,
                'source_asset_sha256':asset['sha256'],'episode_sha256':core.digest(identity),'raw_offset':o,'length':n,
                'family_id':family,'family_evidence':evidence,'source_seed':seed,'planning_starts':planning.tolist(),
                'open_loop_starts':offline.tolist(),'reset_metadata':{'column_shapes':{k:list(v.shape[1:]) for k,v in reset.items()}},
                'exposure':{'OFFICIAL_PRETRAIN_EXPOSURE':'SOURCE_IS_ORIGINAL_AUTHOR_DATA; PER_EPISODE_EXPOSURE_UNVERIFIED',
                    'R3_REFIT_EXPOSURE':'PENDING','PRIOR_USER_STUDY_EXPOSURE':'UNVERIFIED'}})
        roles=core.r3('roles'); previous=roles.TASKS
        try:
            roles.TASKS=core.TASKS
            result=roles.build_roles(task,rows,history_size=3,frameskip=5)
        finally: roles.TASKS=previous
        if len(result['cases']['EVAL'])!=100: raise RuntimeError('X1 requires exactly 100 EVAL cases; no budget reduction')
        for row in result['episodes']: row['exposure']['R3_REFIT_EXPOSURE']=row['role']=='REFIT_TRAIN'
        for cases in result['cases'].values():
            for c in cases:
                i=c['source_episode_idx'];s=c['start_raw_index'];g=c['goal_raw_index']
                c['reset_metadata'].update({'start':{k:raw.array(i,k,s,s+1)[0].tolist() for k in reset_keys},
                    'goal':{k:raw.array(i,k,g,g+1)[0].tolist() for k in reset_keys},
                    'source_start_pixel_sha256':hashlib.sha256(raw.array(i,'pixels',s,s+1).tobytes()).hexdigest(),
                    'source_goal_pixel_sha256':hashlib.sha256(raw.array(i,'pixels',g,g+1).tobytes()).hexdigest()})
        result.update(source=asset,normalization_sha256=core.sha(core.ROOT/'manifests'/f'{task}_normalization.json'),
            x1_extension='R3 metadata role builder unmodified except supported-task allowlist, restored after call')
        core.freeze(core.ROOT/'manifests'/f'{task}_data_roles.json',result)
        return {'status':result['status'],'task':task,'episodes':len(rows),'EVAL':100,
            'reset_seed_missing':len(result['reset_validation_pending_case_ids']),'raw_action_dim':raw.action_dim}

def save_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f: np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def build_cache(task,device,batch=128,shard=0,shards=1):
    import torch
    if batch not in (32,64,128): raise ValueError('Encoding microbatch must be 32/64/128')
    if not 1<=shards<=2 or not 0<=shard<shards:raise ValueError('Cache may use at most the two assigned GPUs')
    core.policy(); roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json')
    normal=core.ROOT/'manifests'/f'{task}_normalization.json'; transform=core.r3('data').image_transform();scale=processor(task)
    source=core.verify(roles['source']);model=core.load_model(task,device);m=core.r3('model');frozen=m.frozen_hashes(model)
    identity={'model':model.r3_identity,'roles_sha256':core.sha(core.ROOT/'manifests'/f'{task}_data_roles.json'),
        'normalization_sha256':core.sha(normal),'source':roles['source'],'batch':batch,'shards':shards,'precision':'FP32_NO_AMP_NO_TF32',
        'implementation_sha256':{'x1_data':core.sha(__file__),'x1_core':core.sha(core.Path(core.__file__)),
            'r3_model':core.sha(core.R3_ROOT/'r3/model.py'),'r3_image_transform':core.sha(core.R3_ROOT/'r3/data.py')}}
    complete=core.ROOT/'manifests'/f'{task}_cache.json'
    if complete.exists():
        old=core.read(complete)
        if old['identity']!=identity or old['status']!='FROZEN_OBSERVED_CACHE_COMPLETE':raise RuntimeError('Completed cache identity changed')
        for record in old['episodes'].values():core.verify(record)
        return {'task':task,'status':'REUSED_VERIFIED_COMPLETE_CACHE','episodes':len(old['episodes'])}
    record={}; moments=None; squares=None; count=0; started=time.monotonic()
    with RawH5(source,keys=['pixels','action']) as raw:
        episodes=[r for r in roles['episodes'] if r['role'] in ('REFIT_TRAIN','MONITOR','TECH','EVAL')]
        # TECH encodes first and is independently complete before all-training cache.
        episodes.sort(key=lambda r:(r['role']!='TECH',r['episode_id']))
        episodes=episodes[shard::shards]
        for row in episodes:
            eid=row['episode_id'];name=hashlib.sha256(eid.encode()).hexdigest();path=core.ROOT/'cache'/(name+'.npz');side=path.with_suffix('.json')
            bind={'inputs':identity,'episode':row}
            if side.exists():
                old=core.read(side)
                if old['identity']!=bind: raise RuntimeError('Cache identity changed')
                core.verify(old['file']);r=old['file']
                with np.load(path,allow_pickle=False) as f:z=f['z'].copy()
            else:
                if path.exists(): raise RuntimeError('Orphan cache requires audit')
                i=row['source_episode_idx'];n=row['length'];aa=raw.array(i,'action');zz=[]
                rawhash=hashlib.sha256(aa.tobytes())
                with torch.inference_mode():
                    for s in range(0,n,batch):
                        pixels=raw.array(i,'pixels',s,min(n,s+batch));rawhash.update(pixels.tobytes());x=torch.from_numpy(pixels)
                        if pixels.shape[-1]==3:x=x.permute(0,3,1,2)
                        zz.append(model.encode({'pixels':transform(x).to(device)[:,None]})['emb'][:,0].cpu().numpy().astype(np.float32))
                z=np.concatenate(zz)
                if z.shape!=(n,model.r3_contract['latent_dim']) or not np.isfinite(z).all():raise RuntimeError('Frozen latent schema differs')
                if shutil.disk_usage(core.ROOT).free < (30<<30)+z.nbytes+aa.nbytes+(2<<20):raise RuntimeError('Cache must preserve 30GiB shared free space')
                save_npz(path,z=z,actions=scale.transform(aa).astype(np.float32),raw_indices=np.arange(n,dtype=np.int64),
                    stride=np.array(5,dtype=np.int64),legal_starts=legal_windows(aa,n))
                r={**core.file_record(path),'role':row['role'],'length':n,'raw_pixel_action_sha256':rawhash.hexdigest()}
                core.atomic(side,{'identity':bind,'file':r,'state_labels_read':0})
            record[eid]=r
            if row['role']=='REFIT_TRAIN':
                d=z.astype(np.float64)
                if moments is None:moments=np.zeros(d.shape[1]);squares=moments.copy()
                moments+=d.sum(0);squares+=np.square(d).sum(0);count+=len(d)
            if len(record)%100==0:m.assert_frozen(model,frozen)
            core.atomic(core.ROOT/'state'/f'{task}_cache_progress_{shard}.json',{'complete':len(record),'total':len(episodes),'shard':shard,'shards':shards,'seconds':time.monotonic()-started})
    m.assert_frozen(model,frozen);variance=float(np.mean(squares/count-(moments/count)**2))
    if not np.isfinite(variance) or variance<=0:raise RuntimeError('Invalid TRAIN latent variance')
    result={'status':'FROZEN_OBSERVED_CACHE_COMPLETE' if shards==1 else 'FROZEN_OBSERVED_CACHE_SHARD_COMPLETE','identity':identity,'episodes':record,'train_scalar_variance':variance,
        'train_variance_frames':count,'frozen':frozen,'optimizer_updates':0,'state_labels_read':0,'seconds':time.monotonic()-started}
    if shards>1:result.update(shard=shard,train_moments=moments.tolist(),train_squared_moments=squares.tolist())
    destination=core.ROOT/'manifests'/(f'{task}_cache.json' if shards==1 else f'{task}_cache_part_{shard}.json')
    # Resumed completed shards retain their original wall-time receipt.
    if destination.exists():
        prior=core.read(destination)
        if {k:v for k,v in prior.items() if k!='seconds'}!={k:v for k,v in result.items() if k!='seconds'}:raise RuntimeError('Completed cache shard identity differs')
        result=prior
    else:core.freeze(destination,result)
    return {'task':task,'status':result['status'],'episodes':len(record),'seconds':result['seconds']}

def merge_cache(task,shards=2):
    if shards!=2:raise ValueError('Two assigned cache shards required')
    parts=[core.read(core.ROOT/'manifests'/f'{task}_cache_part_{i}.json') for i in range(shards)]
    identity=parts[0]['identity'];episodes={};moments=None;squares=None;count=0
    for i,p in enumerate(parts):
        if p['status']!='FROZEN_OBSERVED_CACHE_SHARD_COMPLETE' or p['shard']!=i or p['identity']!=identity or p['frozen']!=parts[0]['frozen']:
            raise RuntimeError('Cache shard identity/frozen base differs')
        if set(episodes)&set(p['episodes']):raise RuntimeError('Duplicate cache episode across shards')
        for rec in p['episodes'].values():core.verify(rec)
        episodes.update(p['episodes']);mm=np.asarray(p['train_moments']);ss=np.asarray(p['train_squared_moments'])
        moments=mm if moments is None else moments+mm;squares=ss if squares is None else squares+ss;count+=p['train_variance_frames']
    roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json')
    expected={r['episode_id'] for r in roles['episodes'] if r['role'] in ('REFIT_TRAIN','MONITOR','TECH','EVAL')}
    if set(episodes)!=expected:raise RuntimeError('Merged cache does not cover exact selected roles')
    variance=float(np.mean(squares/count-(moments/count)**2))
    result={'status':'FROZEN_OBSERVED_CACHE_COMPLETE','identity':identity,'episodes':episodes,'train_scalar_variance':variance,
        'train_variance_frames':count,'frozen':parts[0]['frozen'],'optimizer_updates':0,'state_labels_read':0,
        'seconds':max(p['seconds'] for p in parts),'shard_receipts':[core.file_record(core.ROOT/'manifests'/f'{task}_cache_part_{i}.json') for i in range(shards)]}
    core.freeze(core.ROOT/'manifests'/f'{task}_cache.json',result)
    return {'status':result['status'],'task':task,'episodes':len(episodes),'frames':sum(r['length'] for r in episodes.values()),'seconds':result['seconds']}

def load_cache(task,role):
    cm=core.read(core.ROOT/'manifests'/f'{task}_cache.json');roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json')
    if cm['identity']['roles_sha256']!=core.sha(core.ROOT/'manifests'/f'{task}_data_roles.json'):raise RuntimeError('Cache role lock changed')
    if cm['identity']['normalization_sha256']!=core.sha(core.ROOT/'manifests'/f'{task}_normalization.json'):raise RuntimeError('Cache normalization changed')
    expected={r['episode_id'] for r in roles['episodes'] if r['role']==role};selected={e:r for e,r in cm['episodes'].items() if r['role']==role}
    if set(selected)!=expected:raise RuntimeError('Incomplete role cache')
    cache={}
    for eid,r in selected.items():
        with np.load(core.verify(r),allow_pickle=False) as f:
            if set(f.files)!={'z','actions','raw_indices','stride','legal_starts'}:raise ValueError('Input cache has unregistered fields')
            cache[eid]={k:f[k].copy() for k in f.files};cache[eid]['stride']=int(cache[eid]['stride'])
    return cache
