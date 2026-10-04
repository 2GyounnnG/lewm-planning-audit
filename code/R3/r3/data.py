"""Explicit pinned HDF5 inputs and observed-latent cache; no task-state training inputs."""
from __future__ import annotations
from pathlib import Path
import json, os
import numpy as np
from . import common

ROOT=common.ROOT
ROLES=('REFIT_TRAIN','MONITOR','TECH','EVAL')
IMAGE_CONTRACT={'pipeline':['ToImage','ToDtype(float32,scale=True)','Normalize(ImageNet)','Resize(bilinear,antialias=True)'],
                'mean':[.485,.456,.406],'std':[.229,.224,.225],'size':[224,224],
                'source':'fixed lewm/eval.py img_transform; actual torchvision version in environment receipt'}

def image_transform():
    from torchvision.transforms import v2
    return v2.Compose([v2.ToImage(),v2.ToDtype(__import__('torch').float32,scale=True),
                       v2.Normalize(mean=IMAGE_CONTRACT['mean'],std=IMAGE_CONTRACT['std']),
                       v2.Resize((224,224),interpolation=v2.InterpolationMode.BILINEAR,antialias=True)])

def checked_local(relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Relative R3 source/cache path required')
    result=ROOT/p
    if not result.resolve().is_relative_to(ROOT):raise ValueError('Data path escapes R3')
    return result

def verified_file(record):
    p=checked_local(record['path'])
    if p.stat().st_size!=record['bytes'] or common.sha256(p)!=record['sha256']:
        raise RuntimeError('Data/cache identity changed: '+str(p))
    return p

class RawH5:
    """Only the observed actual flat SWM ep_len/ep_offset format is supported.

    A different schema is an explicit intake failure, not a reconstructed dataset.
    Loading a chunk follows the pinned HDF5Dataset frameskip=1 tensor convention.
    """
    def __init__(self,path,keys=None):
        import h5py, hdf5plugin
        self.path=Path(path);self.file=h5py.File(path,'r',swmr=True,rdcc_nbytes=256*1024**2)
        self._datasets={}
        f=self.file
        if not {'ep_len','ep_offset','pixels','action'}.issubset(f):
            self.close();raise ValueError('Actual source lacks required flat SWM HDF5 keys')
        self.lengths=np.asarray(f['ep_len'][:],dtype=np.int64);self.offsets=np.asarray(f['ep_offset'][:],dtype=np.int64)
        if self.lengths.ndim!=1 or self.offsets.shape!=self.lengths.shape or np.any(self.lengths<=0):
            self.close();raise ValueError('Invalid actual episode lengths/offsets')
        if np.any(self.offsets<0) or np.any(self.offsets[1:]<self.offsets[:-1]+self.lengths[:-1]):
            self.close();raise ValueError('Overlapping source episodes')
        total=int(max(self.offsets+self.lengths));self.column_names=list(keys or [k for k in f if k not in ('ep_len','ep_offset')])
        for k in self.column_names:
            if k not in f or len(f[k])<total:raise ValueError('Source column length differs: '+k)
        pixels=f['pixels'];actions=f['action']
        if pixels.ndim!=4 or pixels.dtype!=np.uint8 or (pixels.shape[-1]!=3 and pixels.shape[1]!=3):raise ValueError('Unsupported actual pixel schema')
        if actions.shape!=(len(pixels),2) or actions.dtype.kind!='f':raise ValueError('Actual raw action schema differs from official 5x2 input')
    def array(self,episode_idx,key,start=0,end=None):
        n=int(self.lengths[episode_idx]);end=n if end is None else int(end)
        if not 0<=start<=end<=n:raise IndexError('Source read crossed episode bounds')
        # Keep real HDF5 dataset handles alive so adjacent episode reads share
        # the native decompressed chunk cache; values and read indices unchanged.
        if key not in self._datasets:self._datasets[key]=self.file[key]
        a=int(self.offsets[episode_idx]);return np.asarray(self._datasets[key][a+start:a+end])
    def load_chunk(self,episodes,starts,ends):
        import torch
        chunks=[]
        for ep,start,end in zip(episodes,starts,ends,strict=True):
            d={}
            for k in self.column_names:
                x=self.array(int(ep),k,int(start),int(end))
                if x.dtype.kind in 'OSU':
                    value=x[0] if len(x) else b'';d[k]=value.decode() if isinstance(value,bytes) else value
                else:
                    t=torch.from_numpy(x)
                    if x.ndim==4 and x.shape[-1] in (1,3):t=t.permute(0,3,1,2)
                    d[k]=t
            chunks.append(d)
        return chunks
    def close(self):
        if getattr(self,'file',None) is not None:self._datasets.clear();self.file.close();self.file=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

def admissible_windows(actions,length,history=3,horizon=1,stride=5):
    """All raw phases, conservative source span; no error-based filtering."""
    actions=np.asarray(actions)
    if actions.shape!=(length,2):raise ValueError('Raw actions not aligned with actual episode')
    span=(history+horizon)*stride
    candidates=np.arange(max(0,length-span+1),dtype=np.int64)
    needed=(history+horizon-1)*stride
    bad=np.r_[0,np.cumsum(~np.isfinite(actions).all(axis=1))]
    return candidates[(bad[candidates+needed]-bad[candidates])==0]

def action_processor(task):
    from sklearn.preprocessing import StandardScaler
    d=common.read_json(f'manifests/{task}_normalization.json')
    if d['convention']!='official_eval_StandardScaler_population_ddof0' or d['image']!=IMAGE_CONTRACT:
        raise RuntimeError('Unregistered official input transform')
    p=StandardScaler();p.mean_=np.asarray(d['action']['mean'],dtype=np.float64)
    p.var_=np.asarray(d['action']['variance'],dtype=np.float64);p.scale_=np.asarray(d['action']['scale'],dtype=np.float64)
    p.n_samples_seen_=int(d['action']['finite_rows']);p.n_features_in_=2
    return p

def load_cache(task,role):
    if role not in ROLES:raise ValueError('Explicit authorized data role required')
    m=common.read_json(f'manifests/{task}_cache.json');roles=common.read_json(f'manifests/{task}_data_roles.json')
    if m['task']!=task or m['roles_manifest_sha256']!=common.sha256(ROOT/f'manifests/{task}_data_roles.json'):
        raise RuntimeError('Cache/role identity differs')
    for relative,digest in m['input_hashes'].items():
        if common.sha256(checked_local(relative))!=digest:raise RuntimeError('Frozen cache input changed: '+relative)
    expected=sorted(x['episode_id'] for x in roles['episodes'] if x['role']==role)
    selected={k:v for k,v in m['episodes'].items() if v['role']==role}
    if sorted(selected)!=expected:raise RuntimeError('Exact role cache is incomplete')
    result={}
    for eid,r in selected.items():
        p=verified_file(r)
        with np.load(p,allow_pickle=False) as f:
            if set(f.files)!={'z','actions','raw_indices','stride','legal_starts'}:raise RuntimeError('Unexpected cache fields, including possible state labels')
            item={k:f[k].copy() for k in f.files};item['stride']=int(item['stride']);result[eid]=item
    return result
