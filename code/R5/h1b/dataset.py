"""All TRAIN rows, bounded-memory matrices; EVAL loaded only by evaluation call."""
from pathlib import Path
import argparse, hashlib, os, time
import numpy as np
from common import *

def target_specs(task):
    p=read(Path(__file__).with_name('FIT_PROTOCOL.json'))
    return [(k,v) for k,v in p['state_targets'][task].items() if v is not None]

def prepare(task,output,technical_episodes=None):
    import h5py,hdf5plugin
    c=task_config(task);roles=read(c['roles']);cache=read(c['cache']);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return read(out/'manifest.json')
    eps=[e for e in roles['episodes'] if e['role']=='REFIT_TRAIN']
    if technical_episodes is not None:eps=eps[:technical_episodes]
    n=sum(e['length'] for e in eps);specs=target_specs(task);dy=sum(len(v['indices']) for _,v in specs);da=5 if task=='cube' else 2
    arrays={'z':np.lib.format.open_memmap(out/'z.npy',mode='w+',dtype='float32',shape=(n,192)),'actions':np.lib.format.open_memmap(out/'actions.npy',mode='w+',dtype='float32',shape=(n,da)),'state':np.lib.format.open_memmap(out/'state.npy',mode='w+',dtype='float64',shape=(n,dy))}
    hold_order=sorted(eps,key=lambda e:hashlib.sha256(f"R5_H1B_HOLDOUT_V1/{task}/{e['episode_id']}".encode()).digest())
    hold={e['episode_id'] for e in hold_order[:int(np.ceil(.1*len(eps)))]};metadata=[];off=0;receipts=[]
    with h5py.File(c['h5'],'r',swmr=True) as f:
        for ei,e in enumerate(eps):
            p=checked(cache['episodes'][e['episode_id']],c['root']);d=np.load(p,allow_pickle=False);nn=e['length'];assert d['z'].shape==(nn,192)
            arrays['z'][off:off+nn]=d['z'];arrays['actions'][off:off+nn]=d['actions'];raw=int(f['ep_offset'][e['source_episode_idx']]);assert int(f['ep_len'][e['source_episode_idx']])==nn
            arrays['state'][off:off+nn]=np.concatenate([np.asarray(f[v['key']][raw:raw+nn])[:,v['indices']] for _,v in specs],axis=1)
            metadata.append({'episode_id':e['episode_id'],'offset':off,'length':nn,'fold':fold(e,task),'holdout':e['episode_id'] in hold,'source_cache_sha256':cache['episodes'][e['episode_id']]['sha256']});off+=nn
            if (ei+1)%1000==0:print('prepare',task,ei+1,len(eps),flush=True)
    for v in arrays.values():v.flush()
    labels=[];groups={};pos=0
    for k,v in specs:groups[k]=list(range(pos,pos+len(v['indices'])));labels.extend(v['labels']);pos+=len(v['indices'])
    rec={'status':'COMPLETE','task':task,'kind':'TECHNICAL' if technical_episodes else 'FORMAL','protocol_sha256':sha(Path(__file__).with_name('FIT_PROTOCOL.json')),'roles':file(c['roles']),'source_cache_manifest':file(c['cache']),'episodes':metadata,'frames':n,'holdout_episodes':len(hold),'state_labels':labels,'state_groups':groups,'files':{k:file(out/(k+'.npy')) for k in arrays},'source_h5_original_sha256':read(Path(output).parents[1]/'INPUT_AUDIT.json')['tasks'][task]['source_h5']['sha256'] if technical_episodes is None else None}
    atomic(out/'manifest.json',rec);return rec

class Dataset:
    def __init__(self,path,target,history):
        self.path=Path(path);self.meta=read(self.path/'manifest.json');self.z=np.load(self.path/'z.npy',mmap_mode='r');self.actions=np.load(self.path/'actions.npy',mmap_mode='r');self.state=np.load(self.path/'state.npy',mmap_mode='r');self.target=target;self.history=history;self.h=int(target.split('_')[1]) if target.startswith('latent_') else 0
        self.components=list(range(self.state.shape[1])) if target=='state' else self.meta['state_groups'].get(target)
        anchors=[];folds=[];hold=[]
        for ep in self.meta['episodes']:
            ix=np.arange(ep['offset']+10,ep['offset']+ep['length']-5*self.h,dtype=np.int64)
            good=np.isfinite(self.z[ix]).all(1)&np.isfinite(self.z[ix-5]).all(1)&np.isfinite(self.z[ix-10]).all(1)
            if self.h:
                good&=np.isfinite(self.z[ix+5*self.h]).all(1)
                for j in range(5*self.h):good&=np.isfinite(self.actions[ix+j]).all(1)
            else:good&=np.isfinite(self.state[ix][:,self.components]).all(1)
            ix=ix[good];anchors.append(ix);folds.extend([ep['fold']]*len(ix));hold.extend([ep['holdout']]*len(ix))
        self.anchors=np.concatenate(anchors);self.folds=np.asarray(folds,dtype=np.int8);self.holdout=np.asarray(hold,dtype=bool)
    def xy(self,rows,residual=False):
        ix=self.anchors[rows];x=self.z[ix] if self.history=='single' else np.concatenate([self.z[ix-10],self.z[ix-5],self.z[ix]],axis=1)
        if self.h:
            act=np.stack([self.actions[ix+j] for j in range(5*self.h)],axis=1).reshape(len(ix),-1);x=np.concatenate([x,act],axis=1);y=np.asarray(self.z[ix+5*self.h]);y=y-self.z[ix] if residual else y
        else:y=np.asarray(self.state[ix])[:,self.components]
        return np.asarray(x),np.asarray(y)

def evaluation(task,target,history):
    import h5py,hdf5plugin
    c=task_config(task);roles=read(c['roles']);cases=roles['cases']['EVAL'];specs=target_specs(task);h=int(target.split('_')[1]) if target.startswith('latent_') else 0
    all_labels=[s for _,v in specs for s in v['labels']];group={};offset=0
    for k,v in specs:group[k]=list(range(offset,offset+len(v['indices'])));offset+=len(v['indices'])
    components=list(range(len(all_labels))) if target=='state' else group.get(target)
    refp=c['root']/'artifacts/open_loop'/task/'inputs.npz' if task in ('pusht','reacher') else c['root']/'recovery/bundle/replay/fixed_open_loop_inputs.npz'
    ref=np.load(refp,allow_pickle=False);lookup={str(v):i for i,v in enumerate(ref['case_ids'])};refsha=sha(refp)
    targetp=refp.with_name('targets.npz') if task in ('pusht','reacher') else refp
    targets=np.load(targetp,allow_pickle=False)['target_z'];targetsha=sha(targetp)
    assert set(lookup)=={e['case_id'] for e in cases} and targets.shape==(100,5,192)
    normal=read(c['normalization']);am=np.asarray(normal['action']['mean']);asc=np.asarray(normal['action']['scale'])
    xs=[];ys=[];curr=[];provenance=[];rolesha=sha(c['roles'])
    with h5py.File(c['h5'],'r',swmr=True) as f:
        for e in cases:
            t=e['open_loop_anchor_raw'];assert t>=10;j=lookup[e['case_id']]
            raw=int(f['ep_offset'][e['source_episode_idx']])+t;past=ref['initial_z'][j];future=targets[j,h-1] if h else None
            if h:
                recorded=np.asarray(f['action'][raw:raw+5*h]);actions=ref['macro_actions'][j,2:2+h].reshape(-1)
                normalized=((recorded-am)/asc).astype(np.float32).reshape(-1)
                assert np.allclose(normalized,actions,atol=1e-6,rtol=1e-6),('ACTION_ANCHOR_MISMATCH',e['case_id'])
            x=past[-1] if history=='single' else past.reshape(-1)
            if h:x=np.r_[x,actions];y=future
            else:y=np.concatenate([np.asarray(f[v['key']][raw])[v['indices']] for _,v in specs])[components]
            assert np.isfinite(x).all() and np.isfinite(y).all()
            xs.append(x);ys.append(y);curr.append(past[-1]);provenance.append({'case_id':e['case_id'],'episode_id':e['episode_id'],'family_id':e.get('family_id'),'anchor_raw':t,'source_asset_sha256':e['source_asset_sha256'],'anchor_reference_sha256':refsha,'target_reference_sha256':targetsha,'roles_sha256':rolesha})
    return np.asarray(xs),np.asarray(ys),np.asarray(curr),provenance,all_labels if target=='state' else ([all_labels[i] for i in components] if not h else [f'z_{i}' for i in range(192)])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--output',required=True);p.add_argument('--technical-episodes',type=int);a=p.parse_args();prepare(a.task,a.output,a.technical_episodes)
