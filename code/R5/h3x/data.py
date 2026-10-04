"""Read-only frozen H1b input cache; fixed R3 diagnostic windows in new H3X."""
from pathlib import Path
import numpy as np
import torch
from . import common

def verify(rec):
    p=Path(rec['path'])
    if p.stat().st_size!=rec['bytes'] or common.sha256(p)!=rec['sha256']:raise RuntimeError('Input identity changed '+str(p))
    return p

def load_cache(task,role):
    assert task=='reacher' and role in ('TECH','REFIT_TRAIN')
    source=Path('/workspace/r5/H1b/cache/reacher/manifest.json')
    audit=common.read_json(common.ROOT/'inputs/CACHE_AUDIT.json')
    if common.sha256(source)!=audit['h1b_cache_manifest']['sha256']:raise RuntimeError('Frozen cache manifest changed')
    m=common.read_json(source);roles=common.read_json('manifests/reacher_data_roles.json')
    assert m['roles_sha256']==common.sha256(common.R3_ROOT/'manifests/reacher_data_roles.json')
    expected=sorted(e['episode_id'] for e in roles['episodes'] if e['role']==role)
    records={k:v for k,v in m['episodes'].items() if v['role']==role}
    assert sorted(records)==expected
    from r3.data import admissible_windows
    result={}
    for eid,r in records.items():
        with np.load(verify(r),allow_pickle=False) as f:
            assert set(f.files)=={'z','actions','raw_indices','stride'}
            x={k:f[k].copy() for k in f.files};x['stride']=int(x['stride'])
        x['legal_starts']=admissible_windows(x['actions'],len(x['z']),3,1,5)
        result[eid]=x
    return result

class FixedMonitorAdapter:
    """Exactly original 256 MONITOR windows; encode only required observation frames."""
    def __init__(self,contract):
        audit=common.read_json(common.ROOT/'inputs/CACHE_AUDIT.json');r=audit['monitor_windows'];p=verify(r)
        with np.load(p,allow_pickle=False) as f:
            self.z=f['z'].copy();self.actions=f['actions'].copy();self.cursor=list(zip(f['episode_ids'].tolist(),f['starts'].tolist()))
        self.L=contract['history_size'];self.ids=sorted({e for e,s in self.cursor});self.lookup={tuple(c):i for i,c in enumerate(self.cursor)}
        roles=common.read_json('manifests/reacher_data_roles.json');assert [list(c) for c in self.cursor]==roles['monitor_windows']
        allowed={e['episode_id'] for e in roles['episodes'] if e['role']=='MONITOR'};assert set(self.ids)<=allowed
        assert self.z.shape==(256,8,192) and self.actions.shape==(256,7,10)
        assert np.isfinite(self.z).all() and np.isfinite(self.actions).all()
    def batch(self,cursors,device,horizon=5):
        assert horizon==5
        idx=[self.lookup[tuple(c)] for c in cursors]
        return torch.as_tensor(self.z[idx],device=device),torch.as_tensor(self.actions[idx],device=device)
