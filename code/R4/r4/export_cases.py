"""Extract exact R3 EVAL/TECH source windows without copying full training data.

Run with a Python containing numpy, h5py and hdf5plugin. Source is opened read-only.
These per-case files retain original source indices; they do not define new cases.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .common import atomic_json,atomic_npz,file_record,sha256

def export(root,output,tasks=('pusht','reacher')):
    import numpy as np
    import h5py,hdf5plugin
    root=Path(root);output=Path(output)
    result=json.loads((output/'CASE_WINDOWS.json').read_text()) if (output/'CASE_WINDOWS.json').exists() else {'version':'R4_EXACT_CASE_WINDOWS_V1','tasks':{},'source_read_only':True}
    for task in tasks:
        roles_path=root/f'manifests/{task}_data_roles.json';roles=json.loads(roles_path.read_text())
        source=json.loads((root/f'manifests/{task}_source_map.json').read_text())
        fallback=json.loads((root/'manifests/RESET_FALLBACK_MANIFEST.json').read_text())['tasks'][task]
        entries=[]
        for raw in roles['cases']['TECH']+roles['cases']['EVAL']:
            case=dict(raw)
            if case['reset_seed'] is None:
                if case['case_id'] not in fallback['affected_case_ids']:raise RuntimeError('Unregistered fallback case')
                case.update(reset_seed=fallback['fallback_seed'],reset_seed_validation_sha256=fallback['validation_receipt']['sha256'],reset_seed_provenance=fallback['provenance'])
            item=source['assets'][case['source_asset_sha256']];p=root/item['path']
            if p.stat().st_size!=item['bytes']:raise RuntimeError('Source byte count differs')
            start=min(case['start_raw_index'],case['open_loop_anchor_raw']-10)
            end=max(case['goal_raw_index'],case['open_loop_anchor_raw']+25)+1
            target=output/task/(case['case_id']+'.npz')
            with h5py.File(p,'r',swmr=True) as h:
                offset=int(h['ep_offset'][case['source_episode_idx']]);length=int(h['ep_len'][case['source_episode_idx']])
                if not 0<=start<end<=length:raise RuntimeError('Illegal exact source window')
                keys=['pixels','action','state'] if task=='pusht' else ['pixels','action','qpos','qvel']
                arrays={k:np.asarray(h[k][offset+start:offset+end]) for k in keys}
                if arrays['pixels'].shape[-1]==3:arrays['pixels']=arrays['pixels'].transpose(0,3,1,2)
                arrays.update(source_start_raw=np.int64(start),source_episode_idx=np.int64(case['source_episode_idx']))
            if target.exists():
                with np.load(target,allow_pickle=False) as existing:
                    if set(existing.files)!=set(arrays) or any(not np.array_equal(existing[k],v,equal_nan=True) for k,v in arrays.items()):raise RuntimeError('Existing source export differs')
            else:atomic_npz(target,**arrays)
            entries.append({'case':case,'file':str(target.relative_to(output)),**file_record(target),'source_asset':item})
        result['tasks'][task]={'roles_sha256':sha256(roles_path),'cases':entries}
        atomic_json(output/'CASE_WINDOWS.json',result)
    return result

class CaseWindows:
    def __init__(self,path,entry):
        import numpy as np
        self.entry=entry;p=Path(path)/entry['file']
        if file_record(p)!={k:entry[k] for k in ('bytes','sha256')}:raise RuntimeError('Exported case bytes changed')
        with np.load(p,allow_pickle=False) as f:self.arrays={k:f[k].copy() for k in f.files}
        self.column_names=[k for k in self.arrays if not k.startswith('source_')]
    def load_chunk(self,episodes,starts,ends):
        import torch
        result=[]
        for ep,start,end in zip(episodes,starts,ends):
            if int(ep)!=int(self.arrays['source_episode_idx']):raise RuntimeError('Different source episode')
            lo=int(start)-int(self.arrays['source_start_raw']);hi=int(end)-int(self.arrays['source_start_raw'])
            if not 0<=lo<hi<=len(self.arrays['pixels']):raise RuntimeError('Outside exported legal source')
            result.append({k:torch.from_numpy(v[lo:hi].copy()) for k,v in self.arrays.items() if k in self.column_names})
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--output',required=True);p.add_argument('--task',choices=['pusht','reacher'],action='append');a=p.parse_args()
    result=export(a.r3_root,a.output,tuple(a.task or ('pusht','reacher')));print(json.dumps({t:len(v['cases']) for t,v in result['tasks'].items()}))
