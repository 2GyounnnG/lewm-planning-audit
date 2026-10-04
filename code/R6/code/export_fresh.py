from __future__ import annotations
import json,hashlib,sys
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 root=Path(sys.argv[1]); out=Path(sys.argv[2]); sel=json.loads((root/'R6_CASE_SELECTION.json').read_text()); r3=Path('/workspace/shared_data/r3')
 sm=json.loads((r3/'manifests/reacher_source_map.json').read_text()); entries=[]
 import h5py,hdf5plugin
 assets={}
 for case in sel['cases']: assets[case['source_asset_sha256']]=sm['assets'][case['source_asset_sha256']]
 handles={}
 try:
  for ah,a in assets.items():
   src=r3/a['path']; assert src.stat().st_size==a['bytes']; handles[ah]=h5py.File(src,'r',swmr=True,rdcc_nbytes=512*1024**2)
  for i,case in enumerate(sel['cases']):
   a=assets[case['source_asset_sha256']];h=handles[case['source_asset_sha256']]; target=out/(case['case_id']+'.npz'); start=case['start_raw_index']; end=case['goal_raw_index']+1
   off=int(h['ep_offset'][case['source_episode_idx']]); length=int(h['ep_len'][case['source_episode_idx']]); assert 0<=start<end<=length
   arr={k:np.asarray(h[k][off+start:off+end]) for k in ('pixels','action','qpos','qvel')}
   if arr['pixels'].shape[-1]==3:arr['pixels']=arr['pixels'].transpose(0,3,1,2)
   arr.update(source_start_raw=np.int64(start),source_episode_idx=np.int64(case['source_episode_idx']))
   np.savez_compressed(target,**arr); entries.append({'case':case,'file':target.name,**record(target),'source_asset':a})
   if i%10==0: print(json.dumps({'exported':i+1,'total':len(sel['cases'])}),flush=True)
 finally:
  for h in handles.values():h.close()
 manifest={'version':'R6_FRESH_CASE_WINDOWS_V1','evidence_label':'PREREGISTERED_FRESH_CASE_REPLICATION','source_roles_manifest':record(r3/'manifests/reacher_data_roles.json'),'source_map':record(r3/'manifests/reacher_source_map.json'),'cases':entries}
 (out/'CASE_WINDOWS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'status':'EXPORTED','cases':len(entries),'sha256':sha(out/'CASE_WINDOWS.json')}))
if __name__=='__main__':main()
