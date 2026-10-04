from __future__ import annotations
import csv,hashlib,json,os
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/r8')
TASK_ROOT={'cube':Path('/workspace/x1_cube'),'tworoom':Path('/workspace/x1_tworoom')}
SOURCE={'cube':Path('/dev/shm/x1_source_unpacked/cube/cube_single_expert.h5'),'tworoom':Path('/workspace/shared_data/unpacked/tworooms/tworoom.h5')}
RESET={'cube':['qpos','qvel','privileged_block_0_pos','privileged_block_0_quat'],'tworoom':['proprio']}

def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def sha_file(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rank(namespace,*parts): return hashlib.sha256(namespace.encode()+b'\n'+canon(list(parts))).hexdigest()
def record(p): return {'path':str(Path(p)),'bytes':Path(p).stat().st_size,'sha256':sha_file(p)}
def source_record(task):
 r=json.loads((TASK_ROOT[task]/'manifests'/f'{task}_source_receipt.json').read_text()); f=r['files'][0]; return {'path':f['path'],'bytes':f['bytes'],'sha256':f['sha256'],'archive_sha256':r['archive']['sha256'],'receipt':str(TASK_ROOT[task]/'manifests'/f'{task}_source_receipt.json')}


def prior(task,roles):
 # Existing X1 EVAL/TECH are the R3-R7 case universe. Compact R6/R7/R8
 # manifests are sufficient for cross-round exclusion; raw trees are skipped.
 ids=set(); eps=set()
 by_case={c['case_id']:c for k in ('TECH','EVAL') for c in roles['cases'].get(k,[])}
 for c in by_case.values(): eps.add(c['episode_id'])
 paths=[]
 for root in (ROOT/'ops',Path('/workspace/r6/ops'),Path('/workspace/r7/ops')):
  if root.exists():
   for p in root.iterdir():
    if p.is_file() and p.suffix in ('.json','.csv','.jsonl') and p.stat().st_size<=12_000_000: paths.append(p)
 for p in paths:
  try:
   if p.suffix=='.csv':
    with p.open(newline='') as f:
     for row in csv.DictReader(f):
      if isinstance(row.get('case_id'),str): ids.add(row['case_id'])
      if isinstance(row.get('episode_id'),str): eps.add(row['episode_id'])
   else:
    obj=json.loads(p.read_text()); stack=[obj]
    while stack:
     x=stack.pop()
     if isinstance(x,dict):
      if isinstance(x.get('case_id'),str): ids.add(x['case_id'])
      if isinstance(x.get('episode_id'),str): eps.add(x['episode_id'])
      stack.extend(x.values())
     elif isinstance(x,list): stack.extend(x)
  except Exception: pass
 for cid in ids:
  if cid in by_case: eps.add(by_case[cid]['episode_id'])
 prefix=task+':'
 return ids,{e for e in eps if e.startswith(prefix)}

def choose_start(task,row):
 parts=(row['episode_id'],row['source_asset_sha256'],int(row['source_episode_idx']))
 legal=[int(i) for i in row['planning_starts'] if int(i)+25<int(row['length'])]
 if not legal: raise ValueError('No legal planning anchor')
 return min(legal,key=lambda i:(rank('R3_CASE_START_20261002',task,*parts,i),i))

def select(task):
 xr=TASK_ROOT[task]; roles=json.loads((xr/'manifests'/f'{task}_data_roles.json').read_text()); ids,eps=prior(task,roles)
 cand=[]
 for row in roles['episodes']:
  if row.get('role') not in ('MONITOR','EVAL_POOL_UNUSED') or not row.get('metadata_eligible'): continue
  if row['episode_id'] in eps: continue
  legal=[i for i in row.get('planning_starts',[]) if int(i)+25<int(row['length'])]
  if not legal: continue
  s=choose_start(task,row); selector_sha=hashlib.sha256(('R8_FRESH/'+task+'/'+row['episode_id']).encode()).hexdigest()
  seed=row.get('source_seed'); reset_seed=int(seed) if seed is not None else 0
  c={'task':task,'case_id':f'R8_{task}_{selector_sha[:24]}','role':'EVAL','episode_id':row['episode_id'],'source_episode_idx':int(row['source_episode_idx']),'source_asset_sha256':row['source_asset_sha256'],'episode_sha256':row['episode_sha256'],'family_id':row.get('family_id'),'length':int(row['length']),'start_raw_index':int(s),'goal_raw_index':int(s+25),'goal_offset_raw':25,'source_seed':seed,'reset_seed':reset_seed,'requires_validated_reset_fallback':seed is None,'reset_metadata':row.get('reset_metadata',{}),'selection_sha256':selector_sha,'selection_rule':f'sha256("R8_FRESH/{task}/"+episode_id)','construction_rule':'R3 metadata planning-start hash, target offset 25 raw, budget 50','selection_used_model_outputs':False,'reset_seed_provenance':'C1_SYMMETRIC_OFFICIAL' if task=='cube' else ('SOURCE_SEED' if seed is not None else 'X1_TECH_FALLBACK_SEED0'),'reset_mode':'C1_SYMMETRIC_OFFICIAL' if task=='cube' else 'OFFICIAL_FALLBACK_SEED0'}
  cand.append((selector_sha,row['episode_id'],c))
 cand.sort(key=lambda x:(x[0],x[1])); chosen=[x[2] for x in cand[:100]]
 status='STOP_BELOW_MINIMUM_50' if len(chosen)<50 else 'SEALED_BEFORE_CLOSED_LOOP'
 out={'version':'R8_M2_FRESH_CASE_SELECTION_V1','evidence_label':'R8_FRESH_CASE_REPLICATION','task':task,'status':status,'selection_rule':f'sha256("R8_FRESH/{task}/"+episode_id)','construction_rule':'R3 metadata case construction; goal offset 25 raw; budget 50 raw','offset_raw':25,'budget_raw':50,'cases':chosen,'candidate_count':len(cand),'formal_count':len(chosen),'excluded_case_ids':len(ids),'excluded_episode_ids':len(eps),'selection_used_model_outputs':False,'planned_formal_trajectories':len(chosen)*12,'source_roles_sha256':sha_file(xr/'manifests'/f'{task}_data_roles.json'),'source_h5':source_record(task)}
 p=ROOT/'ops'/f'M2_{task.upper()}_CASE_SELECTION.json'; p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and json.loads(p.read_text())!=out: raise RuntimeError(f'Frozen selection differs: {p}')
 p.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 return out

def export(task,sel):
 import h5py
 try:
  import hdf5plugin
 except ImportError:
  pass
 outdir=ROOT/'assets'/'M2'/task;outdir.mkdir(parents=True,exist_ok=True); h5=h5py.File(SOURCE[task],'r',swmr=True,rdcc_nbytes=512*1024**2); entries=[]
 try:
  for c in sel['cases']:
   ep=int(c['source_episode_idx']); off=int(h5['ep_offset'][ep]); s=int(c['start_raw_index']);e=int(c['goal_raw_index'])+1
   keys=['pixels','action']+RESET[task]; arr={k:np.asarray(h5[k][off+s:off+e]) for k in keys}
   if arr['pixels'].ndim==4 and arr['pixels'].shape[-1]==3: arr['pixels']=arr['pixels'].transpose(0,3,1,2)
   arr.update(source_start_raw=np.int64(s),source_episode_idx=np.int64(ep))
   p=outdir/(c['case_id']+'.npz'); tmp=p.with_suffix('.npz.tmp')
   with tmp.open('wb') as f: np.savez_compressed(f,**arr)
   os.replace(tmp,p); entries.append({'case':c,'file':p.name,**record(p)})
 finally: h5.close()
 m={'version':'R8_M2_CASE_WINDOWS_V1','evidence_label':'R8_FRESH_CASE_REPLICATION','task':task,'source':source_record(task),'cases':entries}
 mp=ROOT/'ops'/f'M2_CASE_WINDOWS_{task.upper()}.json';mp.write_text(json.dumps(m,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 return m

if __name__=='__main__':
 for task in ('cube','tworoom'):
  sel=select(task); m=export(task,sel); print(json.dumps({'task':task,'candidate_count':sel['candidate_count'],'formal_count':sel['formal_count'],'window_manifest_sha256':sha_file(ROOT/'ops'/f'M2_CASE_WINDOWS_{task.upper()}.json')}))
