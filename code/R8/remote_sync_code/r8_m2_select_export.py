from __future__ import annotations
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
R3=Path('/workspace/shared_data/r3');ROOT=Path('/workspace/r8')
def sha_file(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha_file(p)}
def prior(task):
 ids=set();eps=set(); roots=[Path('/workspace/r4_reacher'),Path('/workspace/r4_pusht'),Path('/workspace/r4_cube'),Path('/workspace/r4_tworoom'),Path('/workspace/r5'),Path('/workspace/r6'),Path('/workspace/r7'),ROOT]
 for root in roots:
  if not root.exists():continue
  for p in root.rglob('*.csv'):
   try:
    for r in csv.DictReader(p.open()):
     cid=r.get('case_id') or r.get('case')
     if cid:ids.add(cid)
     eid=r.get('episode_id')
     if eid:eps.add(eid)
   except Exception:pass
  for p in root.rglob('*.json'):
   if p.stat().st_size>20_000_000:continue
   try:d=json.loads(p.read_text())
   except Exception:continue
   stack=[d]
   while stack:
    x=stack.pop()
    if isinstance(x,dict):
     if x.get('task')==task and x.get('case_id'):ids.add(x['case_id'])
     if x.get('task')==task and x.get('episode_id'):eps.add(x['episode_id'])
     stack.extend(x.values())
    elif isinstance(x,list):stack.extend(x)
 roles=json.loads((R3/'manifests'/f'{task}_data_roles.json').read_text())
 by={c['case_id']:c for k in ('TECH','EVAL') for c in roles['cases'][k]}
 eps.update(by[c]['episode_id'] for c in ids if c in by)
 return ids,eps,roles
def start(task,row,offset):
 legal=[int(i) for i in row['planning_starts'] if int(i)+offset<int(row['length'])]
 def key(i):
  s=json.dumps([row['episode_id'],row['source_asset_sha256'],row['source_episode_idx'],i],sort_keys=True,separators=(',',':'))
  return hashlib.sha256(('R3_CASE_START_20261002'+task+s).encode()).hexdigest()
 return min(legal,key=lambda i:(key(i),i))
def select(task):
 ids,eps,roles=prior(task);fallback=json.loads((R3/'manifests'/'RESET_FALLBACK_MANIFEST.json').read_text())['tasks'][task]
 cand=[]
 for row in roles['episodes']:
  if row.get('role') not in ('MONITOR','EVAL_POOL_UNUSED') or row.get('episode_id') in eps or not row.get('metadata_eligible'):continue
  legal=[int(i) for i in row.get('planning_starts',[]) if int(i)+25<int(row['length'])]
  if not legal:continue
  s=start(task,row,25);sh=hashlib.sha256(('R8_FRESH/'+task+'/'+row['episode_id']).encode()).hexdigest();seed=row.get('source_seed')
  c={'task':task,'case_id':f'R8_{task}_{sh[:24]}','role':'EVAL','episode_id':row['episode_id'],'source_episode_idx':int(row['source_episode_idx']),'source_asset_sha256':row['source_asset_sha256'],'episode_sha256':row['episode_sha256'],'family_id':row.get('family_id'),'length':int(row['length']),'start_raw_index':s,'goal_raw_index':s+25,'goal_offset_raw':25,'source_seed':seed,'reset_seed':int(seed) if seed is not None else 0,'requires_validated_reset_fallback':seed is None,'reset_metadata':row.get('reset_metadata',{}),'selection_sha256':sh,'selection_rule':f'sha256("R8_FRESH/{task}/"+episode_id)','construction_rule':'R3 case-start hash with target offset 25 raw','selection_used_model_outputs':False,'reset_seed_validation_sha256':fallback['validation_receipt']['sha256'] if seed is None else None,'reset_seed_provenance':fallback['provenance'] if seed is None else 'SOURCE_SEED'}
  cand.append((sh,row['episode_id'],c))
 cand.sort(key=lambda x:(x[0],x[1]));chosen=[x[2] for x in cand[:100]]
 if len(chosen)<50: status='STOP_BELOW_MINIMUM_50'
 else:status='SEALED_BEFORE_CLOSED_LOOP'
 out={'version':'R8_M2_FRESH_CASE_SELECTION_V1','evidence_label':'R8_FRESH_CASE_REPLICATION','task':task,'status':status,'selection_rule':f'sha256("R8_FRESH/{task}/"+episode_id)','offset_raw':25,'budget_raw':50,'cases':chosen,'candidate_count':len(cand),'formal_count':len(chosen),'excluded_case_ids':len(ids),'excluded_episode_ids':len(eps),'selection_used_model_outputs':False,'planned_formal_trajectories':len(chosen)*12}
 p=ROOT/'ops'/f'M2_{task.upper()}_CASE_SELECTION.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');return out,roles
def export(task,sel,roles):
 import h5py,hdf5plugin
 sm=json.loads((R3/'manifests'/f'{task}_source_map.json').read_text());outdir=ROOT/'assets'/'M2'/task;outdir.mkdir(parents=True,exist_ok=True);handles={};entries=[]
 for c in sel['cases']:
  ah=c['source_asset_sha256'];src=R3/sm['assets'][ah]['path'];handles.setdefault(ah,h5py.File(src,'r',swmr=True,rdcc_nbytes=512*1024**2));h=handles[ah];off=int(h['ep_offset'][c['source_episode_idx']]);s,e=c['start_raw_index'],c['goal_raw_index']+1;keys=['pixels','action','state'] if task=='pusht' else ['pixels','action','qpos','qvel'];arr={k:np.asarray(h[k][off+s:off+e]) for k in keys};
  if arr['pixels'].shape[-1]==3:arr['pixels']=arr['pixels'].transpose(0,3,1,2)
  arr.update(source_start_raw=np.int64(s),source_episode_idx=np.int64(c['source_episode_idx']));p=outdir/(c['case_id']+'.npz');
  with p.open('wb') as f:np.savez_compressed(f,**arr)
  entries.append({'case':c,'file':p.name,**record(p)})
 for h in handles.values():h.close()
 m={'version':'R8_M2_CASE_WINDOWS_V1','evidence_label':'R8_FRESH_CASE_REPLICATION','task':task,'cases':entries};(ROOT/'ops'/f'M2_CASE_WINDOWS_{task.upper()}.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
for t in ('reacher','pusht'):
 s,r=select(t);export(t,s,r);print(t,s['candidate_count'],s['formal_count'])
