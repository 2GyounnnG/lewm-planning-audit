from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
roles=json.loads((R/'ops/reacher_data_roles.json').read_text())
fallback=json.loads((R/'ops/RESET_FALLBACK_MANIFEST.json').read_text())
# R3 cases/TECH are explicitly excluded. R5 Reacher uses exactly those R3 EVAL cases;
# independently ingest its raw table to bind the exclusion audit.
used_cases={c['case_id']:c for c in roles['cases']['TECH']+roles['cases']['EVAL']}
for csv_path in [R.parent/'r5_execution/h2/reports/reacher_MAIN/ALL_RAW_VALUES.csv',R.parent/'r4_v23_execution/r4/reports/remote_small_002/r4_reacher/reproduction/R3_REPRODUCTION_RAW.csv']:
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                cid=row.get('case_id') or row.get('case')
                if cid: used_cases.setdefault(cid,None)
used_eps={c['episode_id'] for c in used_cases.values() if isinstance(c,dict) and 'episode_id' in c}
# Cases in R5 can be joined to R3 case identities; all must remain in the exclusion set.
for cid in list(used_cases):
    if cid.startswith('R3_reacher_'):
        c=next((x for x in roles['cases']['TECH']+roles['cases']['EVAL'] if x['case_id']==cid),None)
        if c: used_eps.add(c['episode_id'])

def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def rank(ns,*parts): return hashlib.sha256(ns.encode()+b'\n'+canon(list(parts))).hexdigest()
def r3_start(row):
    parts=(row['episode_id'],row['source_asset_sha256'],row['source_episode_idx'])
    start=min(row['planning_starts'],key=lambda i:(rank('R3_CASE_START_20261002','reacher',*parts,i),i))
    if start in row['open_loop_starts']:
        offline=start;relation='SAME_AS_PLANNING_START'
    else:
        offline=min(row['open_loop_starts'],key=lambda i:(rank('R3_OPEN_LOOP_ANCHOR_20261002','reacher',*parts,i),i));relation='OPEN_LOOP_ANCHOR_NOT_PLANNING_START'
    return start,offline,relation
cand=[]
for row in roles['episodes']:
    if row.get('role') not in ('MONITOR','EVAL_POOL_UNUSED'): continue
    if row['role']=='REFIT_TRAIN' or row['episode_id'] in used_eps: continue
    if not row.get('metadata_eligible'): continue
    s,o,rel=r3_start(row)
    selection_hash=hashlib.sha256(('R6_FRESH/reacher/'+row['episode_id']).encode()).hexdigest()
    cid='R6_reacher_'+selection_hash[:24]
    case=dict(task='reacher',case_id=cid,role='R6_FRESH',episode_id=row['episode_id'],source_episode_idx=row['source_episode_idx'],source_asset_sha256=row['source_asset_sha256'],episode_sha256=row['episode_sha256'],family_id=row['family_id'],length=row['length'],start_raw_index=s,goal_raw_index=s+25,goal_offset_raw=25,open_loop_anchor_raw=o,open_loop_window_start_raw=o-10,open_loop_target_raw={str(h):o+h*5 for h in (1,2,5)},open_loop_anchor_relation=rel,source_seed=row.get('source_seed'),reset_seed=0,requires_validated_reset_fallback=True,reset_metadata=row.get('reset_metadata',{}),selection_used_model_outputs=False,selection_sha256=selection_hash,selection_rule='sha256("R6_FRESH/reacher/"+episode_id) ascending',construction_rule='R3 CASE EVAL construction: R3_CASE_START_20261002 hash legal planning start, goal=start+25 raw, R3_OPEN_LOOP_ANCHOR_20261002 fallback',reset_seed_validation_sha256=fallback['tasks']['reacher']['validation_receipt']['sha256'],reset_seed_provenance=fallback['tasks']['reacher']['provenance'])
    cand.append((selection_hash,row['episode_id'],case))
cand.sort(key=lambda x:(x[0],x[1]))
selected=cand[:min(100,len(cand))]
result=dict(version='R6_FRESH_CASE_SELECTION_V1',evidence_label='PREREGISTERED_FRESH_CASE_REPLICATION',status='SEALED_BEFORE_CLOSED_LOOP',task='reacher',created_utc='2026-10-04T03:00:00+08:00',selection_rule='sha256("R6_FRESH/reacher/"+episode_id) ascending',source_roles_manifest={'path':str((R/'ops/reacher_data_roles.json').resolve()),'sha256':hashlib.sha256((R/'ops/reacher_data_roles.json').read_bytes()).hexdigest()},fallback_manifest={'path':str((R/'ops/RESET_FALLBACK_MANIFEST.json').resolve()),'sha256':hashlib.sha256((R/'ops/RESET_FALLBACK_MANIFEST.json').read_bytes()).hexdigest()},exclusion={'excluded_R3_cases_and_TECH':len(roles['cases']['TECH'])+len(roles['cases']['EVAL']),'excluded_R3_episode_ids':len(used_eps),'R5_reacher_raw_case_ids_checked':sum(1 for x in used_cases if x.startswith('R3_reacher_')),'candidate_roles':['MONITOR','EVAL_POOL_UNUSED'],'candidate_count':len(cand),'REFIT_TRAIN_excluded':True,'selection_used_model_outputs':False},case_count=len(selected),minimum_case_count=50 if len(cand)>=50 else len(cand),cases=[x[2] for x in selected],candidate_hash_order=[{'selection_sha256':x[0],'episode_id':x[1]} for x in selected],planned_trajectories=24*len(selected),planned_case_order=[x[2]['case_id'] for x in selected])
if len(cand)<50: result['status']='STOP_BELOW_MINIMUM_50'
(R/'ops/R6_CASE_SELECTION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'candidates':len(cand),'selected':len(selected),'planned_trajectories':result['planned_trajectories'],'first_cases':result['planned_case_order'][:3]}))
