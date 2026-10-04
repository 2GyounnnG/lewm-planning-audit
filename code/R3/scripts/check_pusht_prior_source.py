"""Read-only exact metadata/action comparison with the preserved G1 source."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from r3 import common
from r3.data import RawH5,verified_file

def run():
    root=common.ROOT;old=root.parent/'g1_pusht_2x5090_v1/g1/data'
    paths=[old/x for x in ('index_metadata.npz','initial_input_proprio_for_group_audit.npz','split_manifest.json','family_manifest.json','family_pixel_validation.json')]
    if not all(p.is_file() for p in paths):raise RuntimeError('Missing actual old G1 metadata: '+str([str(p) for p in paths if not p.is_file()]))
    before={str(p):common.sha256(p) for p in paths}
    unpack=common.read_json('manifests/pusht_data_unpacked.json')
    assets=[x for x in unpack['files'] if x['path'].endswith('.h5')]
    if len(assets)!=1:raise RuntimeError('Ambiguous actual HDF5')
    asset=assets[0];source=verified_file(asset)
    with np.load(paths[0],allow_pickle=False) as f:meta={k:f[k] for k in f.files}
    with np.load(paths[1],allow_pickle=False) as f:initial={k:f[k] for k in f.files}
    split=json.loads(paths[2].read_text());families=json.loads(paths[3].read_text())
    checks={};details={}
    with RawH5(source,keys=['pixels','action']) as raw:
        ids,starts,lengths=np.unique(meta['episode_idx'],return_index=True,return_counts=True)
        checks['episode_ids_equal']=bool(np.array_equal(ids,np.arange(len(raw.lengths))))
        checks['episode_lengths_equal']=bool(np.array_equal(lengths,raw.lengths))
        checks['episode_offsets_equal']=bool(np.array_equal(starts,raw.offsets))
        for key in ('episode_idx','step_idx','action'):
            actual=np.asarray(raw.file[key][:]);checks[key+'_dtype_equal']=str(actual.dtype)==str(meta[key].dtype)
            checks[key+'_all_values_exact']=bool(np.array_equal(actual,meta[key]))
            details[key]={'old_sha256':hashlib.sha256(meta[key].tobytes()).hexdigest(),'actual_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'old_dtype':str(meta[key].dtype),'actual_dtype':str(actual.dtype),'shape':list(actual.shape)}
        actual_initial=np.asarray(raw.file['proprio'][raw.offsets])
        checks['initial_proprio_episode_ids_equal']=bool(np.array_equal(initial['episode_idx'],ids))
        checks['initial_proprio_dtype_equal']=str(actual_initial.dtype)==str(initial['proprio'].dtype)
        checks['initial_proprio_all_values_exact']=bool(np.array_equal(actual_initial,initial['proprio']))
        unique,inverse,counts=np.unique(actual_initial,axis=0,return_inverse=True,return_counts=True)
        checks['old_family_membership_equal']=all(len(set(inverse[f['source_episode_indices']]))==1 for f in families['families']) and len(unique)==len(families['families'])
        selected=[]
        for ep in split['episodes']:
            aa=raw.array(ep['source_episode_idx'],'action')
            good=hashlib.sha256(aa.tobytes()).hexdigest()==ep['source_action_sha256']
            selected.append({'source_episode_idx':ep['source_episode_idx'],'G1_role':ep['split'],'family_id':ep['family_id'],'action_sha256_exact':good})
        checks['all_896_selected_episode_actions_exact']=all(x['action_sha256_exact'] for x in selected)
    if any(common.sha256(Path(p))!=h for p,h in before.items()):raise RuntimeError('Old read-only evidence changed')
    # Index storage width is a source format detail. Require integer kinds and
    # exact values; action/proprio bytes and dtypes still must match exactly.
    checks['index_columns_are_integer']=all(meta[k].dtype.kind in 'iu' and details[k]['actual_dtype'].startswith(('int','uint')) for k in ('episode_idx','step_idx'))
    gates={k:v for k,v in checks.items() if k not in ('episode_idx_dtype_equal','step_idx_dtype_equal')}
    result={'status':'EXACT_METADATA_ACTION_AND_INITIAL_INPUT_MATCH' if all(gates.values()) else 'CROSS_FORMAT_MATCH_NOT_ESTABLISHED','checked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checks':checks,'identity_gates':gates,'index_dtype_differences_are_disclosed_format_metadata':True,'old_read_only_files':before,'R3_source':asset,'old_source_revision':split['revision'],'array_evidence':details,'source_episodes':len(ids),'conservative_initial_input_families':len(unique),'family_size_histogram':{str(int(n)):int((counts==n).sum()) for n in np.unique(counts)},'old_selected_episodes':selected,'pixel_byte_identity_checked':False,'limit':'Exact full action/episode metadata and initial proprio match establish source episode alignment; this check does not assert pixel byte identity or original independent demonstration lineage. G1/R2 exposure is carried separately from R3 refit role.','optimizer_updates':0,'model_outputs_read':False,'old_files_modified':False}
    common.atomic_json('state/PUSHT_PRIOR_SOURCE_ALIGNMENT_V2.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('old_selected_episodes','array_evidence','old_read_only_files')}),flush=True)
    if not all(gates.values()):raise RuntimeError('Actual source alignment differs; no inferred G1 exposure/family evidence applied')
if __name__=='__main__':run()
