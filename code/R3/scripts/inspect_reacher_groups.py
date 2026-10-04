"""Inspect only actual episode/initial-condition metadata, never model outcomes."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from r3 import common
from r3.data import RawH5,verified_file
def run():
    source=common.read_json('manifests/reacher_data_unpacked.json')['files']
    if len(source)!=1:raise RuntimeError('Explicit actual source audit required')
    path=verified_file(source[0])
    with RawH5(path,keys=['pixels','action']) as raw:
        starts=raw.offsets;lengths=raw.lengths
        qpos=np.asarray(raw.file['qpos'][starts]);qvel=np.asarray(raw.file['qvel'][starts]);initial=np.concatenate([qpos,qvel],axis=1)
        _,counts=np.unique(initial,axis=0,return_counts=True)
        action=np.asarray(raw.file['action'][:]);finite=np.isfinite(action).all(1)
        bad=np.flatnonzero(~finite);terminal=starts+lengths-1
        steps=np.asarray(raw.file['step_idx'][:]);ep_idx=np.asarray(raw.file['ep_idx'][:]);ids=np.asarray(raw.file['id'][:])
        checks={'lengths201':bool(np.all(lengths==201)),'step_indices_exact':all(np.array_equal(steps[s:s+n],np.arange(n)) for s,n in zip(starts,lengths)),'ep_indices_exact':all(np.all(ep_idx[s:s+n]==i) for i,(s,n) in enumerate(zip(starts,lengths))),'nonfinite_actions_only_terminal':bool(np.array_equal(bad,terminal)),'initial_qpos_qvel_finite':bool(np.isfinite(initial).all())}
        id_constant=all(np.all(ids[s:s+n]==ids[s]) for s,n in zip(starts,lengths))
        result={'status':'METADATA_INSPECTED' if all(checks.values()) else 'METADATA_REQUIRES_REVIEW','source':source[0],'checks':checks,'episodes':len(lengths),'raw_frames':int(lengths.sum()),'exact_initial_qpos_qvel_groups':len(counts),'initial_group_size_histogram':{str(int(x)):int((counts==x).sum()) for x in np.unique(counts)},'source_seed_column_present':'seed' in raw.file,'source_id_constant_within_episode':id_constant,'source_initial_ids_unique':len(np.unique(ids[starts])),'source_id_interpreted_as_seed':False,'family_rule':'FAMILY_UNKNOWN; source provides episode metadata but no verified demonstration lineage','action_terminal_nan_excluded_from_normalization':int(len(bad)),'state_goal_reward_score_success_values_used_for_selection':False,'model_or_scientific_outcomes_read':False,'optimizer_updates':0}
    common.atomic_json('state/REACHER_METADATA_AUDIT.json',result);print(json.dumps(result),flush=True)
    if not all(checks.values()):raise RuntimeError('Preserve actual metadata differences for review')
if __name__=='__main__':run()
