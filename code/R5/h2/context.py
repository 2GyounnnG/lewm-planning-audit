"""Only R5 H2's registered history/action-prefix intervention."""
from pathlib import Path
import json
import numpy as np
from r4.common import file_record

def history_indices(raw_count):
    if raw_count == 0:return [0]
    if raw_count != 25:raise RuntimeError('Unexpected replan outside registered 50-raw/25-receding budget')
    return [15,20,25]

def real_history_observation(info,observations,raw_count):
    # Buffered raw actions still use the untouched official policy. Only the
    # second planning call receives three real observations from this rollout.
    if raw_count != 25:return info
    if len(observations)!=26:raise RuntimeError('History observation/raw alignment')
    out=dict(info)
    out['pixels']=np.stack([observations[i]['pixels'] for i in history_indices(raw_count)])[None]
    return out

def candidates_with_prefix(info,candidates,rows,action_processor):
    import torch
    history=int(info['pixels'].shape[2])
    if history==1:
        if rows:raise RuntimeError('A second plan must contain real history')
        return candidates
    if history!=3 or len(rows)!=25:raise RuntimeError('Unregistered real context/action prefix')
    raw=np.stack([r['action'] for r in rows[15:25]])
    if raw.shape!=(10,2):raise RuntimeError('Expected ten executed raw actions')
    # Each prefix macro leaves its corresponding real frame: 15->20,20->25.
    prefix=action_processor.transform(raw).reshape(1,1,2,10)
    prefix=torch.as_tensor(prefix,device=candidates.device,dtype=candidates.dtype)
    prefix=prefix.expand(candidates.shape[0],candidates.shape[1],2,10)
    if candidates.shape[-2:]!=(5,10):raise RuntimeError('CEM horizon/action block changed')
    return torch.cat([prefix,candidates],dim=2)

def assert_first_plan(folder,plan):
    folder=Path(folder);complete=json.loads((folder/'COMPLETE.json').read_text())
    for name,expected in complete['files'].items():
        if file_record(folder/name)!=expected:raise RuntimeError('Sealed R4 control bytes differ: '+str(folder/name))
    with np.load(folder/'trajectory.npz',allow_pickle=False) as f:
        reference=f['returned_plans_normalized'][0]
    if not np.array_equal(reference,plan):
        raise RuntimeError('H2 first plan differs bitwise from same-model/case/stream R4 control; requires explicit control re-run per R5 section6')
    return {'status':'BITWISE_EQUAL','control_complete_sha256':file_record(folder/'COMPLETE.json')['sha256'],
            'control_trajectory_sha256':complete['files']['trajectory.npz']['sha256'],'max_abs_difference':0.0}

def assert_control_provenance(folder,case,provenance):
    folder=Path(folder)
    complete=json.loads((folder/'COMPLETE.json').read_text())
    from r4.common import digest
    for p in sorted(folder.glob('attempt_*/STARTED.json')):
        identity=json.loads(p.read_text())['identity']
        if digest(identity)==complete['identity_sha256']:
            if identity['case']!=case or identity['provenance']!=provenance:
                raise RuntimeError('H2 source/model provenance differs from sealed control')
            return
    raise RuntimeError('Control completed identity has no STARTED receipt')
