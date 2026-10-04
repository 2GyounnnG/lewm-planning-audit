"""Minimal recovery derivative; redundant final CEM populations remain on host."""
import argparse,hashlib,json,shutil
from pathlib import Path
from .common import atomic_json,atomic_npz,file_record,sha256
from .compact_recovery import frame_hash

DROP=('raw_pixels','last_generation_normalized','last_generation_cost')
def derive(source,target):
    import numpy as np
    source,target=Path(source),Path(target);completion=json.loads((source/'COMPLETE.json').read_text())
    for name,record in completion['files'].items():
        if file_record(source/name)!=record:raise RuntimeError('Original source seal differs')
    if (target/'MINIMUM_DERIVATION.json').exists():
        old=json.loads((target/'MINIMUM_DERIVATION.json').read_text())
        if old['source_trajectory']!=completion['files']['trajectory.npz'] or file_record(target/'trajectory.minimum.npz')!=old['derived_trajectory']:raise RuntimeError('Existing minimum derivative differs')
        return old
    removed={}
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files if k not in DROP}
        for key in DROP:
            if key not in f:continue
            value=f[key];removed[key]={'shape':list(value.shape),'dtype':str(value.dtype),'c_order_bytes_sha256':hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
        if 'raw_pixels' in f:
            value=f['raw_pixels'];arrays['raw_pixel_sha256']=np.asarray([frame_hash(x) for x in value],dtype='U64');arrays['initial_policy_pixels']=value[0].copy()
    atomic_npz(target/'trajectory.minimum.npz',**arrays)
    for name in ('result.json','COMPLETE.json'):shutil.copyfile(source/name,target/('source_'+name))
    for path in reversed(sorted(source.glob('attempt_*/STARTED.json'))):
        started=json.loads(path.read_text())
        if started['identity_sha256']==completion['identity_sha256']:shutil.copyfile(path,target/'source_STARTED.json');break
    else:raise RuntimeError('Missing original STARTED identity')
    record={'version':'R4_V23_MINIMUM_RECOVERY_V1','status':'DERIVED_NOT_ORIGINAL','source_directory':str(source),'source_trajectory':completion['files']['trajectory.npz'],'source_result':completion['files']['result.json'],'source_complete':file_record(source/'COMPLETE.json'),'identity_sha256':completion['identity_sha256'],'derived_trajectory':file_record(target/'trajectory.minimum.npz'),'retained_arrays':sorted(arrays),'removed_arrays':removed,
        'CEM_reconstruction':'Rerun exact official pinned solver and frozen model from each logged observation/history and goal, with result.replans seed_uint64 and original warm-start/action-block semantics. Verify returned plan and removed-array C-order byte SHA. Exact regeneration requires pinned original GPU/backend; a hash mismatch remains a recovery mismatch, never tolerance substitution.',
        'pixel_reconstruction':'Original reset seed and exact dataset start/goal callables, then raw_actions; raw0 is retained initial_policy_pixels. Subsequent reconstructed RGB must match each raw_pixel_sha256. Exact source CASE_WINDOWS map and original public H5 SHA are retained in sidecars; complete source windows/full trajectory logs remain on the host.',
        'actual_menu_evidence':'S2 complete initial_proposals plus normalized/native64 menus are recovered separately. S3 complete actual64 menus/alias/cost/selection are recovered separately. No sampled candidate used in a scientific menu is discarded from that menu.',
        'full_original_logs_retained_on_host':True,'code_sha256':sha256(__file__)}
    atomic_json(target/'MINIMUM_DERIVATION.json',record);return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args();source=Path(a.source);folders=sorted({p.parent for p in source.glob('**/COMPLETE.json') if (p.parent/'trajectory.npz').exists()})
    if a.limit:folders=folders[:a.limit]
    rows=[]
    for folder in folders:
        r=derive(folder,Path(a.output)/folder.relative_to(source));rows.append({'source':str(folder.relative_to(source)),'source_trajectory':r['source_trajectory'],'derived_trajectory':r['derived_trajectory']})
    result={'version':'R4_V23_MINIMUM_RECOVERY_V1','completed_cases':len(rows),'source_bytes':sum(r['source_trajectory']['bytes'] for r in rows),'derived_npz_bytes':sum(r['derived_trajectory']['bytes'] for r in rows),'source_logs_retained':True,'rows':rows};atomic_json(Path(a.output)/'MINIMUM_RECOVERY_INDEX.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
