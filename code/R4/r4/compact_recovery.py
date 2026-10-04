"""Lossless nonpixel recovery derivatives; sealed full source logs stay untouched."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
from .common import atomic_json, atomic_npz, file_record, sha256

def frame_hash(frame):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()

def compact_case(source, target, *, assets=None, replay_receipts=None):
    import numpy as np
    source, target = Path(source), Path(target)
    completion = json.loads((source/'COMPLETE.json').read_text())
    for name, record in completion['files'].items():
        if file_record(source/name) != record:
            raise RuntimeError('Unsealed or changed source: '+str(source/name))
    result = json.loads((source/'result.json').read_text())
    source_record = file_record(source/'trajectory.npz')
    if (target/'DERIVATION.json').exists():
        previous=json.loads((target/'DERIVATION.json').read_text())
        if previous['source_trajectory'] != source_record:
            raise RuntimeError('Derivative source changed')
        if file_record(target/'trajectory.compact.npz') != previous['compact_trajectory']:
            raise RuntimeError('Derivative array changed')
        return previous
    with np.load(source/'trajectory.npz', allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files if k!='raw_pixels'}
        pixel_metadata=None
        if 'raw_pixels' in data:
            pixels=data['raw_pixels']
            arrays['raw_pixel_sha256']=np.asarray([frame_hash(frame) for frame in pixels], dtype='U64')
            pixel_metadata={'shape':list(pixels.shape), 'dtype':str(pixels.dtype), 'hash_bytes_order':'C_CONTIGUOUS_ROW_MAJOR', 'hash_algorithm':'SHA256'}
    target.mkdir(parents=True,exist_ok=True)
    atomic_npz(target/'trajectory.compact.npz', **arrays)
    # Copy metadata verbatim, explicitly under names describing its source.
    for name in ('result.json','COMPLETE.json'):
        shutil.copyfile(source/name,target/('source_'+name))
    started=sorted(source.glob('attempt_*/STARTED.json'))
    identity=None
    for path in reversed(started):
        candidate=json.loads(path.read_text())
        if candidate['identity_sha256']==completion['identity_sha256']:
            identity=candidate['identity'];shutil.copyfile(path,target/'source_STARTED.json');break
    if identity is None:raise RuntimeError('Missing original case identity')
    case_entry=None
    if assets:
        manifest=json.loads((Path(assets)/'CASE_WINDOWS.json').read_text())
        case_entry=next(e for e in manifest['tasks'][result['task']]['cases'] if e['case']['case_id']==result['case_id'])
    receipts=[]
    if replay_receipts:
        receipts=[{'path':str(p),'file':file_record(p),'status':json.loads(p.read_text()).get('status')} for p in sorted(Path(replay_receipts).glob('*S0*.json'))]
    record={'version':'R4_V23_DERIVED_NONPIXEL_RECOVERY_V1','status':'DERIVED_NOT_ORIGINAL',
        'source_directory':str(source.resolve()), 'source_trajectory':source_record,
        'source_completion':file_record(source/'COMPLETE.json'), 'source_result':file_record(source/'result.json'),
        'identity_sha256':completion['identity_sha256'], 'compact_trajectory':file_record(target/'trajectory.compact.npz'),
        'removed_keys':['raw_pixels'] if pixel_metadata else [], 'retained_keys':sorted(arrays),
        'raw_pixel_metadata':pixel_metadata, 'case_window_entry':case_entry,
        'reconstruction':{
            'implementation':'r4.replay.ReplayBranch with exact source case window and raw_actions',
            'reset':'fresh official World; original reset seed; original dataset state/reset callables exactly once',
            'frame_0':'source dataset planning input pixel after reset, not an assumed physical rerender',
            'frame_i_positive':'actual observation render after raw_actions[i-1]; same renderer/environment locks',
            'verify':'Every reconstructed frame must match raw_pixel_sha256. A mismatch is a reconstruction failure, never replaced by tolerance.',
            'dmc_last':'Stop at recorded terminal LAST; never step after LAST because dm_control auto-resets.',
            'capability':'S0 receipts describe evidence only; no assertion that untested frames have already been regenerated.',
            'source_logs_retained':True},
        's0_receipts':receipts,'derivation_code_sha256':sha256(__file__)}
    atomic_json(target/'DERIVATION.json',record)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--assets');p.add_argument('--s0');p.add_argument('--limit',type=int);a=p.parse_args()
    root=Path(a.source); folders=sorted({path.parent for path in root.glob('**/COMPLETE.json') if (path.parent/'trajectory.npz').exists()})
    if a.limit:folders=folders[:a.limit]
    rows=[]
    for folder in folders:
        record=compact_case(folder,Path(a.output)/folder.relative_to(root),assets=a.assets,replay_receipts=a.s0)
        rows.append({'relative_source':str(folder.relative_to(root)), 'source':record['source_trajectory'],'compact':record['compact_trajectory']})
    result={'version':'R4_V23_DERIVED_NONPIXEL_RECOVERY_V1','completed_cases':len(rows),'source_bytes':sum(r['source']['bytes'] for r in rows),'compact_npz_bytes':sum(r['compact']['bytes'] for r in rows),'rows':rows,'source_logs_retained':True}
    atomic_json(Path(a.output)/'RECOVERY_INDEX.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
