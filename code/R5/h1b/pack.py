"""Compact exact-byte recovery, with explicit excluded large resumable inputs."""
import argparse,tarfile
from pathlib import Path
from common import *

def run(a):
    root=Path(a.root);out=root/'recovery';out.mkdir(parents=True,exist_ok=True);manifest=out/f'{a.task}_RECOVERY_MANIFEST.json';dest=out/f'{a.task}_probe_recovery_v1.tar.gz'
    if dest.exists():return
    members=[];excluded=[]
    for p in sorted((root/'fits'/a.task).rglob('*')):
        if not p.is_file():continue
        rec={'relative_path':str(p.relative_to(root)),**file(p)}
        if p.name=='resume.pt':excluded.append({**rec,'reason':'Optimizer/current state retained remotely; exact selected best.pt recovered. No input or original source is deleted.'})
        elif p.suffix in ('.json','.csv','.npz','.pt'):members.append(rec)
    assert members and all((p.parent/'COMPLETE.json').exists() for p in (root/'fits'/a.task).glob('**/fit.json'))
    for p in [root/'matrices'/a.task/'manifest.json',root/'INPUT_AUDIT.json',root/'STARTED.json']:
        members.append({'relative_path':str(p.relative_to(root)),**file(p)})
    for p in (root/'reports'/a.task).glob('*'):
        if p.is_file():members.append({'relative_path':str(p.relative_to(root)),**file(p)})
    atomic(manifest,{'status':'EXACT_SELECTED_FIT_AND_EVALUATION_RECOVERY','task':a.task,'members':members,'excluded_remote_only':excluded,'TRAIN_matrices_scope':'Large all-TRAIN matrices remain remotely, their exact hashes and all source episode/cache hashes are recovered in matrices/task/manifest.json. Recovery CPU evaluation uses exact saved EVAL features/labels and selected fit parameters, no training or encoder inference.','local_CPU_scope':'Every recovered selected probe, every original100 EVAL anchors; no new simulator steps/world-model updates/probe fits.'})
    tmp=dest.with_name(dest.name+'.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=3) as tf:
        for r in members:tf.add(root/r['relative_path'],arcname=r['relative_path'],recursive=False)
        tf.add(manifest,arcname=str(manifest.relative_to(root)),recursive=False)
    tmp.rename(dest);atomic(out/f'{a.task}_ARCHIVE.json',{'status':'COMPLETE','archive':file(dest),'manifest':file(manifest),'uncompressed_member_bytes':sum(r['bytes'] for r in members),'members':len(members),'optimizer_checkpoints_retained_remote':len(excluded)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task',required=True);run(p.parse_args())
