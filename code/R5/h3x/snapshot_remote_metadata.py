"""Snapshot finished H3X metadata/code and catalogue explicitly remote-only files."""
import json,os,shutil,tarfile
from pathlib import Path
from . import common
from .evaluate import file

def main():
    root=common.ROOT;out=root/'recovery/module_metadata';out.mkdir(parents=True,exist_ok=True)
    assert common.read_json(root/'SUPERVISOR_COMPLETE.json')['status']=='COMPLETE'
    originals=[]
    def copy(p,rel):
        q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        assert common.sha256(q)==common.sha256(p)
        originals.append({'original':file(p),'relative_path':rel,'copied':file(q)})
    for p in sorted(common.CODE.iterdir()):
        if p.is_file() and p.suffix in ('.py','.json','.md'):copy(p,'code/'+p.name)
    for rel in ('FORMAL_AUTHORIZATION.json','PRESTART_FAILURE_AUDIT.json','SUPERVISOR_STATUS.json','SUPERVISOR_COMPLETE.json','state/TECHNICAL_LEDGER.json','inputs/CACHE_AUDIT.json','inputs/REFERENCE_CONTROL_AUDIT.json'):
        p=root/rel
        if p.exists():copy(p,'run/'+rel)
    for s in common.SEEDS:
        copy(root/f'evaluation/closed_loop/CLOSED_COMPLETE_{s}.json',f'run/CLOSED_COMPLETE_{s}.json')
        copy(root/f'artifacts/train/H3X_reacher_s{s}/JOB_SHA256.json',f'run/JOB_SHA256_{s}.json')
    for p in sorted((root/'reports/training').iterdir()):
        if p.is_file():copy(p,'reports/training/'+p.name)
    # Includes original monitor cache/input identities without copying large TRAIN cache.
    remoteonly=[]
    for p in sorted((root/'inputs').rglob('*')):
        if p.is_file():remoteonly.append(file(p))
    for s in common.SEEDS:
        d=common.read_json(root/f'recovery/bundles/{s}/RECOVERY_MANIFEST.json')
        remoteonly.extend(d['remote_only_files'])
        remoteonly.extend(x['source_original'] for x in d['derived_trajectories'])
    common.atomic_json(out/'METADATA_MANIFEST.json',{'status':'COMPLETED_H3X_METADATA_SNAPSHOT','label':common.LABEL,'files':originals,'remote_only_originals':remoteonly,'scope':'Code/authorization/budget/source identities and full training raw table copied exactly; optimizer/intermediate weights/original full pixels/monitor cache retained at original remote paths. TRAIN cache remains read-only referenced by CACHE_AUDIT.'})
    dest=root/'recovery/H3X_METADATA.tar.gz'
    with tarfile.open(dest.with_suffix('.tmp'),'w:gz',compresslevel=1) as t:t.add(out,arcname='module_metadata')
    dest.with_suffix('.tmp').replace(dest)
    common.atomic_json(dest.with_suffix('.json'),{'archive':file(dest),'manifest':file(out/'METADATA_MANIFEST.json')})
    print(json.dumps({'archive':file(dest),'files':len(originals),'remote_only':len(remoteonly)}),flush=True)

if __name__=='__main__':main()
