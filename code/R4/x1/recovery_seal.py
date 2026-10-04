"""Local per-file recovery seal with a retained revision chain."""
import argparse,datetime
from . import core

def seal(bundle,extra=None):
    bundle=core.Path(bundle).resolve();manifest=core.read(bundle/'RECOVERY_MANIFEST.json');task=manifest['task']
    cpufolder=bundle.parent/(task+'_CPU_RECOVERY');cpu=core.read(cpufolder/'CPU_RECOVERY_STATUS.json')
    if cpu['status']!='PASS' or cpu['cases']!=100 or cpu['arms']!=4 or cpu['new_optimizer_updates']!=0:raise RuntimeError('Fixed100 four-arm CPU check must pass')
    records={}
    for rel,rec in manifest['files'].items():
        p=bundle/rel;core.verify(dict(rec,path=str(p)));records[str(p)]=core.file_record(p)
    for p in [bundle/'RECOVERY_MANIFEST.json',*cpufolder.iterdir()]:
        if p.is_file():records[str(p)]=core.file_record(p)
    if extra:
        for p in sorted(core.Path(extra).resolve().rglob('*')):
            if p.is_file():records[str(p)]=core.file_record(p)
    previous=bundle.parent/'RECOVERY_SEAL.json';old=None
    if previous.exists():
        old=core.read(previous)
        for path,rec in old['local_files'].items():
            core.verify(rec)
            if path in records and records[path]['sha256']!=rec['sha256']:raise RuntimeError('Existing recovered SHA changed')
            records[path]=rec
    revision=1 if old is None else old['revision']+1
    result={'status':'PASS','task':task,'revision':revision,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'local_files':records,'CPU_recovery':core.file_record(cpufolder/'CPU_RECOVERY_STATUS.json'),'CPU_scope':cpu['scope'],
        'full_training_resume_environment_recovered':False,'full_original_HDF5_or_latent_cache_local':False,
        'remote_rebuild_identity':core.read(bundle/'manifests'/f'{task}_source_receipt.json'),'remote_cache_manifest_sha256':core.sha(bundle/'manifests'/f'{task}_cache.json'),
        'new_optimizer_updates':0,'new_case_selection':False,'previous_seal':core.file_record(bundle.parent/f'RECOVERY_SEAL_v{revision-1}.json') if old else None}
    core.freeze(bundle.parent/f'RECOVERY_SEAL_v{revision}.json',result);core.atomic(previous,result)
    print(core.canonical({'status':'PASS','task':task,'revision':revision,'files':len(records),'seal':core.file_record(previous)}).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--extra');a=p.parse_args();seal(a.bundle,a.extra)
