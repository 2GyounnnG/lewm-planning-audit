"""Byte-complete recovery with an explicit, unrepaired strict CPU failure."""
import argparse,datetime
from . import core

def seal(bundle,extras=()):
    bundle=core.Path(bundle).resolve();base=bundle.parent;manifest=core.read(bundle/'RECOVERY_MANIFEST.json');task=manifest['task'];cpu_folder=base/(task+'_CPU_RECOVERY');cpu=core.read(cpu_folder/'CPU_RECOVERY_STATUS.json')
    if cpu['status']!='FAILED_FROZEN_CPU_TOLERANCE' or cpu['cases']!=100 or cpu['arms']!=4 or cpu['new_optimizer_updates']!=0:raise RuntimeError('A recorded strict CPU failure is required')
    records={}
    def add(p):
        p=core.Path(p).resolve()
        if p.is_file():records[str(p)]=core.file_record(p)
    for rel,r in manifest['files'].items():
        p=bundle/rel;core.verify(dict(r,path=str(p)));add(p)
    add(bundle/'RECOVERY_MANIFEST.json');add(base/'RECOVERY_ARCHIVE.json')
    for folder in [cpu_folder,base/'MAC_OUTLIER_DIAGNOSTIC'/(task+'_CPU_RECOVERY'),base/'LINUX_CPU_RECOVERY',*[core.Path(p) for p in extras]]:
        if folder.exists():
            for p in folder.rglob('*'):add(p)
    for name in ('recovery_cpu.py','mac_outliers.py','limited_seal.py'):add(core.Path(__file__).parent/name)
    previous=base/'RECOVERY_SEAL.json';old=core.read(previous) if previous.exists() else None
    if old:
        for path,r in old['local_files'].items():
            core.verify(r)
            if path in records and records[path]['sha256']!=r['sha256']:raise RuntimeError('Existing evidence changed; never delete earlier SHA')
            records[path]=r
    revision=old['revision']+1 if old else 1
    result={'status':'COMPLETE_WITH_CPU_TOLERANCE_FAILURE','task':task,'revision':revision,'strict_cpu_gate_passed':False,'release_gate':'HOLD',
        'reason':'FROZEN_CPU_REPLAY_TOLERANCE_EXCEEDED','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_files':records,
        'CPU_recovery':core.file_record(cpu_folder/'CPU_RECOVERY_STATUS.json'),'CPU_scope':cpu['scope'],'new_optimizer_updates':0,'new_case_selection':False,
        'primary_scientific_values_modified':False,'tolerance_relaxed':False,'failed_receipts_preserved':True,'full_training_resume_environment_recovered':False,
        'full_original_HDF5_or_latent_cache_local':False,'remote_rebuild_identity':core.read(bundle/'manifests'/f'{task}_source_receipt.json'),
        'remote_cache_manifest_sha256':core.sha(bundle/'manifests'/f'{task}_cache.json'),
        'CPU_diagnostics':[core.file_record(p) for p in [base/'MAC_OUTLIER_DIAGNOSTIC'/(task+'_CPU_RECOVERY')/'DIAGNOSTIC_RECEIPT.json',base/'LINUX_CPU_RECOVERY'/(task+'_CPU_RECOVERY')/'CPU_RECOVERY_STATUS.json'] if p.exists()],
        'previous_seal':core.file_record(base/f'RECOVERY_SEAL_v{revision-1}.json') if old else None}
    core.freeze(base/f'RECOVERY_SEAL_v{revision}.json',result);core.atomic(previous,result)
    print(core.canonical({'status':result['status'],'release_gate':'HOLD','task':task,'revision':revision,'files':len(records),'seal':core.file_record(previous)}).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--extra',action='append',default=[]);a=p.parse_args();seal(a.bundle,a.extra)
