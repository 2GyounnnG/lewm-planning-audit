"""Start Cube intake as soon as the separately running verified download completes."""
import fcntl,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from . import core

def main():
    if core.ROOT!=Path('/workspace/x1_cube'):raise RuntimeError('Cube output root required')
    state=core.ROOT/'state';state.mkdir(parents=True,exist_ok=True)
    lock=(state/'cube_pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assets=Path('/workspace/shared_data/x1_assets');receipt=assets/'cube/datasets_download.json'
    def status(phase,**kwargs):
        value={'status':phase,'pid':os.getpid(),'time':time.time(),**kwargs};core.atomic(state/'cube_pipeline.json',value);print(json.dumps(value),flush=True)
    def call(label,args):
        status(label,args=args)
        with (core.ROOT/'logs'/(label+'.log')).open('a') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    status('WAITING_VERIFIED_CUBE_DOWNLOAD')
    while not receipt.exists():time.sleep(30)
    record=core.read(receipt)
    if record['status']!='VERIFIED' or record['revision']!=core.CONFIG['cube']['data_revision']:raise RuntimeError('Cube download not verified/pinned')
    entries=list(record['files'].values())
    if len(entries)!=1 or entries[0]['sha256']!=core.CONFIG['cube']['archive_sha256']:raise RuntimeError('Cube archive SHA differs')
    source=assets/entries[0]['path'];status('VERIFY_ARCHIVE',bytes=entries[0]['bytes'])
    core.verify({**entries[0],'path':str(source)})
    import zstandard
    with source.open('rb') as f,zstandard.ZstdDecompressor().stream_reader(f) as stream,tarfile.open(fileobj=stream,mode='r|') as archive:
        member=next((m for m in archive if m.isfile() and m.name.endswith('.h5')),None)
        if member is None:raise RuntimeError('Cube verified tar has no HDF5')
        info={'member':member.name,'expanded_bytes':member.size,'scratch':'/dev/shm/x1_source_unpacked','scratch_free_bytes':shutil.disk_usage('/dev/shm').free,
            'persistent_free_bytes':shutil.disk_usage('/workspace').free,'reserve_bytes':30<<30,'archive_sha256':entries[0]['sha256']}
    core.freeze(core.ROOT/'manifests/CUBE_STORAGE_PLAN.json',info)
    if info['scratch_free_bytes']<member.size+(30<<30) or info['persistent_free_bytes']<(30<<30):raise RuntimeError('Cube scratch/persistent reserve insufficient')
    status('STORAGE_PREFLIGHT_PASS',**info)
    unpacked=Path('/dev/shm/x1_source_unpacked/cube/UNPACKED.json')
    if not unpacked.exists():call('unpack',['/workspace/r4_v23_execution/ops/unpack_verified.py','--root',str(assets),'--task','cube','--output-root','/dev/shm/x1_source_unpacked'])
    call('intake',['-m','x1.intake','cube','--unpacked-root','/dev/shm/x1_source_unpacked'])
    schema=core.read(core.ROOT/'manifests/cube_SCHEMA.json')
    # New explicit metadata must be audited rather than guessed by this coordinator.
    if schema['family_columns']:raise RuntimeError('Explicit Cube family/demo fields require source audit before role freeze: '+str(schema['family_columns']))
    call('freeze_data',['-m','x1.runner','freeze-data','cube','--h5',schema['source']['path'],'--source-receipt',str(core.ROOT/'manifests/cube_source_receipt.json')])
    call('line',['-m','x1.line','cube'])
    call('tech_report',['-m','x1.tech_report','cube'])
    status('MAIN_COMPLETE_WAIT_ROOT_DELIVERY_FOR_SECOND_BATCH')

if __name__=='__main__':main()
