"""Summarize the authorized alternate800 and seal only the new raw evidence."""
import shutil,subprocess,sys,tarfile,time
from . import core

if __name__=='__main__':
    root=core.ROOT;gate=root/'state/SECOND_BATCH_COMPLETE.json'
    if root!=core.Path('/workspace/x1_tworoom'):raise RuntimeError('TwoRoom only')
    while not gate.exists():time.sleep(10)
    if core.read(gate)['status']!='COMPLETE':raise RuntimeError('Alternate800 incomplete')
    subprocess.run([sys.executable,'-m','x1.randomness','report'],check=True)
    out=root/'recovery';stage=out/'second_bundle';stage.mkdir(exist_ok=True);records={}
    paths=[]
    for stream in ('R4_ALT_CEM_1','R4_ALT_CEM_2'):paths.extend((root/'closed_loop/EVAL'/stream).rglob('*'))
    paths.extend((root/'reports/tworoom_RANDOMNESS').rglob('*'))
    paths.extend(root/p for p in ('SECOND_BATCH_GATE.json','GLOBAL_MAIN_BATCH_DELIVERED.json','state/SECOND_BATCH_SCOPE.json','state/SECOND_BATCH_COMPLETE.json'))
    paths.extend((root/'logs').glob('alternate_*.log'))
    for source in sorted(paths):
        if not source.is_file():continue
        rel=str(source.relative_to(root));target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True);rec=core.file_record(source)
        if target.exists():
            if core.sha(target)!=rec['sha256']:raise RuntimeError('Second recovery snapshot changed')
        else:shutil.copy2(source,target)
        records[rel]={'original':rec,'bytes':target.stat().st_size,'sha256':core.sha(target)}
    core.freeze(stage/'RECOVERY_INCREMENT_MANIFEST.json',{'task':'tworoom','scope':'Only additional800 trajectories, frozen planner-stream statistics and gates; original400/weights remain in main recovery','new_optimizer_updates':0,'additional_trajectories':800,'files':records,'main_recovery_manifest':core.file_record(root/'recovery/bundle/RECOVERY_MANIFEST.json')})
    archive=out/'tworoom_second.tar.gz'
    if not archive.exists():
        temp=archive.with_suffix('.tmp')
        with tarfile.open(temp,'w:gz',compresslevel=1) as tar:tar.add(stage,arcname='tworoom_second')
        temp.replace(archive)
    core.freeze(out/'SECOND_RECOVERY_ARCHIVE.json',{'status':'COMPLETE','archive':core.file_record(archive),'manifest':core.file_record(stage/'RECOVERY_INCREMENT_MANIFEST.json'),'files':len(records)})
    print('SECOND_RANDOMNESS_AND_RECOVERY_COMPLETE',flush=True)
