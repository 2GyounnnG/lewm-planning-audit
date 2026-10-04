"""Queue the small alternate-stream archive behind the sole Cube bulk transfer."""
import subprocess,sys,tarfile,time
from . import core

if __name__=='__main__':
    root=core.Path(__file__).parent;cube=root/'recovery/cube';target=root/'recovery/tworoom';before=cube/'cube_main.tar.gz';expected=core.read(cube/'RECOVERY_ARCHIVE.json')['archive']
    while not before.exists() or before.stat().st_size!=expected['bytes']:time.sleep(10)
    core.verify(dict(expected,path=str(before)));receipt=core.read(target/'SECOND_RECOVERY_ARCHIVE.json')
    ssh='ssh -c aes128-gcm@openssh.com -o ControlMaster=no -o ControlPath=none -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -p 46720 -o UserKnownHostsFile=/Users/richwang/Documents/ChatGPT/热/r4_v23_execution/ops/known_hosts'
    for attempt in range(1,4):
        code=subprocess.run(['rsync','-rt','--partial','-e',ssh,'root@211.72.13.202:/workspace/x1_tworoom/recovery/tworoom_second.tar.gz',str(target)+'/']).returncode
        core.atomic(target/f'SECOND_TRANSPORT_{attempt}.json',{'returncode':code,'completed_at':time.time(),'optimizer_updates':0})
        if not code:break
        if attempt==3:raise RuntimeError('Second archive transport failed three times with retained partials')
        time.sleep(10*attempt)
    archive=target/'tworoom_second.tar.gz';core.verify(dict(receipt['archive'],path=str(archive)));folder=target/'tworoom_second'
    if not folder.exists():
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar.getmembers():
                p=core.Path(member.name)
                if p.is_absolute() or '..' in p.parts or p.parts[0]!='tworoom_second' or member.issym() or member.islnk():raise RuntimeError('Unsafe member')
            tar.extractall(target,filter='data')
    path=folder/'RECOVERY_INCREMENT_MANIFEST.json';core.verify(dict(receipt['manifest'],path=str(path)));manifest=core.read(path)
    if manifest['additional_trajectories']!=800 or manifest['new_optimizer_updates']!=0:raise RuntimeError('Incorrect increment scope')
    for rel,r in manifest['files'].items():core.verify(dict(r,path=str(folder/rel)))
    core.atomic(target/'SECOND_INCREMENT_LOCAL_VERIFIED.json',{'status':'PASS','additional_trajectories':800,'new_optimizer_updates':0,'archive':core.file_record(archive),'manifest':core.file_record(path)})
    subprocess.run([sys.executable,'-m','x1.limited_seal',str(target/'tworoom'),'--extra',str(folder),'--extra',str(root/'reports/tworoom_RANDOMNESS')],check=True)
    print('SECOND_INCREMENT_VERIFIED_AND_LIMITED_SEAL_UPDATED',flush=True)
