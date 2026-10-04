"""One X1 bulk transfer at a time; preserve partials across transient transport errors."""
import subprocess,time
from . import core

if __name__=='__main__':
    recovery=core.Path(__file__).parent/'recovery';before=recovery/'tworoom';target=recovery/'cube';expected=core.read(before/'RECOVERY_ARCHIVE.json')['archive'];archive=before/'tworoom_main.tar.gz'
    while not archive.exists() or archive.stat().st_size!=expected['bytes']:time.sleep(10)
    core.verify(dict(expected,path=str(archive)))
    ssh='ssh -c aes128-gcm@openssh.com -o ControlMaster=no -o ControlPath=none -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -p 46720 -o UserKnownHostsFile=/Users/richwang/Documents/ChatGPT/热/r4_v23_execution/ops/known_hosts'
    for attempt in range(1,4):
        began=time.time();code=subprocess.run(['rsync','-rt','--partial','-e',ssh,'root@211.72.13.202:/workspace/x1_cube/recovery/cube_main.tar.gz',str(target)+'/']).returncode
        core.atomic(target/f'BULK_ATTEMPT_{attempt}.json',{'returncode':code,'started_at':began,'ended_at':time.time(),'preserve_partial':True,'additional_optimizer_updates':0})
        if code==0:
            core.verify(dict(core.read(target/'RECOVERY_ARCHIVE.json')['archive'],path=str(target/'cube_main.tar.gz')));print('CUBE_BULK_SHA_PASS',flush=True);break
        if attempt==3:raise RuntimeError('Three preserved-partial transport attempts failed')
        time.sleep(10*attempt)
