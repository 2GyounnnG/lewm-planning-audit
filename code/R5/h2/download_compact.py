"""Single bulk stream, resumable partials, actual SHA acceptance per archive."""
import subprocess,json,hashlib,time,shlex,datetime
from pathlib import Path

BASE=Path(__file__).resolve().parent
DEST=BASE/'recovery/archives'
HOST='root@211.72.13.202'
KNOWN=BASE.parents[1]/'r4_v23_execution/ops/known_hosts'
REMOTE='/workspace/r5/H2/compact_archives/'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def save(name,d):
    temp=DEST/(name+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(DEST/name)
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    control=shlex.join(['ssh','-S','/private/tmp/r5_control','-p','46720','-o','UserKnownHostsFile='+str(KNOWN)])
    bulk=shlex.join(['ssh','-c','aes128-gcm@openssh.com','-o','ControlMaster=no','-o','ControlPath=none','-p','46720','-o','UserKnownHostsFile='+str(KNOWN)])
    subprocess.run(['rsync','-rtz','--partial','-e',control,'--include=*_ARCHIVE.json','--include=*_MANIFEST.json','--exclude=*',HOST+':'+REMOTE,str(DEST)+'/'],check=True)
    receipts=sorted(DEST.glob('*_COMPACT_V1_ARCHIVE.json'))
    if len(receipts)!=8:raise RuntimeError('Need all eight completed compact receipts before bulk recovery')
    recovered=[]
    for rp in receipts:
        d=json.loads(rp.read_text());record=d['archive'];name=Path(record['path']).name;p=DEST/name
        manifest=DEST/Path(d['manifest']['path']).name
        assert sha(manifest)==d['manifest']['sha256'] and manifest.stat().st_size==d['manifest']['bytes']
        if record['path']!=REMOTE+name or not name.endswith('_COMPACT_V1.tar.gz'):raise RuntimeError('Unexpected archive mapping')
        for attempt in range(1,6):
            if p.exists() and p.stat().st_size==record['bytes'] and sha(p)==record['sha256']:break
            save('DOWNLOAD_STATUS.json',{'status':'RUNNING','archive':name,'expected_bytes':record['bytes'],'attempt':attempt,'already_verified':recovered,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            with (DEST/'transfer.log').open('ab') as log:
                r=subprocess.run(['rsync','-rt','--partial','--progress','--timeout=120','-e',bulk,HOST+':'+record['path'],str(p)],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode==0:
                if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise RuntimeError('Transferred archive SHA differs: '+name)
                break
            if attempt==5:raise RuntimeError('Transport failed five attempts, partial preserved: '+name)
            time.sleep(5)
        recovered.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'manifest_sha256':d['manifest']['sha256']})
        print(json.dumps({'status':'ARCHIVE_SHA_VERIFIED','archive':name,'bytes':p.stat().st_size}),flush=True)
    save('DOWNLOAD_COMPLETE.json',{'status':'ALL8_ARCHIVE_SHA_VERIFIED','archives':recovered,'full_raw_RGB_local_recovery_claimed':False})
    save('DOWNLOAD_STATUS.json',{'status':'COMPLETE','archives':8,'verified_bytes':sum(r['bytes'] for r in recovered)})

if __name__=='__main__':main()
