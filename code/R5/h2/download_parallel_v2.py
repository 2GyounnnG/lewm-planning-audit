"""Two independent SSH connections, unique archive queue, verified partial resume."""
import concurrent.futures,datetime,fcntl,hashlib,json,queue,shlex,subprocess,threading,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
DEST=BASE/'recovery/archives'
KNOWN=BASE.parents[1]/'r4_v23_execution/ops/known_hosts'
guard=threading.Lock();active={};verified={}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def status():
    with guard:
        d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active':dict(active),'verified':dict(verified)}
        p=DEST/'PARALLEL_DOWNLOAD_STATUS.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def main():
    todo=queue.Queue();receipts=sorted(DEST.glob('*_COMPACT_V1_ARCHIVE.json'));assert len(receipts)==8
    records={}
    for p in receipts:
        r=json.loads(p.read_text())['archive'];name=Path(r['path']).name;records[name]=r
        local=DEST/name
        if local.exists() and local.stat().st_size==r['bytes'] and sha(local)==r['sha256']:verified[name]=r
        else:todo.put(r)
    control=DEST/'DOWNLOAD_PARALLELISM.json'
    if not control.exists():control.write_text('{"workers":2}\n')
    ssh=shlex.join(['ssh','-c','aes128-gcm@openssh.com','-o','ControlMaster=no','-o','ControlPath=none','-p','46720','-o','UserKnownHostsFile='+str(KNOWN)])
    start=time.monotonic()
    def received():
        total=0
        for name,r in records.items():
            paths=[DEST/name,*DEST.glob('.'+name+'.*')];total+=max([p.stat().st_size for p in paths if p.is_file()]+[0])
        return total
    baseline=received();status()
    def worker(index):
        while True:
            if index==1 and json.loads(control.read_text())['workers']<2:return
            try:r=todo.get_nowait()
            except queue.Empty:return
            name=Path(r['path']).name;p=DEST/name
            if r['path']!='/workspace/r5/H2/compact_archives/'+name:raise RuntimeError('Unexpected mapped archive')
            with (DEST/(name+'.lock')).open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with guard:active[str(index)]={'archive':name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                status()
                for attempt in range(1,6):
                    with (DEST/f'parallel_worker_{index}.log').open('ab') as log:
                        code=subprocess.run(['rsync','-rt','--partial','--progress','--timeout=120','-e',ssh,'root@211.72.13.202:'+r['path'],str(p)],stdout=log,stderr=subprocess.STDOUT).returncode
                    if code==0:
                        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise RuntimeError('Archive SHA mismatch: '+name)
                        break
                    if attempt==5:raise RuntimeError('Five transport failures; partial retained: '+name)
                    time.sleep(5)
                with guard:verified[name]=r;active.pop(str(index),None)
                status();todo.task_done();print(json.dumps({'status':'ARCHIVE_SHA_VERIFIED','worker':index,'archive':name,'bytes':r['bytes']}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,i) for i in range(2)]
        while not all(j.done() for j in jobs):
            d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'received_total_bytes':received(),'baseline_bytes':baseline,'total_bytes':sum(r['bytes'] for r in records.values())}
            d['incremental_MBps']=(d['received_total_bytes']-baseline)/max(d['elapsed_seconds'],1)/1e6
            with (DEST/'TRANSFER_PROFILE_V2.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
            time.sleep(10)
        for j in jobs:j.result()
    if len(verified)!=8:raise RuntimeError('Unique queue not fully drained')
    (DEST/'DOWNLOAD_COMPLETE_V2.json').write_text(json.dumps({'status':'ALL8_ARCHIVE_SHA_VERIFIED','archives':verified,'full_raw_RGB_local_recovery_claimed':False},indent=2)+'\n')

if __name__=='__main__':main()
