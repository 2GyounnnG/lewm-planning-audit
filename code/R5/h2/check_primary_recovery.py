"""Read-only primary SHA verification; writes only a new H2 check receipt."""
import argparse,datetime,hashlib,json
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise RuntimeError('Never overwrite primary check receipt')
    d=json.loads(a.manifest.read_text());fail=[]
    for r in d['remote_files']:
        p=Path(r['path'])
        if not p.is_file():fail.append({'path':str(p),'reason':'MISSING'});continue
        actual={'bytes':p.stat().st_size,'sha256':sha(p)}
        if (actual['bytes'],actual['sha256'])!=(r['bytes'],r['sha256']):fail.append({'path':str(p),'reason':'MISMATCH','actual':actual,'expected':r})
    result={'status':'PASS_ALL_PRIMARY_REFERENCED_BYTES' if not fail else 'FAIL_PRIMARY_REFERENCED_BYTES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':d['task'],'manifest_sha256':sha(a.manifest),'files_checked':len(d['remote_files']),'failures':fail,'producer_sha256':sha(__file__),'new_optimizer_updates':0,'new_simulator_steps':0}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
