"""Content-addressed delivery manifest, excluding redundant weight staging copies."""
import argparse,datetime,hashlib,json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','verify']);p.add_argument('root',type=Path);a=p.parse_args()
path=a.root/'R10_SEAL.json'
if a.mode=='seal':
    records={}
    for f in sorted(a.root.rglob('*')):
        rel=f.relative_to(a.root)
        if not f.is_file() or rel.parts[0]=='transfer' or f.name in ['.DS_Store','R10_SEAL.json','R10_MIRROR_VERIFICATION.json'] or '__pycache__' in rel.parts:continue
        records[str(rel)]=dict(bytes=f.stat().st_size,sha256=sha(f))
    seal=dict(version='R10_DELIVERY_SEAL_V1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              exclusions=['transfer/ redundant staging weights; exact source weight identities are bound in prereg/inputs/raw','R10_SEAL.json self','R10_MIRROR_VERIFICATION.json post-copy receipt','.DS_Store','__pycache__'],files=records)
    path.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),seal_sha256=sha(path),bytes=sum(x['bytes'] for x in records.values()))))
else:
    seal=json.loads(path.read_text())
    for rel,record in seal['files'].items():
        f=a.root/rel
        assert f.is_file() and f.stat().st_size==record['bytes'] and sha(f)==record['sha256'],rel
    print(json.dumps(dict(status='PASS',root=str(a.root),files=len(seal['files']),seal_sha256=sha(path))))
