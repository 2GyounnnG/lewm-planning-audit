"""Unpack one SHA-verified public dataset without path traversal or symlinks."""
import argparse, hashlib, json, os, pathlib, shutil, tarfile, time
import zstandard

def run(root,task,output_root='/workspace/shared_data/unpacked'):
    root=pathlib.Path(root)
    receipt=json.loads((root/task/'datasets_download.json').read_text())
    assert receipt['status']=='VERIFIED'
    out=pathlib.Path(output_root)/task;out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,rec in receipt['files'].items():
        p=root/rec['path']
        assert p.stat().st_size==rec['bytes']
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(16<<20),b''):h.update(b)
        assert h.hexdigest()==rec['sha256']
        with p.open('rb') as src,zstandard.ZstdDecompressor().stream_reader(src) as stream:
            if name.endswith('.tar.zst'):
                archive=tarfile.open(fileobj=stream,mode='r|')
                items=((m.name,m.size,archive.extractfile(m)) for m in archive if m.isfile())
            else:items=[(name.removesuffix('.zst'),None,stream)]
            for member,size,handle in items:
                rel=pathlib.PurePosixPath(member)
                if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe archive path')
                target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if not member.endswith('.h5'):continue
                print(json.dumps({'status':'EXTRACT','task':task,'member':member,'bytes':size}),flush=True)
                if size and shutil.disk_usage(out).free < size+(30<<30):raise RuntimeError('DISK_CAPACITY_BLOCKED: preserve 30GiB reserve')
                temp=target.with_suffix(target.suffix+'.partial')
                if target.exists():
                    if size is not None and target.stat().st_size!=size:raise RuntimeError('Existing extraction size mismatch')
                    # Input digest remains authoritative; do not silently overwrite.
                    continue
                if temp.exists():raise RuntimeError('Incomplete extraction preserved; inspect before resuming')
                h=hashlib.sha256();written=0
                with temp.open('xb') as f:
                    while True:
                        b=handle.read(16<<20)
                        if not b:break
                        if shutil.disk_usage(out).free < len(b)+(30<<30):raise RuntimeError('DISK_CAPACITY_BLOCKED')
                        f.write(b);h.update(b);written+=len(b)
                    f.flush();os.fsync(f.fileno())
                if size is not None:assert written==size
                temp.replace(target);records.append({'path':str(target),'bytes':written,'sha256':h.hexdigest(),'archive_sha256':rec['sha256']})
                print(json.dumps(records[-1]),flush=True)
    (out/'UNPACKED.json').write_text(json.dumps({'task':task,'records':records,'status':'COMPLETE'},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task',required=True);p.add_argument('--output-root',default='/workspace/shared_data/unpacked');a=p.parse_args();run(a.root,a.task,a.output_root)
