"""Decode only authenticated public R3 data, guarding space and archive paths."""
from pathlib import Path, PurePosixPath
import argparse, fcntl, hashlib, json, math, os, shutil, subprocess, tarfile, time

ROOT=Path(__file__).resolve().parents[1]
FLOOR=30*(1<<30)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def space(n=0):
    if shutil.disk_usage(ROOT).free < FLOOR+n: raise RuntimeError('30 GiB reserved free space would be violated')

def commit(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    with q.open('w') as f:
        f.write(json.dumps(data,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':str(value)}
    if isinstance(value,list):return [json_safe(x) for x in value]
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    return value

def checked_path(p):
    """Do not follow existing symlink ancestors into another dataset/project."""
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    rel=p.relative_to(ROOT)
    if not rel.parts or '..' in rel.parts:raise RuntimeError('Unpack path escaped R3 root')
    current=ROOT
    for part in rel.parts:
        current=current/part
        if current.is_symlink():raise RuntimeError('Unpack path contains a symlink: '+str(current))
    return p

def member_receipt_path(p):
    rel=str(checked_path(p).relative_to(ROOT))
    return ROOT/'state/unpack_members'/(hashlib.sha256(rel.encode()).hexdigest()+'.json')

def _stream_digest(src,expected=None):
    n=0;h=hashlib.sha256()
    while True:
        b=src.read(8<<20)
        if not b:break
        n+=len(b);h.update(b)
    if expected is not None and n!=expected:raise RuntimeError('Archive member size differs')
    return n,h.hexdigest()

def _read_exact(src,n):
    chunks=[];remaining=n
    while remaining:
        b=src.read(remaining)
        if not b:break
        chunks.append(b);remaining-=len(b)
    return b''.join(chunks)

def _validate_decoded(p,record):
    checked_path(p)
    if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:
        raise RuntimeError('Registered unpacked member differs: '+str(p))

def _completed_member(p,receipt,record):
    return {'path':str(p.relative_to(ROOT)),'bytes':record['bytes'],'sha256':record['sha256'],
            'member_receipt':str(receipt.relative_to(ROOT)),'member_receipt_sha256':sha(receipt)}

def copy_stream(src,p,expected=None,*,archive_identity,member_name,stream_validator=None):
    """Resume only a registered output, comparing every retained prefix byte.

    Registration precedes the first write; READY precedes rename. Therefore a
    crash before/after rename, or in later schema inspection, is recoverable.
    Existing unregistered files/partials are never adopted, truncated or removed.
    The caller holds the task lock and authenticates the compressed archive.
    """
    p=checked_path(p);temp=checked_path(p.with_suffix(p.suffix+'.unpack.tmp'))
    if expected is not None and (type(expected) is not int or expected<0):raise ValueError('Invalid member byte count')
    if (not isinstance(archive_identity,dict) or set(('path','bytes','sha256'))-set(archive_identity)
            or type(archive_identity['bytes']) is not int or archive_identity['bytes']<0
            or len(archive_identity['sha256'])!=64):raise ValueError('Authenticated archive identity required')
    checked_path(archive_identity['path'])
    identity={'archive':archive_identity,'member_name':member_name,'expected_bytes':expected,
              'output_path':str(p.relative_to(ROOT)),'temporary_path':str(temp.relative_to(ROOT))}
    receipt=checked_path(member_receipt_path(p))
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record.get('version')!='R3_UNPACK_MEMBER_V1' or record.get('identity')!=identity:
            raise RuntimeError('Existing member receipt has a different archive/member identity')
    else:
        if p.exists() or temp.exists():raise RuntimeError('Unregistered existing unpack output retained: '+str(p))
        record={'version':'R3_UNPACK_MEMBER_V1','status':'COPYING','identity':identity}
        commit(receipt,record)
    if record['status'] in ('READY_TO_INSTALL','COMPLETE'):
        if p.exists() and temp.exists():raise RuntimeError('Both registered final and partial exist; preserve for inspection')
        existing=p if p.exists() else temp
        if record['status']=='COMPLETE' and existing!=p:raise RuntimeError('Completed member is missing; do not silently recreate it')
        _validate_decoded(existing,record)
        # A reused direct .zst stream must be drained too, or zstd.wait can hang.
        n,digest=_stream_digest(src,expected)
        if stream_validator is not None:stream_validator()
        if n!=record['bytes'] or digest!=record['sha256']:raise RuntimeError('Archive/member content no longer matches registered output')
        if record['status']=='READY_TO_INSTALL':
            if existing==temp:os.replace(temp,p)
            record['status']='COMPLETE';commit(receipt,record)
        return _completed_member(p,receipt,record)
    if record['status']!='COPYING':raise RuntimeError('Unknown member recovery status')
    if p.exists():raise RuntimeError('Final file exists without READY/COMPLETE receipt; retained unchanged')
    if temp.exists() and not temp.is_file():raise RuntimeError('Registered partial is not a regular file')
    prefix=temp.stat().st_size if temp.exists() else 0
    if expected is not None and prefix>expected:raise RuntimeError('Partial is larger than the pinned member')
    space(max(0,expected-prefix) if expected is not None else 0)
    p.parent.mkdir(parents=True,exist_ok=True)
    n=0;h=hashlib.sha256()
    if prefix:
        with temp.open('rb') as old:
            while True:
                saved=old.read(8<<20)
                if not saved:break
                decoded=_read_exact(src,len(saved))
                if decoded!=saved:raise RuntimeError('Own partial prefix differs from authenticated archive; retained unchanged')
                n+=len(saved);h.update(saved)
    with temp.open('ab' if temp.exists() else 'xb') as dst:
        while True:
            data=src.read(8<<20)
            if not data:break
            if expected is not None and n+len(data)>expected:raise RuntimeError('Decoded member exceeds pinned byte count')
            space(len(data));dst.write(data);n+=len(data);h.update(data)
        dst.flush();os.fsync(dst.fileno())
    if expected is not None and n!=expected:raise RuntimeError('Archive member size differs')
    # For raw .h5.zst, only the decoder exit status authenticates EOF/length.
    if stream_validator is not None:stream_validator()
    record.update(status='READY_TO_INSTALL',bytes=n,sha256=h.hexdigest())
    commit(receipt,record)
    if p.exists():raise RuntimeError('Output appeared during unpack; retained unchanged')
    os.replace(temp,p);record['status']='COMPLETE';commit(receipt,record)
    return _completed_member(p,receipt,record)

def inspect_h5(p):
    import h5py, hdf5plugin, numpy as np
    out={'path':str(p.relative_to(ROOT)),'datasets':{}}
    with h5py.File(p,'r') as f:
        for k in f.keys():
            a=f[k]
            if not isinstance(a,h5py.Dataset):
                out['datasets'][k]={'type':'group','keys':list(a.keys())};continue
            rec={'shape':list(a.shape),'dtype':str(a.dtype),'compression':a.compression,'chunks':a.chunks}
            if a.size and k!='pixels':
                first=np.asarray(a[0]);last=np.asarray(a[-1]);rec['first']=first.tolist();rec['last']=last.tolist()
                if first.dtype.kind in 'SUO': rec['first']=str(first);rec['last']=str(last)
            out['datasets'][k]=rec
        for k in ('ep_len','ep_offset'):
            if k in f:
                v=np.asarray(f[k][:]);out[k]={'count':len(v),'min':int(v.min()),'max':int(v.max()),'sum':int(v.sum()),'first10':v[:10].tolist()}
    return json_safe(out)

def _run_locked(task,wait):
    receipt=ROOT/'manifests'/f'{task}_data_assets.json'
    while not receipt.exists():
        if not wait:raise FileNotFoundError(receipt)
        space();time.sleep(15)
    d=json.loads(receipt.read_text()); outputs=[]; members=[]
    done=ROOT/'manifests'/f'{task}_data_unpacked.json'
    if done.exists():
        old=json.loads(done.read_text())
        if old['source_receipt_sha256']!=sha(receipt):raise RuntimeError('Original data source receipt changed')
        for rec in old['files']:
            _validate_decoded(checked_path(rec['path']),rec)
            member=checked_path(rec['member_receipt'])
            if sha(member)!=rec['member_receipt_sha256']:raise RuntimeError('Completed per-member receipt changed')
            saved=json.loads(member.read_text())
            if saved['status']!='COMPLETE' or saved['bytes']!=rec['bytes'] or saved['sha256']!=rec['sha256']:
                raise RuntimeError('Incomplete member receipt in completed dataset')
        return old
    start=time.time()
    for name,rec in d['files'].items():
        if not name.endswith('.zst'):continue
        archive=checked_path(rec['path'])
        if archive.stat().st_size!=rec['bytes'] or sha(archive)!=rec['sha256']:raise RuntimeError('Authenticated archive differs')
        space()
        z=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE)
        archive_identity={k:rec[k] for k in ('path','bytes','sha256')}
        def decoder_finished():
            if z.wait()!=0:raise RuntimeError('Zstd decompression failed; partial retained for verified resume')
        try:
            if name.endswith('.tar.zst'):
                with tarfile.open(fileobj=z.stdout,mode='r|') as tar:
                    for m in tar:
                        p=PurePosixPath(m.name)
                        if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk():raise RuntimeError('Unsafe archive member')
                        members.append({'name':m.name,'bytes':m.size,'is_file':m.isfile()})
                        if not m.isfile():continue
                        if p.suffix not in ('.h5','.hdf5','.json','.md','.txt'):continue
                        outputs.append(copy_stream(tar.extractfile(m),ROOT/'data/unpacked'/task/p,m.size,
                                                   archive_identity=archive_identity,member_name=m.name))
                # Drain any tar padding/trailer before waiting on the producer.
                while z.stdout.read(8<<20):pass
            else:
                outputs.append(copy_stream(z.stdout,ROOT/'data/unpacked'/task/name[:-4],
                                           archive_identity=archive_identity,member_name=name[:-4],
                                           stream_validator=decoder_finished))
            decoder_finished()
        except BaseException:
            if z.poll() is None:z.terminate()
            z.wait();raise
        finally:z.stdout.close()
    schemas=[inspect_h5(ROOT/r['path']) for r in outputs if Path(r['path']).suffix in ('.h5','.hdf5')]
    result={'status':'SOURCE_UNPACKED_SHA_VERIFIED','task':task,'source_receipt_sha256':sha(receipt),
            'files':outputs,'archive_members':members,'schemas':schemas,'seconds':time.time()-start,
            'source_archives_retained_shared_once':True,'optimizer_updates':0,'data_selection_by_risk':False}
    commit(done,result);print(json.dumps(result),flush=True);return result

def run(task,wait):
    if task not in ('pusht','reacher'):raise ValueError('Unknown task')
    p=checked_path(ROOT/'state'/(task+'_unpack.lock'));p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_locked(task,wait)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['pusht','reacher']);p.add_argument('--wait',action='store_true')
    a=p.parse_args();run(a.task,a.wait)
