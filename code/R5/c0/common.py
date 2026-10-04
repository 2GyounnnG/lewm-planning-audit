from __future__ import annotations
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get('R5_C0_ROOT', '/workspace/r5/C0'))
R3 = Path(os.environ.get('R5_R3_ROOT', '/workspace/shared_data/r3'))
X1 = Path(os.environ.get('R5_X1_ROOT', '/workspace/x1_cube'))

def canonical(x): return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def record(p):
    p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p): return json.loads(Path(p).read_text())
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:f.write(canonical(x)+b'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def freeze(p,x):
    if Path(p).exists():
        if read(p)!=x:raise RuntimeError('Frozen identity differs: '+str(p))
    else:atomic(p,x)
def verify(r):
    p=Path(r['path'])
    if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise RuntimeError('Identity mismatch: '+str(p))
    return p
def save_npz(p,**x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**x);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def csv_write(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(p)
def comparisons(a,b):
    rows=[]
    for key in sorted(set(a)|set(b)):
        if key not in a or key not in b:raise RuntimeError('Snapshot schema changed: '+key)
        x,y=np.asarray(a[key]),np.asarray(b[key]);schema=x.dtype==y.dtype and x.shape==y.shape
        finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
        exact=schema and finite and x.tobytes()==y.tobytes()
        delta=np.abs(x.astype(np.float64)-y.astype(np.float64)) if schema else np.array([np.inf])
        rows.append({'field':key,'dtype':str(x.dtype),'shape':str(list(x.shape)),'elements':int(x.size),
                     'schema_equal':schema,'finite':finite,'bitwise_equal':exact,
                     'max_abs_difference':float(delta.max()) if delta.size else 0.0,
                     'different_elements':int(np.count_nonzero(x!=y)) if schema else -1})
    return rows
