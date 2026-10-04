"""H3X isolated authorization/accounting; original R3 trees stay read-only."""
from pathlib import Path
import contextlib, datetime, fcntl, hashlib, json, os, random, shutil, subprocess, tempfile
ROOT=Path(os.environ.get('H3X_ROOT','/workspace/r5/H3X')).resolve()
R3_ROOT=Path(os.environ.get('R3_ROOT','/workspace/shared_data/r3')).resolve()
CODE=Path(__file__).resolve().parent
TASKS=('reacher',)
SEEDS=(103201,103202,103203)
FLOOR=30*(1<<30)
LABEL='H3X_EXPLORATORY_AFTER_TRIGGER_DEFECT'
def read_json(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    return json.loads(path.read_text())

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''): h.update(part)
    return h.hexdigest()

def atomic_json(path, data):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def ensure_space(additional_bytes=0):
    free = shutil.disk_usage(ROOT).free
    if free < FLOOR + additional_bytes: raise RuntimeError(f'Required 30 GiB reserve plus {additional_bytes} bytes; only {free} free')
    return free

def seed_all(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def fp32_policy():
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {'dtype': 'float32', 'AMP': False, 'TF32': False}

@contextlib.contextmanager
def ledger_lock():
    p = ROOT / 'state/BUDGET.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX); yield

def reserve_technical(run_id, job_id, steps):
    if type(steps) is not int or not 0 <= steps <= 1024: raise ValueError('Invalid technical reservation')
    with ledger_lock():
        p = ROOT / 'state/TECHNICAL_LEDGER.json'
        d = read_json(p) if p.exists() else {'optimizer_limit': 1024, 'runs': {}, 'complete_trajectory_limit': 16, 'trajectories': {}}
        if run_id in d['runs']: raise RuntimeError('Technical run ID already reserved; use saved resume state or a new explicitly budgeted run')
        used = sum(x['reserved_updates'] if x.get('actual_updates') is None else x['actual_updates'] for x in d['runs'].values())
        if used + steps > 1024: raise RuntimeError('Technical optimizer budget exhausted')
        d['runs'][run_id] = {'job_id': job_id, 'reserved_updates': steps, 'actual_updates': None, 'reserved_at': now()}
        atomic_json(p, d)
    return d['runs'][run_id]

def finish_technical(run_id, actual_updates, detail=None):
    with ledger_lock():
        p = ROOT / 'state/TECHNICAL_LEDGER.json'; d = read_json(p); x = d['runs'][run_id]
        if actual_updates is None:
            if x['actual_updates'] is not None or not isinstance(detail, dict) or detail.get('status') != 'UNCERTAIN_OPTIMIZER_ATTEMPT':
                raise RuntimeError('Unknown update count requires explicit uncertain optimizer evidence')
            x.update(actual_updates=None, status='UNCERTAIN_OPTIMIZER_ATTEMPT', detail=detail, finished_at=now())
            atomic_json(p, d)
            return
        if x.get('status') == 'UNCERTAIN_OPTIMIZER_ATTEMPT': raise RuntimeError('Cannot release an uncertain reservation without an explicit audit')
        if type(actual_updates) is not int or not 0 <= actual_updates <= x['reserved_updates']: raise RuntimeError('Invalid actual optimizer count')
        if x['actual_updates'] is not None and x['actual_updates'] != actual_updates: raise RuntimeError('Completed optimizer ledger cannot change')
        x.update(actual_updates=actual_updates, detail=detail, finished_at=now()); atomic_json(p, d)

def publish_job(folder, job):
    folder = Path(folder); result = read_json(folder / 'result.json')
    if result.get('actual_updates') != 30000: raise RuntimeError('Only complete fixed30k jobs may be published')
    files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'JOB_SHA256.json' and not p.name.endswith('.tmp')}
    receipt = {'job_id': job['job_id'], 'published_at': now(), 'files': files}
    atomic_json(folder / 'JOB_SHA256.json', receipt)
    atomic_json(ROOT / 'state/recovery_queue' / (job['job_id']+'.json'), receipt)
    return receipt

_original_read_json=read_json
def read_json(path):
    p=Path(path)
    if not p.is_absolute() and str(p)=='manifests/reacher_data_roles.json':p=R3_ROOT/p
    return _original_read_json(p)

def jobs():
    return [{'task':'reacher','refit_seed':s,'updates':30000,'job_id':f'H3X_reacher_s{s}'} for s in SEEDS]

def source_hashes():
    p=read_json(CODE/'CODE_FREEZE.json')
    for name,digest in p['files'].items():
        if sha256(CODE/name)!=digest:raise RuntimeError('Frozen H3X code changed: '+name)
    for name,digest in p['r3_source'].items():
        if sha256(R3_ROOT/name)!=digest:raise RuntimeError('Frozen inherited R3 source changed: '+name)
    return p

def identity_hashes(job):
    names=[R3_ROOT/'manifests/reacher_model_assets.json',R3_ROOT/'manifests/reacher_data_roles.json',ROOT/'inputs/CACHE_AUDIT.json',CODE/'TRAIN_PROTOCOL.json',CODE/'CODE_FREEZE.json']
    return {str(p):sha256(p) for p in names}

def require_authorization(job,technical=False):
    if os.uname().nodename!='72531573d994':raise RuntimeError('Wrong authorized host')
    if job not in jobs():raise ValueError('Job outside H3X authorization')
    source_hashes()
    if technical:return {'label':LABEL,'technical':True}
    d=read_json(ROOT/'FORMAL_AUTHORIZATION.json')
    if d['label']!=LABEL or d['status']!='TECH_PASS_AND_THREE_GPUS_ASSIGNED' or job['job_id'] not in d['jobs']:raise RuntimeError('Formal technical/resource gate incomplete')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if d['gpu_by_job'][job['job_id']]!=gpu:raise RuntimeError('Worker GPU not assigned by root')
    return d
