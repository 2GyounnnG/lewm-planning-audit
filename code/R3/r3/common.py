"""Shared identities and finite-budget accounting for R3; no implicit launch."""
from __future__ import annotations
from pathlib import Path
import contextlib, datetime, fcntl, hashlib, json, os, random, shutil, subprocess, tempfile

ROOT = Path(os.environ.get('R3_ROOT', Path(__file__).resolve().parents[1])).resolve()
TASKS = ('pusht', 'reacher')
SEEDS = (103201, 103202, 103203)
FLOOR = 30 * (1 << 30)

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

def jobs(): return read_json('manifests/JOBS.json')['jobs']

def official_asset(task):
    if task not in TASKS: raise ValueError(task)
    d = read_json(f'manifests/{task}_model_assets.json')
    w, c = d['files']['weights.pt'], d['files']['config.json']
    for record in (w, c):
        p = ROOT / record['path']
        if p.stat().st_size != record['bytes'] or sha256(p) != record['sha256']: raise RuntimeError('Official asset differs: '+str(p))
    return {'repo': d['repo'], 'revision': d['revision'], 'weights_path': ROOT / w['path'],
            'config_path': ROOT / c['path'], 'weights_sha256': w['sha256'], 'config_sha256': c['sha256']}

def identity_hashes(job):
    task = job['task']
    names = [f'manifests/{task}_model_assets.json', f'manifests/{task}_data_roles.json',
             f'manifests/{task}_cache.json', 'manifests/EXPERIMENT_LOCK.json', 'manifests/JOBS.json']
    return {n: sha256(ROOT / n) for n in names}

def require_authorization(job, technical=False):
    identity = read_json('state/INSTANCE_REUSE.json')
    if identity['hostname'] != '6494ba5e1b1a' or os.uname().nodename != identity['hostname']: raise RuntimeError('Wrong authorized instance')
    actual = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
    if actual != identity['gpu_uuids']: raise RuntimeError('Authorized GPU identity differs')
    if technical: return identity
    d = read_json('manifests/TRAINING_AUTHORIZATION.json')
    job_id = job['job_id'] if isinstance(job, dict) else str(job)
    if job_id not in d['job_ids'] or d.get('status') != 'TECHNICAL_GATES_PASS': raise RuntimeError('Formal job not authorized by technical gate')
    for name, value in d['source_hashes'].items():
        if sha256(ROOT / name) != value: raise RuntimeError('Locked formal source changed: '+name)
    return d

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

def reserve_trajectory(run_id, task, case_id, arm, formal=False, retry_of=None):
    with ledger_lock():
        path = ROOT / ('state/FORMAL_TRAJECTORY_LEDGER.json' if formal else 'state/TECHNICAL_LEDGER.json')
        d = read_json(path) if path.exists() else {'runs': {}, 'trajectories': {}, 'optimizer_limit': 1024, 'complete_trajectory_limit': 800 if formal else 16}
        if run_id in d['trajectories']: raise RuntimeError('Existing trajectory must be resumed/audited, not counted twice')
        same=[(k,v) for k,v in d['trajectories'].items() if (v['task'],v['case_id'],v['arm'])==(task,case_id,arm)]
        if same:
            if retry_of not in dict(same) or any(v['status']!='INFRASTRUCTURE_FAILURE_INCOMPLETE' for _,v in same):
                raise RuntimeError('Only an explicitly documented incomplete infrastructure attempt may be retried')
        elif retry_of is not None: raise RuntimeError('Retry must reference the exact same case and arm')
        occupied=sum(v['status']!='INFRASTRUCTURE_FAILURE_INCOMPLETE' for v in d['trajectories'].values())
        if occupied >= d['complete_trajectory_limit']: raise RuntimeError('Trajectory budget exhausted')
        d['trajectories'][run_id] = {'task': task, 'case_id': case_id, 'arm': arm, 'status': 'RESERVED', 'reserved_at': now(), 'retry_of':retry_of}
        atomic_json(path, d)

def finish_trajectory(run_id,result_path,formal=False):
    result_path=Path(result_path)
    if not result_path.is_absolute():result_path=ROOT/result_path
    if not result_path.resolve().is_relative_to(ROOT):raise RuntimeError('Trajectory result escaped R3')
    result=read_json(result_path);digest=sha256(result_path)
    with ledger_lock():
        path=ROOT/('state/FORMAL_TRAJECTORY_LEDGER.json' if formal else 'state/TECHNICAL_LEDGER.json')
        d=read_json(path);record=d['trajectories'][run_id]
        if (result['task'],result['case_id'],result['arm'])!=(record['task'],record['case_id'],record['arm']):raise RuntimeError('Trajectory ledger/result identity mismatch')
        if result['phase']!=('FORMAL' if formal else 'TECH'):raise RuntimeError('Wrong trajectory phase')
        if record['status']!='RESERVED':
            if record.get('result_sha256')!=digest:raise RuntimeError('Completed trajectory ledger cannot change')
            return record
        if result['status']=='COMPLETE' and result['evaluation_complete']:
            receipt=read_json(result_path.parent/'TRAJECTORY_SHA256.json')
            if receipt['identity_sha256']!=result['identity_sha256'] or receipt['files']['result.json']['sha256']!=digest:raise RuntimeError('Completed trajectory receipt mismatch')
            status='COMPLETE'
        elif result['status']=='INFRASTRUCTURE_FAILURE' and not result['evaluation_complete']:
            # An incomplete infrastructure attempt is charged separately and may
            # be retried only with explicit identical case/seed/config provenance.
            # Reaching any terminal success or 50 steps consumes the reserved
            # completion slot conservatively until its evidence is resolved.
            status=('INFRASTRUCTURE_FAILURE_INCOMPLETE' if result['executed_raw_steps']<50 and not result['any_step_success']
                    else 'COMPLETION_UNCERTAIN')
        else:raise RuntimeError('Unknown trajectory completion classification')
        record.update(status=status,result_path=str(result_path.relative_to(ROOT)),result_sha256=digest,
                      executed_raw_steps=result['executed_raw_steps'],replan_calls=result['replan_calls'],
                      trajectory_wall_seconds=result['trajectory_wall_seconds'],finished_at=now())
        atomic_json(path,d)
        return record

def publish_job(folder, job):
    folder = Path(folder); result = read_json(folder / 'result.json')
    if result.get('actual_updates') != 30000: raise RuntimeError('Only complete fixed30k jobs may be published')
    files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'JOB_SHA256.json' and not p.name.endswith('.tmp')}
    receipt = {'job_id': job['job_id'], 'published_at': now(), 'files': files}
    atomic_json(folder / 'JOB_SHA256.json', receipt)
    atomic_json(ROOT / 'state/recovery_queue' / (job['job_id']+'.json'), receipt)
    return receipt
