"""Create the final R3 recovery snapshot after completed computation.

This is a metadata/hash publisher, not CPU scientific acceptance. It never loads
models, scores arrays, connects remotely, queries GPUs, or starts/stops workers.
The caller must stop recovery watchers before invoking the explicit --run entry.
"""
from __future__ import annotations
import argparse, contextlib, datetime, fcntl, hashlib, json, os, stat
from pathlib import Path, PurePosixPath

VERSION = 'R3_FINAL_RECOVERY_SNAPSHOT_V1'
TARGET = 'manifests/RECOVERY_FINAL.json'
ASSETS = 'manifests/OFFICIAL_ASSET_MANIFEST.json'
LOCK = 'manifests/MODELS_AND_SELECTION_LOCK.json'
TASKS = ('pusht', 'reacher')
SEEDS = (103201, 103202, 103203)
ARMS = ('H0',) + tuple('REFIT_'+str(s) for s in SEEDS)
REPORTS = ('INTAKE_AND_INSTANCE_REUSE.md', 'OFFICIAL_SOURCE_AND_PROTOCOL_AUDIT.md',
           'FREEZE_AND_CACHE_VALIDATION.md', 'BASELINE_TECHNICAL_REPRODUCTION.md',
           'TRAINING_COMPLETION.md', 'OPEN_LOOP_RESULTS.md', 'CEM_PAIRED_RESULTS.md',
           'COMPUTE_STORAGE_AND_ETA.md', 'FINAL_SCIENTIFIC_REPORT_ZH.md')
TREES = ('r3', 'scripts', 'tests', 'protocol', 'source', 'official', 'support',
         'manifests', 'state', 'artifacts', 'data/cache', 'reports', 'tables')
OMIT_PARTS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
              'recovery', 'logs', 'failed_partial_transfers'}
OMIT_SUFFIXES = ('.tmp', '.partial', '.lock', '.pyc', '.pyo', '.log', '.out')
SELF_NAMES = {TARGET, 'RELEASE_GATE.json', 'RELEASE_GATE_SEAL.json'}
STAT_FILES = ('RAW_CASE_ROWS.json', 'PAIRED_CASES.csv', 'PAIRED_COUNTS.csv',
              'NUMERIC_RESULTS.json', 'BOOTSTRAP_INDICES.npz',
              'BOOTSTRAP_DISTRIBUTIONS.npz', 'STATISTICS_RECEIPT.json')


def read(path): return json.loads(Path(path).read_text())
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def encoded(value): return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def signature(path):
    s = path.stat(); return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
def local(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or any(ord(c) < 32 for c in name):
        raise ValueError('Canonical ROOT-relative publication path required: '+str(name))
    current = root
    for part in p.parts:
        current /= part
        if current.is_symlink(): raise ValueError('Symlink in publication dependency: '+name)
    if not current.resolve().is_relative_to(root.resolve()): raise ValueError('Publication dependency escapes ROOT')
    return current
def record(expected):
    d = {'sha256': expected} if isinstance(expected, str) else expected
    if not isinstance(d.get('sha256'), str) or len(d['sha256']) != 64: raise ValueError('Invalid evidence SHA')
    if 'bytes' in d and (type(d['bytes']) is not int or d['bytes'] < 0): raise ValueError('Invalid evidence byte count')
    return d


class Inventory:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.files = {}; self.signatures = {}; self.restorable = {}
    def add(self, name, expected=None):
        if name in SELF_NAMES or PurePosixPath(name).name in ('RELEASE_GATE.json', 'RELEASE_GATE_SEAL.json'):
            raise ValueError('Final manifest/gate/seal cannot enter its own input closure')
        p = local(self.root, name)
        if name in self.restorable:
            d = record(expected or self.restorable[name]); r = self.restorable[name]
            if d['sha256'] != r['sha256'] or ('bytes' in d and d['bytes'] != r['bytes']): raise ValueError('Restorable identity differs: '+name)
            return p
        if name.startswith(('data/source/', 'data/unpacked/')) or p.suffix.lower() in ('.h5', '.hdf5'):
            raise ValueError('Bulk source lacks an explicit model-lock restorable entry: '+name)
        if not p.is_file() or not stat.S_ISREG(p.stat().st_mode): raise FileNotFoundError('Required ordinary publication file: '+name)
        before = signature(p)
        if name not in self.files:
            self.files[name] = {'sha256': sha(p), 'bytes': before[2]}; self.signatures[name] = before
            if signature(p) != before: raise RuntimeError('File changed while hashing: '+name)
        elif self.signatures[name] != before: raise RuntimeError('File changed after verification: '+name)
        if expected is not None:
            d = record(expected); actual = self.files[name]
            if actual['sha256'] != d['sha256'] or ('bytes' in d and actual['bytes'] != d['bytes']): raise ValueError('Evidence SHA/bytes differ: '+name)
        return p
    def json(self, name, expected=None): return read(self.add(name, expected))
    def map(self, files, prefix=''):
        if not isinstance(files, dict) or not files: raise ValueError('Required evidence map is empty')
        for name, expected in files.items(): self.add(str(PurePosixPath(prefix)/name) if prefix else name, expected)
    def receipt(self, name, prefix=''):
        d = self.json(name); self.map(d['files'], prefix); return d
    def unchanged(self):
        for name, sig in self.signatures.items():
            if signature(local(self.root, name)) != sig: raise RuntimeError('Publication input changed: '+name)
    def tree(self, tree):
        directory = local(self.root, tree)
        if not directory.exists(): return
        for folder, dirs, names in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in OMIT_PARTS)
            for d in dirs:
                if (Path(folder)/d).is_symlink(): raise ValueError('Symlink directory inside publication tree')
            for filename in sorted(names):
                name = str((Path(folder)/filename).relative_to(self.root)); p = PurePosixPath(name)
                if name in SELF_NAMES or filename in ('RELEASE_GATE.json','RELEASE_GATE_SEAL.json') or filename.startswith('.'): continue
                if filename.endswith(OMIT_SUFFIXES) or '.tmp.' in filename or '.partial.' in filename: continue
                self.add(name)


def bulk_identity(inv, lock):
    """Only the pinned archives and their exact HDF5 members can be omitted."""
    actual = {}
    for task in TASKS:
        assets_name = f'manifests/{task}_data_assets.json'; source_name = f'manifests/{task}_source_map.json'
        assets = inv.json(assets_name); source = inv.json(source_name); unpack = inv.json(f'manifests/{task}_data_unpacked.json')
        if unpack['source_receipt_sha256'] != inv.files[assets_name]['sha256']: raise ValueError('Unpack source receipt differs')
        for item in assets['files'].values():
            local(inv.root, item['path']); actual[item['path']] = record(item)
        members = {m['path']: m for m in unpack['files']}
        for source_sha, item in source['assets'].items():
            if item['sha256'] != source_sha: raise ValueError('Source asset identity differs')
            member = members[item['path']]
            if any(member[k] != item[k] for k in ('sha256','bytes')): raise ValueError('Source/unpacked member differs')
            proof = inv.json(member['member_receipt'], member['member_receipt_sha256'])
            if proof['status'] != 'COMPLETE' or any(proof[k] != item[k] for k in ('sha256','bytes')): raise ValueError('Member receipt is incomplete/different')
            archive = proof['identity']['archive']
            if archive['path'] not in actual or any(archive[k] != actual[archive['path']][k] for k in ('sha256','bytes')): raise ValueError('Member has no exact pinned archive chain')
            actual[item['path']] = record(item)
    omitted = lock.get('restorable_input_files', {})
    if set(omitted) != set(actual): raise ValueError('Every omitted archive/HDF5 must be explicitly identified by the model lock')
    for name, item in omitted.items():
        if name not in lock['files'] or any(item[k] != actual[name][k] for k in ('sha256','bytes')) or any(record(lock['files'][name]).get(k) != item[k] for k in ('sha256','bytes')):
            raise ValueError('Model-lock restorable entry differs: '+name)
        inv.restorable[name] = dict(item)
    return omitted


def training(inv):
    jobs = inv.json('manifests/JOBS.json')['jobs']; done = inv.json('state/FORMAL_TRAINING_COMPLETE.json')
    if len(jobs) != 6 or {(j['task'],j['refit_seed']) for j in jobs} != {(t,s) for t in TASKS for s in SEEDS}: raise ValueError('Six fixed formal jobs required')
    if done['status'] != 'COMPLETE' or done['complete_jobs'] != 6 or done['actual_updates'] != 180000 or set(done['jobs']) != {j['job_id'] for j in jobs}: raise ValueError('Formal completion is incomplete')
    allowed={j['job_id'] for j in jobs}
    for path in local(inv.root,'artifacts/train').iterdir():
        if path.is_dir() and path.name not in allowed and any(path.iterdir()): raise ValueError('Unexpected extra formal-job evidence: '+path.name)
    total = 0
    for job in jobs:
        if job['updates'] != 30000 or job['job_id'] != f'R3_{job["task"]}_PRED_REFIT_s{job["refit_seed"]}': raise ValueError('Fixed formal recipe differs')
        prefix = 'artifacts/train/'+job['job_id']; result = inv.json(prefix+'/result.json')
        if result['status'] != 'REFIT_TRAINING_COMPLETE_UNSCORED' or result['actual_updates'] != 30000 or result['technical'] or result['new_encoder_updates'] or result['state_labels_read'] or result['formal_initialization_inherits_technical_updates']:
            raise ValueError('Formal job lacks exact completion/freeze declaration')
        if result['frozen_before'] != result['frozen_after']: raise ValueError('Frozen tensors changed')
        rows = [json.loads(x) for x in inv.add(prefix+'/updates.jsonl').read_text().splitlines()]
        if len(rows) != 30000 or any(r['step'] != i or r['technical'] is not False or r['sampled_windows'] != 128 for i,r in enumerate(rows,1)): raise ValueError('Formal journal is not exact1..30000')
        total += len(rows)
        for name in ('resume.pt','resume.json','last.json','checkpoint_3000.pt','checkpoint_10000.pt','checkpoint_30000.pt',
                     'RUN_IDENTITY.json','PARAMETER_WHITELIST.json','FROZEN_INITIAL.json','freeze_checks.jsonl','monitor.jsonl','sampling_counts.npz'):
            inv.add(prefix+'/'+name)
        inv.receipt(prefix+'/JOB_SHA256.json')
        for item in (done['jobs'][job['job_id']][k] for k in ('result','publication','recovery_queue')): inv.add(item['path'], item)
        if local(inv.root,prefix+'/OPTIMIZER_INFLIGHT.json').exists(): raise ValueError('Unresolved optimizer attempt')
    if total != 180000: raise ValueError('Formal updates differ')
    return {'jobs':6,'updates':total,'new_encoder_updates':0}


def evaluations(inv, lock_sha):
    plan = inv.json('state/planning/FORMAL/EXECUTION_PLAN.json'); complete = inv.receipt('state/PLANNING_FORMAL_COMPLETE.json')
    if plan['phase'] != 'FORMAL' or plan['models_lock_sha256'] != lock_sha or complete['status'] != 'COMPLETE' or complete['phase'] != 'FORMAL' or complete['execution_plan_sha256'] != digest(plan): raise ValueError('Formal planning completion/lock differs')
    ledger = inv.json('state/FORMAL_TRAJECTORY_LEDGER.json')['trajectories']; expected_rids = set(); counts = {}
    inv.map(plan['files'])
    for task in TASKS:
        roles = inv.json(f'manifests/{task}_data_roles.json'); cases = roles['cases']['EVAL']; counts[task] = len(cases)
        if not 20 <= len(cases) <= 100 or len({c['case_id'] for c in cases}) != len(cases) or len({c['episode_id'] for c in cases}) != len(cases): raise ValueError('Fixed EVAL case count/identity invalid')
        if {c['episode_id'] for c in cases} != {e['episode_id'] for e in roles['episodes'] if e['role']=='EVAL'}: raise ValueError('EVAL case population differs from roles')
        selected = [x['case'] for x in plan['cases'] if x['task']==task]
        if len(selected)!=len(cases) or {x['case_id'] for x in selected}!={x['case_id'] for x in cases}: raise ValueError('Formal execution omitted/added a fixed case')
        for case in cases:
            cp = f'artifacts/planning/FORMAL/{task}/{case["case_id"]}'; cs = inv.receipt(cp+'/CASE_SHA256.json')
            if cs['status']!='COMPLETE' or cs['complete_trajectories']!=4 or cs['execution_plan_sha256']!=digest(plan): raise ValueError('Incomplete fixed four-arm case')
            for arm in ARMS:
                prefix = cp+'/'+arm+'/attempt_0'; result = inv.json(prefix+'/result.json'); seal = inv.receipt(prefix+'/TRAJECTORY_SHA256.json',prefix)
                if result['status']!='COMPLETE' or result['evaluation_complete'] is not True or (result['task'],result['case_id'],result['arm'],result['phase'])!=(task,case['case_id'],arm,'FORMAL') or result['failure_category'] not in (None,'METHOD') or result['optimizer_updates']!=0: raise ValueError('Formal trajectory incomplete/different')
                if not {'result.json','trajectory.npz'} <= set(seal['files']): raise ValueError('Raw full trajectory is absent from its seal')
                rid=f'R3/FORMAL/{task}/{case["case_id"]}/{arm}/attempt_0'; expected_rids.add(rid); entry=ledger[rid]
                if entry['status']!='COMPLETE' or entry['result_path']!=prefix+'/result.json' or entry['result_sha256']!=inv.files[prefix+'/result.json']['sha256']: raise ValueError('Formal trajectory ledger differs')
        prefix='artifacts/open_loop/'+task; opened=inv.receipt(prefix+'/OPEN_LOOP_RECEIPT.json')
        if opened['status']!='COMPLETE' or opened['evaluation_kind']!='OPEN_LOOP' or opened['task']!=task or opened['cases']!=len(cases) or opened['arms']!=10 or opened['optimizer_updates'] or not opened['all_future_targets_scoring_only']: raise ValueError('Actual ten-arm open-loop matrix incomplete')
        expected_arms={'H0'}|{f'REFIT_{s}_{step}' for s in SEEDS for step in (3000,10000,30000)}
        if len(opened['arm_receipts'])!=10 or {r['route']['arm'] for r in opened['arm_receipts']}!=expected_arms: raise ValueError('Open-loop arms differ')
        for arm in expected_arms:
            for suffix in ('.npz','.json'): inv.add(prefix+'/predictions/'+arm+suffix)
        for name in ('inputs.npz','targets.npz','per_case.json','TRAIN_coordinates.npz','TRAIN_variance.json'): inv.add(prefix+'/'+name)
    occupied={rid for rid,r in ledger.items() if r['status']!='INFRASTRUCTURE_FAILURE_INCOMPLETE'}
    if occupied!=expected_rids or complete['complete_trajectories']!=len(expected_rids) or plan['expected_complete_trajectories']!=len(expected_rids) or len(expected_rids)>800: raise ValueError('Formal trajectory budget/matrix differs')
    return {'cases':counts,'complete_formal_trajectories':len(expected_rids),'open_loop_arms_per_task':10}


def build(root):
    inv=Inventory(root); instance=inv.json('state/INSTANCE_REUSE.json'); exited=inv.json('state/FINAL_GPU_EXIT.json')
    if exited['status']!='ALL_R3_GPU_JOBS_EXITED' or exited['hostname']!=instance['hostname'] or exited['active_compute_pids']!=[] or exited['R3_processes']!=[] or not exited['checked_at']: raise ValueError('Actual final GPU/process-exit receipt required')
    lock=inv.json(LOCK)
    if lock['status']!='MODELS_AND_SELECTION_LOCKED' or ASSETS not in lock['files']: raise ValueError('Models and official-asset identity must be locked before final publication')
    bulk_identity(inv,lock); inv.map(lock['files'])
    assetmap=inv.json(ASSETS,lock['files'][ASSETS])
    if assetmap['version']!='R3_OFFICIAL_ASSET_MANIFEST_V1' or assetmap['status']!='COMPLETE_METADATA_ASSET_MAPPING' or set(assetmap['tasks'])!=set(TASKS): raise ValueError('Official asset metadata mapping is incomplete')
    inv.map(assetmap['files'])
    train=training(inv); evaluated=evaluations(inv,inv.files[LOCK]['sha256'])
    output=inv.json('manifests/ANALYSIS_OUTPUTS.json')
    if output['status']!='COMPLETE' or set(output['tasks'])!=set(TASKS) or output['models_lock_sha256']!=inv.files[LOCK]['sha256'] or output['formal_training_jobs']!=6 or output['formal_training_updates']!=180000 or output['bootstrap_replicates']!=5000: raise ValueError('Actual complete two-task analysis required')
    inv.map(output['inputs']); inv.map(output['files'])
    for task in TASKS:
        row=output['tasks'][task]
        if row['statistics_dir']!='artifacts/statistics/'+task or row['open_loop_dir']!='artifacts/open_loop/'+task or row['planning_phase']!='artifacts/planning/FORMAL' or row['cases']!=evaluated['cases'][task] or row['CEM_arms']!=4 or row['open_loop_arms']!=10: raise ValueError('Analysis matrix/schema differs')
        for filename in STAT_FILES: inv.add(row['statistics_dir']+'/'+filename)
        stats=inv.receipt(row['statistics_dir']+'/STATISTICS_RECEIPT.json',row['statistics_dir'])
        if stats['status']!='COMPLETE' or stats['task']!=task or stats['cases']!=evaluated['cases'][task] or stats['arms']!=list(ARMS): raise ValueError('Actual fixed paired-statistics receipt incomplete')
    # Full delivered trees include failed technical evidence, raw journals,
    # checkpoint deltas/resume, caches and all reproducibility source. Unsealed
    # operational logs/locks are omitted; a sealed dependency is never omitted.
    for tree in TREES: inv.tree(tree)
    reports={}
    for base in REPORTS:
        found=[name for name in inv.files if PurePosixPath(name).name==base]
        if len(found)!=1 or not local(inv.root,found[0]).read_text().strip(): raise ValueError('Exactly one nonempty protocol report required: '+base)
        reports[base]=found[0]
    for name in ('scripts/final_acceptance.py','scripts/publish_final.py','manifests/PRIOR_EVIDENCE.json',
                 'manifests/NUMERICAL_TOLERANCES.json','manifests/OPEN_LOOP_ROUTING.json','state/TECHNICAL_LEDGER.json'):
        inv.add(name)
    inv.unchanged()
    manifest={'version':VERSION,'status':'FINAL_SNAPSHOT_READY_FOR_RECOVERY_AND_ACCEPTANCE',
              'files':dict(sorted(inv.files.items())),'restorable_input_files':dict(sorted(inv.restorable.items())),
              'models_lock_sha256':inv.files[LOCK]['sha256'],'analysis_outputs_sha256':inv.files['manifests/ANALYSIS_OUTPUTS.json']['sha256'],
              'official_asset_manifest_sha256':inv.files[ASSETS]['sha256'],'final_GPU_exit_sha256':inv.files['state/FINAL_GPU_EXIT.json']['sha256'],
              'completion':{**train,**evaluated},'reports':reports,
              'bulk_policy':'Only exact pinned archives/HDF5 members already marked restorable in the model lock are omitted. This publisher does not claim a new binary hash of absent bulk sources.',
              'transient_policy':'Unsealed logs, locks, temporary/partial files, environments and bytecode are excluded. Bound evidence is retained.',
              'CPU_acceptance_still_required':True,'contains_CPU_gate_or_seal':False,'contains_itself':False,
              'optimizer_updates':0,'environment_steps':0,'GPU_queries':0,'SSH_connections':0,
              'scientific_results_modified':False,'training_launched':False,'resources_destroyed':False}
    return manifest,inv


def publish(root, manifest, inv):
    path=local(Path(root).resolve(),TARGET); inv.unchanged(); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='published_at'}!=manifest: raise RuntimeError('Different existing final manifest preserved; never overwrite')
        return old,False
    value={**manifest,'published_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}; data=encoded(value)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    inv.unchanged()
    try: os.link(tmp,path)
    except FileExistsError: raise RuntimeError('Concurrent publication preserved; inspect the existing manifest')
    tmp.unlink(); fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return value,True


def run(root):
    root=Path(root).resolve()
    if os.uname().nodename!=read(local(root,'state/INSTANCE_REUSE.json'))['hostname']: raise RuntimeError('Run only on the already authorized instance')
    with contextlib.ExitStack() as stack:
        for name in ('state/final_publication.lock','state/execution_phase.lock','state/model_freeze.lock','state/training_technical_pipeline.lock','state/input_publication.lock','state/BUDGET.lock'):
            path=local(root,name); path.parent.mkdir(parents=True,exist_ok=True); f=stack.enter_context(path.open('a+')); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,inv=build(root); return publish(root,manifest,inv)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    parser.add_argument('--root',default=os.environ.get('R3_ROOT',str(Path(__file__).resolve().parents[1]))); args=parser.parse_args()
    if not args.run: parser.error('Explicit --run required; no recovery or CPU release gate runs automatically')
    out,created=run(args.root); print(json.dumps({'status':out['status'],'created':created,'files':len(out['files']),
        'bytes':sum(x['bytes'] for x in out['files'].values()),'CPU_acceptance_still_required':True}),flush=True)
