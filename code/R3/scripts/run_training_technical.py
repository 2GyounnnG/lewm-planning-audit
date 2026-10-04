"""Isolated R3 process-resume smoke and fixed 2-versus-4-worker profile.

The root supervisor invokes this on the authorized instance. Import is inert.
It never launches formal training, changes a recipe, or replays an interrupted
profile to obtain a preferred timing. All optimizer calls belong to train_worker
and its existing per-segment technical ledger.
"""
from __future__ import annotations
import argparse, contextlib, csv, fcntl, hashlib, json, math, os, re, subprocess, sys
import threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3 import common

ROOT = common.ROOT
VERSION = 'R3_TRAINING_TECHNICAL_320_V1'
TASKS = ('pusht', 'reacher')
BATCH = 128
STEPS = 48


def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(obj): return hashlib.sha256(canonical(obj)).hexdigest()
def read(path): return common.read_json(path)
def write(path, obj): common.atomic_json(path, obj)
def journal(path): return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []
def record(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': common.sha256(path)}


def verify_records(files):
    for item in files:
        path = ROOT / item['path']
        if path.stat().st_size != item['bytes'] or common.sha256(path) != item['sha256']:
            raise RuntimeError('Registered technical artifact changed: ' + item['path'])


def policy():
    return {'version': VERSION, 'smoke': 'Each task: continuous8 vs separate PID4+resume4; exact equality',
            'profile_A': '2 workers total, PushT GPU0 and Reacher GPU1, 48 updates each',
            'profile_B': '4 workers total, one PushT and one Reacher per GPU, 48 updates each',
            'planned_optimizer_updates': 320, 'smoke_updates': 32, 'profile_A_updates': 96, 'profile_B_updates': 192,
            'microbatch': 128, 'effective_batch': 128, 'initialization': 'official base; new optimizer; isolated technical seed offset',
            'choice': '4 only if finite/frozen/no OOM and both measured end-to-end aggregate throughput and conservative observed steady-update throughput improve by at least10%; ambiguous means2',
            'scientific_metrics_used_for_scheduling': False, 'interrupted_profile': 'preserve and block; no automatic timing rerun',
            'formal_launch': False, 'technical_parameters_inherit_into_formal': False,
            'planning_environment_gate': 'separate root-owned check; not asserted by this helper'}


def identity():
    names = ['scripts/run_training_technical.py', 'r3/model.py', 'r3/train_worker.py', 'r3/data.py', 'r3/common.py',
             'manifests/JOBS.json', 'manifests/EXPERIMENT_LOCK.json', 'state/INSTANCE_REUSE.json']
    for task in TASKS:
        names += [f'manifests/{task}_{suffix}.json' for suffix in ('model_assets', 'data_roles', 'cache')]
    hashes = {name: common.sha256(ROOT / name) for name in names}
    for task in TASKS:
        cache = read(f'manifests/{task}_cache.json')
        if cache['status'] != 'FROZEN_OBSERVED_CACHE_COMPLETE': raise RuntimeError('Observed cache is not complete')
        roles = read(f'manifests/{task}_data_roles.json')
        if not roles.get('technical_monitor_windows'): raise RuntimeError('Actual fixed TECH monitoring windows missing')
        for name, expected in cache['input_hashes'].items():
            if common.sha256(ROOT / name) != expected: raise RuntimeError('Cache input changed: ' + name)
            hashes[name] = expected
    return {'policy': policy(), 'files': hashes}


def assert_identity(frozen):
    if identity() != frozen: raise RuntimeError('Frozen source/cache/experiment identity changed during technical checks')


def gpu_inventory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total', '--format=csv,noheader,nounits'], text=True)
    result = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 3: raise RuntimeError('Unexpected GPU inventory output')
        result.append({'index': int(row[0]), 'uuid': row[1].strip(), 'memory_total_MiB': int(row[2])})
    if [x['index'] for x in result] != [0, 1]: raise RuntimeError('Exactly the authorized two-GPU instance is required')
    return result


def require_idle_gpu():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    pids = [int(x.strip()) for x in output.splitlines() if x.strip()]
    if pids: raise RuntimeError('GPU compute processes already active; no signals sent: ' + str(pids))


def append(path, obj):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(obj, allow_nan=False) + '\n'); handle.flush(); os.fsync(handle.fileno())


def gpu_sample():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=10)
    values = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 5: raise RuntimeError('Unexpected GPU telemetry output')
        values.append({'index': int(row[0]), 'uuid': row[1].strip(), 'utilization_percent': float(row[2]),
                       'memory_used_MiB': float(row[3]), 'memory_total_MiB': float(row[4])})
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemTotal'): memory[key + '_bytes'] = int(value.split()[0]) * 1024
    return {'time_unix': time.time(), 'monotonic': time.monotonic(), 'gpus': values, **memory}


def worker_folder(spec): return ROOT / 'artifacts/technical' / spec['run_id'] / spec['job']['job_id']
def worker_command(spec):
    command = [sys.executable, '-m', 'r3.train_worker', '--job', spec['job']['job_id'], '--device', 'cuda',
               '--microbatch', '128', '--technical-steps', str(spec['endpoint']), '--run-id', spec['run_id'], '--slot', str(spec['slot'])]
    if spec.get('stop_after') is not None: command += ['--stop-after', str(spec['stop_after'])]
    return command


def inspect_worker(spec, expected=None, load_checkpoint=True):
    import torch
    folder = worker_folder(spec); expected = expected if expected is not None else spec['endpoint']
    result = read(folder / 'result.json'); rows = journal(folder / 'updates.jsonl')
    if result['actual_updates'] != expected or not result['technical'] or result['effective_batch'] != 128 or result['microbatch'] != 128:
        raise RuntimeError('Worker endpoint/technical/batch identity differs')
    if result['new_encoder_updates'] != 0 or result['state_labels_read'] != 0 or result['formal_initialization_inherits_technical_updates']:
        raise RuntimeError('Forbidden update/exposure or technical inheritance')
    if result['frozen_before'] != result['frozen_after']: raise RuntimeError('Frozen model changed')
    if (folder / 'OPTIMIZER_INFLIGHT.json').exists(): raise RuntimeError('Unresolved optimizer attempt')
    if [row['step'] for row in rows] != list(range(1, expected + 1)): raise RuntimeError('Technical successful-step journal differs')
    for row in rows:
        if not row['technical'] or row['sampled_windows'] != 128 or row['predicted_tokens'] != 384:
            raise RuntimeError('Per-batch token or technical contract differs')
        if any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) for k in ('loss_raw_MSE', 'lr', 'preclip_gradient_norm', 'compute_seconds')):
            raise RuntimeError('Nonfinite technical training journal')
    checks = journal(folder / 'freeze_checks.jsonl')
    if not checks or checks[-1]['step'] != expected or any(not x['passed'] or x['frozen_sha256'] != result['frozen_before'] for x in checks):
        raise RuntimeError('Missing or failed frozen checks')
    for row in journal(folder / 'monitor.jsonl'):
        for key in ('teacher_forced_one_step_raw_MSE', 'free_H5_mean_raw_MSE', 'free_H5_terminal_raw_MSE'):
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]): raise RuntimeError('Nonfinite technical monitor')
    pointer = read(folder / 'last.json'); path = folder / pointer['path']
    if path.parent != folder or common.sha256(path) != pointer['sha256']: raise RuntimeError('Technical checkpoint pointer differs')
    evidence = {'result': result, 'journal': rows, 'folder': str(folder.relative_to(ROOT)),
                'files': [record(folder / name) for name in ('result.json', 'updates.jsonl', 'last.json', 'RUN_IDENTITY.json', 'PARAMETER_WHITELIST.json', 'FROZEN_INITIAL.json', 'freeze_checks.jsonl', 'sampling_counts.npz')] + [record(path)]}
    if (folder / 'monitor.jsonl').exists(): evidence['files'].append(record(folder / 'monitor.jsonl'))
    if load_checkpoint:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['step'] != expected or checkpoint['new_encoder_updates'] != 0 or checkpoint['state_labels_read'] != 0:
            raise RuntimeError('Checkpoint endpoint/exposure differs')
        replacements = checkpoint['delta']['replacement_parameters']
        if not replacements or any(not name.startswith(('predictor.', 'pred_proj.')) or not torch.isfinite(value).all() for name, value in replacements.items()):
            raise RuntimeError('Unexpected/nonfinite replacement parameters')
        evidence['checkpoint'] = checkpoint
    return evidence


def equal_tree(left, right, path='root'):
    import torch
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise AssertionError('Exact resume mismatch: ' + path)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right): raise AssertionError('Exact resume keys mismatch: ' + path)
        for key in left: equal_tree(left[key], right[key], path + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right): raise AssertionError('Exact resume sequence mismatch: ' + path)
        for i, value in enumerate(left): equal_tree(value, right[i], path + '/' + str(i))
    elif type(left) is not type(right) or left != right: raise AssertionError('Exact resume value mismatch: ' + path)


def compare_smoke(continuous, split):
    a, b = inspect_worker(continuous), inspect_worker(split)
    ca, cb = a['checkpoint'], b['checkpoint']
    for key in ('delta', 'optimizer', 'scheduler', 'rng', 'sampler', 'next_batch_sha256', 'frozen', 'sampling_summary'):
        equal_tree(ca[key], cb[key], key)
    fields = ('step', 'loss_raw_MSE', 'lr', 'preclip_gradient_norm', 'batch_cursor_sha256', 'sampled_windows', 'predicted_tokens', 'raw_action_exposures')
    equal_tree([{k: r[k] for k in fields} for r in a['journal']], [{k: r[k] for k in fields} for r in b['journal']], 'all8_per_batch')
    equal_tree(read(worker_folder(continuous) / 'PARAMETER_WHITELIST.json'), read(worker_folder(split) / 'PARAMETER_WHITELIST.json'), 'same_official_initialization')
    first = {r['pid'] for r in b['journal'][:4]}; last = {r['pid'] for r in b['journal'][4:]}
    if len(first) != 1 or len(last) != 1 or first == last or len({r['pid'] for r in a['journal']}) != 1:
        raise RuntimeError('Resume did not use distinct real worker processes')
    return {'status': 'PASS', 'task': continuous['job']['task'], 'exact_equality': True, 'tolerance': 0,
            'checked': ['official initial parameters', 'all8 batch cursor/loss/gradient norm/LR', 'replacement weights', 'optimizer', 'sampler and counts', 'Python/NumPy/Torch CPU/CUDA RNG', 'scheduler', 'frozen parameters and all buffers'],
            'split_process_ids': [next(iter(first)), next(iter(last))], 'files': a['files'] + b['files'],
            'optimizer_updates': 16, 'formal_inheritance': False}


def run_phase(folder, name, specs, frozen, profile=False):
    path = folder / (name + '.json')
    if path.exists():
        old = read(path)
        if old.get('status') != 'PASS': raise RuntimeError('Incomplete/failed stage preserved; no automatic replay: ' + name)
        if old['identity_sha256'] != digest(frozen): raise RuntimeError('Completed stage identity differs')
        verify_records(old['files']); return old
    assert_identity(frozen); require_idle_gpu(); common.ensure_space()
    initial_rows = {s['run_id']: journal(worker_folder(s) / 'updates.jsonl') for s in specs}
    initial_loader = {s['run_id']: read(worker_folder(s) / 'result.json')['timing']['loader_seconds'] if (worker_folder(s) / 'result.json').exists() else 0. for s in specs}
    if profile and any(initial_rows.values()): raise RuntimeError('Profile workers must start from separate official initialization')
    record0 = {'initial_journal_steps': {k: len(v) for k, v in initial_rows.items()}, 'status': 'STARTED', 'stage': name, 'identity_sha256': digest(frozen), 'specs': specs,
               'wall_start_unix': time.time(), 'commands': [worker_command(s) for s in specs]}
    write(path, record0)
    telemetry = folder / (name + '_gpu.jsonl'); observations = folder / (name + '_journal_observations.jsonl')
    started = time.monotonic(); telemetry_stop = threading.Event(); sample_errors = []
    def sample_loop():
        while not telemetry_stop.is_set():
            try: append(telemetry, gpu_sample())
            except BaseException as error: sample_errors.append(type(error).__name__)
            telemetry_stop.wait(1.)
    sampling = threading.Thread(target=sample_loop, daemon=True) if profile else None
    if sampling: sampling.start()
    processes = []; handles = []; histories = {s['run_id']: [] for s in specs}; previous_poll = started; last_counts = {}
    try:
        for spec in specs:
            log = folder / (name + '_' + spec['job']['task'] + '_gpu' + str(spec['gpu']) + '_slot' + str(spec['slot']) + '.log')
            handle = log.open('x'); handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['gpu']), PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            process = subprocess.Popen(worker_command(spec), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((spec, process, log))
            record0.setdefault('processes', []).append({'run_id': spec['run_id'], 'pid': process.pid, 'gpu': spec['gpu'], 'slot': spec['slot'], 'log': str(log.relative_to(ROOT))})
            write(path, record0)
        while True:
            now = time.monotonic()
            for spec, process, log in processes:
                jpath = worker_folder(spec) / 'updates.jsonl'
                try: rows = journal(jpath)
                except json.JSONDecodeError: continue  # A final line may be mid-write; next poll rechecks.
                n = len(rows)
                if last_counts.get(spec['run_id']) != n:
                    entry = {'run_id': spec['run_id'], 'successful_steps': n, 'observed_monotonic': now,
                             'previous_poll_monotonic': previous_poll, 'process_pid': process.pid}
                    histories[spec['run_id']].append(entry); append(observations, entry); last_counts[spec['run_id']] = n
            previous_poll = now
            if all(p.poll() is not None for _, p, _ in processes): break
            time.sleep(.1)
        ended = time.monotonic()
    finally:
        # Signal only the helper's telemetry thread; never terminate unrelated or child processes.
        telemetry_stop.set()
        if sampling: sampling.join(timeout=12)
        for handle in handles: handle.close()
    failures = []
    for spec, process, log in processes:
        if process.returncode != 0: failures.append({'run_id': spec['run_id'], 'returncode': process.returncode, 'log': str(log.relative_to(ROOT))})
    if failures:
        failure_files = [record(log) for _, _, log in processes]
        for spec in specs:
            if worker_folder(spec).exists():
                failure_files += [record(p) for p in sorted(worker_folder(spec).rglob('*')) if p.is_file() and not p.name.endswith(('.lock', '.tmp'))]
        for p in (telemetry, observations):
            if p.exists(): failure_files.append(record(p))
        result = {**record0, 'status': 'FAILED', 'failures': failures, 'launch_wall_seconds': ended - started,
                  'files': failure_files, 'no_automatic_retry': True}
        write(path, result); raise RuntimeError('Technical worker phase failed; all outputs preserved: ' + name)
    evidence = [inspect_worker(s, s.get('stop_after') or s['endpoint'], load_checkpoint=True) for s in specs]
    files = [item for e in evidence for item in e['files']] + [record(log) for _, _, log in processes]
    if observations.exists(): files.append(record(observations))
    if telemetry.exists(): files.append(record(telemetry))
    update_count = sum((s.get('stop_after') or s['endpoint']) - len(initial_rows[s['run_id']]) for s in specs)
    phase_compute = {s['run_id']: sum(r['compute_seconds'] for r in e['journal'][len(initial_rows[s['run_id']]):]) for s, e in zip(specs, evidence)}
    useful_steps = 0; excluded = {}
    first_bounds = []; last_bounds = []
    for spec in specs:
        points = histories[spec['run_id']]; endpoint = spec.get('stop_after') or spec['endpoint']
        first = next((x for x in points if x['successful_steps'] > 0), None)
        last = next((x for x in points if x['successful_steps'] == endpoint), None)
        if not first or not last or last is first: break
        useful_steps += endpoint - first['successful_steps']; excluded[spec['run_id']] = first['successful_steps']
        first_bounds.append((first['previous_poll_monotonic'], first['observed_monotonic']))
        last_bounds.append((last['previous_poll_monotonic'], last['observed_monotonic']))
    steady = {'status': 'AMBIGUOUS_OBSERVATION_RESOLUTION', 'poll_interval_target_seconds': .1}
    if len(first_bounds) == len(specs) and len(last_bounds) == len(specs):
        lower = max(x[0] for x in last_bounds) - min(x[1] for x in first_bounds)
        upper = max(x[1] for x in last_bounds) - min(x[0] for x in first_bounds)
        if lower > 0:
            steady = {'status': 'BOUNDED_OBSERVED_INTERVAL', 'span_lower_seconds': lower, 'span_upper_seconds': upper,
                      'useful_steps_after_initial_observed_prefix': useful_steps, 'excluded_initial_updates_by_worker': excluded,
                      'aggregate_updates_per_second_lower': useful_steps / upper,
                      'aggregate_updates_per_second_upper': useful_steps / lower,
                      'definition': 'Across all workers from earliest initial observed nonzero journal count to latest endpoint; observed initial prefix excluded per worker; polling bounds reported'}
    result = {**record0, 'status': 'PASS', 'launch_wall_seconds': ended - started,
              'total_successful_updates': update_count, 'end_to_end_updates_per_second': update_count / (ended - started),
              'steady_update_span': steady, 'sum_worker_compute_seconds': sum(phase_compute.values()),
              'sum_worker_loader_seconds': sum(e['result']['timing']['loader_seconds'] - initial_loader[s['run_id']] for s, e in zip(specs, evidence)),
              'compute_equivalent_aggregate_updates_per_second': sum(((s.get('stop_after') or s['endpoint']) - len(initial_rows[s['run_id']])) / phase_compute[s['run_id']] for s in specs),
              'peak_allocated_bytes_by_worker': {s['run_id']: e['result']['peak_allocated_bytes'] for s, e in zip(specs, evidence)},
              'raw_gpu_telemetry': str(telemetry.relative_to(ROOT)) if telemetry.exists() else None,
              'telemetry_errors': sample_errors, 'startup_and_endpoint_monitor_and_save_included_in_launch_wall': True,
              'files': files, 'no_optimizer_state_inheritance': True}
    assert_identity(frozen); write(path, result); return result


def choose_profile(a, b):
    end_gain = b['end_to_end_updates_per_second'] / a['end_to_end_updates_per_second']
    sa, sb = a['steady_update_span'], b['steady_update_span']; steady_gain = None
    if sa['status'] == sb['status'] == 'BOUNDED_OBSERVED_INTERVAL':
        steady_gain = sb['aggregate_updates_per_second_lower'] / sa['aggregate_updates_per_second_upper']
    use4 = end_gain >= 1.1 and steady_gain is not None and steady_gain >= 1.1 and not a['telemetry_errors'] and not b['telemetry_errors']
    return {'selected_workers': 4 if use4 else 2, 'workers_per_gpu': 2 if use4 else 1, 'microbatch': 128,
            'end_to_end_gain_B_over_A': end_gain, 'conservative_steady_gain_B_over_A': steady_gain,
            'reason': 'BOTH_COMPARABLE_GAINS_AT_LEAST_10_PERCENT' if use4 else 'CONSERVATIVE_ONE_PER_GPU_GAIN_INSUFFICIENT_OR_AMBIGUOUS',
            'scientific_outcomes_used': False}


def technical_ledger_summary(prefix, require_exact320=True):
    d = read('state/TECHNICAL_LEDGER.json'); own = {k: v for k, v in d['runs'].items() if k.startswith(prefix + '_')}
    unresolved = {k: v for k, v in own.items() if v.get('actual_updates') is None}
    if require_exact320 and unresolved: raise RuntimeError('Technical reservation remains unresolved')
    actual = sum(v['actual_updates'] for v in own.values() if v.get('actual_updates') is not None)
    own_upper = sum(v['reserved_updates'] if v.get('actual_updates') is None else v['actual_updates'] for v in own.values())
    global_upper = sum(v['reserved_updates'] if v.get('actual_updates') is None else v['actual_updates'] for v in d['runs'].values())
    if (require_exact320 and actual != 320) or own_upper > 320 or global_upper > 1024: raise RuntimeError('Exact320 helper updates/global technical limit differ')
    if not require_exact320 and actual < 128: raise RuntimeError('Lower concurrency smoke/profile128 successful updates not established')
    return {'actual_helper_updates': actual if not unresolved else None, 'known_successful_helper_updates': actual,
            'helper_charged_upper_bound': own_upper, 'unresolved_reserved_segments': list(unresolved),
            'global_charged_upper_bound': global_upper, 'limit': 1024, 'segments': own}


def run(run_id):
    if not re.fullmatch(r'[A-Za-z0-9_\-]+', run_id): raise ValueError('Safe explicit run ID required')
    common.require_authorization({}, technical=True); common.ensure_space()
    if (ROOT / 'manifests/TRAINING_AUTHORIZATION.json').exists(): raise RuntimeError('Formal authorization already exists; do not mix technical and formal phases')
    if any((ROOT / 'artifacts/train').glob('*/updates.jsonl')): raise RuntimeError('Formal optimizer history already exists')
    lock = ROOT / 'state/training_technical_pipeline.lock'; lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = identity(); folder = ROOT / 'state/technical_training' / run_id; folder.mkdir(parents=True, exist_ok=True)
        saved = folder / 'IDENTITY.json'
        if saved.exists():
            if read(saved) != frozen: raise RuntimeError('Technical run source identity differs; no automatic restart')
        else: write(saved, frozen)
        ledger_path = ROOT / 'state/TECHNICAL_LEDGER.json'
        if ledger_path.exists():
            ledger = read(ledger_path)['runs']
            charged = lambda v: v['reserved_updates'] if v.get('actual_updates') is None else v['actual_updates']
            used = sum(charged(v) for v in ledger.values())
            own = sum(charged(v) for k, v in ledger.items() if k.startswith(run_id + '_'))
            if used + max(0, 320 - own) > 1024: raise RuntimeError('Insufficient existing global technical budget for the fixed remaining matrix')
        inventory = gpu_inventory(); jobs = common.jobs()
        selected = {task: next(j for j in jobs if j['task'] == task and int(j['refit_seed']) == 103201) for task in TASKS}
        def spec(task, suffix, gpu, slot=0, endpoint=8, stop_after=None):
            return {'job': selected[task], 'run_id': run_id + '_' + suffix, 'gpu': gpu, 'slot': slot, 'endpoint': endpoint, 'stop_after': stop_after}
        smoke_path = folder / 'SMOKE_COMPLETE.json'
        if smoke_path.exists():
            smoke = read(smoke_path)
            if smoke['status'] != 'PASS' or smoke['identity_sha256'] != digest(frozen): raise RuntimeError('Smoke receipt differs')
            verify_records(smoke['files'])
        else:
            continuous = [spec(t, t + '_continuous8', g) for g, t in enumerate(TASKS)]
            split4 = [spec(t, t + '_split8', g, stop_after=4) for g, t in enumerate(TASKS)]
            split8 = [{**s, 'stop_after': None} for s in split4]
            run_phase(folder, 'smoke_continuous8', continuous, frozen)
            # The partial4 stage's endpoint files are intentionally replaced by
            # the SAME run's resume8 checkpoint; preserve its process receipt.
            partial_path = folder / 'smoke_split4.json'
            if not partial_path.exists(): run_phase(folder, 'smoke_split4', split4, frozen)
            elif read(partial_path)['status'] != 'PASS': raise RuntimeError('Interrupted split4 launch requires explicit audit')
            run_phase(folder, 'smoke_resume4', split8, frozen)
            checks = [compare_smoke(c, s) for c, s in zip(continuous, split8)]
            smoke = {'status': 'PASS', 'identity_sha256': digest(frozen), 'tasks': checks, 'optimizer_updates': 32,
                     'files': [x for c in checks for x in c['files']] + [record(partial_path)], 'genuine_process_exit_resume': True}
            write(smoke_path, smoke)
        write(ROOT / 'state/GPU_SMOKE.json', smoke)
        a_specs = [spec(t, 'profile_A_' + t, g, endpoint=STEPS) for g, t in enumerate(TASKS)]
        b_specs = [spec(t, 'profile_B_' + t + '_gpu' + str(g), g, slot=slot, endpoint=STEPS)
                   for g in (0, 1) for slot, t in enumerate(TASKS)]
        a = run_phase(folder, 'profile_A', a_specs, frozen, True)
        try:
            b = run_phase(folder, 'profile_B', b_specs, frozen, True)
        except RuntimeError:
            failed_path = folder / 'profile_B.json'
            if not failed_path.exists(): raise
            b = read(failed_path)
            if b.get('status') != 'FAILED': raise
            verify_records(b['files'])
            failures = []
            for item in b['failures']:
                match = next(spec for spec in b_specs if spec['run_id'] == item['run_id'])
                receipts = sorted(worker_folder(match).glob('failure_*.json'))
                if not receipts: raise RuntimeError('Higher-concurrency failure has no audited optimizer receipt')
                current = [read(p) for p in receipts]
                if not all('outofmemory' in x['type'].replace('_', '').lower() or 'out of memory' in x['message'].lower() for x in current):
                    raise RuntimeError('Higher-concurrency non-OOM failure needs explicit technical audit')
                failures += [record(p) for p in receipts]
            b = {**b, 'higher_concurrency_rejected': 'AUDITED_OUT_OF_MEMORY', 'files': b['files'] + failures}
        if b['status'] == 'PASS':
            selection = choose_profile(a, b); budget = technical_ledger_summary(run_id)
        else:
            selection = {'selected_workers': 2, 'workers_per_gpu': 1, 'microbatch': 128,
                         'end_to_end_gain_B_over_A': None, 'conservative_steady_gain_B_over_A': None,
                         'reason': 'ONE_PER_GPU_PASSED_HIGHER_CONCURRENCY_OOM_PRESERVED', 'scientific_outcomes_used': False}
            budget = technical_ledger_summary(run_id, require_exact320=False)
        profile = {'status': 'PROFILE_COMPLETE', 'identity_sha256': digest(frozen), **selection, 'A': a, 'B': b,
                   'gpu_inventory': inventory, 'budget': budget, 'no_six_worker_scan': True,
                   'technical_only_no_formal_launch': True}
        write(ROOT / 'state/GPU_PROFILE.json', profile)
        report = ROOT / 'reports/GPU_PROFILE.csv'; report.parent.mkdir(parents=True, exist_ok=True)
        fields = ['stage', 'workers', 'updates', 'launch_wall_seconds', 'end_to_end_updates_per_second', 'steady_lower_updates_per_second', 'steady_upper_updates_per_second', 'sum_worker_compute_seconds', 'sum_worker_loader_seconds']
        with report.open('w') as output:
            writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
            for label, count, phase in [('A', 2, a), ('B', 4, b)]:
                steady = phase.get('steady_update_span', {})
                writer.writerow({'stage': label, 'workers': count, 'updates': phase.get('total_successful_updates'),
                    'launch_wall_seconds': phase['launch_wall_seconds'], 'end_to_end_updates_per_second': phase.get('end_to_end_updates_per_second'),
                    'steady_lower_updates_per_second': steady.get('aggregate_updates_per_second_lower'), 'steady_upper_updates_per_second': steady.get('aggregate_updates_per_second_upper'),
                    'sum_worker_compute_seconds': phase.get('sum_worker_compute_seconds'), 'sum_worker_loader_seconds': phase.get('sum_worker_loader_seconds')})
        gate = {'status': 'TRAINING_TECHNICAL_GATES_PASS', 'identity_sha256': digest(frozen), **selection, 'budget': budget,
                'smoke_receipt': record(ROOT / 'state/GPU_SMOKE.json'), 'profile_receipt': record(ROOT / 'state/GPU_PROFILE.json'),
                'profile_table': record(report), 'formal_training_launched': False, 'formal_updates': 0,
                'planning_environment_gate_covered': False, 'source_files': frozen['files']}
        assert_identity(frozen); write(ROOT / 'state/TRAINING_TECHNICAL_GATE.json', gate)
        return gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', default='r3_training_tech_v1'); args = parser.parse_args()
    try: print(json.dumps(run(args.run_id)), flush=True)
    except BaseException as error:
        print(json.dumps({'status': 'TECHNICAL_PIPELINE_BLOCKED', 'type': type(error).__name__, 'reason': str(error),
                          'no_automatic_extra_updates': True, 'formal_launched': False}), flush=True)
        raise SystemExit(1) from None
