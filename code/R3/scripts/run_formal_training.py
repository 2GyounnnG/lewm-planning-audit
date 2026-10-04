"""Fixed six-job R3 controller; no retry, resume, model choice, or extra updates.

Only the root-approved technical gate chooses two or four worker slots. A failed
worker stops new launches; already running authorized workers finish naturally.
Import and metadata tests do not query hardware or start any worker.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3 import common

ROOT = common.ROOT
VERSION = 'R3_FIXED_FORMAL_TRAINING_CONTROLLER_V1'
AUTH = 'manifests/TRAINING_AUTHORIZATION.json'
GATE = 'state/TRAINING_TECHNICAL_GATE.json'
DIRECTORY = 'state/formal_training'
COMPLETE = 'state/FORMAL_TRAINING_COMPLETE.json'
REQUIRED_HASHES = {'scripts/run_formal_training.py', 'r3/common.py', 'r3/model.py',
                   'r3/train_worker.py', 'r3/data.py', 'manifests/JOBS.json',
                   'manifests/EXPERIMENT_LOCK.json', GATE, 'state/GPU_SMOKE.json',
                   'state/PLANNING_TECH_GATE.json'}


def local_path(relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts: raise ValueError('ROOT-relative path required')
    q = ROOT / p
    if not q.resolve().is_relative_to(ROOT.resolve()): raise ValueError('Path escapes R3 root')
    return q


def read(path): return common.read_json(local_path(path) if not Path(path).is_absolute() else path)
def record(path): return {'sha256': common.sha256(path), 'bytes': Path(path).stat().st_size}
def digest(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def append(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())


def verify_sources(hashes):
    for name, expected in hashes.items():
        if not isinstance(expected, str) or common.sha256(local_path(name)) != expected:
            raise RuntimeError('Frozen formal input differs: ' + name)


def validate_jobs(jobs):
    expected = {(task, seed) for task in common.TASKS for seed in common.SEEDS}
    if len(jobs) != 6 or {(x['task'], x['refit_seed']) for x in jobs} != expected:
        raise ValueError('Exactly the fixed two-task, three-seed matrix is required')
    ids = [x['job_id'] for x in jobs]
    if len(set(ids)) != 6: raise ValueError('Duplicate job identity')
    for job in jobs:
        expected_id = f"R3_{job['task']}_PRED_REFIT_s{job['refit_seed']}"
        if job['job_id'] != expected_id or job['updates'] != 30000 or job['effective_batch'] != 128 or job['microbatch'] != 128:
            raise ValueError('Job differs from fixed formal identity/budget/batch')
    return jobs


def slots_for(workers):
    if type(workers) is not int or workers not in (2, 4): raise ValueError('Only frozen two/four-worker profiles are allowed')
    # FIFO first fills one slot on each card, then the second slot on each card.
    return [(gpu, slot) for slot in range(workers // 2) for gpu in (0, 1)]


def build_plan():
    jobs = validate_jobs(common.jobs()); auth = read(AUTH)
    if auth.get('status') != 'TECHNICAL_GATES_PASS' or sorted(auth.get('job_ids', [])) != sorted(x['job_id'] for x in jobs):
        raise RuntimeError('Exact six-job technical authorization is missing')
    hashes = auth['source_hashes']
    if not REQUIRED_HASHES.issubset(hashes): raise RuntimeError('Formal authorization omits required code/gates')
    verify_sources(hashes)
    gate = read(GATE); smoke = read('state/GPU_SMOKE.json'); cem = read('state/PLANNING_TECH_GATE.json')
    if gate.get('status') != 'TRAINING_TECHNICAL_GATES_PASS' or gate.get('microbatch') != 128:
        raise RuntimeError('Actual fixed-microbatch training technical gate is missing')
    slots = slots_for(gate.get('selected_workers'))
    if gate.get('workers_per_gpu') != len(slots) // 2: raise RuntimeError('Profile slot count disagrees')
    if smoke.get('status') != 'PASS' or smoke.get('genuine_process_exit_resume') is not True or smoke.get('optimizer_updates') != 32:
        raise RuntimeError('Actual process exit/resume smoke is missing')
    if cem.get('status') != 'PASS' or cem.get('complete_trajectories') != 8 or cem.get('clone_exact') is not True:
        raise RuntimeError('Eight complete CEM clone trajectories must pass first')
    data_checks = [read(name) for name in hashes if Path(name).name == 'REAL_DATA_MODEL_CHECK.json']
    if len(data_checks) != 2 or {x.get('task') for x in data_checks} != set(common.TASKS) or any(x.get('status') != 'PASS' for x in data_checks):
        raise RuntimeError('Both actual data/model technical checks must be PASS and hash-bound')
    cost_checks = [read(name) for name in hashes if Path(name).name == 'COST_WRAPPER_CHECK.json']
    if len(cost_checks) != 2 or {x.get('task') for x in cost_checks} != set(common.TASKS) or any(x.get('status') != 'PASS' for x in cost_checks):
        raise RuntimeError('Both actual cost-wrapper checks must be PASS and hash-bound')
    return {'version': VERSION, 'jobs': jobs, 'job_order': [j['job_id'] for j in jobs],
            'authorization_sha256': common.sha256(local_path(AUTH)), 'source_hashes': hashes,
            'selected_workers': gate['selected_workers'], 'slots': [list(x) for x in slots],
            'microbatch': 128, 'formal_updates': 180000, 'automatic_resume': False,
            'failure_policy': 'STOP_NEW_LAUNCHES_ALLOW_RUNNING_AUTHORIZED_WORKERS_TO_FINISH',
            'scheduler_rule': 'FROZEN_JOBS_ORDER_FIFO_FIRST_AVAILABLE_PROFILE_SLOT'}


def assert_plan(plan):
    if common.sha256(local_path(AUTH)) != plan['authorization_sha256']: raise RuntimeError('Formal authorization changed')
    verify_sources(plan['source_hashes'])


def verify_complete(job):
    """Hash every published byte and verify exact update accounting, without loading a model."""
    folder = local_path('artifacts/train/' + job['job_id'])
    if not folder.exists(): return None
    entries = list(folder.iterdir())
    if not entries: return None
    # Even a zero-update prior attempt needs explicit review before another run.
    receipt_path = folder / 'JOB_SHA256.json'
    if not receipt_path.exists(): raise RuntimeError('Incomplete prior job; explicit review required: ' + job['job_id'])
    receipt = read(receipt_path); queue_path = local_path('state/recovery_queue/' + job['job_id'] + '.json')
    if receipt.get('job_id') != job['job_id'] or not queue_path.exists() or read(queue_path) != receipt:
        raise RuntimeError('Completed job publication/queue identity differs')
    actual_files = {str(p.relative_to(ROOT)) for p in folder.rglob('*') if p.is_file() and p.name != 'JOB_SHA256.json' and not p.name.endswith('.tmp')}
    if set(receipt['files']) != actual_files: raise RuntimeError('Published job inventory differs')
    for name, expected in receipt['files'].items():
        path = local_path(name)
        if not path.resolve().is_relative_to(folder.resolve()) or record(path) != expected: raise RuntimeError('Published job bytes differ: ' + name)
    needed = {'RUN_IDENTITY.json', 'result.json', 'updates.jsonl', 'monitor.jsonl', 'last.json', 'resume.json', 'resume.pt',
              'checkpoint_3000.pt', 'checkpoint_10000.pt', 'checkpoint_30000.pt', 'sampling_counts.npz', 'FROZEN_INITIAL.json', 'PARAMETER_WHITELIST.json'}
    if not needed.issubset({p.name for p in folder.iterdir()}): raise RuntimeError('Complete formal artifact set missing')
    if (folder / 'OPTIMIZER_INFLIGHT.json').exists() or list(folder.glob('failure_*.json')):
        raise RuntimeError('Prior optimizer failure/inflight evidence requires explicit review')
    identity = read(folder / 'RUN_IDENTITY.json'); result = read(folder / 'result.json')
    if identity.get('job') != job or identity.get('technical') is not False or identity.get('endpoint') != 30000 or identity.get('microbatch') != 128:
        raise RuntimeError('Completed run construction differs')
    if identity['inputs'] != common.identity_hashes(job): raise RuntimeError('Completed job input identity differs')
    for name, expected in identity['source'].items():
        if common.sha256(ROOT / 'r3' / name) != expected: raise RuntimeError('Completed job source differs')
    if (result.get('status') != 'REFIT_TRAINING_COMPLETE_UNSCORED' or result.get('actual_updates') != 30000
        or result.get('job_id') != job['job_id'] or result.get('task') != job['task'] or result.get('technical') is not False
        or result.get('identity_sha256') != digest(identity) or result.get('frozen_before') != result.get('frozen_after')
        or result.get('base_identity') != identity.get('base_identity') or result.get('microbatch') != 128):
        raise RuntimeError('Completed result/frozen/identity check failed')
    rows = [json.loads(line) for line in (folder / 'updates.jsonl').read_text().splitlines()]
    if len(rows) != 30000 or any(row.get('step') != i or row.get('technical') is not False or row.get('sampled_windows') != 128 for i, row in enumerate(rows, 1)):
        raise RuntimeError('Formal successful-update journal must contain exact steps1..30000')
    monitor = [json.loads(line) for line in (folder / 'monitor.jsonl').read_text().splitlines()]
    if [x.get('step') for x in monitor] != list(range(0, 30001, 1000)): raise RuntimeError('Fixed MONITOR observations are incomplete')
    for pointer_name, result_key, expected_path in [('last.json', 'last_checkpoint', 'checkpoint_30000.pt'), ('resume.json', 'resume_checkpoint', 'resume.pt')]:
        pointer = read(folder / pointer_name)
        if pointer.get('step') != 30000 or pointer.get('path') != expected_path or common.sha256(folder / expected_path) != pointer['sha256'] or result[result_key] != pointer:
            raise RuntimeError('Completed checkpoint pointer differs')
    return {'job_id': job['job_id'], 'actual_updates': 30000, 'result': {'path': str((folder/'result.json').relative_to(ROOT)), **record(folder/'result.json')},
            'publication': {'path': str(receipt_path.relative_to(ROOT)), **record(receipt_path)},
            'recovery_queue': {'path': str(queue_path.relative_to(ROOT)), **record(queue_path)}}


def worker_command(job, slot):
    return [sys.executable, '-m', 'r3.train_worker', '--job', job['job_id'], '--device', 'cuda', '--microbatch', '128', '--slot', str(slot)]


def worker_environment(gpu):
    return dict(os.environ, R3_ROOT=str(ROOT), CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='8',
                OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')


def compute_pids():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    return sorted({int(x.strip()) for x in out.splitlines() if x.strip()})


def gpu_sample():
    command = ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits']
    output = subprocess.check_output(command, text=True, timeout=10)
    rows = list(csv.reader(output.splitlines()))
    if [int(row[0]) for row in rows] != [0, 1] or any(len(row) != 7 for row in rows): raise RuntimeError('Unexpected two-GPU telemetry')
    return {'utc': common.now(), 'time_unix': time.time(), 'query': command, 'raw_csv': output,
            'compute_pids': compute_pids(), 'proc_meminfo': Path('/proc/meminfo').read_text(), 'loadavg': list(os.getloadavg())}


def progress(job_id):
    path = local_path('artifacts/train/' + job_id + '/updates.jsonl')
    if not path.exists(): return {'last_successful_update': 0}
    # Read only the last committed record; retain all raw journals in the job.
    with path.open('rb') as f:
        size = f.seek(0, 2); f.seek(max(0, size - 16384)); lines = f.read().splitlines()
    if not lines: return {'last_successful_update': 0}
    try: value = json.loads(lines[-1])
    except (ValueError, UnicodeDecodeError):
        if len(lines) < 2: return {'last_successful_update': None, 'partial_append_observed': True}
        value = json.loads(lines[-2])
    return {'last_successful_update': value['step'], 'last_journal_record': value}


def control_loop(plan, directory, inherited_fds, completed, *, popen=subprocess.Popen, sample=gpu_sample, sleep=time.sleep, poll_seconds=5.):
    pending = deque(j for j in plan['jobs'] if j['job_id'] not in completed)
    running = {}; errors = []; logs = []; stop = {'requested': False}; previous_handlers = {}
    events = directory/'EVENTS.jsonl'; snapshots = directory/'PROCESS_STATUS.jsonl'; telemetry = directory/'GPU_TELEMETRY.jsonl'
    started = time.monotonic()
    def fail(reason):
        errors.append(reason); stop['requested'] = True
        common.atomic_json(directory/'STOP_REQUESTED.json', {'utc': common.now(), 'errors': errors,
            'no_new_launches': True, 'running_authorized_workers_allowed_to_finish': True})
        append(events, {'utc': common.now(), 'event': 'STOP_NEW_LAUNCHES', 'reason': reason})
    def interrupted(signum, _frame): fail({'reason': 'CONTROLLER_SIGNAL', 'signal': signum})
    for sig in (signal.SIGINT, signal.SIGTERM): previous_handlers[sig] = signal.signal(sig, interrupted)
    try:
        while pending or running:
            # Collect every exit before considering another launch.
            for slot, entry in list(running.items()):
                process, job = entry['process'], entry['job']; code = process.poll()
                if code is None: continue
                if code != 0: fail({'reason': 'WORKER_EXIT', 'job_id': job['job_id'], 'returncode': code})
                else:
                    try:
                        result = verify_complete(job)
                        if result is None: raise RuntimeError('Worker exited0 without complete publication')
                        completed[job['job_id']] = result
                    except Exception as error: fail({'reason': 'COMPLETION_VERIFICATION_FAILED', 'job_id': job['job_id'], 'error': str(error)})
                append(events, {'utc': common.now(), 'event': 'WORKER_EXIT', 'job_id': job['job_id'], 'pid': process.pid, 'returncode': code})
                del running[slot]
            for slot in map(tuple, plan['slots']):
                if stop['requested'] or not pending: break
                if slot in running: continue
                job = pending[0]
                try:
                    assert_plan(plan); common.require_authorization(job); common.ensure_space()
                    # No automatic resume even if another process created evidence since preflight.
                    if verify_complete(job) is not None: raise RuntimeError('Pending job appeared after controller preflight')
                    path = directory/(job['job_id']+'.log'); log = path.open('ab'); logs.append(log)
                    process = popen(worker_command(job, slot[1]), cwd=ROOT, env=worker_environment(slot[0]),
                                    stdout=log, stderr=subprocess.STDOUT, pass_fds=inherited_fds)
                    pending.popleft(); running[slot] = {'process': process, 'job': job}
                    append(events, {'utc': common.now(), 'event': 'WORKER_LAUNCHED', 'job_id': job['job_id'], 'pid': process.pid,
                                    'gpu': slot[0], 'slot': slot[1], 'command': worker_command(job, slot[1])})
                except Exception as error: fail({'reason': 'LAUNCH_PREFLIGHT_OR_SPAWN_FAILED', 'job_id': job['job_id'], 'error': str(error)}); break
            if not running: break
            try: append(telemetry, sample())
            except Exception as error: fail({'reason': 'TELEMETRY_FAILED', 'error': str(error)})
            state = {'utc': common.now(), 'status': 'DRAINING_AFTER_FAILURE' if stop['requested'] else 'RUNNING',
                     'completed_jobs': len(completed), 'completed_updates': 30000*len(completed),
                     'pending': [j['job_id'] for j in pending], 'active': [], 'controller_wall_seconds': time.monotonic()-started}
            for (gpu, slot), entry in running.items():
                try: update = progress(entry['job']['job_id'])
                except Exception as error: update = {'progress_read_error': str(error)}
                state['active'].append({'job_id': entry['job']['job_id'], 'pid': entry['process'].pid,
                                        'gpu': gpu, 'slot': slot, 'returncode': entry['process'].poll(), **update})
            append(snapshots, state); common.atomic_json(directory/'STATUS.json', state)
            sleep(poll_seconds)
        return {'completed': completed, 'errors': errors, 'pending': [j['job_id'] for j in pending],
                'controller_wall_seconds': time.monotonic()-started}
    finally:
        # No terminate/kill. Inherited controller/phase flock descriptors also
        # prevent a second controller/planner if this process exits unexpectedly.
        for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
        for log in logs: log.close()


def run():
    directory = local_path(DIRECTORY); directory.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        handles = []
        for path, mode in [(directory/'controller.lock', fcntl.LOCK_EX), (ROOT/'state/execution_phase.lock', fcntl.LOCK_SH)]:
            f = stack.enter_context(path.open('a+')); fcntl.flock(f, mode|fcntl.LOCK_NB); handles.append(f)
        plan = build_plan()
        for job in plan['jobs']: common.require_authorization(job)
        common.ensure_space()
        completed = {j['job_id']: result for j in plan['jobs'] if (result := verify_complete(j)) is not None}
        saved_plan = directory/'EXECUTION_PLAN.json'
        if saved_plan.exists() and read(saved_plan) != plan: raise RuntimeError('Existing controller plan differs')
        if not saved_plan.exists(): common.atomic_json(saved_plan, plan)
        if local_path(COMPLETE).exists():
            result = read(COMPLETE)
            if len(completed) != 6 or result.get('execution_plan_sha256') != digest(plan) or result.get('jobs') != completed:
                raise RuntimeError('Completed matrix differs from current verified publication')
            return result
        if (directory/'STOP_REQUESTED.json').exists(): raise RuntimeError('Prior interruption/failure requires explicit review; no automatic resume')
        pids = compute_pids()
        if pids: raise RuntimeError('Authorized GPUs must both be idle before launch; no signals sent: '+str(pids))
        summary = control_loop(plan, directory, tuple(f.fileno() for f in handles), completed)
        assert_plan(plan)
        if summary['errors'] or len(summary['completed']) != 6:
            common.atomic_json(directory/'STATUS.json', {'status': 'BLOCKED_EXPLICIT_REVIEW_REQUIRED', **summary})
            raise RuntimeError('Formal matrix incomplete/failed; preserved existing workers/results, no automatic retry')
        # Full final audit, including jobs that were already complete at entry.
        final = {j['job_id']: verify_complete(j) for j in plan['jobs']}
        result = {'version': VERSION, 'status': 'COMPLETE', 'execution_plan_sha256': digest(plan),
                  'jobs': final, 'complete_jobs': 6, 'actual_updates': 180000, 'selected_workers': plan['selected_workers'],
                  'microbatch': 128, 'new_scientific_jobs_added': 0, 'automatic_retries': 0,
                  'controller_wall_seconds': summary['controller_wall_seconds'], 'completed_at': common.now()}
        common.atomic_json(COMPLETE, result); common.atomic_json(directory/'STATUS.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    result = run(); print(json.dumps({'status': result['status'], 'complete_jobs': result['complete_jobs'], 'actual_updates': result['actual_updates']}), flush=True)


if __name__ == '__main__': main()
