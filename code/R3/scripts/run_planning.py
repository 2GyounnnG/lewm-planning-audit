"""Root-invoked fixed R3 CEM controller. Importing launches nothing.

The parent holds an exclusive execution-phase lock, inherited by at most one
worker per GPU. Each case's arms run sequentially on its SHA-assigned GPU.
Interrupted attempts block explicit review; this controller never auto-retries.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3 import common
from r3 import planning
from r3 import reset_fallback
from r3.open_loop import local_path, locked_record, validate_routes, verify_lock, validate_checkpoint

ROOT = common.ROOT
VERSION = 'R3_FIXED_OFFICIAL_CEM_CONTROLLER_V1'
PREFLIGHT = 'state/environment_preflight/attempt_001/ENVIRONMENT_PREFLIGHT.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def record(path):
    p = Path(path)
    return {'sha256': common.sha256(p), 'bytes': p.stat().st_size}


def safe_component(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}', value) or value in ('.', '..'):
        raise ValueError('Unsafe case/arm path component')
    return value


def gpu_for_case(case_id):
    return int.from_bytes(hashlib.sha256(case_id.encode()).digest(), 'big') % 2


def select_cases(roles, task, phase):
    if roles['task'] != task or roles.get('status') == 'BLOCKED_DATA': raise RuntimeError('Task roles are blocked/different')
    key = 'TECH' if phase == 'TECH' else 'EVAL'
    raw_cases = roles['cases'][key][:2] if phase == 'TECH' else roles['cases'][key]
    cases = [reset_fallback.effective_case(task,c) for c in raw_cases]
    if (phase == 'TECH' and len(cases) != 2) or (phase == 'FORMAL' and not 20 <= len(cases) <= 100):
        raise RuntimeError('Case count violates fixed TECH/formal matrix')
    if len({c['case_id'] for c in cases}) != len(cases) or len({c['episode_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case/episode')
    rows = {x['episode_id']: x for x in roles['episodes']}
    for case in cases:
        safe_component(case['case_id']); planning.validate_case(task, case, phase)
        row = rows[case['episode_id']]
        if row['role'] != key or case['role'] != key or case['task'] != task: raise ValueError('Planning role mismatch')
        for k in ('source_asset_sha256', 'episode_sha256', 'source_episode_idx', 'length', 'family_id'):
            if case[k] != row[k]: raise ValueError('Planning/source metadata mismatch')
        if case['start_raw_index'] not in row['planning_starts']: raise ValueError('Start was not metadata-eligible')
    if phase == 'FORMAL' and {c['episode_id'] for c in cases} != {e for e, r in rows.items() if r['role'] == key}:
        raise ValueError('Formal cases do not exactly cover frozen EVAL role')
    return cases


def assert_environment(preflight):
    environment = common.read_json('state/ENVIRONMENT_READY.json'); gate = common.read_json(preflight)
    if environment.get('status') != 'ISOLATED_ENVIRONMENT_READY' or environment.get('old_environment_unchanged') is not True:
        raise RuntimeError('Isolated environment identity not established')
    if gate.get('status') != 'PASS' or set(gate.get('tasks', {})) != set(common.TASKS):
        raise RuntimeError('Both actual environment reset/single-step checks must pass')
    if any(x.get('status') != 'PASS' or not x.get('close_called') for x in gate['tasks'].values()):
        raise RuntimeError('Environment preflight contains incomplete task')
    if gate['counts']['environment_resets'] != 2 or gate['counts']['raw_environment_steps'] != 2:
        raise RuntimeError('Unexpected preflight reset/step count')


def verify_completed_training(route, task):
    identity = common.read_json(route['run_identity']); result = common.read_json(route['result']); job = identity['job']
    if (job['task'] != task or job['refit_seed'] != route['seed'] or job['updates'] != 30000
            or identity.get('technical') is not False or result.get('technical') is not False
            or result.get('actual_updates') != 30000 or result.get('status') != 'REFIT_TRAINING_COMPLETE_UNSCORED'
            or result['identity_sha256'] != digest(identity) or result['frozen_before'] != result['frozen_after']):
        raise RuntimeError('All six fixed30k jobs must be complete before formal planning')
    folder = local_path(route['run_identity']).parent
    with (folder/'updates.jsonl').open() as f:
        count = 0
        for count, line in enumerate(f, 1):
            row = json.loads(line)
            if row['step'] != count or row.get('technical') is not False or row['sampled_windows'] != 128 or row['predicted_tokens'] != 384:
                raise RuntimeError('Formal training update journal differs')
    if count != 30000: raise RuntimeError('Formal training journal is incomplete')
    if (folder/'OPTIMIZER_INFLIGHT.json').exists(): raise RuntimeError('Uncommitted optimizer intent remains')
    return str((folder/'updates.jsonl').relative_to(ROOT))


def build_plan(phase, routing, preflight):
    if phase not in ('TECH', 'FORMAL'): raise ValueError('Explicit TECH or FORMAL phase required')
    assert_environment(preflight); planning.verify_official_sources()
    paths = ['scripts/run_planning.py', 'r3/planning.py', 'r3/open_loop.py', 'r3/model.py', 'r3/common.py', 'r3/data.py', 'r3/roles.py', 'r3/reset_fallback.py', 'r3/env_compat.py',
             'manifests/JOBS.json', 'manifests/EXPERIMENT_LOCK.json', 'state/INSTANCE_REUSE.json',
             'state/ENVIRONMENT_READY.json', preflight, 'state/lewm_source_manifest.json', 'state/spt_source_manifest.json', 'state/swm_compat_source_manifest.json']
    paths += ['source/swm_compat/'+p for p in planning.SOURCE_SHA256]
    paths += ['source/lewm/'+p for p in ('jepa.py', 'module.py', 'train.py', 'utils.py')]
    paths += ['source/spt/stable_pretraining/backbone/utils.py']
    tasks = {}; flat = []; all_routes = common.read_json(routing) if phase == 'FORMAL' else None
    if phase == 'FORMAL': paths += [routing, 'state/PLANNING_TECH_GATE.json']
    for task in common.TASKS:
        names = [f'manifests/{task}_{k}.json' for k in ('data_roles', 'source_map', 'normalization', 'model_assets')]; paths += names
        roles = common.read_json(names[0]); cases = select_cases(roles, task, phase)
        paths += reset_fallback.evidence_paths(task,roles)
        source = common.read_json(names[1]); assets = common.read_json(names[3])
        for item in assets['files'].values():
            path = local_path(item['path'])
            if record(path) != {'sha256': item['sha256'], 'bytes': item['bytes']}: raise RuntimeError('Official weight/config asset differs')
            paths.append(item['path'])
        sources = {}
        for c in cases:
            item = source['assets'][c['source_asset_sha256']]; path = local_path(item['path'])
            if item['sha256'] != c['source_asset_sha256']: raise RuntimeError('Case source SHA differs')
            if item['path'] not in paths:
                if record(path) != {'sha256': item['sha256'], 'bytes': item['bytes']}: raise RuntimeError('Actual source HDF5 differs')
                paths.append(item['path'])
            sources[c['source_asset_sha256']] = item
            flat.append({'task': task, 'case': c, 'gpu': gpu_for_case(c['case_id'])})
        if phase == 'TECH': selected = [{'arm': 'H0', 'step': 0, 'seed': None}, {'arm': 'H0_CLONE', 'step': 0, 'seed': None}]
        else:
            selected = [r for r in validate_routes(all_routes, task) if r['step'] in (0, 30000)]
            for r in selected:
                if r['step']:
                    paths += [r['checkpoint']['path'], r['run_identity'], r['result'], verify_completed_training(r, task)]
        tasks[task] = {'arms': selected, 'sources': sources, 'role_manifest': names[0], 'normalization': names[2]}
    locked = None
    if phase == 'FORMAL':
        gate = common.read_json('state/PLANNING_TECH_GATE.json')
        if gate.get('status') != 'PASS' or gate.get('complete_trajectories') != 8 or gate.get('clone_exact') is not True:
            raise RuntimeError('Real full-CEM eight-trajectory technical clone gate must pass')
        for rel, expected in gate['files'].items():
            if record(local_path(rel)) != expected: raise RuntimeError('Planning technical gate evidence changed')
        locked = verify_lock(paths)
        for task in common.TASKS:
            for r in tasks[task]['arms']:
                if r['step'] and locked_record(locked['files'][r['checkpoint']['path']])[0] != r['checkpoint']['sha256']:
                    raise RuntimeError('30k explicit checkpoint route differs from model lock')
        paths.append('manifests/MODELS_AND_SELECTION_LOCK.json')
    files = {p: record(local_path(p)) for p in sorted(set(paths))}
    return {'version': VERSION, 'phase': phase, 'tasks': tasks, 'cases': flat, 'files': files,
            'models_lock_sha256': None if locked is None else locked['sha256'],
            'assignment': 'int.from_bytes(SHA256(case_id), big) % 2; all arms same card sequential',
            'max_workers_per_gpu': 1, 'expected_complete_trajectories': sum(len(tasks[x['task']]['arms']) for x in flat),
            'case_replacement': False, 'automatic_trajectory_retry': False,
            'separate_warmup_trajectories': 0, 'warmup_accounting': 'cold kernels remain in first counted trajectory; model/setup separately measured',
            'selection_basis': 'frozen source metadata only; no success-gain gate'}


def verify_plan(plan):
    for rel, expected in plan['files'].items():
        if record(local_path(rel)) != expected: raise RuntimeError('Frozen planning input/source differs: '+rel)


def trajectory_folder(plan, task, case, arm):
    return ROOT/'artifacts/planning'/plan['phase']/task/safe_component(case['case_id'])/safe_component(arm)/'attempt_0'


def run_id(plan, task, case, arm):
    return f"R3/{plan['phase']}/{task}/{case['case_id']}/{arm}/attempt_0"


def ledger_record(rid, formal):
    path = ROOT/('state/FORMAL_TRAJECTORY_LEDGER.json' if formal else 'state/TECHNICAL_LEDGER.json')
    with common.ledger_lock():
        return common.read_json(path).get('trajectories', {}).get(rid) if path.exists() else None


def complete_trajectory(folder, expected=None):
    seal = folder/'TRAJECTORY_SHA256.json'
    if not seal.exists(): return None
    saved = common.read_json(seal)
    if saved.get('status') != 'COMPLETE': raise RuntimeError('Invalid trajectory completion seal')
    for name, expected_file in saved['files'].items():
        if Path(name).name != name or record(folder/name) != expected_file: raise RuntimeError('Trajectory artifact differs')
    result = common.read_json(folder/'result.json')
    if result.get('status') != 'COMPLETE' or not result.get('evaluation_complete') or result['identity_sha256'] != saved['identity_sha256']:
        raise RuntimeError('Incomplete/changed trajectory result')
    if expected is not None:
        encoded = hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if result['identity'] != expected or result['identity_sha256'] != encoded: raise RuntimeError('Saved trajectory has a different identity')
    return result


def compare_clone(h0_folder, clone_folder):
    import numpy as np
    h0 = complete_trajectory(h0_folder); clone = complete_trajectory(clone_folder)
    if h0 is None or clone is None: raise RuntimeError('Incomplete technical clone pair')
    if h0['failure_category'] is not None or clone['failure_category'] is not None: raise RuntimeError('Method/infra failure cannot pass technical clone gate')
    keys = ('raw_actions', 'step_success', 'step_truncated', 'physical_state', 'goal_error', 'goal_state', 'rewards')
    with np.load(h0_folder/'trajectory.npz', allow_pickle=False) as a, np.load(clone_folder/'trajectory.npz', allow_pickle=False) as b:
        for k in keys:
            if a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k]): raise RuntimeError('H0/clone exact mismatch: '+k)
            if a[k].dtype.kind in 'fc' and not np.isfinite(a[k]).all(): raise RuntimeError('Nonfinite technical trajectory: '+k)
    for k in ('success', 'any_step_success', 'terminal_success', 'executed_raw_steps', 'replan_calls', 'final_goal_error'):
        if h0[k] != clone[k]: raise RuntimeError('H0/clone result mismatch: '+k)
    return {'task': h0['task'], 'case_id': h0['case_id'], 'status': 'PASS', 'exact_fields': list(keys),
            'timing_fields_excluded_from_equality': True, 'success': h0['success'], 'raw_steps': h0['executed_raw_steps']}


def publish_case(plan, task, case):
    folder = ROOT/'artifacts/planning'/plan['phase']/task/case['case_id']; results = []
    for r in plan['tasks'][task]['arms']:
        f = trajectory_folder(plan, task, case, public_arm(r)); result = complete_trajectory(f)
        if result is None: raise RuntimeError('Cannot publish incomplete case')
        results.append(result)
    clone = compare_clone(trajectory_folder(plan, task, case, 'H0'), trajectory_folder(plan, task, case, 'H0_CLONE')) if plan['phase'] == 'TECH' else None
    files = {str(p.relative_to(ROOT)): record(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'CASE_SHA256.json' and not p.name.endswith('.tmp')}
    receipt = {'status': 'COMPLETE', 'phase': plan['phase'], 'task': task, 'case_id': case['case_id'],
               'execution_plan_sha256': digest(plan), 'complete_trajectories': len(results), 'clone_check': clone, 'files': files}
    common.atomic_json(folder/'CASE_SHA256.json', receipt)
    name = f"planning_{plan['phase']}_{task}_{case['case_id']}.json"
    common.atomic_json(ROOT/'state/recovery_queue'/name, receipt)
    return receipt


def public_arm(route):
    return route['arm'] if not route['step'] else f"REFIT_{route['seed']}"


def worker(plan_path, gpu, phase_fd):
    if gpu not in (0, 1) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu): raise RuntimeError('Worker must bind exactly its assigned physical GPU')
    actual = os.fstat(phase_fd); expected = (ROOT/'state/execution_phase.lock').stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino): raise RuntimeError('Wrong inherited phase-lock descriptor')
    fcntl.flock(phase_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)  # Same inherited open description; parent already owns it.
    common.require_authorization({}, technical=True); plan = common.read_json(plan_path); verify_plan(plan)
    selected = [x for x in plan['cases'] if x['gpu'] == gpu]; formal = plan['phase'] == 'FORMAL'
    directory = local_path(plan_path).parent; stop = directory/'STOP_REQUESTED.json'; started = time.perf_counter()
    info = {'status': 'STARTED', 'pid': os.getpid(), 'gpu': gpu, 'phase': plan['phase'], 'plan_sha256': digest(plan),
            'setup_seconds': 0., 'restoration_and_verification_seconds': 0., 'completed_new': 0, 'reused': 0, 'trajectories': []}
    output = directory/f'worker_gpu{gpu}_pid{os.getpid()}.json'
    common.atomic_json(output, info)
    setup_start = time.perf_counter()
    from r3.data import RawH5, action_processor, image_transform, IMAGE_CONTRACT
    from r3.model import load_official, apply_delta, delta_state, frozen_hashes, assert_frozen, tensor_sha256
    import torch
    torch.cuda.set_device(0); torch.set_num_threads(1); common.fp32_policy(); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    info['setup_seconds'] += time.perf_counter()-setup_start
    model = None; current_task = None; readers = {}; signatures = {}
    def model_sha(): return digest({k: tensor_sha256(v) for k, v in model.state_dict().items()})
    def file_signature(path):
        s = path.stat(); return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    try:
        for spec in selected:
            if stop.exists(): raise RuntimeError('Controller requested stop before next fixed case; no case replaced')
            common.ensure_space(); task, case = spec['task'], spec['case']; t = time.perf_counter()
            if current_task != task:
                model = None; torch.cuda.empty_cache(); model = load_official(task, 'cuda').eval().requires_grad_(False)
                h0 = delta_state(model); frozen = frozen_hashes(model); base_full_sha = model_sha()
                processor, transform = action_processor(task), image_transform(); current_task = task
            asset = plan['tasks'][task]['sources'][case['source_asset_sha256']]; source_path = local_path(asset['path'])
            if asset['path'] not in readers:
                if record(source_path) != {'bytes': asset['bytes'], 'sha256': asset['sha256']}: raise RuntimeError('Planning source changed')
                required = ['pixels', 'state'] if task == 'pusht' else ['pixels', 'qpos', 'qvel']
                reader = RawH5(source_path, keys=required)
                if 'seed' in reader.file: reader.column_names.append('seed')
                readers[asset['path']] = reader; signatures[asset['path']] = file_signature(source_path)
            dataset = readers[asset['path']]
            if int(dataset.lengths[case['source_episode_idx']]) != case['length']: raise RuntimeError('Case source length differs')
            info['setup_seconds'] += time.perf_counter()-t
            for route in plan['tasks'][task]['arms']:
                common.ensure_space(); arm = public_arm(route); folder = trajectory_folder(plan, task, case, arm); rid = run_id(plan, task, case, arm)
                if signatures[asset['path']] != file_signature(source_path): raise RuntimeError('Source file changed during planning')
                t = time.perf_counter()
                if route['step']:
                    cp_path = local_path(route['checkpoint']['path'])
                    if common.sha256(cp_path) != route['checkpoint']['sha256']: raise RuntimeError('30k delta differs from route')
                    checkpoint = torch.load(cp_path, map_location='cpu', weights_only=True)
                    validate_checkpoint(route, checkpoint, common.read_json(route['run_identity']), common.read_json(route['result']), model)
                    del checkpoint; apply_delta(model, cp_path)
                else:
                    with torch.no_grad():
                        parameters = dict(model.named_parameters())
                        for k, v in h0.items(): parameters[k].copy_(v.to(parameters[k].device))
                    if model_sha() != base_full_sha: raise RuntimeError('H0/clone original model restoration differs')
                model.eval().requires_grad_(False); assert_frozen(model, frozen); before = model_sha()
                provenance = {'execution_plan_sha256': digest(plan), 'model_base': model.r3_identity,
                              'model_full_state_sha256': before, 'frozen_sha256': frozen['sha256'], 'route': route,
                              'source_asset': asset, 'dataset_columns': list(dataset.column_names), 'gpu': gpu,
                              'data_roles_sha256': plan['files'][plan['tasks'][task]['role_manifest']]['sha256'],
                              'normalization_sha256': plan['files'][plan['tasks'][task]['normalization']]['sha256'],
                              'image_contract': IMAGE_CONTRACT, 'precision': 'FP32_NO_AMP_NO_TF32_MATH_SDPA'}
                identity = {'task': task, 'case': case, 'arm': arm, 'phase': plan['phase'], 'plan_config': planning.PLAN_CONFIG,
                            'cem_config': planning.CEM_CONFIG, 'eval_budget_raw_steps': 50, 'source': planning.verify_official_sources(),
                            'provenance': provenance, 'planning_code_sha256': common.sha256(ROOT/'r3/planning.py')}
                saved = complete_trajectory(folder, identity); entry = ledger_record(rid, formal)
                info['restoration_and_verification_seconds'] += time.perf_counter()-t
                if saved is not None:
                    if entry is None: raise RuntimeError('Completed trajectory has no original reservation')
                    common.finish_trajectory(rid, folder/'result.json', formal=formal); result = saved; info['reused'] += 1
                else:
                    if entry is not None or (folder/'STARTED.json').exists():
                        raise RuntimeError('Interrupted/reserved trajectory blocks explicit infrastructure review; no automatic retry: '+rid)
                    common.reserve_trajectory(rid, task, case['case_id'], arm, formal=formal)
                    try:
                        result = planning.run_case(model, task, case, dataset, processor, transform, arm=arm, phase=plan['phase'],
                                                   output_dir=folder, provenance=provenance, device='cuda')
                    except BaseException:
                        if (folder/'result.json').exists(): common.finish_trajectory(rid, folder/'result.json', formal=formal)
                        raise
                    common.finish_trajectory(rid, folder/'result.json', formal=formal); info['completed_new'] += 1
                assert_frozen(model, frozen)
                if model_sha() != before: raise RuntimeError('Planning inference changed model parameters or buffers')
                info['trajectories'].append({'run_id': rid, 'result_path': str((folder/'result.json').relative_to(ROOT)),
                    'reused': saved is not None, 'trajectory_wall_seconds': result['trajectory_wall_seconds'],
                    'planning_synchronized_wall_seconds': result['planning_synchronized_wall_seconds'],
                    'device_timing': result['device_timing'], 'executed_raw_steps': result['executed_raw_steps'], 'replan_calls': result['replan_calls']})
                common.atomic_json(output, info)
            publish_case(plan, task, case)
        verify_plan(plan); info['status'] = 'COMPLETE'
    except BaseException as error:
        info.update(status='BLOCKED_REVIEW', error={'type': type(error).__name__, 'message': str(error)}); raise
    finally:
        for reader in readers.values(): reader.close()
        info['worker_wall_seconds'] = time.perf_counter()-started; common.atomic_json(output, info)
    return info


def collect(plan, directory):
    clones, results, files = [], [], {}
    for spec in plan['cases']:
        task, case = spec['task'], spec['case']; folder = ROOT/'artifacts/planning'/plan['phase']/task/case['case_id']
        receipt = common.read_json(folder/'CASE_SHA256.json')
        if receipt['execution_plan_sha256'] != digest(plan): raise RuntimeError('Published case plan identity differs')
        for rel, expected in receipt['files'].items():
            if record(local_path(rel)) != expected: raise RuntimeError('Published case artifact differs')
            files[rel] = expected
        files[str((folder/'CASE_SHA256.json').relative_to(ROOT))] = record(folder/'CASE_SHA256.json')
        for route in plan['tasks'][task]['arms']:
            f = trajectory_folder(plan, task, case, public_arm(route)); result = complete_trajectory(f)
            if result is None: raise RuntimeError('Incomplete planning matrix')
            entry = ledger_record(run_id(plan, task, case, public_arm(route)), plan['phase'] == 'FORMAL')
            if entry is None or entry.get('status') != 'COMPLETE' or entry.get('result_sha256') != common.sha256(f/'result.json'):
                raise RuntimeError('Completed trajectory ledger is absent or differs')
            results.append(result)
        if plan['phase'] == 'TECH': clones.append(compare_clone(trajectory_folder(plan, task, case, 'H0'), trajectory_folder(plan, task, case, 'H0_CLONE')))
    if len(results) != plan['expected_complete_trajectories']: raise RuntimeError('Planning matrix count differs')
    workers = [common.read_json(p) for p in sorted(directory.glob('worker_gpu*_pid*.json'))]
    for p in directory.glob('worker_gpu*_pid*.json'): files[str(p.relative_to(ROOT))] = record(p)
    files[str((directory/'EXECUTION_PLAN.json').relative_to(ROOT))] = record(directory/'EXECUTION_PLAN.json')
    device_timings = {}
    for result in results:
        for label, seconds in result['device_timing'].items(): device_timings[label] = device_timings.get(label, 0.)+seconds
    return {'status': 'PASS' if plan['phase'] == 'TECH' else 'COMPLETE', 'phase': plan['phase'], 'execution_plan_sha256': digest(plan),
            'complete_trajectories': len(results), 'clone_exact': True if clones else None, 'clone_comparisons': clones, 'files': files,
            'total_counted_trajectory_wall_seconds': sum(x['trajectory_wall_seconds'] for x in results),
            'total_planning_synchronized_wall_seconds': sum(x['planning_synchronized_wall_seconds'] for x in results),
            'total_environment_step_seconds': sum(x['environment_step_seconds'] for x in results),
            'total_executed_raw_steps': sum(x['executed_raw_steps'] for x in results), 'total_replan_calls': sum(x['replan_calls'] for x in results),
            'device_timing_totals_seconds': device_timings, 'device_timing_labels_may_be_nested_do_not_sum_across_labels': True,
            'all_worker_attempt_setup_seconds': sum(x['setup_seconds'] for x in workers),
            'all_worker_attempt_wall_seconds': sum(x['worker_wall_seconds'] for x in workers),
            'separate_warmup_trajectories': 0, 'warmup_accounting': plan['warmup_accounting'],
            'optimizer_updates': 0, 'success_gain_threshold': None, 'timings_are_measured_not_ETA': True}


def run(phase, routing='manifests/OPEN_LOOP_ROUTING.json', preflight=PREFLIGHT):
    common.require_authorization({}, technical=True); common.ensure_space()
    phase_path = ROOT/'state/execution_phase.lock'; phase_path.parent.mkdir(parents=True, exist_ok=True)
    with phase_path.open('a+') as phase_handle:
        fcntl.flock(phase_handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = build_plan(phase, routing, preflight); directory = ROOT/'state/planning'/phase; directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory/'EXECUTION_PLAN.json'
        if plan_path.exists() and common.read_json(plan_path) != plan: raise RuntimeError('Existing planning execution plan differs')
        if not plan_path.exists(): common.atomic_json(plan_path, plan)
        output = ROOT/('state/PLANNING_TECH_GATE.json' if phase == 'TECH' else 'state/PLANNING_FORMAL_COMPLETE.json')
        if output.exists():
            existing = common.read_json(output)
            if existing.get('status') != ('PASS' if phase == 'TECH' else 'COMPLETE') or existing['execution_plan_sha256'] != digest(plan): raise RuntimeError('Completed phase identity/status differs')
            for rel, expected in existing['files'].items():
                if record(local_path(rel)) != expected: raise RuntimeError('Completed phase artifact differs')
            return existing
        if (directory/'STOP_REQUESTED.json').exists(): raise RuntimeError('Prior controller/worker interruption requires explicit review')
        pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
        if pids: raise RuntimeError('GPU compute already active; no concurrent planner/trainer allowed: '+str(pids))
        children = []; logs = []; start = time.perf_counter()
        try:
            for gpu in (0, 1):
                if not any(x['gpu'] == gpu for x in plan['cases']): continue
                log = (directory/f'gpu{gpu}_controller{os.getpid()}.log').open('ab'); logs.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), R3_ROOT=str(ROOT), MUJOCO_GL='egl', SDL_VIDEODRIVER='dummy',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command = [sys.executable, str(ROOT/'scripts/run_planning.py'), '--worker-plan', str(plan_path.relative_to(ROOT)),
                           '--gpu', str(gpu), '--phase-lock-fd', str(phase_handle.fileno())]
                children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(phase_handle.fileno(),)))
            while any(p.poll() is None for p in children):
                failures = [{'pid': p.pid, 'returncode': p.returncode} for p in children if p.poll() not in (None, 0)]
                if failures and not (directory/'STOP_REQUESTED.json').exists():
                    common.atomic_json(directory/'STOP_REQUESTED.json', {'reason': 'WORKER_FAILURE', 'children': failures, 'current_trajectory_allowed_to_finish': True})
                time.sleep(.5)
            if any(p.returncode != 0 for p in children): raise RuntimeError('Planning worker failed; preserved attempts need explicit review')
            verify_plan(plan); result = collect(plan, directory); result['controller_wall_seconds'] = time.perf_counter()-start
            common.atomic_json(output, result); return result
        except BaseException as error:
            common.atomic_json(directory/'STOP_REQUESTED.json', {'reason': type(error).__name__, 'message': str(error),
                               'current_trajectory_allowed_to_finish': True, 'children': [{'pid': p.pid, 'returncode': p.poll()} for p in children]})
            raise
        finally:
            for log in logs: log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase', choices=('TECH', 'FORMAL'))
    parser.add_argument('--routing', default='manifests/OPEN_LOOP_ROUTING.json'); parser.add_argument('--environment-preflight', default=PREFLIGHT)
    parser.add_argument('--worker-plan', help=argparse.SUPPRESS); parser.add_argument('--gpu', type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument('--phase-lock-fd', type=int, help=argparse.SUPPRESS); args = parser.parse_args()
    if args.worker_plan:
        if args.phase or args.gpu is None or args.phase_lock_fd is None: parser.error('Private worker requires inherited phase descriptor and assigned GPU')
        result = worker(args.worker_plan, args.gpu, args.phase_lock_fd)
    else:
        if args.phase is None or args.gpu is not None or args.phase_lock_fd is not None: parser.error('Controller requires --phase only')
        result = run(args.phase, args.routing, args.environment_preflight)
    print(json.dumps({'status': result['status'], 'phase': result['phase']}), flush=True)


if __name__ == '__main__': main()
