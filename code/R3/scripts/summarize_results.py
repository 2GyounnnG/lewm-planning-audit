"""Post-completion R3 tables/figures from recorded outputs; never fit or simulate.

Final scientific analysis is blocked until all six fixed30k jobs, both fixed
ten-arm open-loop matrices, and both four-arm formal CEM matrices are complete.
Technical failures and incomplete timing coverage are retained in the ledger.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from r3 import common, statistics
from r3.open_loop import clean, local_path, verify_lock, validate_routes, validate_cases
from r3.reset_fallback import effective_case, evidence_paths

ROOT = common.ROOT
VERSION = 'R3_RECORDED_RESULTS_SUMMARY_V1'
REPORT_NAMES = ('INTAKE_AND_INSTANCE_REUSE.md', 'OFFICIAL_SOURCE_AND_PROTOCOL_AUDIT.md',
                'FREEZE_AND_CACHE_VALIDATION.md', 'BASELINE_TECHNICAL_REPRODUCTION.md',
                'TRAINING_COMPLETION.md', 'OPEN_LOOP_RESULTS.md', 'CEM_PAIRED_RESULTS.md',
                'COMPUTE_STORAGE_AND_ETA.md', 'FINAL_SCIENTIFIC_REPORT_ZH.md')


def record(path): return {'sha256': common.sha256(path), 'bytes': Path(path).stat().st_size}


def atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); q = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with q.open('wb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(q, path)


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(clean(v), ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list, tuple)) else clean(v) for k, v in row.items()})
    data = stream.getvalue().encode()
    atomic_bytes(path, gzip.compress(data, mtime=0) if str(path).endswith('.gz') else data)


class Inputs:
    def __init__(self): self.files = {}
    def bind(self, relative):
        path = local_path(relative); current = record(path)
        if relative in self.files and self.files[relative] != current: raise RuntimeError('Analysis input changed: '+relative)
        self.files[relative] = current; return path
    def json(self, relative): return common.read_json(self.bind(relative))
    def journal(self, relative):
        with self.bind(relative).open() as f: return [json.loads(line) for line in f if line.strip()]
    def receipt(self, relative):
        d = self.json(relative)
        for rel, expected in d['files'].items():
            if record(self.bind(rel)) != expected: raise RuntimeError('Recorded artifact differs: '+rel)
        return d
    def unchanged(self):
        for rel, expected in self.files.items():
            if record(local_path(rel)) != expected: raise RuntimeError('Analysis input changed: '+rel)


def validate_updates(rows, expected=30000, technical=False):
    if len(rows) != expected or [r['step'] for r in rows] != list(range(1, expected+1)):
        raise RuntimeError('Successful update journal is not the exact contiguous budget')
    for r in rows:
        if r.get('technical') is not technical or r['sampled_windows'] != 128 or r['predicted_tokens'] != 384 or r['raw_action_exposures'] != 1920:
            raise RuntimeError('Update exposure/batch contract differs')
        if any(not math.isfinite(float(r[k])) for k in ('loss_raw_MSE', 'lr', 'preclip_gradient_norm', 'compute_seconds')):
            raise RuntimeError('Formal successful journal contains nonfinite training values')


def validate_sampling_totals(totals, sampling):
    keys={'sampled_windows':'sampled_windows','predicted_tokens':'predicted_target_tokens','raw_action_exposures':'raw_action_exposures'}
    if any(sampling[keys[k]] != v for k,v in totals.items()): raise RuntimeError('Training journal/sampler totals differ')


def training_data(inputs):
    jobs = inputs.json('manifests/JOBS.json')['jobs']
    if len(jobs) != 6 or {(j['task'], j['refit_seed']) for j in jobs} != {(t, s) for t in common.TASKS for s in common.SEEDS}:
        raise RuntimeError('Exactly six fixed task/seed jobs required')
    updates, monitors, windows, summary, failures = [], [], [], [], []
    for job in jobs:
        prefix = 'artifacts/train/'+job['job_id']; result = inputs.json(prefix+'/result.json')
        if (result.get('actual_updates') != 30000 or result.get('status') != 'REFIT_TRAINING_COMPLETE_UNSCORED'
                or result.get('technical') is not False or result['frozen_before'] != result['frozen_after']
                or result.get('new_encoder_updates') != 0 or result.get('state_labels_read') != 0):
            raise RuntimeError('Formal six-job training/freeze contract not complete')
        raw = inputs.journal(prefix+'/updates.jsonl'); validate_updates(raw)
        monitor = inputs.journal(prefix+'/monitor.jsonl')
        if [m['step'] for m in monitor] != list(range(0, 30001, 1000)): raise RuntimeError('Fixed monitor curve is incomplete or adaptively sampled')
        tag = {'job_id': job['job_id'], 'task': job['task'], 'refit_seed': job['refit_seed']}
        updates.extend({**tag, **r} for r in raw)
        for m in monitor:
            if m.get('checkpoint_selection') is not False or m.get('role') != 'MONITOR': raise RuntimeError('Monitor selection/role differs')
            monitors.append({**tag, **{k: v for k, v in m.items() if k != 'per_window'}})
            lengths = {len(x) for x in m['per_window'].values()}
            if lengths != {m['windows']}: raise RuntimeError('Per-window monitor values incomplete')
            for i in range(m['windows']):
                windows.append({**tag, 'step': m['step'], 'window_index': i, 'window_identity_sha256': m['window_identity_sha256'],
                                **{k: v[i] for k, v in m['per_window'].items()}})
        for p in sorted((ROOT/prefix).glob('failure_*.json')):
            f = inputs.json(str(p.relative_to(ROOT))); failures.append({**tag, 'file': str(p.relative_to(ROOT)), **f})
            if f.get('uncertain_optimizer_attempt'): raise RuntimeError('Formal optimizer uncertainty must be audited before claiming exact180000 updates')
        if (ROOT/prefix/'OPTIMIZER_INFLIGHT.json').exists(): raise RuntimeError('Uncommitted formal optimizer intent remains')
        totals = {k: sum(r[k] for r in raw) for k in ('sampled_windows', 'predicted_tokens', 'raw_action_exposures')}
        validate_sampling_totals(totals,result['sampling'])
        summary.append({**tag, 'status': result['status'], 'actual_updates': len(raw), **totals,
                        'trainable_parameters': result['trainable_parameters'], 'frozen_parameters': result['frozen_parameters'],
                        'loss_first': raw[0]['loss_raw_MSE'], 'loss_last': raw[-1]['loss_raw_MSE'],
                        'new_encoder_updates': 0, 'state_labels_read': 0, 'frozen_equal': True,
                        'microbatch': result['microbatch'], 'effective_batch': result['effective_batch'],
                        **{k: v for k, v in result['timing'].items()},
                        'journal_compute_seconds': sum(r['compute_seconds'] for r in raw),
                        'journal_optimizer_intent_commit_seconds': sum(r.get('optimizer_intent_commit_seconds', 0.) for r in raw),
                        'peak_allocated_bytes': result['peak_allocated_bytes'], 'peak_reserved_bytes': result['peak_reserved_bytes']})
    return {'updates': updates, 'monitors': monitors, 'monitor_windows': windows, 'summary': summary, 'failures': failures}


def number(value):
    if value is None: return float('nan')
    return float(value)


def extended_mean(values):
    a = np.asarray([number(v) for v in values], dtype=np.float64)
    # A nonfinite predictive risk is never silently removed from its mean.
    return float(a.mean()) if np.isfinite(a).all() else float('inf')


def aggregate_open_rows(rows):
    groups = {}
    for r in rows: groups.setdefault((r['task'], r['arm'], r['refit_seed'], r['checkpoint_step'], r['horizon_macro']), []).append(r)
    result = []
    for (task, arm, seed, step, h), part in sorted(groups.items(), key=lambda x: (x[0][0], x[0][3], x[0][1], x[0][4])):
        if len({r['case_id'] for r in part}) != len(part): raise RuntimeError('Duplicate open-loop case within arm/horizon')
        raw = [number(r['latent_raw_MSE']) for r in part]
        row = {'task': task, 'arm': arm, 'refit_seed': seed, 'checkpoint_step': step, 'horizon_macro': h, 'horizon_raw': h*5,
               'cases': len(part), 'case_weighting': 'EQUAL_CASE_FIXED_SINGLE_ANCHOR',
               'mean_latent_raw_MSE': extended_mean(raw), 'median_latent_raw_MSE': float(np.median(raw)),
               'mean_latent_TRAIN_variance_normalized_MSE': extended_mean(r['latent_TRAIN_variance_normalized_MSE'] for r in part),
               'nonfinite_prediction_cases': sum(not r['prediction_finite'] for r in part),
               'nonfinite_metric_cases': sum(not r['metric_finite'] for r in part),
               'inferential_CI': 'NONE_EXPLORATORY_CHECKPOINT_HORIZON_DIAGNOSTIC'}
        for k in ('prediction_norm_l2', 'target_norm_l2', 'prediction_norm2_over_D', 'target_norm2_over_D', 'prediction_displacement_from_initial_MSE'):
            values = np.asarray([number(r[k]) for r in part]); finite = np.isfinite(values)
            row[k+'_all_case_mean'] = float(values.mean()) if finite.all() else None
            row[k+'_finite_cases'] = int(finite.sum())
        result.append(clean(row))
    return result


def open_loop_data(inputs, routing):
    rows, drift, vectors, metadata = [], [], {}, {}
    routes = inputs.json(routing)
    for task in common.TASKS:
        prefix = f'artifacts/open_loop/{task}'; receipt = inputs.receipt(prefix+'/OPEN_LOOP_RECEIPT.json')
        if receipt.get('status') != 'COMPLETE' or receipt.get('arms') != 10 or receipt.get('evaluation_kind') != 'OPEN_LOOP':
            raise RuntimeError('Complete ten-arm open-loop matrix required')
        roles = inputs.json(f'manifests/{task}_data_roles.json'); cases = validate_cases(roles, task); by_case = {c['case_id']: c for c in cases}
        case_rows = inputs.json(prefix+'/per_case.json'); arms = validate_routes(routes, task)
        if {r['case']['case_id'] for r in case_rows} != set(by_case) or len(case_rows) != len(cases): raise RuntimeError('Open-loop case roster differs')
        for row in case_rows:
            if row['case'] != by_case[row['case']['case_id']] or row['task'] != task or row['phase'] != 'EVAL': raise RuntimeError('Open-loop case/phase identity differs')
            if {r['arm'] for r in row['arms']} != {r['arm'] for r in arms} or len(row['arms']) != 10: raise RuntimeError('Open-loop arm roster differs')
            for arm in row['arms']:
                route = next(r for r in arms if r['arm'] == arm['arm'])
                if arm['refit_seed'] != route['seed'] or arm['checkpoint_step'] != route['step'] or [m['horizon_macro'] for m in arm['metrics']] != [1, 2, 5]:
                    raise RuntimeError('Open-loop checkpoint/horizon routing differs')
                rows.extend({'task': task, 'case_id': row['case']['case_id'], 'episode_id': row['case']['episode_id'],
                             'family_id': row['case']['family_id'], 'arm': arm['arm'], 'refit_seed': arm['refit_seed'],
                             'checkpoint_step': arm['checkpoint_step'], **m} for m in arm['metrics'])
        with np.load(inputs.bind(prefix+'/targets.npz'), allow_pickle=False) as f: target = f['target_z'].astype(np.float64)
        with np.load(inputs.bind(prefix+'/inputs.npz'), allow_pickle=False) as f: saved_ids = f['case_ids'].tolist()
        if saved_ids != [c['case_id'] for c in cases] or target.shape[:2] != (len(cases), 5): raise RuntimeError('Open-loop raw bundle order/shape differs')
        vectors[task] = {}; target_means = []; target_vars = []
        for h in (1, 2, 5): target_means.append(target[:, h-1].mean(0)); target_vars.append(target[:, h-1].var(0))
        vectors[task]['target_mean'] = np.stack(target_means); vectors[task]['target_coordinate_variance'] = np.stack(target_vars)
        pm, pv = [], []
        for route in arms:
            with np.load(inputs.bind(prefix+'/predictions/'+route['arm']+'.npz'), allow_pickle=False) as f:
                prediction = f['prediction'].astype(np.float64)
                if f['case_ids'].tolist() != saved_ids or prediction.shape != target.shape: raise RuntimeError('Raw forecast bundle order differs')
            m, v = [], []
            for j, h in enumerate((1, 2, 5)):
                x = prediction[:, h-1]; finite = bool(np.isfinite(x).all())
                with np.errstate(over='ignore', invalid='ignore'):
                    mean, variance = x.mean(0), x.var(0); shift = float(np.square(mean-target_means[j]).mean())
                m.append(mean); v.append(variance)
                drift.append(clean({'task': task, 'arm': route['arm'], 'refit_seed': route['seed'], 'checkpoint_step': route['step'],
                                   'horizon_macro': h, 'horizon_raw': h*5, 'cases': len(cases), 'all_predictions_finite': finite,
                                   'prediction_mean_norm2_over_D': float(np.square(mean).mean()),
                                   'target_mean_norm2_over_D': float(np.square(target_means[j]).mean()),
                                   'mean_vector_shift_MSE': shift, 'prediction_mean_coordinate_variance': float(variance.mean()),
                                   'target_mean_coordinate_variance': float(target_vars[j].mean()),
                                   'statistics_are_scoring_diagnostics_not_whitening': True}))
            pm.append(m); pv.append(v)
        vectors[task].update(prediction_mean=np.asarray(pm), prediction_coordinate_variance=np.asarray(pv),
                             arms=np.asarray([a['arm'] for a in arms]), horizons=np.asarray([1,2,5]))
        metadata[task] = {'cases': len(cases), 'arms': 10, 'scalar_variance': receipt['train_variance'],
                          'anchor_not_planning_start_cases': sum(c['open_loop_anchor_relation'] == 'OPEN_LOOP_ANCHOR_NOT_PLANNING_START' for c in cases)}
    return {'rows': rows, 'summary': aggregate_open_rows(rows), 'drift': drift, 'vectors': vectors, 'metadata': metadata}


def planning_data(inputs):
    completion = inputs.receipt('state/PLANNING_FORMAL_COMPLETE.json')
    if completion.get('status') != 'COMPLETE': raise RuntimeError('Formal CEM matrix is not complete')
    ledger = inputs.json('state/FORMAL_TRAJECTORY_LEDGER.json'); by_task = {t: [] for t in common.TASKS}; attempts = []
    for rid, entry in ledger['trajectories'].items():
        if entry['status'] in ('RESERVED', 'COMPLETION_UNCERTAIN'): raise RuntimeError('Formal trajectory completion remains uncertain')
        row = {**entry, 'run_id': rid}
        if entry.get('result_path'):
            result = inputs.json(entry['result_path'])
            if record(local_path(entry['result_path']))['sha256'] != entry['result_sha256']: raise RuntimeError('Formal ledger/result hash differs')
            row['failure'] = result.get('failure'); row['failure_category'] = result.get('failure_category')
            if entry['status'] == 'COMPLETE':
                if entry['result_path'] not in completion['files']: raise RuntimeError('Complete trajectory omitted by final planning receipt')
                by_task[result['task']].append(result)
        attempts.append(row)
    for task in common.TASKS:
        roles = inputs.json(f'manifests/{task}_data_roles.json'); cases = {c['case_id']: c for c in roles['cases']['EVAL']}
        for name in evidence_paths(task, roles): inputs.bind(name)
        effective = {case_id: effective_case(task, case) for case_id, case in cases.items()}
        rows = by_task[task]
        if {(r['case_id'], r['arm']) for r in rows} != {(c, a) for c in cases for a in statistics.ARMS} or len(rows) != 4*len(cases):
            raise RuntimeError('Formal paired matrix differs from exact frozen cases/four arms')
        for row in rows:
            case = cases[row['case_id']]
            if row['identity']['case'] != effective[row['case_id']] or row['phase'] != 'FORMAL' or row['episode_id'] != case['episode_id'] or row['family_id'] != case['family_id']:
                raise RuntimeError('Formal planning case provenance differs')
        statistics.normalize_rows(task, rows)
    if sum(map(len, by_task.values())) != completion['complete_trajectories']: raise RuntimeError('Formal completion count differs')
    return by_task, attempts, completion


def compute_data(inputs, train, formal_attempts, planning_complete):
    tech = inputs.json('state/TECHNICAL_LEDGER.json'); segments = []; lower = upper = 0; unresolved = []
    journal_paths = set(); failures = []; components = []
    for rid, row in tech.get('runs', {}).items():
        actual = row.get('actual_updates'); known = actual if actual is not None else row.get('detail', {}).get('known_successful_updates', 0)
        lower += known; upper += actual if actual is not None else row['reserved_updates']
        if actual is None: unresolved.append(rid)
        segments.append({'run_id': rid, **row})
        detail = row.get('detail') or {}
        if detail.get('run_id') and row.get('job_id'): journal_paths.add(f"artifacts/technical/{detail['run_id']}/{row['job_id']}/updates.jsonl")
    if upper > 1024: raise RuntimeError('Technical optimizer charged budget exceeds1024')
    journal_updates = 0; technical_workers = []; technical_journals = []
    for rel in sorted(journal_paths):
        if (ROOT/rel).exists():
            rows = inputs.journal(rel); journal_updates += len(rows)
            if [r['step'] for r in rows] != list(range(1, len(rows)+1)) or any(r.get('technical') is not True for r in rows): raise RuntimeError('Technical journal identity differs')
            technical_journals.append({'path':rel,'successful_updates':len(rows),
                'compute_seconds':sum(r['compute_seconds'] for r in rows),
                'optimizer_intent_commit_seconds':sum(r.get('optimizer_intent_commit_seconds',0.) for r in rows)})
        folder = (ROOT/rel).parent
        if (folder/'result.json').exists():
            r = inputs.json(str((folder/'result.json').relative_to(ROOT))); technical_workers.append({'path': str(folder.relative_to(ROOT)), 'result': r})
        for path in folder.glob('failure_*.json'): failures.append({'file': str(path.relative_to(ROOT)), **inputs.json(str(path.relative_to(ROOT)))})
    lower = max(lower, journal_updates)
    if lower > upper: raise RuntimeError('Technical journal exceeds charged ledger bound')
    tech_attempts = []
    for rid, row in tech.get('trajectories', {}).items():
        item = {'run_id': rid, **row}
        if row.get('result_path'):
            r = inputs.json(row['result_path'])
            if common.sha256(local_path(row['result_path'])) != row['result_sha256']: raise RuntimeError('Technical trajectory ledger/result differs')
            item.update(failure=r.get('failure'), failure_category=r.get('failure_category'))
        tech_attempts.append(item)
    charged_tech_trajectories = sum(r['status'] != 'INFRASTRUCTURE_FAILURE_INCOMPLETE' for r in tech_attempts)
    if charged_tech_trajectories > 16: raise RuntimeError('Technical trajectory charged budget exceeds16')
    def component(stage, source, label, value, scope):
        if value is not None: components.append({'stage': stage, 'source': source, 'measure': label, 'value': value, 'unit': 'seconds', 'scope': scope})
    for row in train['summary']:
        for key in ('worker_seconds','compute_seconds','loader_seconds','checkpoint_seconds','monitor_seconds','journal_optimizer_intent_commit_seconds'):
            component('FORMAL_TRAIN', 'artifacts/train/'+row['job_id']+'/result.json' if not key.startswith('journal') else 'artifacts/train/'+row['job_id']+'/updates.jsonl',
                      key, row.get(key), 'Per-job measured component; worker_seconds overlaps components and excludes pre-loop model/cache setup')
    for worker in technical_workers:
        for key, value in worker['result'].get('timing', {}).items(): component('TECH_OPTIMIZER', worker['path']+'/result.json', key, value, 'Cumulative unique run; resumed segments not added again')
    for journal in technical_journals:
        for key in ('compute_seconds','optimizer_intent_commit_seconds'):
            component('TECH_SUCCESSFUL_UPDATE_JOURNAL',journal['path'],key,journal[key],
                      'Includes successful updates before a failure; overlaps completed worker totals; failed in-flight time may be unavailable')
    for phase, attempts in [('FORMAL_CEM', formal_attempts), ('TECH_CEM', tech_attempts)]:
        for row in attempts: component(phase, row.get('result_path'), 'trajectory_wall_seconds', row.get('trajectory_wall_seconds'), 'Every recorded attempt, including incomplete infrastructure failures')
    setup_sources = [('state/ENVIRONMENT_READY.json','seconds','ISOLATED_DEPENDENCY_SETUP'),
                     ('state/environment_preflight/attempt_001/ENVIRONMENT_PREFLIGHT.json','wall_seconds','RESET_STEP_PREFLIGHT'),
                     ('state/PLANNING_TECH_GATE.json','all_worker_attempt_setup_seconds','TECH_CEM_SETUP'),
                     ('state/PLANNING_FORMAL_COMPLETE.json','all_worker_attempt_setup_seconds','FORMAL_CEM_SETUP')]
    for task in common.TASKS: setup_sources.append((f'manifests/{task}_cache.json','seconds','OBSERVED_LATENT_CACHE_'+task))
    missing = []
    for source, key, stage in setup_sources:
        if (ROOT/source).exists():
            d = inputs.json(source)
            if key in d: component(stage, source, key, d[key], 'Actual recorded scope; not asserted to cover every setup operation')
            else: missing.append({'source': source, 'missing_timing_field': key})
        else: missing.append({'source': source, 'missing_file': True})
    # Preserve all measured per-attempt zero-update check timings, failures too.
    zero_checks=[]
    for base in (ROOT/'artifacts',ROOT/'state'):
        for path in sorted(base.glob('**/REAL_DATA_MODEL_CHECK.json')):
            rel = str(path.relative_to(ROOT)); d = inputs.json(rel); component('ZERO_UPDATE_REAL_MODEL_CHECK', rel, 'wall_seconds', d.get('wall_seconds'), 'Saved attempt status='+d.get('status','UNKNOWN'))
            zero_checks.append({'path':rel,'status':d.get('status'),'optimizer_updates':d.get('optimizer_updates'),
                                'full_policy_trajectories':d.get('full_policy_trajectories'),'wall_seconds':d.get('wall_seconds')})
        for path in sorted(base.glob('**/ENVIRONMENT_PREFLIGHT.json')):
            rel=str(path.relative_to(ROOT));d=inputs.json(rel)
            zero_checks.append({'path':rel,'status':d.get('status'),'counts':d.get('counts')})
            for task,part in d.get('tasks',{}).items():
                component('ENVIRONMENT_PREFLIGHT_'+task,rel,'wall_seconds',part.get('wall_seconds'),'Saved reset/step attempt status='+part.get('status','UNKNOWN'))
        for basename, stage in [('RESET_VALIDATION_RECEIPT.json', 'ZERO_UPDATE_RESET_FALLBACK'),
                                ('COST_WRAPPER_CHECK.json', 'ZERO_UPDATE_COST_WRAPPER')]:
            for path in sorted(base.glob('**/'+basename)):
                rel = str(path.relative_to(ROOT)); d = inputs.json(rel)
                component(stage, rel, 'wall_seconds', d.get('wall_seconds'), 'Every saved diagnostic attempt, including failures; no CEM/optimizer updates')
                zero_checks.append({'path': rel, 'status': d.get('status'), 'task': d.get('task'),
                                    'counts': d.get('counts'), 'optimizer_updates': d.get('optimizer_updates', d.get('counts', {}).get('optimizer_updates')),
                                    'wall_seconds': d.get('wall_seconds'), 'CEM_calls': d.get('CEM_calls', d.get('counts', {}).get('CEM_calls')),
                                    'environment_steps': d.get('environment_steps'), 'diagnostic_only': True})
                if d.get('wall_seconds') is None: missing.append({'source': rel, 'missing_timing_field': 'wall_seconds'})
    profile = inputs.json('state/GPU_PROFILE.json') if (ROOT/'state/GPU_PROFILE.json').exists() else None
    if profile:
        for name in ('A','B'):
            p = profile[name]; component('TECH_PROFILE_'+name, 'state/GPU_PROFILE.json', 'launch_wall_seconds', p.get('launch_wall_seconds'), 'Concurrent wall; overlaps corresponding per-worker timing, not added to it')
    exact_technical = lower if not unresolved and lower == upper else None
    return clean({'formal_optimizer_updates': sum(r['actual_updates'] for r in train['summary']), 'formal_training_jobs': 6,
                  'formal_predicted_tokens': sum(r['predicted_tokens'] for r in train['summary']),
                  'formal_sampled_windows': sum(r['sampled_windows'] for r in train['summary']),
                  'formal_raw_action_exposures': sum(r['raw_action_exposures'] for r in train['summary']),
                  'technical_optimizer_updates_exact': exact_technical, 'technical_optimizer_updates_lower': lower,
                  'technical_optimizer_updates_upper': upper, 'technical_optimizer_limit': 1024,
                  'technical_unresolved_segments': unresolved, 'technical_optimizer_segments': segments,
                  'technical_unique_journal_updates': journal_updates, 'technical_worker_results': technical_workers,
                  'technical_unique_journals':technical_journals,'zero_update_check_attempts':zero_checks,
                  'technical_failures': failures, 'formal_training_failures': train['failures'],
                  'technical_trajectory_attempts': tech_attempts, 'formal_trajectory_attempts': formal_attempts,
                  'technical_complete_trajectories': sum(x['status']=='COMPLETE' for x in tech_attempts),
                  'technical_charged_trajectory_upper': charged_tech_trajectories, 'technical_trajectory_limit':16,
                  'formal_complete_trajectories': planning_complete['complete_trajectories'], 'formal_trajectory_limit':800,
                  'planning_measured_totals': planning_complete, 'measured_time_components': components,
                  'timing_coverage_gaps': missing,
                  'unmeasured_formal_preloop_setup_seconds': None,
                  'total_research_wall_seconds': None,
                  'timing_warning': 'Nested/concurrent component durations are not additive wall time; unmeasured setup is not zero.',
                  'gpu_profile': profile, 'new_analysis_optimizer_updates':0, 'new_analysis_trajectories':0,
                  'scientific_work_remaining_updates':0, 'scientific_work_remaining_trajectories':0,
                  'recovery_and_acceptance_eta_seconds':None, 'release_gate_still_required':True})


def storage_inventory():
    rows = []; seen = set(); totals = {}
    for directory, names, files in os.walk(ROOT, followlinks=False):
        names[:] = [n for n in names if n not in ('recovery', 'recovery_receipts') and not (Path(directory)/n).is_symlink()]
        for name in files:
            p = Path(directory)/name; rel = str(p.relative_to(ROOT))
            if p.is_symlink() or name.endswith(('.tmp', '.lock')): continue
            s = p.stat(); key = (s.st_dev, s.st_ino); unique = key not in seen; seen.add(key)
            parts = Path(rel).parts; category = '/'.join(parts[:2]) if parts[0] in ('artifacts','data') and len(parts)>1 else parts[0]
            allocated = int(getattr(s,'st_blocks',0))*512
            row = {'path':rel,'category':category,'logical_bytes':s.st_size,'allocated_bytes_stat_blocks':allocated,'first_inode_in_snapshot':unique}
            rows.append(row); total = totals.setdefault(category, {'files':0,'logical_bytes':0,'unique_inode_logical_bytes':0,'unique_inode_allocated_bytes':0})
            total['files']+=1; total['logical_bytes']+=s.st_size
            if unique: total['unique_inode_logical_bytes']+=s.st_size; total['unique_inode_allocated_bytes']+=allocated
    return rows, {'scope':'R3 tree snapshot before final compute report/analysis manifest; no old project traversal or recovery copies',
                  'as_of_utc':common.now(),'categories':totals,'logical_bytes':sum(x['logical_bytes'] for x in totals.values()),
                  'unique_inode_allocated_bytes':sum(x['unique_inode_allocated_bytes'] for x in totals.values()),
                  'cross_project_hardlink_allocation_not_resolved':True,'symlinks_not_followed':True}


def figures(train, opened, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','path.simplify':False})
    folder=ROOT/'reports/plots'; paths=[]
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','svg'):
            p=folder/(name+'.'+extension); buf=io.BytesIO(); fig.savefig(buf,format=extension,dpi=200,bbox_inches='tight'); atomic_bytes(p,buf.getvalue()); paths.append(str(p.relative_to(ROOT)))
        plt.close(fig)
    for task in common.TASKS:
        fig,ax=plt.subplots(figsize=(8,4))
        for seed in common.SEEDS:
            rows=[r for r in train['updates'] if r['task']==task and r['refit_seed']==seed]
            ax.plot([r['step'] for r in rows],[r['loss_raw_MSE'] for r in rows],lw=.45,alpha=.7,label=str(seed))
        ax.set(xlabel='Additional optimizer updates',ylabel='Raw minibatch latent MSE',title=f'{task}: every recorded training update (no smoothing)');ax.legend(title='Refit seed');save(fig,task+'_training_raw')
        fig,axes=plt.subplots(1,3,figsize=(13,3.5))
        for ax,key,label in zip(axes,('teacher_forced_one_step_raw_MSE','free_H5_mean_raw_MSE','free_H5_terminal_raw_MSE'),('Teacher-forced one-step','Free H5 mean','Free H5 terminal')):
            nonfinite=0
            for seed in common.SEEDS:
                rows=[r for r in train['monitors'] if r['task']==task and r['refit_seed']==seed]
                values=[number(r[key]) for r in rows];nonfinite+=sum(not math.isfinite(v) for v in values)
                ax.plot([r['step'] for r in rows],[v if math.isfinite(v) else np.nan for v in values],'.-',lw=1,label=str(seed))
            if nonfinite:ax.text(.01,.99,f'Nonfinite values preserved as {nonfinite} gaps',transform=ax.transAxes,va='top',fontsize=7,color='red')
            ax.set(xlabel='Additional updates',ylabel='Raw latent MSE',title=label)
        axes[0].legend();fig.suptitle(f'{task}: fixed MONITOR windows, diagnostic only');save(fig,task+'_monitor_raw')
        fig,axes=plt.subplots(1,3,figsize=(13,3.5))
        for ax,h in zip(axes,(1,2,5)):
            rows=[r for r in opened['summary'] if r['task']==task and r['horizon_macro']==h];base=next(r for r in rows if r['arm']=='H0')
            nonfinite=[]
            for seed in common.SEEDS:
                ordered=[base]+[next(r for r in rows if r['refit_seed']==seed and r['checkpoint_step']==n) for n in (3000,10000,30000)]
                y=[number(r['mean_latent_raw_MSE']) for r in ordered]
                nonfinite += [r['arm'] for r,v in zip(ordered,y) if not math.isfinite(v)]
                ax.plot([0,3000,10000,30000],[v if math.isfinite(v) else np.nan for v in y],'.-',label=str(seed))
            if nonfinite:ax.text(.01,.99,'Nonfinite gaps: '+', '.join(sorted(set(nonfinite))),transform=ax.transAxes,va='top',fontsize=6,color='red',wrap=True)
            ax.set(xlabel='Additional updates (0 = H0)',ylabel='Case-mean raw latent MSE',title=f'h={h} macro / {5*h} raw steps')
        axes[0].legend();fig.suptitle(f'{task}: frozen EVAL open loop; checkpoint/horizon diagnostics');save(fig,task+'_open_loop_raw')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axes,common.TASKS):
        s=stats[task]
        for i,arm in enumerate(statistics.ARMS):
            d=s['schemes']['CASE']['metrics'][arm+'_success_percent'];ax.plot(i,d['estimate'],'o',color='tab:blue');ax.vlines(i,*d['ci95'],color='tab:blue')
        ax.set_xticks(range(4),['H0','seed 103201','seed 103202','seed 103203'],rotation=20)
        ax.set(ylim=(-2,102),ylabel='Success (%) with case bootstrap 95% CI',title=f'{task} (n={s["cases"]} cases)')
    save(fig,'cem_success_by_arm')
    fig,ax=plt.subplots(figsize=(8,4)); labels=[];pos=0
    for task in common.TASKS:
        for scheme,color in [('CASE','tab:blue'),('FAMILY','tab:orange')]:
            data=stats[task]['schemes'][scheme]
            if data['status']!='COMPLETE':continue
            d=data['metrics']['FIXED_THREE_REFIT_MEAN_minus_H0_pp'];ax.hlines(pos,*d['ci95'],color=color,lw=2);ax.plot(d['estimate'],pos,'o',color=color)
            labels.append(task+' / '+scheme);pos+=1
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(pos),labels);ax.set(xlabel='Fixed three-model mean − H0 (percentage points), 95% CI',title='Conditional paired success changes; family sensitivity retained');save(fig,'cem_paired_effect')
    return paths


def table(rows, fields):
    def fmt(x):
        if isinstance(x,float):return f'{x:.8g}'
        if x is None:return '—'
        return str(x).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(k)) for k in fields)+' |' for r in rows)+'\n'


def reports(train, opened, stats, compute, storage):
    output={}
    def save(name,text):
        path='reports/'+name;atomic_bytes(ROOT/path,text.encode());output[name]=path
    save('TRAINING_COMPLETION.md','# R3 固定训练完成记录\n\n六个任务×seed作业均完成固定30,000次更新，共180,000次；这不是六个独立预训练encoder。观测encoder/projector与action encoder未更新，所有BN buffers固定。\n\n'+
         table(train['summary'],['task','refit_seed','actual_updates','predicted_tokens','loss_first','loss_last','worker_seconds'])+
         '\n全部原始更新、梯度、LR与时间见 `tables/training_updates.csv.gz`；MONITOR及逐窗口原值见对应CSV。曲线没有平滑或只保留有利片段。MONITOR不选模型，达到30k不证明充分优化。失败记录在计算账中保留。\n')
    texts=['# R3 开环预测结果\n\n任务分开报告，h=1/2/5 macro对应5/10/25原始步。预测仅使用三帧初始历史与记录动作，自由递归不回填真实未来观测；真实未来latent仅评分。所有checkpoint/horizon比较属描述性诊断，未做多重比较校正。\n']
    for task in common.TASKS:
        texts += ['\n## '+task+'\n\n',table([r for r in opened['summary'] if r['task']==task],['arm','horizon_macro','cases','mean_latent_raw_MSE','mean_latent_TRAIN_variance_normalized_MSE','nonfinite_prediction_cases']),
                  '\n固定TRAIN标量方差与锚点元数据：\n\n```json\n'+json.dumps(opened['metadata'][task],ensure_ascii=False,indent=2)+'\n```\n']
    texts += ['\n均值向量、逐坐标方差、范数及漂移原值见 `tables/open_loop_coordinate_drift.csv` 与NPZ；它们是评分诊断，不是重白化。非有限风险按+Infinity保留，不删case。原坐标不同的任务不合成一个latent风险赢家。预测改善本身不保证CEM成功率改善。\n']
    save('OPEN_LOOP_RESULTS.md',''.join(texts))
    texts=['# R3 配对 CEM 结果\n\n只比较H0与三个固定30k refit；每个case的四个arm独立闭环控制，三模型平均不是ensemble policy。使用既定 `r3.statistics.analyze_task`、5,000次配对case重采样及family敏感性；不按结果选择区间。\n']
    for task in common.TASKS:
        s=stats[task];ci=[]
        for scheme,data in s['schemes'].items():
            if data['status']=='COMPLETE':
                d=data['metrics']['FIXED_THREE_REFIT_MEAN_minus_H0_pp'];ci.append({'scheme':scheme,'delta_pp':d['estimate'],'CI95':d['ci95'],'CI97.5_approx':d['ci97_5_two_task_bonferroni_approximation'],'cluster_warning':data['cluster_warning']})
            else:ci.append({'scheme':scheme,'status':data['status']})
        texts += ['\n## '+task+f'：{s["cases"]}个case\n\n',table(ci,['scheme','delta_pp','CI95','CI97.5_approx','cluster_warning','status']),
                  '\n',table(s['paired_counts'],['arm','H0_successes','REFIT_successes','s01_H0_failure_REFIT_success','s10_H0_success_REFIT_failure','s11_both_success','s00_both_failure','delta_success_pp'])]
    texts += ['\n95%区间条件于固定官方起点、数据和三个refit，不包含重新预训练/重新抽seed的不确定性；97.5%是两任务Bonferroni近似展示，不是有限样本严格FWER保证。family未知或簇少的限制按机器结果原样保留。推断族不证明统计独立。方法失败保留success=0；未完成基础设施尝试不冒充评分。原source可能参与官方预训练，本轮仅可称对refit留出。额外训练成本保留，不宣称新规划算法或对其他planner论文作因果归因。\n']
    save('CEM_PAIRED_RESULTS.md',''.join(texts))
    selected={k:compute[k] for k in ('formal_optimizer_updates','formal_predicted_tokens','formal_sampled_windows','formal_raw_action_exposures','technical_optimizer_updates_exact','technical_optimizer_updates_lower','technical_optimizer_updates_upper','technical_complete_trajectories','technical_charged_trajectory_upper','formal_complete_trajectories')}
    save('COMPUTE_STORAGE_AND_ETA.md','# R3 实际计算、存储与剩余工作\n\n'+table([{'measure':k,'value':v} for k,v in selected.items()],['measure','value'])+
         '\n实际阶段/逐worker时间、失败、重启与技术预约原值见 `tables/compute_components.csv`、`tables/compute_accounting.json`。初始化与冷启动不从记录中删除；同一段的总时间和子组件、并行worker时间不能相加冒充端到端墙钟。未记录的正式pre-loop setup为未知，不填0。\n\n'+
         table([{'category':k,**v} for k,v in storage['categories'].items()],['category','files','logical_bytes','unique_inode_allocated_bytes'])+
         '\n存储是汇总清单生成前的实际R3目录快照，原始文件逐项尺寸见CSV。logical与allocated不同；跨项目硬链接不在此解释为独占新增磁盘。科学训练和指定评价已完成，科学剩余更新/轨迹为0；回收与CPU恢复验收仍独立核验，退卡/最终交付以release gate为准。此报告不把训练吞吐外推为网络或验收ETA。\n')
    return output


def run(routing='manifests/OPEN_LOOP_ROUTING.json'):
    lock_path=ROOT/'state/execution_phase.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        required=['scripts/summarize_results.py','r3/statistics.py','r3/open_loop.py','r3/reset_fallback.py',routing]
        for task in common.TASKS:
            required.extend(evidence_paths(task,common.read_json(f'manifests/{task}_data_roles.json')))
        locked=verify_lock(required)
        output=ROOT/'manifests/ANALYSIS_OUTPUTS.json'
        if output.exists():
            prior=common.read_json(output)
            if prior['source_sha256']!=common.sha256(__file__) or prior['models_lock_sha256']!=locked['sha256']:raise RuntimeError('Existing analysis identity differs')
            for rel,expected in {**prior['inputs'],**prior['files']}.items():
                if record(local_path(rel))!=expected:raise RuntimeError('Existing analysis artifact changed: '+rel)
            return prior
        inputs=Inputs();inputs.bind('manifests/MODELS_AND_SELECTION_LOCK.json');train=training_data(inputs)
        opened=open_loop_data(inputs,routing);paired,attempts,completion=planning_data(inputs)
        compute=compute_data(inputs,train,attempts,completion)
        if compute['formal_optimizer_updates']!=180000:raise RuntimeError('Exact formal180000 budget not achieved')
        stats={t:statistics.analyze_task(t,paired[t],ROOT/'artifacts/statistics'/t) for t in common.TASKS}
        folder=ROOT/'tables';folder.mkdir(parents=True,exist_ok=True)
        tables={'training_updates.csv.gz':train['updates'],'monitor_curves.csv':train['monitors'],
                'monitor_per_window.csv.gz':train['monitor_windows'],'training_summary.csv':train['summary'],
                'open_loop_per_case.csv.gz':opened['rows'],'open_loop_summary.csv':opened['summary'],
                'open_loop_coordinate_drift.csv':opened['drift'],'formal_planning_raw.csv':[r for part in paired.values() for r in part],
                'compute_components.csv':compute['measured_time_components'],'technical_optimizer_segments.csv':compute['technical_optimizer_segments'],
                'trajectory_attempts.csv':[{**r,'ledger_phase':phase} for phase,key in [('TECH','technical_trajectory_attempts'),('FORMAL','formal_trajectory_attempts')] for r in compute[key]]}
        for name,rows in tables.items():write_csv(folder/name,rows)
        for task,arrays in opened['vectors'].items():
            buf=io.BytesIO();np.savez_compressed(buf,**arrays);atomic_bytes(folder/(task+'_open_loop_coordinate_vectors.npz'),buf.getvalue())
        plot_paths=figures(train,opened,stats)
        storage_rows,storage=storage_inventory();write_csv(folder/'storage_inventory.csv.gz',storage_rows)
        common.atomic_json(folder/'storage_summary.json',storage);common.atomic_json(folder/'compute_accounting.json',compute)
        report_paths=reports(train,opened,stats,compute,storage)
        # Root writes the other protocol reports after reviewing actual science.
        # Do not seal an early narrative here and make its later review invalid.
        paths=list(plot_paths)+list(report_paths.values())+[str(p.relative_to(ROOT)) for p in folder.iterdir() if p.is_file() and not p.name.endswith('.tmp')]
        paths += [str(p.relative_to(ROOT)) for p in (ROOT/'artifacts/statistics').rglob('*') if p.is_file() and not p.name.endswith('.tmp')]
        inputs.unchanged()
        if verify_lock()['sha256']!=locked['sha256']:raise RuntimeError('Model lock changed during analysis')
        result={'version':VERSION,'status':'COMPLETE','source_sha256':common.sha256(__file__),'models_lock_sha256':locked['sha256'],
                'inputs':inputs.files,'files':{rel:record(local_path(rel)) for rel in sorted(set(paths))},'reports':report_paths,
                'plots':plot_paths,'tasks':{t:{'statistics_dir':f'artifacts/statistics/{t}','open_loop_dir':f'artifacts/open_loop/{t}',
                    'planning_phase':'artifacts/planning/FORMAL','cases':stats[t]['cases'],'open_loop_arms':10,'CEM_arms':4} for t in common.TASKS},
                'formal_training_jobs':6,'formal_training_updates':180000,'bootstrap_replicates':5000,
                'raw_values_preserved':True,'new_optimizer_updates':0,'new_CEM_trajectories':0,
                'final_scientific_narrative_and_release_gate_are_separate':True}
        common.atomic_json(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--routing',default='manifests/OPEN_LOOP_ROUTING.json');args=parser.parse_args()
    out=run(args.routing);print(json.dumps({'status':out['status'],'tasks':list(out['tasks']),'files':len(out['files'])}))
