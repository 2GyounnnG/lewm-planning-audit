"""One fixed CPU replay failure diagnosis; never an acceptance gate.

Uses only the recovered frozen official adapter and saved original inputs. No
backend/precision/thread/tolerance sweep, fitting, encoder forward or env step.
"""
from __future__ import annotations
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path('/Volumes/MyProj/r3_official_lewm_predictor_refit/recovery')
OUT = ROOT / 'recovery_receipts/runtime_diagnosis_001/CPU_REPLAY_BASELINE'
DEVICE = 16777237
FINAL_SHA = 'f4e0f4c87ae97f3aa57a2ededf97c92f1fe88c1d9bfefb401b508bae0d7d7b4f'
LOCK_SHA = 'cbf032b4471f1615cc48181b382fe0b99d399a648146d815d3941791a7ec4f88'
ATOL = RTOL = 1e-5


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def guard(path):
    p = Path(path)
    if not p.is_absolute() or not p.is_relative_to(Path('/Volumes/MyProj')):
        raise RuntimeError('Output/input is outside required external mount')
    for q in [Path('/Volumes/MyProj')] + list(reversed(p.parents)) + [p]:
        if q == Path('/Volumes/MyProj') or q.is_relative_to(Path('/Volumes/MyProj')):
            if q.exists() and (q.is_symlink() or q.stat().st_dev != DEVICE):
                raise RuntimeError('External path device/ancestor changed: ' + str(q))
    if Path('/Volumes/MyProj').stat().st_dev != DEVICE: raise RuntimeError('External volume absent')


def write_json(path, doc):
    guard(path)
    with path.open('x') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def read(path): return json.loads(Path(path).read_text())


def signature(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def run():
    guard(ROOT); guard(OUT)
    if OUT.exists(): raise RuntimeError('Create-only diagnosis already exists; no implicit rerun')
    if shutil.disk_usage(ROOT).free < 30 * (1 << 30): raise RuntimeError('30 GiB reserve not available')
    manifest_path = ROOT / 'manifests/RECOVERY_FINAL.json'
    if sha(manifest_path) != FINAL_SHA: raise RuntimeError('Published final manifest changed')
    manifest = read(manifest_path)
    if manifest['status'] != 'FINAL_SNAPSHOT_READY_FOR_RECOVERY_AND_ACCEPTANCE': raise RuntimeError('Wrong final snapshot status')
    records = manifest['files']; checked = {}

    def verify(rel, additional=None):
        p = ROOT / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts: raise RuntimeError('Unsafe relative source path')
        guard(p); expected = records[rel]; sig = signature(p)
        if sig[2] != expected['bytes'] or sha(p) != expected['sha256'] or signature(p) != sig:
            raise RuntimeError('Frozen file bytes/SHA differ: ' + rel)
        if additional is not None:
            rec = {'sha256': additional} if isinstance(additional, str) else additional
            if expected['sha256'] != rec['sha256'] or ('bytes' in rec and expected['bytes'] != rec['bytes']):
                raise RuntimeError('Receipt identity differs: ' + rel)
        checked[rel] = {'sha256': expected['sha256'], 'bytes': expected['bytes'], 'signature': sig}
        return p

    for rel in ('r3/__init__.py', 'r3/common.py', 'r3/model.py', 'r3/open_loop.py',
                'manifests/NUMERICAL_TOLERANCES.json', 'manifests/OPEN_LOOP_ROUTING.json',
                'manifests/MODELS_AND_SELECTION_LOCK.json'):
        verify(rel)
    if sha(ROOT / 'manifests/MODELS_AND_SELECTION_LOCK.json') != LOCK_SHA: raise RuntimeError('Model lock differs')
    tolerance = read(ROOT / 'manifests/NUMERICAL_TOLERANCES.json')
    if tolerance['comparisons']['CPU_replay'] != {'atol': ATOL, 'rtol': RTOL}: raise RuntimeError('Original tolerance differs')
    for component, names in {'lewm': ['jepa.py', 'module.py', 'train.py', 'utils.py'],
                             'spt': ['stable_pretraining/backbone/utils.py']}.items():
        source = read(verify(f'state/{component}_source_manifest.json'))
        for name in names: verify(f'source/{component}/{name}', source['files'][name])
    OUT.parent.mkdir(exist_ok=True); guard(OUT.parent); OUT.mkdir(); guard(OUT)
    helper = OUT / Path(__file__).name
    with helper.open('xb') as f:
        f.write(Path(__file__).read_bytes()); f.flush(); os.fsync(f.fileno())
    started = now(); wall = time.perf_counter()
    try:
        os.environ['R3_ROOT'] = str(ROOT); sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        from r3 import common, model as models, open_loop
        for module in (common, models, open_loop):
            if not Path(module.__file__).resolve().is_relative_to(ROOT): raise RuntimeError('Imported unrecovered source')
        if common.ROOT != ROOT: raise RuntimeError('Imported root differs')
        common.fp32_policy()
        versions = {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'numpy')}
        if versions != {'torch': '2.8.0', 'transformers': '4.51.3', 'numpy': '2.3.5'}: raise RuntimeError('Fixed CPU runtime differs: ' + str(versions))
        metadata = {'status': 'DIAGNOSIS_STARTED_NOT_ACCEPTANCE', 'started_utc': started,
                    'scope': '2 tasks x H0 plus three fixed final refits; saved 100 cases, original batch32, five macro steps',
                    'manifest_sha256': FINAL_SHA, 'models_lock_sha256': LOCK_SHA,
                    'diagnostic_helper_sha256': sha(helper), 'versions': versions,
                    'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                    'torch_config': torch.__config__.show(), 'device': 'cpu', 'dtype': 'float32',
                    'threads': torch.get_num_threads(), 'deterministic': torch.are_deterministic_algorithms_enabled(),
                    'precision': torch.get_float32_matmul_precision(), 'atol': ATOL, 'rtol': RTOL,
                    'actual_SDP_flags': {'flash': torch.backends.cuda.flash_sdp_enabled(),
                                         'mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
                                         'math': torch.backends.cuda.math_sdp_enabled()},
                    'original_GPU_producer_SDP_flags': {'flash': False, 'mem_efficient': False, 'math': True},
                    'SDP_flags_modified_by_this_diagnostic': False,
                    'no_backend_precision_thread_or_tolerance_sweep': True,
                    'new_optimizer_updates': 0, 'new_encoder_forward_calls': 0, 'new_environment_steps': 0,
                    'creates_acceptance_or_release_gate': False, 'changes_scientific_conclusions': False}
        write_json(OUT / 'STARTED.json', metadata)
        routes_doc = read(ROOT / 'manifests/OPEN_LOOP_ROUTING.json')
        replays = []; score_summaries = []
        for task in ('pusht', 'reacher'):
            prefix = f'artifacts/open_loop/{task}'
            roles = read(verify(f'manifests/{task}_data_roles.json'))
            cases = open_loop.validate_cases(roles, task)
            routes = open_loop.validate_routes(routes_doc, task)
            if len(cases) != 100: raise RuntimeError('Expected original fixed 100 cases')
            receipt = read(verify(prefix + '/OPEN_LOOP_RECEIPT.json'))
            if receipt['status'] != 'COMPLETE' or receipt['cases'] != 100 or receipt['arms'] != 10:
                raise RuntimeError('Original saved evaluation incomplete')
            for rel, item in receipt['files'].items(): verify(rel, item)
            if receipt['identity']['batch_size'] != 32 or receipt['identity']['models_lock_sha256'] != LOCK_SHA:
                raise RuntimeError('Original batch/model lock differs')
            inputs = {k: v for k, v in np.load(verify(prefix + '/inputs.npz'), allow_pickle=False).items()}
            targets = {k: v for k, v in np.load(verify(prefix + '/targets.npz'), allow_pickle=False).items()}
            per_case = read(verify(prefix + '/per_case.json'))
            variance = read(verify(prefix + '/TRAIN_variance.json'))
            ids = [c['case_id'] for c in cases]
            if inputs['case_ids'].tolist() != ids or len(per_case) != 100: raise RuntimeError('Case order changed')
            assets = read(verify(f'manifests/{task}_model_assets.json'))
            for rec in assets['files'].values(): verify(rec['path'], rec)
            model = models.load_official(task, 'cpu')
            frozen = models.frozen_hashes(model)
            for route in routes:
                arm = route['arm']; pred_rel = prefix + '/predictions/' + arm + '.npz'
                arm_receipt = read(verify(prefix + '/predictions/' + arm + '.json'))
                if arm_receipt['route'] != route or arm_receipt['identity_sha256'] != open_loop.digest(receipt['identity']):
                    raise RuntimeError('Original forecast route identity differs')
                pred_path = verify(pred_rel, arm_receipt['prediction_file'])
                with np.load(pred_path, allow_pickle=False) as saved:
                    expected = saved['prediction'].copy()
                    if saved['case_ids'].tolist() != ids: raise RuntimeError('Forecast case order differs')
                if expected.shape != (100, 5, 192) or expected.dtype != np.float32: raise RuntimeError('Saved forecast shape/dtype differs')
                if route['step']:
                    cp_path = verify(route['checkpoint']['path'], route['checkpoint'])
                    identity = read(verify(route['run_identity'])); result = read(verify(route['result']))
                    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
                    open_loop.validate_checkpoint(route, cp, identity, result, model)
                    del cp
                score_details = []
                for ci, case in enumerate(cases):
                    old = per_case[ci]
                    if old['case'] != case or old['task'] != task or old['phase'] != 'EVAL': raise RuntimeError('Score case identity differs')
                    matched = [a for a in old['arms'] if a['arm'] == arm]
                    if len(matched) != 1: raise RuntimeError('Saved score arm is missing/duplicated')
                    actual_metrics = open_loop.score_case(expected[ci], targets['target_z'][ci], variance['scalar'], inputs['initial_z'][ci])
                    equal = matched[0]['metrics'] == actual_metrics
                    score_details.append({'case_id': case['case_id'], 'exact': equal,
                                          'saved_metrics': matched[0]['metrics'], 'recomputed_metrics': actual_metrics})
                score_doc = {'task': task, 'arm': arm, 'cases': 100,
                             'horizon_records': 300, 'exact_cases': sum(x['exact'] for x in score_details),
                             'all_fields_exact': all(x['exact'] for x in score_details), 'rows': score_details}
                write_json(OUT / f'{task}__{arm}__score_recalculation.json', score_doc)
                score_summaries.append({k: v for k, v in score_doc.items() if k != 'rows'})
                if route['step'] not in (0, 30000): continue
                if route['step']: models.apply_delta(model, cp_path)
                model.eval().requires_grad_(False)
                before = {k: models.tensor_sha256(v) for k, v in model.state_dict().items()}
                state_sha = open_loop.digest(before)
                if state_sha != arm_receipt['model_state_sha256_before_and_after'] or frozen['sha256'] != arm_receipt['frozen_sha256']:
                    raise RuntimeError('CPU model state differs from original forecast receipt')
                parts = []; tick = time.perf_counter()
                with torch.inference_mode():
                    for i in range(0, 100, 32):
                        if not replays and i == 0:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                                value = models.cached_rollout(model, torch.from_numpy(inputs['initial_z'][i:i+32]),
                                                              torch.from_numpy(inputs['macro_actions'][i:i+32]), 5).numpy()
                            events = [{'name': e.key, 'calls': e.count, 'CPU_total_microseconds': e.cpu_time_total,
                                       'input_shapes': e.input_shapes} for e in prof.key_averages(group_by_input_shape=True)]
                            write_json(OUT / 'FIRST_BATCH_CPU_PROFILER.json', {
                                'task': task, 'arm': arm, 'case_indices_zero_based': [0,31],
                                'existing_first_batch_only_no_added_forward': True, 'SDP_flags_modified': False,
                                'attention_events': [e for e in events if 'attention' in e['name'].lower() or 'sdp' in e['name'].lower()],
                                'all_events': events})
                        else:
                            value = models.cached_rollout(model, torch.from_numpy(inputs['initial_z'][i:i+32]),
                                                          torch.from_numpy(inputs['macro_actions'][i:i+32]), 5).numpy()
                        parts.append(value)
                elapsed = time.perf_counter() - tick
                actual = np.concatenate(parts)
                after = {k: models.tensor_sha256(v) for k, v in model.state_dict().items()}
                if before != after: raise RuntimeError('Replay changed model state')
                models.assert_frozen(model, frozen)
                nonfinite_exact = all(np.array_equal(fn(actual), fn(expected)) for fn in (np.isfinite, np.isnan, np.isposinf, np.isneginf))
                finite = np.isfinite(actual) & np.isfinite(expected)
                errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                bounds = ATOL + RTOL * np.abs(expected.astype(np.float64))
                excess = errors - bounds
                bad = finite & (excess > 0)
                positions = np.argwhere(bad)
                def coord(index):
                    c, h, d = map(int, index)
                    return {'case_index_zero_based': c, 'case_id': ids[c], 'horizon_macro': h+1,
                            'coordinate_zero_based': d, 'saved_prediction': float(expected[c,h,d]),
                            'CPU_prediction': float(actual[c,h,d]), 'absolute_error': float(errors[c,h,d]),
                            'original_bound': float(bounds[c,h,d]), 'excess': float(excess[c,h,d])}
                finite_excess = np.where(finite, excess, -np.inf)
                finite_errors = np.where(finite, errors, -np.inf)
                maxex = np.unravel_index(np.argmax(finite_excess), excess.shape)
                maxabs = np.unravel_index(np.argmax(finite_errors), errors.shape)
                row = {'task': task, 'arm': arm, 'cases': 100, 'batch_size': 32, 'macro_steps': 5,
                       'coordinates': int(actual.size), 'finite_coordinates': int(finite.sum()),
                       'nonfinite_pattern_exact': nonfinite_exact, 'over_bound_coordinates': int(bad.sum()),
                       'over_bound_by_horizon_macro_1_to_5': bad.sum(axis=(0,2)).tolist(),
                       'max_abs_error': float(errors[finite].max()), 'max_excess': float(excess[finite].max()),
                       'max_abs_error_location': coord(maxabs), 'max_excess_location': coord(maxex),
                       'first_over_bound': coord(positions[0]) if len(positions) else None,
                       'original_bound_satisfied': nonfinite_exact and not bool(bad.any()),
                       'before_state_sha256': state_sha, 'after_state_sha256': open_loop.digest(after),
                       'frozen_sha256_before': frozen['sha256'], 'frozen_sha256_after': models.frozen_hashes(model)['sha256'],
                       'all_parameters_and_BN_buffers_unchanged': before == after, 'rollout_seconds': elapsed,
                       'encoder_forward_calls': 0, 'optimizer_updates': 0}
                npz_path = OUT / f'{task}__{arm}__CPU_replay.npz'
                with npz_path.open('xb') as f:
                    np.savez_compressed(f, prediction=actual, saved_prediction=expected, absolute_error=errors,
                                        original_bound=bounds, excess=excess, over_bound=bad, case_ids=inputs['case_ids'])
                    f.flush(); os.fsync(f.fileno())
                write_json(OUT / f'{task}__{arm}__replay.json', row)
                write_json(OUT / f'{task}__{arm}__model_hashes.json', {'before': before, 'after': after, 'frozen_before': frozen})
                replays.append(row)
                print(json.dumps({'task': task, 'arm': arm, 'over_bound_coordinates': row['over_bound_coordinates'],
                                  'max_abs_error': row['max_abs_error'], 'max_excess': row['max_excess'],
                                  'rollout_seconds': elapsed}), flush=True)
            del model
        if len(replays) != 8 or len(score_summaries) != 20: raise RuntimeError('Required fixed scope incomplete')
        for rel, item in checked.items():
            p = ROOT / rel
            if signature(p) != tuple(item['signature']) or sha(p) != item['sha256']: raise RuntimeError('Input/source changed during diagnosis: ' + rel)
        if sha(manifest_path) != FINAL_SHA: raise RuntimeError('Final manifest changed during diagnosis')
        write_json(OUT / 'VERIFIED_INPUT_FILES.json', checked)
        report = {**metadata, 'status': 'FIXED_SCOPE_DIAGNOSIS_COMPLETE_NOT_ACCEPTANCE', 'completed_utc': now(),
                  'elapsed_seconds': time.perf_counter()-wall, 'CPU_replay_models': 8, 'saved_score_arms': 20,
                  'saved_score_case_arm_pairs': 2000, 'saved_score_horizon_records': 6000,
                  'all_saved_score_fields_exact': all(x['all_fields_exact'] for x in score_summaries),
                  'routes_outside_original_bounds': [x['task']+'/'+x['arm'] for x in replays if not x['original_bound_satisfied']],
                  'all_input_source_checkpoint_hashes_unchanged': True,
                  'replays': replays, 'saved_score_recalculation': score_summaries,
                  'new_cached_rollout_calls': 32, 'new_batched_predictor_forward_calls': 160,
                  'new_sample_macro_predictions': 4000,
                  'known_LR_gate_failure': 'NOT_RETESTED_NOT_REPLACED_REMAINS_ORIGINAL_FAILED_CHECK'}
        write_json(OUT / 'DIAGNOSIS.json', report)
        fields = ['task','arm','over_bound_coordinates','max_abs_error','max_excess','original_bound_satisfied','all_parameters_and_BN_buffers_unchanged','rollout_seconds']
        with (OUT / 'replay_summary.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for row in replays: w.writerow({k: row[k] for k in fields})
            f.flush(); os.fsync(f.fileno())
        outputs = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()}
        write_json(OUT / 'DIAGNOSTIC_FILES.json', {'status': 'DIAGNOSTIC_ONLY_NOT_A_RELEASE_GATE', 'files': outputs,
                                                'manifest_excludes_itself': True, 'source_manifest_sha256': FINAL_SHA})
        print(json.dumps({'path': str(OUT), 'status': report['status'], 'routes_outside_original_bounds': report['routes_outside_original_bounds'],
                          'all_saved_score_fields_exact': report['all_saved_score_fields_exact']}), flush=True)
    except BaseException:
        failure = {'status': 'DIAGNOSIS_EXECUTION_FAILED_NOT_ACCEPTANCE', 'time': now(), 'traceback': traceback.format_exc(),
                   'creates_acceptance_or_release_gate': False, 'new_optimizer_updates': 0}
        write_json(OUT / 'EXECUTION_FAILURE.json', failure)
        raise


if __name__ == '__main__': run()
