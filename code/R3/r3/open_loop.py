"""Fixed-case, decoder-free R3 open-loop scoring. Importing starts no work.

Only the initial three observed latent tokens and seven recorded macro actions
enter the forecast. All ten arms' forecasts are persisted before future targets
are materialized for scoring. No EVAL-derived normalization is fitted.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from . import common

ROOT = common.ROOT
VERSION = 'R3_OPEN_LOOP_V1'
HORIZONS = (1, 2, 5)
STEPS = (3000, 10000, 30000)
VARIANCE_FORMULA = 'mean_D(E_REFIT_TRAIN_all_raw_frames[z_d^2]-E[z_d]^2), population ddof0, float64 accumulator'
LOCK = 'manifests/MODELS_AND_SELECTION_LOCK.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def clean(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def local_path(relative, root=None):
    root = Path(root or ROOT).resolve(); p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Only ROOT-relative paths are accepted')
    q = root / p
    if not q.resolve().is_relative_to(root): raise ValueError('Path escapes ROOT')
    return q


def locked_record(value):
    sha = value if isinstance(value, str) else value['sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(x not in '0123456789abcdef' for x in sha):
        raise ValueError('Invalid locked SHA256')
    return sha, None if isinstance(value, str) else value.get('bytes')


def verify_lock(required=(), *, root=None):
    root = Path(root or ROOT); path = local_path(LOCK, root)
    if not path.is_file(): raise RuntimeError('BLOCKED_UNSEALED_MODELS: '+LOCK)
    lock = json.loads(path.read_text())
    if lock.get('status') != 'MODELS_AND_SELECTION_LOCKED':
        raise RuntimeError('Models/selection are not frozen')
    files = lock.get('files', {})
    if not files or any(x not in files for x in required):
        raise RuntimeError('Model lock omits required input/source: '+str(sorted(set(required)-set(files))))
    for rel, record in files.items():
        p = local_path(rel, root); sha, size = locked_record(record)
        if not p.is_file() or (size is not None and p.stat().st_size != size) or common.sha256(p) != sha:
            raise RuntimeError('Model lock identity differs: '+rel)
    return {'path': LOCK, 'sha256': common.sha256(path), 'files': files}


def validate_routes(route, task):
    if route.get('version') != 'R3_OPEN_LOOP_ROUTING_V1': raise ValueError('Unknown explicit checkpoint routing')
    arms = route['tasks'][task]['arms']; expected = {(None, 0)} | {(s, n) for s in common.SEEDS for n in STEPS}
    actual = [(r.get('seed'), r.get('step')) for r in arms]
    if len(actual) != 10 or set(actual) != expected: raise ValueError('Exactly H0 plus three seeds x three milestones are required')
    for r in arms:
        seed, step = r.get('seed'), r.get('step')
        if type(step) is not int or (seed is not None and type(seed) is not int): raise ValueError('Integer checkpoint identities required')
        expected_name = 'H0' if seed is None else f'REFIT_{seed}_{step}'
        if r.get('arm') != expected_name: raise ValueError('Arm name must identify its fixed seed/update')
        if seed is None:
            if any(k in r for k in ('checkpoint', 'run_identity', 'result')): raise ValueError('H0 cannot carry refit weights')
        else:
            for key in ('run_identity', 'result'): local_path(r[key])
            local_path(r['checkpoint']['path']); locked_record(r['checkpoint'])
    return sorted(arms, key=lambda r: (r['step'] != 0, r['seed'] or 0, r['step']))


def validate_cases(roles, task):
    if roles.get('task') != task or roles.get('status') == 'BLOCKED_DATA': raise ValueError('Task data roles are blocked/different')
    cases = roles['cases']['EVAL']; rows = {x['episode_id']: x for x in roles['episodes']}
    if not 20 <= len(cases) <= 100: raise ValueError('Frozen EVAL case count must be 20..100')
    if len({c['case_id'] for c in cases}) != len(cases) or len({c['episode_id'] for c in cases}) != len(cases):
        raise ValueError('EVAL needs unique case/episode identities')
    expected = {e for e, r in rows.items() if r['role'] == 'EVAL'}
    if {c['episode_id'] for c in cases} != expected: raise ValueError('Cases differ from frozen EVAL episodes')
    for c in cases:
        row = rows[c['episode_id']]
        if c['task'] != task or c['role'] != 'EVAL': raise ValueError('TECH/other-task case in formal evaluation')
        for key in ('source_asset_sha256', 'episode_sha256', 'source_episode_idx', 'length', 'family_id'):
            if c[key] != row[key]: raise ValueError('Case source identity differs')
        anchor = c['open_loop_anchor_raw']; start = c['open_loop_window_start_raw']
        if type(anchor) is not int or start != anchor-10 or anchor not in row['open_loop_starts']:
            raise ValueError('Open-loop anchor was not frozen as eligible')
        if c['open_loop_target_raw'] != {str(h): anchor+5*h for h in HORIZONS}:
            raise ValueError('Frozen target indexing differs')
    return cases


def build_inputs(cache, cases, contract):
    """No future latent value is indexed or returned by this function."""
    L, D, A = (int(contract[k]) for k in ('history_size', 'latent_dim', 'macro_action_dim'))
    if (L, A) != (3, 10): raise ValueError('R3 registered three-history/five-action-block contract differs')
    if set(cache) != {c['episode_id'] for c in cases}: raise ValueError('Exact EVAL cache membership required')
    histories, actions, history_raw, action_raw = [], [], [], []
    for c in cases:
        x = cache[c['episode_id']]; z, a, raw = (np.asarray(x[k]) for k in ('z', 'actions', 'raw_indices'))
        if set(x) != {'z', 'actions', 'raw_indices', 'stride', 'legal_starts'}: raise ValueError('Input-only cache fields required')
        if int(x['stride']) != 5 or z.dtype != np.float32 or z.shape != (c['length'], D): raise ValueError('Latent/stride contract differs')
        if a.dtype != np.float32 or a.shape != (len(z), 2) or not np.array_equal(raw, np.arange(len(z))):
            raise ValueError('Exact contiguous raw frame/action identities required')
        start = c['open_loop_window_start_raw']; anchor = c['open_loop_anchor_raw']
        if start != anchor-10 or start < 0 or start+40 > len(z): raise ValueError('Conservative official H5 clip span is not legal')
        if start not in np.asarray(x['legal_starts']): raise ValueError('Anchor is not a cache-declared legal one-step start')
        hi = start+np.arange(3)*5; ai = np.arange(start, start+35)
        initial, aa = z[hi], a[ai]
        if not np.isfinite(initial).all() or not np.isfinite(aa).all(): raise ValueError('Nonfinite observed history/recorded legal actions')
        histories.append(initial); actions.append(aa.reshape(7, 10)); history_raw.append(hi); action_raw.append(ai)
    return {'initial_z': np.stack(histories), 'macro_actions': np.stack(actions),
            'history_raw_indices': np.stack(history_raw), 'action_raw_indices': np.stack(action_raw),
            'case_ids': np.asarray([c['case_id'] for c in cases]), 'episode_ids': np.asarray([c['episode_id'] for c in cases])}


def scoring_targets(cache, cases):
    """Called only after all ten raw forecast bundles are persisted."""
    indices = np.asarray([[c['open_loop_anchor_raw']+5*h for h in range(1, 6)] for c in cases])
    targets = np.stack([cache[c['episode_id']]['z'][i] for c, i in zip(cases, indices)])
    if not np.isfinite(targets).all(): raise RuntimeError('Nonfinite frozen future target is a data failure, not model loss')
    return {'target_z': targets, 'target_raw_indices': indices}


def train_variance(cache):
    if not cache: raise ValueError('Empty REFIT_TRAIN cache')
    s1 = s2 = None; count = 0
    for ep in sorted(cache):
        z = np.asarray(cache[ep]['z'], dtype=np.float64)
        if z.ndim != 2 or not len(z) or not np.isfinite(z).all(): raise ValueError('Invalid TRAIN latent')
        if s1 is None: s1 = np.zeros(z.shape[1], dtype=np.float64); s2 = s1.copy()
        if z.shape[1] != len(s1): raise ValueError('Inconsistent TRAIN coordinate dimension')
        s1 += z.sum(0); s2 += np.square(z).sum(0); count += len(z)
    mean = s1/count; per_coordinate = s2/count-np.square(mean); value = float(per_coordinate.mean())
    if not np.isfinite(value) or value <= 0: raise ValueError('TRAIN scalar variance is not positive/finite')
    return {'scalar': value, 'formula': VARIANCE_FORMULA, 'frames': count, 'episodes': len(cache),
            'frame_weighting': 'POOLED_RAW_FRAMES_NOT_EPISODE_EQUAL', 'role': 'REFIT_TRAIN',
            'coordinate_mean': mean, 'coordinate_variance': per_coordinate}


def score_case(pred, target, scalar, history):
    pred, target, history = (np.asarray(x, dtype=np.float64) for x in (pred, target, history))
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[0] != 5: raise ValueError('Expected five terminal raw-coordinate vectors')
    result = []
    for h in HORIZONS:
        p, t = pred[h-1], target[h-1]; finite = bool(np.isfinite(p).all())
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            mse = float(np.square(p-t).mean()) if finite else float('inf')
            if not math.isfinite(mse): mse = float('inf')
            p2, t2 = float(np.square(p).sum()), float(np.square(t).sum())
            drift = float(np.square(p-history[-1]).mean())
        result.append({'horizon_macro': h, 'horizon_raw': 5*h, 'latent_raw_MSE': mse,
                       'latent_TRAIN_variance_normalized_MSE': mse/scalar,
                       'prediction_norm_l2': math.sqrt(p2), 'target_norm_l2': math.sqrt(t2),
                       'prediction_norm2_over_D': p2/len(p), 'target_norm2_over_D': t2/len(t),
                       'prediction_displacement_from_initial_MSE': drift,
                       'history_norm2_over_D': np.square(history).mean(1).tolist(),
                       'prediction_finite': finite, 'prediction_prefix_finite': bool(np.isfinite(pred[:h]).all()),
                       'target_finite': bool(np.isfinite(t).all()), 'metric_finite': math.isfinite(mse),
                       'nonfinite_prediction_coordinates': int((~np.isfinite(p)).sum()),
                       'failure_category': None if finite and math.isfinite(mse) else 'METHOD_NONFINITE',
                       'nonfinite_risk_policy': 'EXTENDED_POSITIVE_INFINITY_NO_CASE_DELETION'})
    return clean(result)


def atomic_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def file_record(path):
    return {'sha256': common.sha256(path), 'bytes': path.stat().st_size}


def write_or_check_npz(path, values):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], v, equal_nan=True) if np.asarray(v).dtype.kind in 'fc' else not np.array_equal(old[k], v) for k, v in values.items()):
                raise RuntimeError('Existing input/target bundle differs: '+str(path))
    else: atomic_npz(path, values)


def validate_checkpoint(route, checkpoint, identity, result, model):
    job = identity['job']
    if identity.get('technical') is not False or identity.get('role') != 'REFIT_TRAIN' or identity.get('endpoint') != 30000:
        raise RuntimeError('Only formal REFIT_TRAIN checkpoints may be scored')
    if job['task'] != model.r3_identity['task'] or job['refit_seed'] != route['seed'] or job['updates'] != 30000:
        raise RuntimeError('Checkpoint task/seed/budget routing differs')
    identity_sha = digest(identity)
    if type(checkpoint.get('step')) is not int or checkpoint.get('step') != route['step'] or checkpoint.get('identity_sha256') != identity_sha:
        raise RuntimeError('Checkpoint update count or run identity differs')
    if (result.get('actual_updates') != 30000 or result.get('technical') is not False
            or result.get('identity_sha256') != identity_sha or result.get('job_id') != job['job_id']
            or result.get('status') != 'REFIT_TRAINING_COMPLETE_UNSCORED'):
        raise RuntimeError('All fixed30k training must finish before any EVAL score')
    if identity['base_identity'] != model.r3_identity or identity['contract'] != model.r3_contract:
        raise RuntimeError('Checkpoint official source/coordinate differs')
    if checkpoint['delta']['base_identity'] != model.r3_identity or checkpoint['delta']['contract'] != model.r3_contract:
        raise RuntimeError('Checkpoint delta identity differs')
    if result.get('frozen_before') != result.get('frozen_after') or result.get('frozen_before') != checkpoint['delta']['frozen_sha256']:
        raise RuntimeError('Checkpoint/result frozen-state invariance differs')


def evaluate(task, routing='manifests/OPEN_LOOP_ROUTING.json', device='cpu', batch_size=32):
    if task not in common.TASKS or type(batch_size) is not int or not 1 <= batch_size <= 100: raise ValueError('Invalid task/batch size')
    local_path(routing)
    # Check the lock BEFORE importing model code/loading any EVAL inputs.
    required = [routing, 'r3/open_loop.py', 'r3/common.py', 'r3/model.py', 'r3/data.py', 'r3/roles.py',
                'manifests/JOBS.json', f'manifests/{task}_data_roles.json', f'manifests/{task}_cache.json',
                f'manifests/{task}_model_assets.json', f'manifests/{task}_normalization.json']
    locked = verify_lock(required)
    routes = validate_routes(common.read_json(routing), task)
    for r in routes:
        if r['step']:
            required += [r['checkpoint']['path'], r['run_identity'], r['result']]
            if locked_record(locked['files'].get(r['checkpoint']['path'], {}))[0] != r['checkpoint']['sha256']:
                raise RuntimeError('Explicit route does not match locked checkpoint SHA')
    assets = common.read_json(f'manifests/{task}_model_assets.json')
    required += [r['path'] for r in assets['files'].values()]
    cm = common.read_json(f'manifests/{task}_cache.json')
    required += list(cm['input_hashes'])
    locked = verify_lock(required)
    roles = common.read_json(f'manifests/{task}_data_roles.json'); cases = validate_cases(roles, task)
    folder = ROOT/'artifacts/open_loop'/task; folder.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        for p, kind in [(ROOT/'state/execution_phase.lock', fcntl.LOCK_EX), (folder/'worker.lock', fcntl.LOCK_EX)]:
            p.parent.mkdir(parents=True, exist_ok=True); handle = stack.enter_context(p.open('a+')); fcntl.flock(handle, kind|fcntl.LOCK_NB)
        return _evaluate_locked(task, routing, routes, device, batch_size, locked, required, cm, cases, folder)


def _evaluate_locked(task, routing, routes, device, batch_size, locked, required, cm, cases, folder):
    from .data import load_cache
    from .model import load_official, apply_delta, cached_rollout, frozen_hashes, assert_frozen, tensor_sha256
    import torch
    identity = {'version': VERSION, 'task': task, 'models_lock_sha256': locked['sha256'], 'routing_sha256': common.sha256(local_path(routing)),
                'case_sha256': digest(cases), 'device': str(device), 'batch_size': batch_size, 'horizons_macro': list(HORIZONS),
                'variance_formula': VARIANCE_FORMULA, 'evaluation_kind': 'OPEN_LOOP', 'precision': 'FP32_NO_AMP_NO_TF32'}
    identity_sha = digest(identity); receipt_path = folder/'OPEN_LOOP_RECEIPT.json'
    if receipt_path.exists():
        receipt = common.read_json(receipt_path)
        if receipt['identity'] != identity: raise RuntimeError('Existing open-loop result uses different frozen inputs')
        for rel, rec in receipt['files'].items():
            if file_record(local_path(rel)) != rec: raise RuntimeError('Completed open-loop artifact differs: '+rel)
        return receipt
    prior = folder/'RUN_IDENTITY.json'
    if prior.exists() and common.read_json(prior) != identity: raise RuntimeError('Unfinished evaluation has different identity')
    common.atomic_json(prior, identity); common.ensure_space()
    common.fp32_policy(); torch.use_deterministic_algorithms(True)
    if str(device).startswith('cuda'):
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    # Validate every explicitly routed checkpoint before EVAL cache access or H0
    # inference; a missing/unfinished seed does not yield a partial comparison.
    model = load_official(task, device); model.eval().requires_grad_(False); frozen = frozen_hashes(model)
    for r in routes:
        if r['step']:
            path = local_path(r['checkpoint']['path'])
            if common.sha256(path) != r['checkpoint']['sha256']: raise RuntimeError('Checkpoint changed after model lock')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            validate_checkpoint(r, checkpoint, common.read_json(r['run_identity']), common.read_json(r['result']), model)
            del checkpoint
    train = load_cache(task, 'REFIT_TRAIN'); variance = train_variance(train); del train
    if (cm.get('train_variance_formula') != VARIANCE_FORMULA or cm.get('train_variance_frames') != variance['frames']
            or not math.isclose(cm['train_scalar_variance'], variance['scalar'], rel_tol=1e-10, abs_tol=1e-12)):
        raise RuntimeError('Independent TRAIN variance check differs from frozen cache manifest')
    coord = {'coordinate_mean': variance.pop('coordinate_mean'), 'coordinate_variance': variance.pop('coordinate_variance')}
    write_or_check_npz(folder/'TRAIN_coordinates.npz', coord)
    common.atomic_json(folder/'TRAIN_variance.json', variance)
    cache = load_cache(task, 'EVAL'); inputs = build_inputs(cache, cases, model.r3_contract)
    write_or_check_npz(folder/'inputs.npz', inputs)
    predictions = {}; arm_receipts = []; reused_arms = []; started = time.perf_counter()
    def state_sha(): return digest({k: tensor_sha256(v) for k, v in model.state_dict().items()})
    for r in routes:
        pred_path = folder/'predictions'/(r['arm']+'.npz'); rec_path = pred_path.with_suffix('.json')
        if rec_path.exists():
            rec = common.read_json(rec_path)
            if rec['identity_sha256'] != identity_sha or rec['route'] != r or rec['prediction_file'] != file_record(pred_path):
                raise RuntimeError('Saved forecast identity differs')
            with np.load(pred_path, allow_pickle=False) as f: pred = f['prediction'].copy()
            reused_arms.append(r['arm'])
        else:
            if r['step']:
                path = local_path(r['checkpoint']['path'])
                if common.sha256(path) != r['checkpoint']['sha256']: raise RuntimeError('Checkpoint changed after model lock')
                cp = torch.load(path, map_location='cpu', weights_only=True)
                validate_checkpoint(r, cp, common.read_json(r['run_identity']), common.read_json(r['result']), model)
                del cp; apply_delta(model, path); model.eval().requires_grad_(False)
            before = state_sha(); t = time.perf_counter(); outputs = []
            with torch.inference_mode():
                for i in range(0, len(cases), batch_size):
                    z = torch.as_tensor(inputs['initial_z'][i:i+batch_size], device=device)
                    a = torch.as_tensor(inputs['macro_actions'][i:i+batch_size], device=device)
                    outputs.append(cached_rollout(model, z, a, 5).cpu().numpy())
            pred = np.concatenate(outputs); elapsed = time.perf_counter()-t
            if state_sha() != before: raise RuntimeError('Inference changed model parameters or buffers')
            assert_frozen(model, frozen)
            if pred.shape != (len(cases), 5, model.r3_contract['latent_dim']) or pred.dtype != np.float32: raise RuntimeError('Forecast shape/dtype differs')
            atomic_npz(pred_path, {'prediction': pred, 'case_ids': inputs['case_ids']})
            rec = {'identity_sha256': identity_sha, 'route': r, 'prediction_file': file_record(pred_path),
                   'seconds': elapsed, 'model_state_sha256_before_and_after': before, 'frozen_sha256': frozen['sha256'],
                   'future_targets_read_by_forecast': False, 'optimizer_updates': 0}
            common.atomic_json(rec_path, rec)
        predictions[r['arm']] = pred; arm_receipts.append(rec)
    # All arms have immutable forecast receipts before target materialization.
    targets = scoring_targets(cache, cases); write_or_check_npz(folder/'targets.npz', targets)
    per_case = []
    for ci, case in enumerate(cases):
        record = {'evaluation_kind': 'OPEN_LOOP', 'phase': 'EVAL', 'task': task, 'case': case, 'arms': []}
        for r in routes:
            record['arms'].append({'arm': r['arm'], 'refit_seed': r['seed'], 'checkpoint_step': r['step'],
                                  'metrics': score_case(predictions[r['arm']][ci], targets['target_z'][ci], variance['scalar'], inputs['initial_z'][ci])})
        per_case.append(record)
        common.atomic_json(folder/'cases'/(hashlib.sha256(case['case_id'].encode()).hexdigest()+'.json'), record)
    common.atomic_json(folder/'per_case.json', per_case)
    final_lock = verify_lock(required)
    if final_lock['sha256'] != locked['sha256']: raise RuntimeError('Models lock changed during evaluation')
    files = {str(p.relative_to(ROOT)): file_record(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name not in ('worker.lock', 'OPEN_LOOP_RECEIPT.json') and not p.name.endswith('.tmp')}
    receipt = {'status': 'COMPLETE', 'evaluation_kind': 'OPEN_LOOP', 'identity': identity, 'task': task,
               'cases': len(cases), 'arms': len(routes), 'case_horizon_records': len(cases)*len(routes)*len(HORIZONS),
               'arm_receipts': arm_receipts, 'train_variance': variance, 'files': files,
               'this_invocation_forecast_and_scoring_seconds': time.perf_counter()-started,
               'reused_forecast_arms': reused_arms,
               'all_future_targets_scoring_only': True, 'decoder_fits': 0, 'optimizer_updates': 0,
               'closed_loop_trajectories': 0, 'physical_state_labels_read': 0,
               'offline_history': 'three real recorded observations; distinct from official closed-loop initial history of one',
               'scope': 'same frozen EVAL episodes and preselected anchors; source-distribution post-refit holdout; no independent-pretrain claim'}
    common.atomic_json(receipt_path, receipt); return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('task', choices=common.TASKS)
    p.add_argument('--routing', default='manifests/OPEN_LOOP_ROUTING.json'); p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32); a = p.parse_args()
    out = evaluate(a.task, a.routing, a.device, a.batch_size)
    print(json.dumps({'status': out['status'], 'task': out['task'], 'cases': out['cases'], 'arms': out['arms']}))


if __name__ == '__main__': main()
