"""Root-invoked, zero-update TECH checks of actual official inputs and model.

Importing this module performs no execution. The two windows are the first two
already frozen technical_monitor_windows. Only pixels/action are opened in H5;
no EVAL, state, reward, optimizer construction or optimizer step is involved.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from r3 import common, data
from r3.model import (load_official, cached_predict, cached_rollout, refit_mode,
                      frozen_hashes, assert_frozen, whitelist_manifest,
                      tensor_sha256, TRAINABLE_PREFIXES)

ROOT = common.ROOT
VERSION = 'R3_REAL_TECH_MODEL_CHECK_V1'
TOLERANCE_PATH = 'manifests/NUMERICAL_TOLERANCES.json'
# Root registered these values before any real-data model GPU check. Protocol07
# itself specifies no numeric tolerance; the separate manifest is the authority.
REGISTERED_TOLERANCES = {'encoder_cache': {'atol': 1e-5, 'rtol': 1e-5},
                         'predict_wrapper': {'atol': 1e-6, 'rtol': 1e-6},
                         'rollout_wrapper': {'atol': 1e-6, 'rtol': 1e-6}}


def tolerances_from(record):
    if record.get('status') != 'PROSPECTIVELY_FROZEN':
        raise RuntimeError('Numeric tolerances must be frozen before the real TECH check')
    found = record.get('comparisons', {})
    if any(found.get(k) != v for k, v in REGISTERED_TOLERANCES.items()):
        raise RuntimeError('Missing or changed pre-registered numerical tolerance')
    return {k: dict(found[k]) for k in REGISTERED_TOLERANCES}


def choose_windows(roles, task, contract):
    if roles.get('task') != task or roles.get('status') != 'METADATA_ROLES_FROZEN':
        raise RuntimeError('Exact frozen task roles are required')
    L = int(contract['history_size'])
    if L != 3 or int(contract['macro_action_dim']) != 10 or roles.get('frameskip') != 5 or roles.get('history_size') != L:
        raise ValueError('Actual official contract differs from registered L3/action5x2')
    windows = roles.get('technical_monitor_windows', [])[:2]
    if len(windows) != 2 or len({tuple(x) for x in windows}) != 2:
        raise ValueError('Two distinct fixed TECH windows required; do not choose substitutes')
    rows = {x['episode_id']: x for x in roles['episodes']}
    out = []
    for ep, start in windows:
        row = rows.get(ep)
        if row is None or row.get('role') != 'TECH':
            raise PermissionError('Only the first two frozen TECH windows may be read')
        if type(start) is not int or start < 0 or start+40 > row['length'] or start+10 not in row['open_loop_starts']:
            raise ValueError('Frozen TECH window lacks its declared conservative H5 span')
        out.append({'episode': row, 'start_raw_index': start,
                    'observation_raw_indices': (start+5*np.arange(8)).tolist(),
                    'history_raw_indices': (start+5*np.arange(3)).tolist(),
                    'teacher_forced_target_raw_indices': (start+5*np.arange(1, 4)).tolist(),
                    'free_rollout_target_raw_indices': (start+5*np.arange(3, 8)).tolist(),
                    'action_raw_indices': list(range(start, start+35))})
    return out


def macro_actions(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) % 5:
        raise ValueError('Explicit consecutive raw actions [5*T,2] required')
    if not np.isfinite(raw).all(): raise ValueError('Nonfinite action in selected legal TECH window')
    return raw.reshape(len(raw)//5, 10)


def pulse_check():
    """All70 scalar action positions, not just block means or first actions."""
    traces = []
    for i in range(35):
        for coordinate in range(2):
            pulse = np.zeros((35, 2), dtype=np.float32); pulse[i, coordinate] = 1
            packed = macro_actions(pulse)
            expected = np.zeros((7, 10), dtype=np.float32)
            expected[i//5, 2*(i%5)+coordinate] = 1
            if not np.array_equal(packed, expected): raise AssertionError('Raw-action pulse timing was changed')
            traces.append({'raw_action_offset': i, 'coordinate': coordinate,
                           'macro_token': i//5, 'macro_channel': 2*(i%5)+coordinate})
    return {'status': 'PASS', 'exact': True, 'scalar_pulses': 70, 'index_trace': traces,
            'rule': 'macro[t,2*j+c] = normalized_raw[5*t+j,c]; j=0..4,c=0..1'}


def compare(actual, reference, tolerance=None):
    a = actual.detach().cpu().double(); b = reference.detach().cpu().double()
    if a.shape != b.shape:
        return {'status': 'FAIL', 'reason': 'SHAPE_DIFFERENCE', 'actual_shape': list(a.shape), 'reference_shape': list(b.shape)}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    out = {'status': 'DIAGNOSTIC_ONLY' if tolerance is None else 'FAIL', 'shape': list(a.shape),
           'finite': finite, 'actual_sha256': tensor_sha256(actual), 'reference_sha256': tensor_sha256(reference),
           'max_abs_difference': None, 'rmse_difference': None, 'tolerance': tolerance}
    if finite:
        error = (a-b).abs(); out['max_abs_difference'] = float(error.max())
        out['rmse_difference'] = float((a-b).square().mean().sqrt())
        if tolerance is not None:
            threshold = tolerance['atol']+tolerance['rtol']*b.abs()
            out['elements_exceeding_tolerance'] = int((error > threshold).sum())
            out['max_threshold_excess'] = float((error-threshold).max())
            out['status'] = 'PASS' if out['elements_exceeding_tolerance'] == 0 else 'FAIL'
    return out


def _parameter_hashes(model):
    return {name: tensor_sha256(p) for name, p in model.named_parameters()}


def model_checks(model, transformed_pixels, cached_z, normalized_macro, tolerance):
    """One real or synthetic batch, no optimizer. Future z only supplies targets.

    Direct original JEPA.rollout is compared on its own freshly encoded initial
    coordinates, separating wrapper arithmetic from already measured cache
    encoding roundoff. The production cache-start difference is also retained.
    """
    model.eval().requires_grad_(False)
    L = int(model.r3_contract['history_size']); H = 5
    if cached_z.ndim != 3 or cached_z.shape[1] != L+H or transformed_pixels.shape[:2] != cached_z.shape[:2]:
        raise ValueError('Aligned three-history plus five-target TECH frames required')
    if normalized_macro.shape != (len(cached_z), L+H-1, 10): raise ValueError('Exactly seven full macro actions required')
    if cached_z.requires_grad or normalized_macro.requires_grad or transformed_pixels.requires_grad:
        raise ValueError('Observed inputs and targets must have no gradient')
    before = frozen_hashes(model); weights_before = _parameter_hashes(model)
    arrays = {}; checks = {}; gradients = []
    try:
        with torch.no_grad():
            direct_z = model.encode({'pixels': transformed_pixels})['emb']
            checks['encoder_cache'] = compare(direct_z, cached_z, tolerance['encoder_cache'])
            direct = model.predict(cached_z[:, :L], model.action_encoder(normalized_macro[:, :L]))
            wrapped = cached_predict(model, cached_z[:, :L], normalized_macro[:, :L])
            checks['predict_wrapper'] = compare(wrapped, direct, tolerance['predict_wrapper'])
            # This is the actual unmodified official rollout, including encode.
            info = model.rollout({'pixels': transformed_pixels[:, None, :L]},
                                 normalized_macro[:, None], history_size=L)
            official_rollout = info['predicted_emb'][:, 0, -H:]
            original_initial_z = info['emb'][:, 0]
            if original_initial_z.shape != cached_z[:, :L].shape:
                raise ValueError('Official rollout no longer exposes its original initial history')
            wrapped_same_initial = cached_rollout(model, original_initial_z, normalized_macro, H)
            checks['rollout_wrapper'] = compare(wrapped_same_initial, official_rollout, tolerance['rollout_wrapper'])
            checks['rollout_initial_encoder_cache'] = compare(original_initial_z, cached_z[:, :L], tolerance['encoder_cache'])
            production_rollout = cached_rollout(model, cached_z[:, :L], normalized_macro, H)
            checks['pixels_vs_cache_start_rollout_diagnostic'] = compare(production_rollout, official_rollout)
            arrays = {'direct_encoded_z': direct_z, 'cached_z': cached_z,
                      'normalized_macro_actions': normalized_macro,
                      'direct_prediction_same_cached_history': direct,
                      'cached_prediction_same_history': wrapped,
                      'official_rollout_initial_z': original_initial_z,
                      'official_rollout_last5': official_rollout,
                      'cached_rollout_same_official_initial_z': wrapped_same_initial,
                      'production_cached_rollout_last5': production_rollout}
        assert_frozen(model, before)
        model.zero_grad(set_to_none=True); refit_mode(model, training=True)
        modes = {'predictor_training': model.predictor.training, 'pred_proj_training': model.pred_proj.training,
                 'all_batchnorm_eval': all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)),
                 'frozen_components_eval': all(not m.training for key in ('encoder', 'projector', 'action_encoder') for m in getattr(model, key).modules())}
        prediction = cached_predict(model, cached_z[:, :L], normalized_macro[:, :L])
        target = cached_z[:, 1:L+1]
        loss = (prediction-target).square().mean()
        finite_loss = bool(torch.isfinite(loss))
        if finite_loss: loss.backward()
        nonzero_by_prefix = {prefix: 0 for prefix in TRAINABLE_PREFIXES}
        count_by_prefix = {prefix: 0 for prefix in TRAINABLE_PREFIXES}
        bad = []
        for name, parameter in model.named_parameters():
            allowed = name.startswith(TRAINABLE_PREFIXES); grad = parameter.grad
            finite = grad is not None and bool(torch.isfinite(grad).all())
            nonzero = int(torch.count_nonzero(grad)) if finite else 0
            row = {'name': name, 'whitelisted': allowed, 'requires_grad': parameter.requires_grad,
                   'gradient_is_none': grad is None, 'gradient_finite': finite if grad is not None else None,
                   'gradient_nonzero_elements': nonzero, 'gradient_sha256': tensor_sha256(grad) if grad is not None else None,
                   'gradient_l2': float(grad.detach().double().norm()) if finite else None}
            gradients.append(row)
            if allowed:
                prefix = next(x for x in TRAINABLE_PREFIXES if name.startswith(x))
                count_by_prefix[prefix] += 1; nonzero_by_prefix[prefix] += int(nonzero > 0)
                if not parameter.requires_grad or grad is None or not finite: bad.append(name)
            elif parameter.requires_grad or grad is not None: bad.append(name)
        if any(count_by_prefix[p] and nonzero_by_prefix[p] == 0 for p in TRAINABLE_PREFIXES):
            bad.append('NO_NONZERO_GRADIENT_IN_WHITELIST_COMPONENT')
        after = assert_frozen(model, before)
        weights_after = _parameter_hashes(model)
        unchanged = weights_after == weights_before
        checks['zero_update_backward'] = {'status': 'PASS' if finite_loss and not bad and unchanged and modes['predictor_training'] and modes['all_batchnorm_eval'] and modes['frozen_components_eval'] else 'FAIL',
            'loss': float(loss.detach()) if finite_loss else None, 'finite_loss': finite_loss,
            'target_requires_grad': target.requires_grad, 'target_alignment': 'all L next-token targets z[:,1:L+1]; no final-token-only substitution',
            'training_modes': modes, 'problem_parameters': bad,
            'nonzero_gradient_parameters_by_prefix': nonzero_by_prefix,
            'whitelist_parameters_by_prefix': count_by_prefix,
            'zero_gradient_policy': 'Every used whitelist parameter must have finite non-None gradients; each nonempty whitelist component must have a nonzero gradient. Individual exact-zero gradients are retained, not assumed bugs.',
            'all_parameter_values_exactly_unchanged': unchanged,
            'frozen_parameters_and_all_buffers_exactly_unchanged': after == before,
            'optimizer_objects_created': 0, 'optimizer_updates': 0}
        arrays['training_mode_prediction'] = prediction.detach(); arrays['training_targets'] = target.detach()
        tensor_finiteness = {name: bool(torch.isfinite(value).all()) for name, value in arrays.items()}
        checks['all_saved_model_arrays_finite'] = {'status': 'PASS' if all(tensor_finiteness.values()) else 'FAIL',
                                                 'arrays': tensor_finiteness}
        report = {'status': 'PASS' if all(x['status'] == 'PASS' for key, x in checks.items() if not key.endswith('_diagnostic')) else 'FAIL',
                  'checks': checks, 'gradients': gradients, 'frozen_hashes_before': before, 'frozen_hashes_after': after,
                  'all_parameter_hashes_before': weights_before, 'all_parameter_hashes_after': weights_after}
        return report, {key: value.detach().cpu().numpy().copy() for key, value in arrays.items()}
    finally:
        model.zero_grad(set_to_none=True); refit_mode(model, training=False)


def _stat_identity(path):
    s = Path(path).stat(); return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def _selected_cache(task, roles, windows, manifest):
    if manifest.get('status') != 'FROZEN_OBSERVED_CACHE_COMPLETE' or manifest.get('task') != task:
        raise RuntimeError('Actual complete frozen cache required')
    if manifest['roles_manifest_sha256'] != common.sha256(ROOT/f'manifests/{task}_data_roles.json'):
        raise RuntimeError('Cache roles identity differs')
    for name, digest in manifest['input_hashes'].items():
        if common.sha256(data.checked_local(name)) != digest: raise RuntimeError('Frozen cache input changed: '+name)
    expected = {r['episode_id'] for r in roles['episodes'] if r['role'] == 'TECH'}
    if {ep for ep, r in manifest['episodes'].items() if r['role'] == 'TECH'} != expected:
        raise RuntimeError('TECH cache roster differs from roles')
    result = {}; records = {}
    for window in windows:
        ep = window['episode']['episode_id']
        if ep in result: continue
        record = manifest['episodes'][ep]
        if record['role'] != 'TECH': raise PermissionError('Selected cache is not TECH')
        path = data.verified_file(record)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'z', 'actions', 'raw_indices', 'stride', 'legal_starts'}:
                raise ValueError('Cache contains unregistered fields, possibly task state')
            result[ep] = {key: archive[key].copy() for key in archive.files}
        records[ep] = record
    return result, records


def collect_inputs(task, model, roles, cache_manifest, source_map):
    windows = choose_windows(roles, task, model.r3_contract)
    cache, records = _selected_cache(task, roles, windows, cache_manifest)
    processor = data.action_processor(task); transform = data.image_transform()
    readers = {}; source_records = {}; watched = {}; zz = []; aa = []; pp = []; traces = []
    try:
        for window in windows:
            row = window['episode']; ep = row['episode_id']; item = cache[ep]
            n, D = row['length'], int(model.r3_contract['latent_dim'])
            if item['z'].shape != (n, D) or item['z'].dtype != np.float32 or item['actions'].shape != (n, 2) or item['actions'].dtype != np.float32:
                raise ValueError('Cache shape/dtype differs from actual official coordinate')
            if int(item['stride']) != 5 or not np.array_equal(item['raw_indices'], np.arange(n)):
                raise ValueError('Cache raw time identities differ')
            start = window['start_raw_index']
            if start not in item['legal_starts']: raise ValueError('TECH start was not declared cache-legal')
            asset = row['source_asset_sha256']; record = source_map['assets'][asset]
            if record['sha256'] != asset: raise RuntimeError('Source map key must be the actual uncompressed H5 SHA')
            if asset not in readers:
                path = data.verified_file(record); watched[str(path)] = _stat_identity(path)
                readers[asset] = data.RawH5(path, keys=['pixels', 'action'])
                source_records[asset] = record
            reader = readers[asset]; source_index = row['source_episode_idx']
            if int(reader.lengths[source_index]) != n: raise RuntimeError('Actual source episode length changed')
            iz = np.asarray(window['observation_raw_indices'], dtype=np.int64)
            # Reading individually avoids inspecting intervening observation frames.
            pixels = np.concatenate([reader.array(source_index, 'pixels', int(i), int(i)+1) for i in iz])
            raw_actions = reader.array(source_index, 'action', start, start+35)
            normalized = processor.transform(raw_actions).astype(np.float32)
            cached_actions = item['actions'][start:start+35]
            if not np.array_equal(normalized, cached_actions): raise RuntimeError('Exact same-processor cached action normalization differs')
            macro = macro_actions(normalized)
            if not np.array_equal(macro.reshape(35, 2), normalized): raise AssertionError('Macro action time/coordinate order changed')
            x = torch.from_numpy(pixels)
            if pixels.shape[-1] == 3: x = x.permute(0, 3, 1, 2)
            x = transform(x)
            if x.shape != (8, 3, 224, 224) or x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
                raise ValueError('Actual transformed TECH image contract differs')
            pp.append(x); zz.append(item['z'][iz]); aa.append(macro)
            traces.append({**window, 'raw_pixels_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                           'raw_pixels_shape': list(pixels.shape), 'raw_pixels_dtype': str(pixels.dtype),
                           'raw_actions_sha256': hashlib.sha256(raw_actions.tobytes()).hexdigest(),
                           'raw_actions_dtype': str(raw_actions.dtype), 'raw_actions': raw_actions.tolist(),
                           'normalized_raw_actions': normalized.tolist(), 'macro_actions': macro.tolist(),
                           'action_normalization_cache_exact': True,
                           'transformed_images_sha256': tensor_sha256(x),
                           'source_raw_keys_read': ['pixels', 'action'], 'state_reward_goal_labels_read': 0})
        return torch.stack(pp), torch.from_numpy(np.stack(zz)), torch.from_numpy(np.stack(aa)), {
            'windows': traces, 'source_files': source_records, 'cache_files': records,
            'source_stat_guards': watched, 'actual_source_columns_read': ['pixels', 'action'],
            'actual_raw_pixel_frames_read': 16, 'actual_raw_action_rows_read': 70,
            'other_role_cache_arrays_read': 0, 'source_state_reward_goal_labels_read': 0}
    finally:
        for reader in readers.values(): reader.close()


def save_arrays(path, arrays):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(task, device, output_dir):
    if task not in common.TASKS: raise ValueError('Only two official tasks')
    common.require_authorization({}, technical=True); common.ensure_space(); common.fp32_policy()
    output = data.checked_local(output_dir)
    if output.exists(): raise RuntimeError('Keep prior technical attempt evidence; choose a new output directory')
    output.mkdir(parents=True)
    report = {'version': VERSION, 'task': task, 'status': 'STARTED', 'started_at': common.now(),
              'scope': 'ZERO_UPDATE_TECHNICAL_CHECK_NOT_SCIENTIFIC_RESULTS', 'device': str(device),
              'optimizer_objects_created': 0, 'optimizer_updates': 0, 'full_policy_trajectories': 0,
              'EVAL_arrays_read': 0, 'source_state_reward_goal_labels_read': 0}
    started = time.perf_counter(); model = None
    try:
        needed = [TOLERANCE_PATH, f'manifests/{task}_cache.json', f'manifests/{task}_data_roles.json',
                  f'manifests/{task}_source_map.json', f'manifests/{task}_normalization.json',
                  'r3/data.py', 'r3/model.py', 'r3/common.py', 'r3/roles.py', 'scripts/check_real_data_model.py',
                  'protocol/07_PROTOCOL_CHECKS.py']
        report['input_hashes'] = {name: common.sha256(data.checked_local(name)) for name in needed}
        tolerance = tolerances_from(common.read_json(TOLERANCE_PATH)); report['tolerances'] = tolerance
        report['numeric_contract_source'] = 'Separately prospectively frozen NUMERICAL_TOLERANCES.json; protocol07 contains no numeric tolerances'
        roles = common.read_json(f'manifests/{task}_data_roles.json')
        cache_manifest = common.read_json(f'manifests/{task}_cache.json')
        source_map = common.read_json(f'manifests/{task}_source_map.json')
        model = load_official(task, device); report['official_identity'] = model.r3_identity
        report['loaded_frozen_hashes_before'] = frozen_hashes(model)
        report['loaded_parameter_hashes_before'] = _parameter_hashes(model)
        report['official_contract'] = model.r3_contract; report['whitelist'] = whitelist_manifest(model)
        pixels, z, actions, inputs = collect_inputs(task, model, roles, cache_manifest, source_map)
        report['inputs'] = inputs; report['macro_action_pulse_check'] = pulse_check()
        seed = int.from_bytes(hashlib.sha256(f'R3_REAL_TECH_MODEL_CHECK_20261002/{task}'.encode()).digest()[:8], 'big')
        common.seed_all(seed); report['technical_dropout_seed'] = seed
        report['technical_dropout_seed_rule'] = 'SHA256(R3_REAL_TECH_MODEL_CHECK_20261002/{task}) first8 bytes big endian; no relation to formal training RNG'
        checks, arrays = model_checks(model, pixels.to(device), z.to(device), actions.to(device), tolerance)
        report['model_checks'] = checks
        if torch.device(device).type == 'cuda': torch.cuda.synchronize(torch.device(device))
        for name, digest in report['input_hashes'].items():
            if common.sha256(data.checked_local(name)) != digest: raise RuntimeError('Input identity changed during TECH check: '+name)
        for record in inputs['cache_files'].values(): data.verified_file(record)
        for name, before in inputs['source_stat_guards'].items():
            if _stat_identity(name) != before: raise RuntimeError('Source H5 changed after its verified SHA')
        save_arrays(output/'TECH_MODEL_ARRAYS.npz', arrays)
        report['status'] = checks['status']
    except Exception as error:
        report.update(status='FAIL', error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        if model is not None:
            # Preserve exact post-failure identities as well as the exception.
            report['loaded_frozen_hashes_after'] = frozen_hashes(model)
            report['loaded_parameter_hashes_after'] = _parameter_hashes(model)
            report['loaded_model_tensors_unchanged'] = (
                report.get('loaded_frozen_hashes_before') == report['loaded_frozen_hashes_after']
                and report.get('loaded_parameter_hashes_before') == report['loaded_parameter_hashes_after'])
            if not report['loaded_model_tensors_unchanged']: report['status'] = 'FAIL'
            model.zero_grad(set_to_none=True)
        report['finished_at'] = common.now(); report['wall_seconds'] = time.perf_counter()-started
        common.atomic_json(output/'REAL_DATA_MODEL_CHECK.json', report)
        files = {p.name: {'sha256': common.sha256(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file() and p.name != 'SHA256.json'}
        common.atomic_json(output/'SHA256.json', {'status': report['status'], 'files': files})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=common.TASKS); parser.add_argument('--device', required=True)
    parser.add_argument('--output-dir', required=True, help='Unique ROOT-relative technical attempt folder')
    args = parser.parse_args()
    lock_path = ROOT/f'state/{args.task}_real_data_model_check.lock'; lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(args.task, args.device, args.output_dir)
    print(json.dumps({'task': args.task, 'status': result['status'], 'optimizer_updates': 0, 'output': args.output_dir}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__': main()
