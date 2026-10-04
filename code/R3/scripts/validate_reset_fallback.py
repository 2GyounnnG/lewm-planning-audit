"""Finite PushT missing-seed reset diagnostic; importing performs no work.

Root may execute --run on the authorized host. This never discovers a source
seed, changes roles, loads a model, constructs an optimizer, or runs CEM. Sixteen
fresh resets and sixteen raw zero-action steps are separately accounted.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from r3 import common
from r3.env_compat import normalize_reset_seed, SEED_COMPATIBILITY

ROOT = common.ROOT
VERSION = 'R3_PUSHT_MISSING_SEED_RESET_VALIDATION_V1'
SEEDS = (0, 1)
REPEATS = (0, 1)
PATHS = ('ORIGINAL_WORLD_EVALUATE', 'EXPLICIT_EQUIVALENT_RESET_WRAPPER')
OVERRIDDEN_DEFAULT_VARIATIONS = ('agent.start_position', 'block.start_position', 'block.angle')
SOURCE_MEMBERS = ('stable_worldmodel/world/world.py', 'stable_worldmodel/world/env_pool.py',
                  'stable_worldmodel/envs/pusht/env.py', 'stable_worldmodel/envs/utils.py',
                  'stable_worldmodel/spaces.py', 'stable_worldmodel/wrapper/default.py')


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite_float__': 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'}
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)): return value
    raise TypeError('Opaque reset evidence: '+type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def contract():
    return {'version': VERSION, 'task': 'pusht', 'case_selection': 'Frozen roles.cases.TECH[:2], no replacement',
            'seeds': list(SEEDS), 'repeats_per_seed_and_path': 2, 'paths': list(PATHS),
            'action': [0.0, 0.0], 'raw_steps_per_reset': 1, 'expected_resets': 16, 'expected_raw_steps': 16,
            'hard_comparisons': 'Exact finite effective physical state/render and one-step outcome equality within seed, across seeds, and across original/wrapper path. Non-overridden variation values and physics configuration must also be exact.',
            'overridden_random_default_variations': list(OVERRIDDEN_DEFAULT_VARIATIONS),
            'source_state_or_pixel_reconstruction_bias': 'All original differences retained as diagnostics; no zero-bias assertion because official _set_state advances space.step(dt).',
            'candidate_identity': 'SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED', 'candidate_reset_seed': 0,
            'reset_seed_type_compatibility': SEED_COMPATIBILITY,
            'scope_limit': 'Two fixed TECH cases and two tested seeds in the exact source/environment only; not proof of original-seed recovery, arbitrary-seed independence, or all-EVAL-case equivalence.',
            'optimizer_updates': 0, 'CEM_calls': 0, 'complete_formal_or_technical_CEM_trajectories': 0,
            'diagnostic_one_step_segments': 16, 'changes_existing_roles': False}


def selected_cases(roles):
    if roles.get('task') != 'pusht' or roles.get('status') != 'METADATA_ROLES_FROZEN':
        raise ValueError('Frozen actual PushT metadata roles required')
    cases = roles['cases']['TECH'][:2]
    if len(cases) != 2 or len({c['case_id'] for c in cases}) != 2 or len({c['episode_id'] for c in cases}) != 2:
        raise ValueError('Two distinct fixed TECH cases required')
    rows = {r['episode_id']: r for r in roles['episodes']}
    for c in cases:
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}', c['case_id']) or c['case_id'] in ('.', '..'):
            raise ValueError('Unsafe frozen case ID')
        row = rows[c['episode_id']]
        if c['role'] != 'TECH' or row['role'] != 'TECH' or c['task'] != 'pusht': raise PermissionError('Only TECH reset cases')
        for key in ('source_asset_sha256', 'source_episode_idx', 'episode_sha256', 'length'):
            if c[key] != row[key]: raise ValueError('Case/source identity differs')
        if c.get('source_seed') is not None or c.get('reset_seed') is not None or row.get('source_seed') is not None:
            raise ValueError('Fallback diagnostic only applies to explicitly missing source seeds')
        if c['start_raw_index'] not in row['planning_starts'] or c['goal_raw_index'] != c['start_raw_index']+25:
            raise ValueError('Case start/goal differs from frozen metadata')
    return copy.deepcopy(cases)


def flatten(value, prefix=''):
    out = {}
    for k, v in value.items():
        name = prefix+'.'+k if prefix else k
        if isinstance(v, dict): out.update(flatten(v, name))
        else: out[name] = plain(v)
    return out


def physical_snapshot(env):
    """Include hidden velocities that source state7 does not completely set."""
    dynamic = []
    configuration = {'dt': env.dt, 'control_hz': env.control_hz, 'k_p': env.k_p, 'k_v': env.k_v,
                     'relative': env.relative, 'action_scale': env.action_scale,
                     'space_damping': env.space.damping, 'space_gravity': list(env.space.gravity),
                     'goal_pose': np.asarray(env.goal_pose).tolist()}
    for label in ('agent', 'block'):
        b = getattr(env, label)
        dynamic.extend([*b.position, b.angle, *b.velocity, b.angular_velocity, *b.force, b.torque])
        configuration[label] = {'body_type': b.body_type, 'mass': b.mass, 'moment': b.moment,
                                'center_of_gravity': list(b.center_of_gravity)}
    variations = flatten(env.variation_space.value)
    return {'arrays': {'state7': np.asarray(env._get_obs(), dtype=np.float64).copy(),
                       'agent_block_dynamic': np.asarray(dynamic, dtype=np.float64),
                       'goal_state': np.asarray(env.goal_state, dtype=np.float64).copy(),
                       'render_rgb': np.asarray(env.render()).copy()},
            'configuration': plain(configuration), 'variations_all': variations,
            'variations_not_overridden': {k: v for k, v in variations.items() if k not in OVERRIDDEN_DEFAULT_VARIATIONS}}


class FixedCaseDataset:
    """Source chunk plus explicitly synthetic seed; never claims seed provenance."""
    column_names = ['pixels', 'state', 'seed']
    def __init__(self, case, pixels, states, seed):
        self.case, self.pixels, self.states, self.seed = case, pixels, states, seed
        self.reads = 0
    def load_chunk(self, episodes, starts, ends):
        import torch
        c = self.case
        if list(episodes) != [c['source_episode_idx']] or list(starts) != [c['start_raw_index']] or list(ends) != [c['goal_raw_index']+1]:
            raise PermissionError('Reset read escaped the fixed TECH case')
        self.reads += 1
        if self.reads != 1: raise RuntimeError('Unexpected repeated source-window access')
        pixels = torch.from_numpy(self.pixels.copy())
        if pixels.shape[-1] == 3: pixels = pixels.permute(0, 3, 1, 2)
        return [{'pixels': pixels, 'state': self.states.copy(), 'seed': np.full(26, self.seed, dtype=np.int64)}]


class ZeroActionProbe:
    def __init__(self): self.calls = 0; self.inputs = None
    def set_env(self, env): self.env = env
    def get_action(self, info):
        self.calls += 1
        if self.calls != 1: raise RuntimeError('Only one diagnostic raw step is allowed')
        self.inputs = {k: np.asarray(info[k]).copy() for k in ('pixels', 'goal')}
        return np.zeros((1, 2), dtype=np.float32)


def one_trial(factory, helpers, case, pixels, states, seed, repeat, path):
    """Actual World reference or explicit equivalent reset, each in fresh env."""
    if seed not in SEEDS or repeat not in REPEATS or path not in PATHS: raise ValueError('Outside fixed diagnostic matrix')
    world = None; started = time.perf_counter(); arrays = {}; snapshots = {}; probe = ZeroActionProbe()
    row = {'case_id': case['case_id'], 'seed': seed, 'repeat': repeat, 'path': path,
           'status': 'STARTED', 'reset_attempts': 0, 'resets': 0, 'step_attempts': 0, 'raw_steps': 0,
           'CEM_calls': 0, 'complete_formal_or_technical_CEM_trajectories': 0, 'close_called': False}
    try:
        world = factory(); environment = world.envs.envs[0].unwrapped
        reset, step = world.reset, world.envs.step
        def observed_reset(*args, **kwargs):
            row['reset_attempts'] += 1
            if row['reset_attempts'] != 1: raise RuntimeError('Unexpected additional reset')
            if args and 'seed' in kwargs: raise TypeError('Reset seed supplied twice')
            raw_seed = args[0] if args else kwargs.get('seed')
            normalized_seed = normalize_reset_seed(raw_seed)
            row['reset_seed_type_compatibility'] = {'identity': SEED_COMPATIBILITY,
                'input_type': type(raw_seed).__name__, 'input_value': plain(raw_seed),
                'normalized_type': type(normalized_seed).__name__, 'normalized_value': plain(normalized_seed),
                'values_unchanged': plain(raw_seed) == plain(normalized_seed)}
            if args: args = (normalized_seed, *args[1:])
            else: kwargs['seed'] = normalized_seed
            result = reset(*args, **kwargs); row['resets'] += 1; return result
        def observed_step(action, *args, **kwargs):
            row['step_attempts'] += 1
            if row['step_attempts'] != 1 or not np.array_equal(action, np.zeros((1, 2), dtype=np.float32)):
                raise RuntimeError('Unexpected diagnostic action or additional step')
            snapshots['before_zero_action'] = physical_snapshot(environment)
            result = step(action, *args, **kwargs); row['raw_steps'] += 1
            snapshots['after_zero_action'] = physical_snapshot(environment)
            _, reward, terminated, truncated, info = result
            arrays['step_reward'] = np.asarray(reward).copy()
            arrays['step_terminated'] = np.asarray(terminated).copy(); arrays['step_truncated'] = np.asarray(truncated).copy()
            arrays['step_pixels'] = np.asarray(info['pixels']).copy(); arrays['step_state'] = np.asarray(info['state']).copy()
            return result
        world.reset = observed_reset; world.envs.step = observed_step; world.set_policy(probe)
        dataset = FixedCaseDataset(case, pixels, states, seed)
        args = {'dataset': dataset, 'episodes_idx': [case['source_episode_idx']],
                'start_steps': [case['start_raw_index']], 'goal_offset': 25}
        if path == PATHS[0]:
            world.evaluate(**args, eval_budget=1, callables=helpers['callables'], video=None)
        else:
            init, goal, _ = helpers['extract'](dataset, args['episodes_idx'], args['start_steps'], 25)
            world.reset(seed=init.get('seed'))
            helpers['apply'](environment, helpers['callables'], {k: v[0] for k, v in {**init, **goal}.items()})
            # Exact source World._evaluate_from_dataset observation override.
            prefix = world.infos['pixels'].shape[:2]
            for source in (init, goal):
                for key, value in source.items():
                    if key in world.infos or key in goal:
                        world.infos[key] = np.broadcast_to(value[:, None, ...], prefix+value.shape[1:]).copy()
            goal_snapshot = {k: world.infos[k].copy() for k in goal}
            world.envs.step(probe.get_action(world.infos)); world.infos.update(goal_snapshot)
        if row['resets'] != 1 or row['raw_steps'] != 1 or probe.calls != 1 or dataset.reads != 1:
            raise RuntimeError('Fixed diagnostic counts differ')
        for label, snapshot in snapshots.items():
            for key, value in snapshot.pop('arrays').items(): arrays[label+'__'+key] = value
        arrays.update({'source_injected_pixels': probe.inputs['pixels'], 'source_injected_goal': probe.inputs['goal'],
                       'source_start_state': states[0], 'source_goal_state': states[-1],
                       'source_state_minus_reset_state': states[0].astype(np.float64)-arrays['before_zero_action__state7']})
        raw_start = pixels[0]
        if raw_start.shape[0] == 3 and raw_start.shape[-1] != 3: raw_start = np.moveaxis(raw_start, 0, -1)
        render = arrays['before_zero_action__render_rgb']
        row['source_pixel_bias'] = {'source_shape': list(raw_start.shape), 'render_shape': list(render.shape),
                                   'equal_shape': raw_start.shape == render.shape, 'exact_equal': bool(np.array_equal(raw_start, render))}
        if raw_start.shape == render.shape:
            arrays['source_pixel_minus_reset_render'] = raw_start.astype(np.int16)-render.astype(np.int16)
            row['source_pixel_bias'].update(max_abs_difference=int(np.abs(arrays['source_pixel_minus_reset_render']).max()),
                                          rms_difference=float(np.sqrt(np.square(arrays['source_pixel_minus_reset_render'].astype(np.float64)).mean())))
        row['source_state_bias'] = {'max_abs_difference': float(np.abs(arrays['source_state_minus_reset_state']).max()),
                                    'coordinate_differences': arrays['source_state_minus_reset_state'].tolist(),
                                    'diagnostic_only_no_zero_bias_requirement': True}
        row['snapshots'] = snapshots
        row['finite_arrays'] = {key: bool(np.isfinite(value).all()) for key, value in arrays.items()}
        row['status'] = 'PASS' if all(row['finite_arrays'].values()) else 'BLOCKED_NONFINITE'
    except Exception as error:
        row.update(status='BLOCKED_EXCEPTION', error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        if world is not None:
            try: world.close(); row['close_called'] = True
            except Exception as error: row.update(status='BLOCKED_CLOSE', close_error=str(error))
        row['wall_seconds'] = time.perf_counter()-started
    return row, arrays


def compare_trials(a, b, arrays_a, arrays_b, label):
    checked = [key for key in arrays_a if not key.startswith('source_state_minus_') and not key.startswith('source_pixel_minus_')]
    differences = {}; exact = True
    if set(checked) != {k for k in arrays_b if not k.startswith('source_state_minus_') and not k.startswith('source_pixel_minus_')}:
        return {'label': label, 'status': 'BLOCKED', 'reason': 'Output field sets differ'}
    for key in checked:
        x, y = arrays_a[key], arrays_b[key]
        equal = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y) and np.isfinite(x).all() and np.isfinite(y).all()
        differences[key] = {'exact': bool(equal), 'shape_a': list(x.shape), 'shape_b': list(y.shape),
                            'max_abs_difference': float(np.abs(x.astype(np.float64)-y.astype(np.float64)).max()) if x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all() else None}
        exact = exact and bool(equal)
    metadata_equal = all(a['snapshots'][phase][key] == b['snapshots'][phase][key]
                         for phase in ('before_zero_action', 'after_zero_action')
                         for key in ('configuration', 'variations_not_overridden'))
    exact = exact and metadata_equal
    return {'label': label, 'status': 'PASS' if exact else 'BLOCKED', 'exact_numeric_tolerance': 0,
            'array_differences': differences, 'non_overridden_variations_and_physics_exact': metadata_equal,
            'random_initial_variations_are_diagnostic_not_asserted_seed_independent': list(OVERRIDDEN_DEFAULT_VARIATIONS)}


def evaluate_matrix(cases, run_one):
    rows, arrays, comparisons = [], {}, []
    for case in cases:
        by = {}
        for seed in SEEDS:
            for repeat in REPEATS:
                for path in PATHS:
                    key = f'{len(rows):02d}'; row, values = run_one(case, seed, repeat, path)
                    row['trial_id'] = key; rows.append(row); arrays[key] = values; by[seed, repeat, path] = row
        if any(row['status'] != 'PASS' or not row['close_called'] for row in by.values()): continue
        # Every trial is compared to the same fixed reference. This covers each
        # seed repeat and same-seed original/wrapper equality without selection.
        reference = by[0, 0, PATHS[0]]
        for key, row in by.items():
            if row is reference: continue
            comparisons.append(compare_trials(reference, row, arrays[reference['trial_id']], arrays[row['trial_id']],
                                                {'case_id': case['case_id'], 'reference_trial': reference['trial_id'], 'other_trial': row['trial_id'],
                                                 'other_seed_repeat_path': list(key)}))
    counts = {k: sum(r[k] for r in rows) for k in ('reset_attempts', 'resets', 'step_attempts', 'raw_steps')}
    good = len(cases) == 2 and len(rows) == 16 and all(r['status'] == 'PASS' and r['close_called'] for r in rows)
    good = good and counts['resets'] == 16 and counts['raw_steps'] == 16 and len(comparisons) == 14 and all(r['status'] == 'PASS' for r in comparisons)
    return {'task': 'pusht', 'status': 'PASS_TECH_RESET_FALLBACK' if good else 'BLOCKED_RESET_FALLBACK',
            'routing_eligible': bool(good), 'candidate_reset_seed': 0 if good else None,
            'candidate_identity': contract()['candidate_identity'], 'contract': contract(), 'trials': rows,
            'comparisons': comparisons, 'counts': {**counts, 'optimizer_updates': 0, 'CEM_calls': 0,
                                                 'complete_formal_or_technical_CEM_trajectories': 0}}, arrays


def _save_npz(path, arrays):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f: np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_receipt(path, *, roles_sha256, source_asset_sha256):
    """No env execution; for the later separate routing-manifest builder."""
    p = Path(path); d = common.read_json(p)
    if d.get('task') != 'pusht' or d.get('status') != 'PASS_TECH_RESET_FALLBACK' or d.get('routing_eligible') is not True or d.get('candidate_reset_seed') != 0:
        raise RuntimeError('A blocked/incomplete reset validation cannot authorize a fallback')
    if d['roles_sha256'] != roles_sha256 or source_asset_sha256 not in d['source_files'] or d['contract'] != contract():
        raise RuntimeError('Reset fallback source/role/diagnostic contract identity differs')
    trial_ids = [f'{i:02d}' for i in range(16)]
    if len(d.get('cases', [])) != 2 or len({c['case_id'] for c in d['cases']}) != 2:
        raise RuntimeError('Reset validation must bind exactly two fixed TECH cases')
    if [t['trial_id'] for t in d['trials']] != trial_ids or any(t['status'] != 'PASS' or not t['close_called'] for t in d['trials']):
        raise RuntimeError('Reset trial identities or successful closure are incomplete')
    required = {f'trial_{i}.{extension}' for i in trial_ids for extension in ('npz', 'json')}
    required.update(c['case_id']+'_source.npz' for c in d['cases'])
    required.add('PROGRESS.json')
    if not required.issubset(d['files']): raise RuntimeError('Reset receipt omits raw diagnostic evidence')
    for name, expected in d['files'].items():
        file = p.parent/name
        if Path(name).name != name or not file.is_file() or file.stat().st_size != expected['bytes'] or common.sha256(file) != expected['sha256']:
            raise RuntimeError('Reset fallback evidence changed')
    if len(d['comparisons']) != 14 or not all(x['status'] == 'PASS' for x in d['comparisons']) or d['counts']['resets'] != 16 or d['counts']['raw_steps'] != 16:
        raise RuntimeError('Finite reset validation matrix is incomplete')
    return d


def run(output_dir):
    from r3 import planning, data
    common.require_authorization({}, technical=True); common.ensure_space()
    out = data.checked_local(output_dir)
    if out.exists(): raise RuntimeError('Preserve previous reset evidence; use a new attempt folder')
    out.mkdir(parents=True); started = time.perf_counter()
    report = {'version': VERSION, 'task': 'pusht', 'status': 'BLOCKED_RESET_FALLBACK', 'routing_eligible': False, 'started_at': common.now(),
              'contract': contract(), 'optimizer_updates': 0, 'CEM_calls': 0, 'EVAL_source_windows_read': 0}
    try:
        sources = planning.verify_official_sources()
        source_manifest = common.read_json('state/swm_compat_source_manifest.json')
        paths = ['manifests/pusht_data_roles.json', 'manifests/pusht_source_map.json', 'state/ENVIRONMENT_READY.json',
                 'state/swm_compat_source_manifest.json', 'r3/planning.py', 'r3/data.py', 'r3/env_compat.py', 'scripts/validate_reset_fallback.py']
        for rel in SOURCE_MEMBERS:
            path = ROOT/'source/swm_compat'/rel; expected = source_manifest['files'][rel]
            if path.stat().st_size != expected['bytes'] or common.sha256(path) != expected['sha256']: raise RuntimeError('Official reset source changed: '+rel)
            paths.append('source/swm_compat/'+rel)
        report['input_hashes'] = {p: common.sha256(data.checked_local(p)) for p in paths}
        roles = common.read_json(paths[0]); cases = selected_cases(roles); report['cases'] = cases
        report['roles_sha256'] = report['input_hashes'][paths[0]]; report['source_files'] = {}
        source_map = common.read_json(paths[1]); case_arrays = {}; signatures = {}
        for case in cases:
            asset = case['source_asset_sha256']; record = source_map['assets'][asset]
            if record['sha256'] != asset: raise RuntimeError('Actual H5 source identity differs')
            path = data.checked_local(record['path']) if asset in report['source_files'] else data.verified_file(record)
            if str(path) not in signatures:
                signatures[str(path)] = (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
            report['source_files'][asset] = record
            with data.RawH5(path, keys=['pixels', 'state']) as reader:
                if 'seed' in reader.file: raise RuntimeError('Source has a seed column; missing-seed fallback is inapplicable')
                if int(reader.lengths[case['source_episode_idx']]) != case['length']: raise RuntimeError('Source episode length differs')
                pixels = reader.array(case['source_episode_idx'], 'pixels', case['start_raw_index'], case['goal_raw_index']+1)
                states = reader.array(case['source_episode_idx'], 'state', case['start_raw_index'], case['goal_raw_index']+1)
            if len(pixels) != 26 or states.shape != (26, 7) or pixels.dtype != np.uint8 or not np.isfinite(states[[0, -1]]).all():
                raise RuntimeError('Actual TECH source window does not match PushT state7/pixels contract')
            metadata = case['reset_metadata']
            for where, index in (('start', 0), ('goal', -1)):
                if not np.array_equal(states[index], np.asarray(metadata[where]['state'])):
                    raise RuntimeError('Source reset values differ from frozen TECH case')
                expected = metadata['source_'+where+'_pixel_sha256']
                if hashlib.sha256(pixels[index].tobytes()).hexdigest() != expected: raise RuntimeError('Source case image differs')
            case_arrays[case['case_id']] = (pixels, states)
            _save_npz(out/(case['case_id']+'_source.npz'), {'pixels': pixels, 'state': states})
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        swm, _, _, _ = planning.load_official_api()
        module = importlib.import_module('stable_worldmodel.world.world')
        env_module = importlib.import_module('stable_worldmodel.envs.pusht.env')
        if tuple(env_module.DEFAULT_VARIATIONS) != OVERRIDDEN_DEFAULT_VARIATIONS: raise RuntimeError('Default variations changed')
        report['official_source'] = sources
        report['environment_versions'] = {name: importlib.metadata.version(name) for name in ('numpy', 'pygame', 'pymunk', 'gymnasium', 'torch')}
        helpers = {'extract': module._extract_init_goal, 'apply': module._apply_callables, 'callables': planning.CALLABLES['pusht']}
        def factory(): return swm.World(env_name='swm/PushT-v1', num_envs=1, max_episode_steps=100, image_shape=(224, 224))
        completed = []
        def actual(case, seed, repeat, path):
            trial_id = f'{len(completed):02d}'
            progress = {'status': 'TRIAL_IN_PROGRESS', 'trial_id': trial_id, 'case_id': case['case_id'],
                        'seed': seed, 'repeat': repeat, 'path': path, 'completed_trials': len(completed),
                        'completed_counts': {key: sum(r[key] for r in completed) for key in ('reset_attempts', 'resets', 'step_attempts', 'raw_steps')},
                        'interrupted_trial_requires_explicit_audit_not_automatic_repetition': True}
            common.atomic_json(out/'PROGRESS.json', progress)
            row, values = one_trial(factory, helpers, case, *case_arrays[case['case_id']], seed, repeat, path)
            row['trial_id'] = trial_id
            _save_npz(out/('trial_'+trial_id+'.npz'), values)
            common.atomic_json(out/('trial_'+trial_id+'.json'), plain(row)); completed.append(row)
            common.atomic_json(out/'PROGRESS.json', {**progress, 'status': 'TRIAL_SAVED', 'completed_trials': len(completed),
                               'completed_counts': {key: sum(r[key] for r in completed) for key in ('reset_attempts', 'resets', 'step_attempts', 'raw_steps')}})
            return row, values
        result, arrays = evaluate_matrix(cases, actual); report.update(result)
        for name, expected in report['input_hashes'].items():
            if common.sha256(data.checked_local(name)) != expected: raise RuntimeError('Reset validation input changed: '+name)
        for name, expected in signatures.items():
            s = Path(name).stat()
            if (s.st_size, s.st_mtime_ns, s.st_ino) != expected: raise RuntimeError('Verified source H5 changed')
    except Exception as error:
        report.update(status='BLOCKED_RESET_FALLBACK', routing_eligible=False,
                      error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    report['finished_at'] = common.now(); report['wall_seconds'] = time.perf_counter()-started
    report['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': common.sha256(p)} for p in sorted(out.iterdir()) if p.is_file()}
    common.atomic_json(out/'RESET_VALIDATION_RECEIPT.json', plain(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', help='Unique R3-root-relative attempt directory')
    args = parser.parse_args()
    if args.run:
        if not args.output_dir: parser.error('--run requires --output-dir')
        result = run(args.output_dir); print(json.dumps({'status': result['status'], 'counts': result.get('counts'), 'output': args.output_dir}))
        raise SystemExit(0 if result['status'] == 'PASS_TECH_RESET_FALLBACK' else 1)
    print(json.dumps(contract(), indent=2))
