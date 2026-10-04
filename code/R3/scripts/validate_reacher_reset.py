"""Finite Reacher qpos_match missing-seed reset diagnostic; importing performs no work.

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
VERSION = 'R3_REACHER_QPOS_MATCH_MISSING_SEED_RESET_VALIDATION_V2_EFFECTIVE_CONTACT'
MJDSBL_CONTACT = 16  # Read from the actually installed MuJoCo3.11.0 enum; run() verifies it.
ORIGINAL_XML_SHA256 = '849e1e3dd8e79b8f5796bd6e32b0d7a51b916355584402d8f803d74ad35ad193'
SEEDS = (0, 1)
REPEATS = (0, 1)
PATHS = ('ORIGINAL_WORLD_EVALUATE', 'EXPLICIT_EQUIVALENT_RESET_WRAPPER')
# Cross-seed exclusions apply ONLY to these explicitly diagnostic prefixes.
# Same-seed/path comparisons retain every one of them as a hard exact check.
DIAGNOSTIC_PREFIX = 'point_task_diagnostic__'
SOURCE_BIAS_PREFIX = 'source_bias__'
SOURCE_MEMBERS = ('stable_worldmodel/world/world.py', 'stable_worldmodel/world/env_pool.py',
                  'stable_worldmodel/envs/dmcontrol/reacher.py',
                  'stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py',
                  'stable_worldmodel/envs/dmcontrol/dmcontrol.py',
                  'stable_worldmodel/spaces.py', 'stable_worldmodel/wrapper/default.py')
DATA_FIELDS = ('qpos', 'qvel', 'act', 'ctrl', 'qacc', 'qacc_warmstart',
               'qfrc_applied', 'xfrc_applied', 'qfrc_bias', 'qfrc_passive',
               'qfrc_actuator', 'mocap_pos', 'mocap_quat', 'userdata', 'time')
MODEL_FIELDS = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                'dof_damping', 'dof_armature', 'dof_frictionloss', 'jnt_stiffness',
                'jnt_range', 'jnt_limited', 'actuator_ctrlrange', 'actuator_gear',
                'actuator_dynprm', 'actuator_gainprm', 'actuator_biasprm',
                'actuator_dyntype', 'actuator_gaintype', 'actuator_biastype',
                'geom_contype', 'geom_conaffinity', 'geom_size', 'geom_friction')


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
    return {'version': VERSION, 'task': 'reacher', 'environment_task': 'qpos_match',
            'case_selection': 'Frozen roles.cases.TECH[:2], no replacement',
            'seeds': list(SEEDS), 'repeats_per_seed_and_path': 2, 'paths': list(PATHS),
            'action': [0.0, 0.0], 'raw_steps_per_reset': 1, 'expected_resets': 16, 'expected_raw_steps': 16,
            'expected_comparisons': {'SAME_SEED_ALL_FIELDS_EXACT': 12, 'CROSS_SEED_TASK_FIELDS_EXACT': 2},
            'comparison_graph': 'Within each case: seed0 reference to its other3; seed1 reference to its other3; seed0 reference to seed1 reference.',
            'hard_comparisons': 'All seeds: exact finite joint state, target_qpos, hidden dynamic state, physical parameters, renderer, source-injected planner pixels/goal, zero-action outcome and qpos_match termination. Same seed: point-target location, native observations, reward and score must also be exact.',
            'cross_seed_diagnostic_only': ['point target geometry position', 'native point-task observation', 'point reward', 'cumulative point score'],
            'diagnostic_basis': 'Official qpos_match termination uses only all abs(qpos-target_qpos)<0.05; official planner consumes pixels/goal, not point reward or native observations. Hidden non-colliding transparent point target is checked, not assumed.',
            'required_target_configuration': {'render_target': 0, 'material_alpha': 0.0, 'contact_globally_disabled': True},
            'effective_contact_gate': {'installed_mjDSBL_CONTACT': MJDSBL_CONTACT, 'formula': 'bool(model.opt.disableflags & installed_mjDSBL_CONTACT)',
                'original_xml_sha256': ORIGINAL_XML_SHA256,
                'correction_identity': 'LOCAL_GEOM_BITS_ARE_NOT_EFFECTIVE_CONTACT_STATE; preserve attemptedV1 evidence; no environment/tolerance/case changes'},
            'hidden_dynamic_fields': list(DATA_FIELDS), 'physical_model_fields': list(MODEL_FIELDS),
            'source_state_or_pixel_reconstruction_bias': 'All original differences retained as diagnostics, not a zero-bias gate.',
            'candidate_identity': 'SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED', 'candidate_reset_seed': 0,
            'reset_seed_type_compatibility': SEED_COMPATIBILITY,
            'scope_limit': 'Two fixed TECH cases and two tested seeds in the exact source/environment only; not original-seed recovery, arbitrary-seed independence, or proof of all-EVAL equivalence.',
            'optimizer_updates': 0, 'CEM_calls': 0, 'complete_formal_or_technical_CEM_trajectories': 0,
            'diagnostic_one_step_segments': 16, 'changes_existing_roles': False}


def selected_cases(roles):
    if roles.get('task') != 'reacher' or roles.get('status') != 'METADATA_ROLES_FROZEN':
        raise ValueError('Frozen actual Reacher metadata roles required')
    cases = roles['cases']['TECH'][:2]
    if len(cases) != 2 or len({c['case_id'] for c in cases}) != 2 or len({c['episode_id'] for c in cases}) != 2:
        raise ValueError('Two distinct fixed TECH cases required')
    rows = {r['episode_id']: r for r in roles['episodes']}
    for c in cases:
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}', c['case_id']) or c['case_id'] in ('.', '..'):
            raise ValueError('Unsafe frozen case ID')
        row = rows[c['episode_id']]
        if c['role'] != 'TECH' or row['role'] != 'TECH' or c['task'] != 'reacher': raise PermissionError('Only TECH reset cases')
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
    """Separate only source-justified point-task diagnostics from hard fields."""
    physics = env.env.physics; model = physics.model; task = env.env.task
    if env._task_name != 'qpos_match' or float(task.qpos_threshold) != .05:
        raise RuntimeError('Actual qpos_match task/threshold differs')
    variations = flatten(env.variation_space.value)
    target_id = int(model.name2id('target', 'geom'))
    target_material = int(model.geom_matid[target_id])
    flags = {'render_target': int(variations['rendering.render_target']),
             'material_alpha': float(model.mat_rgba[target_material, 3]) if target_material >= 0 else None,
             'geom_contype': int(model.geom_contype[target_id]), 'geom_conaffinity': int(model.geom_conaffinity[target_id]),
             'global_disableflags': int(model.opt.disableflags), 'mjDSBL_CONTACT': MJDSBL_CONTACT,
             'contact_globally_disabled': bool(int(model.opt.disableflags) & MJDSBL_CONTACT)}
    if {k: flags[k] for k in contract()['required_target_configuration']} != contract()['required_target_configuration']:
        raise RuntimeError('Point target is not proven invisible/non-colliding: '+str(flags))
    arrays = {'physics_state': np.asarray(physics.get_state()).copy(),
              'target_qpos': np.asarray(task.target_qpos).copy(), 'render_rgb': np.asarray(env.render()).copy()}
    for key in DATA_FIELDS: arrays['data__'+key] = np.asarray(getattr(physics.data, key)).copy()
    for key in MODEL_FIELDS: arrays['model__'+key] = np.asarray(getattr(model, key)).copy()
    keep = np.arange(model.ngeom) != target_id
    arrays['model__non_target_geom_pos'] = np.asarray(model.geom_pos)[keep].copy()
    arrays['data__non_target_geom_xpos'] = np.asarray(physics.data.geom_xpos)[keep].copy()
    arrays['point_task_diagnostic__target_model_position'] = np.asarray(model.geom_pos[target_id]).copy()
    arrays['point_task_diagnostic__target_world_position'] = np.asarray(physics.data.geom_xpos[target_id]).copy()
    arrays['point_task_diagnostic__cumulative_score'] = np.asarray(env._cumulative_reward, dtype=np.float64)
    observations = task.get_observation(physics)
    for key, value in observations.items(): arrays['point_task_diagnostic__native_observation__'+key] = np.asarray(value).copy()
    target = arrays['target_qpos']; position = arrays['data__qpos']
    if target.shape != position.shape or position.shape != (2,): raise RuntimeError('Actual Reacher joint/target shape differs')
    success = bool(np.all(np.abs(position-target) < .05))
    if (task.get_termination(physics) is not None) != success: raise RuntimeError('Actual source termination differs from qpos_match')
    arrays['qpos_match_success'] = np.asarray(success)
    arrays['qpos_goal_error'] = np.asarray(np.linalg.norm(position-target))
    options = {k: plain(np.asarray(getattr(model.opt, k))) for k in ('timestep', 'gravity', 'integrator', 'solver', 'iterations', 'tolerance')}
    return {'arrays': arrays, 'configuration': {'task': env._task_name, 'qpos_threshold': float(task.qpos_threshold),
            'action_repeat': int(env.action_repeat), 'camera_id': int(env.camera_id),
            'target_configuration': flags, 'physics_options': options,
            'action_minimum': plain(env.action_space.low), 'action_maximum': plain(env.action_space.high)},
            'variations_all': variations}


class FixedCaseDataset:
    """Only source26 TECH frames/qpos/qvel and an explicit synthetic seed."""
    column_names = ['pixels', 'qpos', 'qvel', 'seed']
    def __init__(self, case, pixels, qpos, qvel, seed):
        self.case, self.pixels, self.qpos, self.qvel, self.seed = case, pixels, qpos, qvel, seed
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
        return [{'pixels': pixels, 'qpos': self.qpos.copy(), 'qvel': self.qvel.copy(), 'seed': np.full(26, self.seed, dtype=np.int64)}]


class ZeroActionProbe:
    def __init__(self): self.calls = 0; self.inputs = None
    def set_env(self, env): self.env = env
    def get_action(self, info):
        self.calls += 1
        if self.calls != 1: raise RuntimeError('Only one diagnostic raw step is allowed')
        self.inputs = {k: np.asarray(info[k]).copy() for k in ('pixels', 'goal')}
        return np.zeros((1, 2), dtype=np.float32)


def one_trial(factory, helpers, case, pixels, qpos, qvel, seed, repeat, path):
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
            arrays['point_task_diagnostic__step_reward'] = np.asarray(reward).copy()
            arrays['step_terminated'] = np.asarray(terminated).copy(); arrays['step_truncated'] = np.asarray(truncated).copy()
            arrays['step_pixels'] = np.asarray(info['pixels']).copy()
            arrays['step_qpos'] = np.asarray(info['qpos']).copy(); arrays['step_qvel'] = np.asarray(info['qvel']).copy()
            return result
        world.reset = observed_reset; world.envs.step = observed_step; world.set_policy(probe)
        dataset = FixedCaseDataset(case, pixels, qpos, qvel, seed)
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
            for key, value in snapshot.pop('arrays').items(): arrays[(DIAGNOSTIC_PREFIX+label+'__'+key[len(DIAGNOSTIC_PREFIX):]) if key.startswith(DIAGNOSTIC_PREFIX) else label+'__'+key] = value
        arrays.update({'source_injected_pixels': probe.inputs['pixels'], 'source_injected_goal': probe.inputs['goal'],
                       'source_start_qpos': qpos[0], 'source_start_qvel': qvel[0], 'source_goal_qpos': qpos[-1],
                       'source_goal_qvel': qvel[-1],
                       'source_bias__qpos': qpos[0].astype(np.float64)-arrays['before_zero_action__data__qpos'],
                       'source_bias__qvel': qvel[0].astype(np.float64)-arrays['before_zero_action__data__qvel']})
        raw_start = pixels[0]
        if raw_start.shape[0] == 3 and raw_start.shape[-1] != 3: raw_start = np.moveaxis(raw_start, 0, -1)
        render = arrays['before_zero_action__render_rgb']
        row['source_pixel_bias'] = {'source_shape': list(raw_start.shape), 'render_shape': list(render.shape),
                                   'equal_shape': raw_start.shape == render.shape, 'exact_equal': bool(np.array_equal(raw_start, render))}
        if raw_start.shape == render.shape:
            arrays['source_bias__pixels'] = raw_start.astype(np.int16)-render.astype(np.int16)
            row['source_pixel_bias'].update(max_abs_difference=int(np.abs(arrays['source_bias__pixels']).max()),
                                          rms_difference=float(np.sqrt(np.square(arrays['source_bias__pixels'].astype(np.float64)).mean())))
        row['source_state_bias'] = {key: arrays['source_bias__'+key].tolist() for key in ('qpos', 'qvel')}
        row['source_bias_diagnostic_only_no_zero_requirement'] = True
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
    cross_seed = a['seed'] != b['seed']
    checked = {k for k in arrays_a if not k.startswith(SOURCE_BIAS_PREFIX)}
    other = {k for k in arrays_b if not k.startswith(SOURCE_BIAS_PREFIX)}
    if checked != other:
        return {'label': label, 'status': 'BLOCKED', 'reason': 'Output field sets differ'}
    differences = {}; exact = True
    for key in sorted(checked):
        x, y = arrays_a[key], arrays_b[key]
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        equal = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y) and finite
        diagnostic = cross_seed and key.startswith(DIAGNOSTIC_PREFIX)
        differences[key] = {'exact': bool(equal), 'hard_gate': not diagnostic, 'finite': finite,
                            'shape_a': list(x.shape), 'shape_b': list(y.shape),
                            'max_abs_difference': float(np.abs(x.astype(np.float64)-y.astype(np.float64)).max(initial=0)) if x.shape == y.shape and finite else None}
        exact = exact and finite and (bool(equal) or diagnostic)
    metadata_equal = all(a['snapshots'][phase][key] == b['snapshots'][phase][key]
                         for phase in ('before_zero_action', 'after_zero_action')
                         for key in ('configuration', 'variations_all'))
    exact = exact and metadata_equal
    return {'label': label, 'status': 'PASS' if exact else 'BLOCKED', 'exact_numeric_tolerance': 0,
            'comparison_class': 'CROSS_SEED_TASK_FIELDS_EXACT' if cross_seed else 'SAME_SEED_ALL_FIELDS_EXACT',
            'array_differences': differences, 'configuration_and_variations_exact': metadata_equal,
            'point_task_fields_are_diagnostic_only_across_seed': cross_seed}


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
        edges = []
        for seed in SEEDS:
            reference = by[seed, 0, PATHS[0]]
            edges += [(reference, row) for (s, _, _), row in by.items() if s == seed and row is not reference]
        edges.append((by[0, 0, PATHS[0]], by[1, 0, PATHS[0]]))
        for reference, row in edges:
            comparisons.append(compare_trials(reference, row, arrays[reference['trial_id']], arrays[row['trial_id']],
                {'case_id': case['case_id'], 'reference_trial': reference['trial_id'], 'other_trial': row['trial_id'],
                 'other_seed_repeat_path': [row['seed'], row['repeat'], row['path']]}))
    counts = {k: sum(r[k] for r in rows) for k in ('reset_attempts', 'resets', 'step_attempts', 'raw_steps')}
    classes = {name: sum(r.get('comparison_class') == name for r in comparisons) for name in contract()['expected_comparisons']}
    good = len(cases) == 2 and len(rows) == 16 and all(r['status'] == 'PASS' and r['close_called'] for r in rows)
    good = good and counts['resets'] == 16 and counts['raw_steps'] == 16 and classes == contract()['expected_comparisons'] and all(r['status'] == 'PASS' for r in comparisons)
    return {'task': 'reacher', 'status': 'PASS_TECH_RESET_FALLBACK' if good else 'BLOCKED_RESET_FALLBACK',
            'routing_eligible': bool(good), 'candidate_reset_seed': 0 if good else None,
            'candidate_identity': contract()['candidate_identity'], 'contract': contract(), 'trials': rows,
            'comparisons': comparisons, 'comparison_class_counts': classes,
            'counts': {**counts, 'optimizer_updates': 0, 'CEM_calls': 0, 'complete_formal_or_technical_CEM_trajectories': 0}}, arrays


def _save_npz(path, arrays):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f: np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_receipt(path, *, roles_sha256, source_asset_sha256):
    """No env execution; for the later separate routing-manifest builder."""
    p = Path(path); d = common.read_json(p)
    if d.get('task') != 'reacher' or d.get('status') != 'PASS_TECH_RESET_FALLBACK' or d.get('routing_eligible') is not True or d.get('candidate_reset_seed') != 0:
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
    if d.get('comparison_class_counts') != contract()['expected_comparisons'] or {k: sum(c.get('comparison_class') == k for c in d['comparisons']) for k in contract()['expected_comparisons']} != contract()['expected_comparisons']:
        raise RuntimeError('Reset validation comparison classes differ from the registered14-edge graph')
    expected_trials = [(c['case_id'], s, r, p) for c in d['cases'] for s in SEEDS for r in REPEATS for p in PATHS]
    if [(r['case_id'], r['seed'], r['repeat'], r['path']) for r in d['trials']] != expected_trials:
        raise RuntimeError('Reset trial order differs from the fixed matrix')
    expected_edges = []
    for offset, c in enumerate(d['cases']):
        base = 8*offset
        for reference, other in [(base, base+1), (base, base+2), (base, base+3),
                                 (base+4, base+5), (base+4, base+6), (base+4, base+7), (base, base+4)]:
            r = d['trials'][other]
            expected_edges.append({'case_id': c['case_id'], 'reference_trial': f'{reference:02d}',
                                   'other_trial': f'{other:02d}', 'other_seed_repeat_path': [r['seed'], r['repeat'], r['path']]})
    if [c['label'] for c in d['comparisons']] != expected_edges:
        raise RuntimeError('Reset comparison graph differs from the registered14 edges')
    if any(d['counts'][k] != 0 for k in ('optimizer_updates', 'CEM_calls', 'complete_formal_or_technical_CEM_trajectories')):
        raise RuntimeError('Reset diagnostic contains an unexpected optimizer/CEM budget')
    return d


def run(output_dir):
    from r3 import planning, data
    common.require_authorization({}, technical=True); common.ensure_space()
    out = data.checked_local(output_dir)
    if out.exists(): raise RuntimeError('Preserve previous reset evidence; use a new attempt folder')
    out.mkdir(parents=True); started = time.perf_counter()
    report = {'version': VERSION, 'task': 'reacher', 'status': 'BLOCKED_RESET_FALLBACK', 'routing_eligible': False, 'started_at': common.now(),
              'contract': contract(), 'optimizer_updates': 0, 'CEM_calls': 0, 'EVAL_source_windows_read': 0}
    try:
        sources = planning.verify_official_sources()
        source_manifest = common.read_json('state/swm_compat_source_manifest.json')
        paths = ['manifests/reacher_data_roles.json', 'manifests/reacher_source_map.json', 'state/ENVIRONMENT_READY.json',
                 'state/swm_compat_source_manifest.json', 'r3/planning.py', 'r3/data.py', 'r3/env_compat.py', 'scripts/validate_reacher_reset.py',
                 'state/REACHER_RESET_IMPLEMENTATION_CORRECTION.json',
                 'state/technical_source_snapshots/reacher_reset_attempt1/validate_reacher_reset.py',
                 'state/technical_source_snapshots/reacher_reset_attempt1/test_reacher_reset.py']
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
            with data.RawH5(path, keys=['pixels', 'qpos', 'qvel']) as reader:
                if 'seed' in reader.file: raise RuntimeError('Source has a seed column; missing-seed fallback is inapplicable')
                if int(reader.lengths[case['source_episode_idx']]) != case['length']: raise RuntimeError('Source episode length differs')
                pixels = reader.array(case['source_episode_idx'], 'pixels', case['start_raw_index'], case['goal_raw_index']+1)
                qpos = reader.array(case['source_episode_idx'], 'qpos', case['start_raw_index'], case['goal_raw_index']+1)
                qvel = reader.array(case['source_episode_idx'], 'qvel', case['start_raw_index'], case['goal_raw_index']+1)
            if len(pixels) != 26 or qpos.shape != (26, 2) or qvel.shape != (26, 2) or pixels.dtype != np.uint8 or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise RuntimeError('Actual TECH source window does not match Reacher joint/pixels contract')
            metadata = case['reset_metadata']
            for where, index in (('start', 0), ('goal', -1)):
                if any(not np.array_equal(value[index], np.asarray(metadata[where][key])) for key, value in (('qpos', qpos), ('qvel', qvel))):
                    raise RuntimeError('Source reset values differ from frozen TECH case')
                expected = metadata['source_'+where+'_pixel_sha256']
                if hashlib.sha256(pixels[index].tobytes()).hexdigest() != expected: raise RuntimeError('Source case image differs')
            case_arrays[case['case_id']] = (pixels, qpos, qvel)
            _save_npz(out/(case['case_id']+'_source.npz'), {'pixels': pixels, 'qpos': qpos, 'qvel': qvel})
        if os.environ.get('MUJOCO_GL') not in (None, 'egl'): raise RuntimeError('Use a fresh process with official MUJOCO_GL=egl')
        os.environ['MUJOCO_GL'] = 'egl'
        swm, _, _, _ = planning.load_official_api()
        module = importlib.import_module('stable_worldmodel.world.world')
        import inspect
        report['installed_reset_sources'] = {}
        for module_name in ('dm_control.suite.reacher', 'dm_control.rl.control', 'dm_control.suite.wrappers.action_scale'):
            installed = importlib.import_module(module_name); path = Path(inspect.getsourcefile(installed))
            name = module_name.replace('.', '_')+'.py'; payload = path.read_bytes(); (out/name).write_bytes(payload)
            report['installed_reset_sources'][module_name] = {'installed_path': str(path), 'saved_source': name, 'sha256': hashlib.sha256(payload).hexdigest()}
        dm_reacher = importlib.import_module('dm_control.suite.reacher'); xml, assets = dm_reacher.get_model_and_assets()
        import mujoco
        if int(mujoco.mjtDisableBit.mjDSBL_CONTACT) != MJDSBL_CONTACT:
            raise RuntimeError('Actual installed MuJoCo contact-disable enum differs from the verified identity')
        if hashlib.sha256(xml).hexdigest() != ORIGINAL_XML_SHA256:
            raise RuntimeError('Actual original Reacher XML differs from the audited contact-disable source')
        report['effective_contact_source_evidence'] = {'original_xml_sha256': ORIGINAL_XML_SHA256,
            'installed_mjDSBL_CONTACT': int(mujoco.mjtDisableBit.mjDSBL_CONTACT),
            'mujoco_version': mujoco.__version__, 'actual_compiled_flags_checked_per_snapshot': True}
        (out/'dm_control_reacher.xml').write_bytes(xml)
        report['installed_mjcf_assets'] = {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()} for name, value in (assets or {}).items()}
        report['official_source'] = sources
        report['environment_versions'] = {name: importlib.metadata.version(name) for name in ('numpy', 'dm-control', 'dm-env', 'mujoco', 'gymnasium', 'torch')}
        helpers = {'extract': module._extract_init_goal, 'apply': module._apply_callables, 'callables': planning.CALLABLES['reacher']}
        def factory(): return swm.World(env_name='swm/ReacherDMControl-v0', num_envs=1, max_episode_steps=100, image_shape=(224, 224), task='qpos_match')
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
