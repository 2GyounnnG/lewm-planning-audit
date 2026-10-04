"""Instrument the fixed author's CEM/policy/World; never implement a new solver.

No environment, GPU, model, network or trajectory is created at import time.
The caller reserves the TECH/FORMAL trajectory budget before calling run_case.
The single-case interface intentionally makes CEM RNG independent of scheduling.
"""
from __future__ import annotations
import hashlib, importlib, json, os, sys, time
from pathlib import Path
from .common import ROOT, atomic_json, ensure_space, fp32_policy, now, sha256

SWM_REVISION = 'abdced49809d5eae38e24b27dc7b635c502c4812'
SOURCE_SHA256 = {
    'stable_worldmodel/solver/cem.py': 'a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b',
    'stable_worldmodel/solver/utils.py': 'd30c6225a9c4c83782132180a94c94188b07e0381d2c938c658fe5a49aed48e1',
    'stable_worldmodel/policy.py': '4967e7e3d5b20eb7a1d0b00e5d60fd701cce1c750ae7ec4a9b02529b9366db22',
    'stable_worldmodel/world/world.py': 'f0d2834a04c7ce8915522773f6f4f53396cdf79bf3a1c92c3560d8d2c378fe56',
    'stable_worldmodel/world/env_pool.py': 'b6e73253199e1a39f0fcadfad7dbea09270cbd559e366d1bc9bf09c9e7da0606',
    'stable_worldmodel/protocols.py': '35b3e08fa2df0239277d2004356375af58b207edbfa555d46d24baca646fc401',
    'stable_worldmodel/envs/__init__.py': '118a262677a647c6d6ef8889443b096abe082db4e365fc0a5b1249715e7e3c1a',
    'stable_worldmodel/envs/pusht/env.py': 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2',
    'stable_worldmodel/envs/dmcontrol/reacher.py': '7d58845f0a3412a1f8d52c0e2305fd8ca2889745ea38429ff08ccf2a8b561f65',
    'stable_worldmodel/envs/dmcontrol/custom_tasks/reacher.py': 'e5e2e82c0621efca7b4c3c194f422c946db59edb12db7735fbe8bdcf12e37e7c',
    'stable_worldmodel/envs/dmcontrol/dmcontrol.py': '13d55bf73f9b7f8a9b5a2b380e12750e253a391716e64e65f795b1c9fff8bc08',
    'stable_worldmodel/wrapper/default.py': '1099cbbdf9a79d4f409ef99dd95529aa6c9baced16d5102f47999110e35c0a51',
}
PLAN_CONFIG = {'horizon': 5, 'receding_horizon': 5, 'action_block': 5}
CEM_CONFIG = {'batch_size': 1, 'num_samples': 300, 'var_scale': 1.0, 'n_steps': 30, 'topk': 30}
CALLABLES = {
    'pusht': [{'method': '_set_state', 'args': {'state': {'value': 'state'}}},
              {'method': '_set_goal_state', 'args': {'goal_state': {'value': 'goal_state'}}}],
    'reacher': [{'method': 'set_state', 'args': {'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'}}},
                {'method': 'set_target_qpos', 'args': {'target_qpos': {'value': 'goal_qpos'}}}],
}
PLANNER_KEYS = frozenset(('pixels', 'goal', 'action', '_needs_flush', 'terminated'))


class PlanningMethodFailure(RuntimeError):
    """A nonfinite model/plan outcome counts as a method failure, not a retry."""


def case_seed(task, case_id, replan_index=0):
    if task not in CALLABLES or not isinstance(case_id, str) or not case_id:
        raise ValueError('Authorized task and explicit case ID required')
    if type(replan_index) is not int or replan_index < 0:
        raise ValueError('Nonnegative replan index required')
    # Exact helper convention supplied in protocol/07_PROTOCOL_CHECKS.py.
    return int.from_bytes(hashlib.sha256(f'R3_CEM_CASE_20261002/{task}/{case_id}/{replan_index}'.encode()).digest()[:8], 'big')


def replan_seed(task, case_id, replan_index):
    return case_seed(task, case_id, replan_index)


def planner_observation(info):
    """Keep task images and policy bookkeeping; no state/reward/expert future."""
    if not {'pixels', 'goal', 'action'}.issubset(info):
        raise ValueError('Official policy requires pixels, goal and action fields')
    return {key: value for key, value in info.items() if key in PLANNER_KEYS}


def validate_case(task, case, phase):
    if task not in CALLABLES or phase not in ('TECH', 'FORMAL'):
        raise ValueError('Only authorized task and phase values are accepted')
    required = ('case_id', 'episode_id', 'source_episode_idx', 'start_raw_index', 'goal_raw_index', 'reset_seed', 'family_id')
    if any(key not in case for key in required):
        raise ValueError('Case manifest missing fields: '+str([k for k in required if k not in case]))
    case_seed(task, case['case_id'])
    if any(type(case[k]) is not int for k in ('source_episode_idx', 'start_raw_index', 'goal_raw_index')):
        raise ValueError('Exact integer source indices required')
    if case['start_raw_index'] < 0 or case['goal_raw_index'] != case['start_raw_index'] + 25:
        raise ValueError('Official goal offset is exactly 25 raw steps')
    if type(case['reset_seed']) is not int or not 0 <= case['reset_seed'] < 2**32:
        raise ValueError('Explicit source/technically verified uint32 reset_seed required; never guessed')
    if 'role' in case and case['role'] != ('TECH' if phase == 'TECH' else 'EVAL'):
        raise ValueError('TECH and formal EVAL case identities cannot overlap')


def verify_official_sources(source_root=None):
    directory = Path(source_root or ROOT / 'source/swm_compat').resolve()
    for name, expected in SOURCE_SHA256.items():
        if sha256(directory / name) != expected:
            raise RuntimeError('Incompatible official SWM planning source: '+name)
    return {'revision': SWM_REVISION, 'files': dict(SOURCE_SHA256)}


def load_official_api(source_root=None):
    directory = Path(source_root or ROOT / 'source/swm_compat').resolve()
    identity = verify_official_sources(directory)
    loaded = sys.modules.get('stable_worldmodel')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(directory):
        raise RuntimeError('Another SWM version is already imported; use an isolated process with source/swm_compat')
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    swm = importlib.import_module('stable_worldmodel')
    cem = importlib.import_module('stable_worldmodel.solver.cem')
    policy = importlib.import_module('stable_worldmodel.policy')
    for module in (swm, cem, policy):
        if not Path(module.__file__).resolve().is_relative_to(directory):
            raise RuntimeError('Official planning import escaped pinned source root')
    return swm, cem.CEMSolver, policy.WorldModelPolicy, identity


class CaseDatasetView:
    """Official load_chunk interface, narrowed to one frozen reset/goal case.

    The source supplies actual 26-step tensors, pixels in T,C,H,W uint8 form.
    Expert action columns are never returned. World uses endpoint images/state;
    all later observations come from its real environment transitions.
    """
    def __init__(self, dataset, task, case):
        self.dataset, self.task, self.case = dataset, task, case
        required = ['pixels', 'state'] if task == 'pusht' else ['pixels', 'qpos', 'qvel']
        if not set(required).issubset(dataset.column_names):
            raise ValueError('Reset dataset lacks required task columns: '+str(required))
        self.column_names = required + ['seed']
        self.goal_state = None
        self.source_seed_present = 'seed' in dataset.column_names
        self.reads = 0

    def load_chunk(self, episodes, starts, ends):
        import numpy as np
        import torch
        case = self.case
        if (list(episodes) != [case['source_episode_idx']] or list(starts) != [case['start_raw_index']]
                or list(ends) != [case['goal_raw_index'] + 1]):
            raise ValueError('Dataset read escaped the frozen planning case')
        chunks = self.dataset.load_chunk(episodes, starts, ends)
        if len(chunks) != 1:
            raise ValueError('Planning is one explicit case per policy process')
        source = chunks[0]
        result = {key: source[key] for key in self.column_names if key != 'seed'}
        for key, value in result.items():
            if not isinstance(value, (np.ndarray, torch.Tensor)) or len(value) != 26:
                raise ValueError('Expected exact 26-row official source chunk: '+key)
        pixels = result['pixels']
        if not torch.is_tensor(pixels) or pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.dtype != torch.uint8:
            raise ValueError('World.load_chunk expects uint8 torch pixels[T,3,H,W]')
        if self.source_seed_present:
            seeds = np.asarray(source['seed']).reshape(26, -1)
            if not np.all(seeds == case['reset_seed']):
                raise ValueError('Frozen reset seed differs from source episode seed')
        elif not case.get('reset_seed_validation_sha256'):
            raise ValueError('Missing source seed needs a separately saved real reset validation receipt')
        result['seed'] = np.full(26, case['reset_seed'], dtype=np.int64)
        key = 'state' if self.task == 'pusht' else 'qpos'
        self.goal_state = np.asarray(result[key][-1], dtype=np.float64).copy()
        self.reads += 1
        if self.reads != 1:
            raise RuntimeError('Unexpected repeated case source access')
        return [result]


class _Timer:
    def __init__(self, device):
        import torch
        self.torch, self.device = torch, torch.device(device)
        self.records = []

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def begin(self, label):
        if self.device.type == 'cuda':
            start = self.torch.cuda.Event(enable_timing=True); start.record(self.torch.cuda.current_stream(self.device))
            return label, start
        return label, time.perf_counter()

    def end(self, token):
        label, start = token
        if self.device.type == 'cuda':
            end = self.torch.cuda.Event(enable_timing=True); end.record(self.torch.cuda.current_stream(self.device))
            self.records.append((label, start, end))
        else:
            self.records.append((label, start, time.perf_counter()))

    def totals(self):
        self.synchronize(); result = {}
        for label, start, end in self.records:
            seconds = start.elapsed_time(end) / 1000 if self.device.type == 'cuda' else end - start
            result[label] = result.get(label, 0.) + seconds
        return result


class AuditedCost:
    def __init__(self, model, timer):
        self.model, self.timer, self.calls = model, timer, 0

    def parameters(self):
        return self.model.parameters()

    def get_cost(self, info, candidates):
        import torch
        # JEPA overwrites action with candidate actions. The observed action
        # field is retained because its official get_cost explicitly pops it.
        inputs = {k: info[k] for k in ('pixels', 'goal', 'action')}
        token = self.timer.begin('model_cost_seconds')
        try:
            cost = self.model.get_cost(inputs, candidates)
        finally:
            self.timer.end(token)
        self.calls += 1
        if not torch.isfinite(cost).all():
            raise PlanningMethodFailure('NONFINITE_CEM_MODEL_COST')
        return cost


def instrument_solver(solver, task, case_id, timer, replans):
    """Seed and time the author's bound solve method, preserving its algorithm."""
    official_solve = solver.solve
    def solve(info, init_action=None):
        index = len(replans); seed = replan_seed(task, case_id, index)
        solver.torch_gen.manual_seed(seed)
        timer.synchronize(); start = time.perf_counter()
        record = {'replan_index': index, 'seed_uint64': seed, 'status': 'STARTED',
                  'observed_history_frames': int(info['pixels'].shape[1])}
        replans.append(record)
        try:
            result = official_solve(info, init_action=init_action)
            import torch
            if not torch.isfinite(result['actions']).all():
                raise PlanningMethodFailure('NONFINITE_CEM_RETURNED_ACTIONS')
            record.update(status='COMPLETE', elite_mean_costs=result.get('costs'))
            return result
        except BaseException as error:
            record.update(status='FAILED', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            timer.synchronize(); record['synchronized_wall_seconds'] = time.perf_counter() - start
    solver.solve = solve
    return solver


class IsolatedPolicy:
    def __init__(self, official_policy):
        self.official_policy = official_policy

    def set_env(self, env):
        self.official_policy.set_env(env)

    def get_action(self, info, **kwargs):
        import numpy as np
        action = self.official_policy.get_action(planner_observation(info), **kwargs)
        if not np.isfinite(action).all():
            raise PlanningMethodFailure('NONFINITE_RAW_POLICY_ACTION')
        return action


def _save_arrays(path, arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_case(model, task, case, dataset, action_processor, image_transform, *, arm,
             phase, output_dir, provenance, device='cuda', source_root=None):
    """Execute exactly one already-reserved trajectory using official World.evaluate.

    provenance binds caller-authenticated model/delta, source data, transform and
    normalization identities. Infrastructure exceptions are saved and re-raised;
    nonfinite model/planning outcomes are COMPLETE method failures with success=0.
    This function does not reserve or silently retry a trajectory.
    """
    import numpy as np
    import torch
    validate_case(task, case, phase)
    valid_arms = {'H0', 'REFIT_103201', 'REFIT_103202', 'REFIT_103203'}
    if phase == 'TECH': valid_arms.add('H0_CLONE')
    if arm not in valid_arms: raise ValueError('Arm outside fixed official/refit matrix')
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('Explicit asset/data/normalization/transform provenance is required')
    source_identity = verify_official_sources(source_root)
    identity = {'task': task, 'case': case, 'arm': arm, 'phase': phase,
                'plan_config': PLAN_CONFIG, 'cem_config': CEM_CONFIG, 'eval_budget_raw_steps': 50,
                'source': source_identity, 'provenance': provenance, 'planning_code_sha256': sha256(__file__)}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    folder = Path(output_dir)
    complete = folder / 'TRAJECTORY_SHA256.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['identity_sha256'] != identity_sha: raise RuntimeError('Existing trajectory identity differs')
        for name, record in saved['files'].items():
            path = folder / name
            if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise RuntimeError('Completed trajectory artifact differs: '+name)
        return json.loads((folder / 'result.json').read_text())
    if (folder / 'STARTED.json').exists():
        raise RuntimeError('Interrupted trajectory cannot silently restart; caller must account infrastructure failure and new attempt')
    ensure_space(); fp32_policy()
    if any(p.dtype != torch.float32 for p in model.parameters()): raise ValueError('Planning model must use FP32')
    requested_device = torch.device(device); actual_device = next(model.parameters()).device
    if requested_device.type != actual_device.type or (requested_device.index is not None and requested_device.index != actual_device.index):
        raise ValueError('Model/planning device mismatch')
    device = actual_device
    model.eval(); model.requires_grad_(False)
    swm, solver_type, policy_type, _ = load_official_api(source_root)
    timer = _Timer(device); replans = []; cost = AuditedCost(model, timer)
    solver = instrument_solver(solver_type(model=cost, device=device, seed=case_seed(task, case['case_id']), **CEM_CONFIG),
                               task, case['case_id'], timer, replans)
    policy = policy_type(solver=solver, config=swm.PlanConfig(**PLAN_CONFIG),
                         process={'action': action_processor}, transform={'pixels': image_transform, 'goal': image_transform})
    view = CaseDatasetView(dataset, task, case)
    rows = []; hooks = []; stacks = {}
    def enter(label):
        def hook(_module, _inputs): stacks.setdefault(label, []).append(timer.begin(label))
        return hook
    def leave(label):
        def hook(_module, _inputs, _outputs): timer.end(stacks[label].pop())
        return hook
    for label, module in (('visual_encoder_seconds', model.encoder), ('observation_projector_seconds', model.projector)):
        hooks += [module.register_forward_pre_hook(enter(label)), module.register_forward_hook(leave(label))]
    started = time.perf_counter(); world = None; failure = None; infrastructure_error = None; metrics = None
    atomic_json(folder / 'STARTED.json', {'identity_sha256': identity_sha, 'identity': identity,
                'started_at': now(), 'caller_reserved_budget': True})
    try:
        env_kwargs = {'task': 'qpos_match'} if task == 'reacher' else {}
        world = swm.World(env_name='swm/PushT-v1' if task == 'pusht' else 'swm/ReacherDMControl-v0',
                          num_envs=1, max_episode_steps=100, image_shape=(224, 224), **env_kwargs)
        # Pinned World extracts a NumPy seed array from recorded data, whereas
        # Gymnasium requires Python integer scalars. Preserve every numeric
        # value and only normalize its representation at the API boundary.
        from .env_compat import normalize_reset_seed
        original_reset=world.reset
        def compatible_reset(seed=None,options=None):
            return original_reset(seed=normalize_reset_seed(seed),options=options)
        world.reset=compatible_reset
        environment = world.envs.envs[0].unwrapped
        for spec in CALLABLES[task]:
            if not callable(getattr(environment, spec['method'], None)):
                raise RuntimeError('Required official reset/goal method absent: '+spec['method'])
        if task == 'reacher' and getattr(environment, '_task_name', None) != 'qpos_match':
            raise RuntimeError('Reacher task must be qpos_match')
        if tuple(world.envs.single_action_space.shape) != (2,):
            raise RuntimeError('Unexpected raw action dimension for fixed official tasks')
        if getattr(model, 'r3_contract', {}).get('macro_action_dim') != 10:
            raise RuntimeError('Official model macro-action dimension must match five raw 2D actions')
        world.set_policy(IsolatedPolicy(policy))
        official_step = world.envs.step
        def observed_step(actions, *args, **kwargs):
            before = time.perf_counter(); returned = official_step(actions, *args, **kwargs)
            elapsed = time.perf_counter() - before
            _, reward, terminated, truncated, infos = returned
            key = 'state' if task == 'pusht' else 'qpos'
            physical = np.asarray(infos[key][0, -1], dtype=np.float64).copy()
            target = view.goal_state
            if task == 'pusht':
                goal_error = float(world.envs.envs[0].unwrapped.eval_state(target, physical)[1])
            else:
                goal_error = float(np.linalg.norm(physical - target))
            rows.append({'action': np.asarray(actions[0]).copy(), 'reward': float(reward[0]),
                         'terminated': bool(terminated[0]), 'truncated': bool(truncated[0]),
                         'goal_error': goal_error, 'physical_state': physical, 'environment_seconds': elapsed})
            return returned
        world.envs.step = observed_step
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            metrics = world.evaluate(dataset=view, episodes_idx=[case['source_episode_idx']],
                start_steps=[case['start_raw_index']], goal_offset=25, eval_budget=50,
                callables=CALLABLES[task], video=None)
    except PlanningMethodFailure as error:
        failure = {'category': 'METHOD_FAILURE', 'type': type(error).__name__, 'message': str(error)}
    except BaseException as error:
        infrastructure_error = error
        failure = {'category': 'INFRASTRUCTURE_OR_INTERFACE_FAILURE', 'type': type(error).__name__, 'message': str(error)}
    finally:
        for hook in hooks: hook.remove()
        if world is not None: world.close()
    elapsed = time.perf_counter() - started
    timings = timer.totals()
    any_success = any(row['terminated'] for row in rows)
    success = bool(metrics['episode_successes'][0]) if metrics is not None else False
    if metrics is not None and success != any_success: raise RuntimeError('Official success differs from recorded termination events')
    result = {'status': 'COMPLETE' if infrastructure_error is None else 'INFRASTRUCTURE_FAILURE',
        'evaluation_kind': 'CLOSED_LOOP_CEM', 'evaluation_complete': infrastructure_error is None,
        'failure_category': 'INFRASTRUCTURE' if infrastructure_error is not None else 'METHOD' if failure else None,
        'identity_sha256': identity_sha, 'identity': identity, 'task': task, 'case_id': case['case_id'],
        'episode_id': case['episode_id'], 'family_id': case['family_id'], 'arm': arm, 'phase': phase,
        'success': int(success), 'any_step_success': int(any_success),
        'terminal_success': int(bool(rows and rows[-1]['terminated'])),
        'stop_reason': 'METHOD_FAILURE' if failure and infrastructure_error is None else 'INFRASTRUCTURE_FAILURE' if infrastructure_error is not None else 'SUCCESS' if success else 'RAW_STEP_BUDGET',
        'failure': failure, 'executed_raw_steps': len(rows), 'replan_calls': len(replans), 'replans': replans,
        'final_goal_error': rows[-1]['goal_error'] if rows else None,
        'trajectory_wall_seconds': elapsed, 'planning_synchronized_wall_seconds': sum(r['synchronized_wall_seconds'] for r in replans),
        'environment_step_seconds': sum(r['environment_seconds'] for r in rows), 'model_cost_calls': cost.calls,
        'device_timing': timings, 'device': str(device), 'source_dataset_window_reads': view.reads,
        'history_contract': 'one current image at every official replan; no expert past or future observations supplied',
        'action_contract': '5 consecutive raw actions concatenated per macro; official inverse_transform; no extra action clipping',
        'success_contract': 'official World.evaluate OR of env termination under wait mode',
        'goal_error_contract': 'official PushT full-state L2 distance' if task == 'pusht' else 'qpos L2 radians; success remains official all-joint abs<0.05 termination',
        'optimizer_updates': 0, 'completed_at': now()}
    _save_arrays(folder / 'trajectory.npz', {
        'raw_actions': np.stack([r['action'] for r in rows]) if rows else np.empty((0, 2)),
        'rewards': np.asarray([r['reward'] for r in rows]), 'step_success': np.asarray([r['terminated'] for r in rows]),
        'step_truncated': np.asarray([r['truncated'] for r in rows]), 'goal_error': np.asarray([r['goal_error'] for r in rows]),
        'physical_state': np.stack([r['physical_state'] for r in rows]) if rows else np.empty((0, 0)),
        'goal_state': view.goal_state if view.goal_state is not None else np.empty(0),
        'environment_step_seconds': np.asarray([r['environment_seconds'] for r in rows])})
    atomic_json(folder / 'result.json', result)
    if infrastructure_error is not None: raise infrastructure_error
    files = {name: {'sha256': sha256(folder / name), 'bytes': (folder / name).stat().st_size}
             for name in ('STARTED.json', 'result.json', 'trajectory.npz')}
    atomic_json(complete, {'status': 'COMPLETE', 'identity_sha256': identity_sha, 'files': files})
    return result
