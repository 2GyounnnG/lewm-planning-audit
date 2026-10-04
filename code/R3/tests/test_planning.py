"""Synthetic CPU checks only: no environment, GPU, real data or optimizer."""
import ast, copy, hashlib, json, logging, time, types, unittest
from collections import deque
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
from r3 import planning as p

ROOT = Path(__file__).resolve().parents[1]


def case():
    return {'case_id': 'synthetic-case', 'episode_id': 'episode_9', 'source_episode_idx': 9,
            'start_raw_index': 4, 'goal_raw_index': 29, 'reset_seed': 17, 'family_id': 'synthetic-family', 'role': 'TECH'}


class Dataset:
    column_names = ['pixels', 'state', 'qpos', 'qvel', 'seed', 'action', 'reward']
    def load_chunk(self, episodes, starts, ends):
        return [{'pixels': torch.zeros(26, 3, 4, 4, dtype=torch.uint8),
                 'state': np.arange(26 * 7).reshape(26, 7), 'qpos': np.zeros((26, 2)), 'qvel': np.ones((26, 2)),
                 'seed': np.full(26, 17), 'action': np.full((26, 2), 987654321.), 'reward': np.arange(26)}]


def official_class(filename, class_name, namespace):
    """Compile an unmodified official class while omitting environment imports."""
    source = ROOT / 'source/swm_compat/stable_worldmodel' / filename
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[class_name]


def synthetic_cem_type():
    class Box:
        shape = (1, 2)
    namespace = {'time': time, 'Any': Any, 'torch': torch, 'np': np,
                 'gym': types.SimpleNamespace(Space=Box), 'Box': Box, 'logging': logging,
                 'Costable': object, 'Callback': object, 'Actionable': type('UnusedActorProtocol', (), {})}
    source = ROOT / 'source/swm_compat/stable_worldmodel/solver/utils.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_init_action')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return official_class('solver/cem.py', 'CEMSolver', namespace), Box


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_pinned_real_source_hashes_without_importing_environments(self):
        identity = p.verify_official_sources()
        self.assertEqual(identity['revision'], 'abdced49809d5eae38e24b27dc7b635c502c4812')
        self.assertEqual(len(identity['files']), 12)

    def test_device_timer_uses_requested_stream_without_any_cuda_call(self):
        seen = []
        class Event:
            def __init__(self, **kwargs): pass
            def record(self, stream): seen.append(stream)
            def elapsed_time(self, other): return 12.5
        timer = p._Timer('cuda:1')  # Constructing torch.device never initializes CUDA.
        timer.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            Event=Event, current_stream=lambda device: 'fake-stream-'+str(device), synchronize=lambda device: None))
        timer.end(timer.begin('encode'))
        self.assertEqual(seen, ['fake-stream-cuda:1', 'fake-stream-cuda:1'])
        self.assertEqual(timer.totals(), {'encode': 0.0125})

    def test_case_and_replan_random_streams_are_arm_schedule_independent(self):
        base = p.case_seed('pusht', 'a')
        self.assertEqual(base, int.from_bytes(hashlib.sha256(b'R3_CEM_CASE_20261002/pusht/a/0').digest()[:8], 'big'))
        expected = {name: [p.replan_seed('pusht', name, i) for i in range(2)] for name in ('a', 'b')}
        self.assertEqual(expected, {name: [p.replan_seed('pusht', name, i) for i in range(2)] for name in ('b', 'a')})
        self.assertNotEqual(expected['a'][0], expected['a'][1])
        self.assertNotEqual(base, p.case_seed('reacher', 'a'))
        with self.assertRaises(ValueError): p.replan_seed('pusht', 'a', -1)

    def test_case_metadata_cannot_change_offset_phase_or_guess_seed(self):
        p.validate_case('pusht', case(), 'TECH')
        with self.assertRaises(ValueError): p.validate_case('pusht', case(), 'FORMAL')
        for replacement in ({'goal_raw_index': 30}, {'reset_seed': None}, {'reset_seed': 2**32}, {'start_raw_index': -1}):
            with self.assertRaises(ValueError): p.validate_case('pusht', {**case(), **replacement}, 'TECH')

    def test_endpoint_dataset_has_exact_indices_and_no_expert_actions(self):
        view = p.CaseDatasetView(Dataset(), 'pusht', case())
        chunk = view.load_chunk([9], [4], [30])[0]
        self.assertEqual(set(chunk), {'pixels', 'state', 'seed'})
        self.assertNotIn('action', chunk); np.testing.assert_array_equal(view.goal_state, np.arange(26 * 7).reshape(26, 7)[-1])
        with self.assertRaises(RuntimeError): view.load_chunk([9], [4], [30])
        with self.assertRaises(ValueError): p.CaseDatasetView(Dataset(), 'pusht', case()).load_chunk([9], [5], [31])
        with self.assertRaises(ValueError): p.CaseDatasetView(Dataset(), 'pusht', {**case(), 'reset_seed': 18}).load_chunk([9], [4], [30])
        no_seed = Dataset(); no_seed.column_names = [k for k in no_seed.column_names if k != 'seed']
        with self.assertRaises(ValueError): p.CaseDatasetView(no_seed, 'pusht', case()).load_chunk([9], [4], [30])

    def test_planner_only_receives_images_actions_and_bookkeeping(self):
        info = {k: object() for k in ('pixels', 'goal', 'action', '_needs_flush', 'terminated', 'state', 'qpos', 'qvel', 'goal_state', 'goal_qpos', 'reward', 'expert_future_actions')}
        filtered = p.planner_observation(info)
        self.assertEqual(set(filtered), p.PLANNER_KEYS)
        self.assertIs(filtered['pixels'], info['pixels'])
        self.assertIn('state', info)  # Source dict remains untouched.

    def test_audited_cost_is_exact_and_rejects_nonfinite_model_results(self):
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
            def get_cost(self, info, action):
                self.seen = set(info)
                return action.square().sum((-1, -2)) * self.weight
        model = Model(); timer = p._Timer('cpu'); wrapped = p.AuditedCost(model, timer)
        candidates = torch.arange(30.).reshape(1, 3, 5, 2)
        info = {k: torch.zeros(1) for k in ('pixels', 'goal', 'action', 'state', 'goal_qpos')}
        actual = wrapped.get_cost(info, candidates)
        torch.testing.assert_close(actual, model.get_cost({k: info[k] for k in ('pixels', 'goal', 'action')}, candidates), rtol=0, atol=0)
        self.assertEqual(model.seen, {'pixels', 'goal', 'action'})
        with self.assertRaises(p.PlanningMethodFailure): wrapped.get_cost(info, candidates * float('nan'))

    def test_official_cem_clone_same_seed_actions_without_environment(self):
        solver_type, box = synthetic_cem_type()
        class Quadratic(torch.nn.Module):
            def __init__(self): super().__init__(); self.dummy = torch.nn.Parameter(torch.tensor(0.))
            def get_cost(self, info, actions): return (actions - 0.25).square().sum((-1, -2))
        config = types.SimpleNamespace(**p.PLAN_CONFIG)
        model = Quadratic(); inputs = {'pixels': torch.zeros(1, 1, 3, 2, 2), 'goal': torch.ones(1, 1, 3, 2, 2), 'action': torch.zeros(1, 1, 2)}
        traces = []
        for _ in range(2):
            solver = solver_type(model=copy.deepcopy(model), device='cpu', seed=999, **p.CEM_CONFIG)
            solver.configure(action_space=box(), n_envs=1, config=config)
            log = []; p.instrument_solver(solver, 'pusht', 'synthetic', p._Timer('cpu'), log)
            actions = [solver(copy.deepcopy(inputs))['actions'] for _ in range(2)]
            traces.append((actions, log))
        for a, b in zip(traces[0][0], traces[1][0]): torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual([r['seed_uint64'] for r in traces[0][1]], [r['seed_uint64'] for r in traces[1][1]])
        self.assertTrue(all(r['observed_history_frames'] == 1 for r in traces[0][1]))

    def test_official_policy_unpacks_25_distinct_raw_actions_then_replans(self):
        namespace = {'Any': Any, 'np': np, 'torch': torch, 'deque': deque, 'Solver': object,
                     'PlanConfig': object, 'Transformable': object, 'Callable': Callable}
        official_class('policy.py', 'BasePolicy', namespace)
        policy_type = official_class('policy.py', 'WorldModelPolicy', namespace)
        class Solver:
            calls = 0
            def configure(self, **kwargs): pass
            def __call__(self, info, init_action=None):
                self.calls += 1
                return {'actions': torch.arange(50.).reshape(1, 5, 10)}
        solver = Solver(); config = types.SimpleNamespace(**p.PLAN_CONFIG, warm_start=True)
        policy = policy_type(solver=solver, config=config)
        env = types.SimpleNamespace(num_envs=1, action_space=types.SimpleNamespace(shape=(1, 2)), single_action_space=types.SimpleNamespace(shape=(2,)))
        policy.set_env(env)
        info = {'pixels': np.zeros((1, 1, 2, 2, 3), dtype=np.uint8), 'goal': np.zeros((1, 1, 2, 2, 3), dtype=np.uint8), 'action': np.zeros((1, 1, 2))}
        actions = [policy.get_action(info).copy() for _ in range(25)]
        np.testing.assert_array_equal(np.stack(actions)[:, 0], np.arange(50).reshape(25, 2))
        self.assertEqual(solver.calls, 1); policy.get_action(info); self.assertEqual(solver.calls, 2)


if __name__ == '__main__': unittest.main()
