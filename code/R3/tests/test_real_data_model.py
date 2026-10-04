"""Synthetic CPU gate checks; no official weights, real data, GPU or optimizer."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import types
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from scripts import check_real_data_model as c
from r3.model import frozen_hashes, tensor_sha256

ROOT = Path(__file__).resolve().parents[1]


def original_jepa():
    spec = importlib.util.spec_from_file_location('_real_gate_test_official_jepa', ROOT/'source/lewm/jepa.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.JEPA


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(3, 4); self.bn = nn.BatchNorm1d(4)
    def forward(self, pixels, **kwargs):
        value = self.bn(self.linear(pixels.mean((-1, -2))))
        return types.SimpleNamespace(last_hidden_state=value[:, None])


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__(); self.z = nn.Linear(4, 4); self.a = nn.Linear(4, 4); self.dropout = nn.Dropout(.1)
    def forward(self, z, a): return self.dropout(self.z(z)+self.a(a))


def fixture():
    torch.manual_seed(123)
    model = original_jepa()(TinyEncoder(), TinyPredictor(), nn.Linear(10, 4),
                            nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)),
                            nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))
    model.r3_contract = {'history_size': 3, 'latent_dim': 4, 'macro_action_dim': 10}
    model.eval().requires_grad_(False)
    pixels = torch.rand(2, 8, 3, 4, 4)
    with torch.no_grad(): z = model.encode({'pixels': pixels})['emb']
    actions = torch.linspace(-1, 1, 140).reshape(2, 7, 10)
    return model, pixels, z, actions


def roles():
    rows = [{'episode_id': 'tech', 'role': 'TECH', 'length': 80, 'open_loop_starts': [11, 14]},
            {'episode_id': 'eval', 'role': 'EVAL', 'length': 80, 'open_loop_starts': [11, 14]}]
    return {'task': 'pusht', 'status': 'METADATA_ROLES_FROZEN', 'history_size': 3, 'frameskip': 5,
            'technical_monitor_windows': [['tech', 1], ['tech', 4], ['eval', 1]], 'episodes': rows}


class RealDataModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_numeric_contract_cannot_be_guessed_or_relaxed(self):
        record = {'status': 'PROSPECTIVELY_FROZEN', 'comparisons': copy.deepcopy(c.REGISTERED_TOLERANCES)}
        self.assertEqual(c.tolerances_from(record), c.REGISTERED_TOLERANCES)
        for changed in ({}, {'status': 'DRAFT', 'comparisons': record['comparisons']}):
            with self.assertRaises(RuntimeError): c.tolerances_from(changed)
        record['comparisons']['rollout_wrapper']['atol'] = 1e-4
        with self.assertRaises(RuntimeError): c.tolerances_from(record)

    def test_first_two_fixed_windows_and_all_raw_phases(self):
        result = c.choose_windows(roles(), 'pusht', {'history_size': 3, 'macro_action_dim': 10})
        self.assertEqual([r['start_raw_index'] for r in result], [1, 4])
        self.assertEqual(result[0]['history_raw_indices'], [1, 6, 11])
        self.assertEqual(result[0]['free_rollout_target_raw_indices'], [16, 21, 26, 31, 36])
        self.assertEqual(result[0]['action_raw_indices'], list(range(1, 36)))
        bad = roles(); bad['technical_monitor_windows'][0][0] = 'eval'
        with self.assertRaises(PermissionError): c.choose_windows(bad, 'pusht', {'history_size': 3, 'macro_action_dim': 10})

    def test_every_pulse_preserves_action_time_and_coordinate(self):
        result = c.pulse_check(); self.assertEqual(result['scalar_pulses'], 70)
        raw = np.arange(70, dtype=np.float32).reshape(35, 2)
        np.testing.assert_array_equal(c.macro_actions(raw)[1], np.arange(10, 20))
        np.testing.assert_array_equal(c.macro_actions(raw).reshape(35, 2), raw)
        with self.assertRaises(ValueError): c.macro_actions(raw[:-1])
        raw[2, 1] = np.nan
        with self.assertRaises(ValueError): c.macro_actions(raw)

    def test_actual_original_jepa_predict_rollout_and_backward_no_optimizer(self):
        model, pixels, z, actions = fixture(); before = frozen_hashes(model)
        weights = {n: tensor_sha256(p) for n, p in model.named_parameters()}
        # An accidental optimizer construction is a hard fixture failure.
        with patch('torch.optim.AdamW', side_effect=AssertionError('Optimizer forbidden')):
            report, arrays = c.model_checks(model, pixels, z, actions, c.REGISTERED_TOLERANCES)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(arrays['official_rollout_last5'].shape, (2, 5, 4))
        np.testing.assert_allclose(arrays['official_rollout_last5'], arrays['cached_rollout_same_official_initial_z'], rtol=1e-6, atol=1e-6)
        gradient = report['checks']['zero_update_backward']
        self.assertTrue(gradient['all_parameter_values_exactly_unchanged'])
        self.assertTrue(all(gradient['nonzero_gradient_parameters_by_prefix'].values()))
        self.assertTrue(all(g['gradient_is_none'] for g in report['gradients'] if not g['whitelisted']))
        self.assertEqual(before, frozen_hashes(model))
        self.assertEqual(weights, {n: tensor_sha256(p) for n, p in model.named_parameters()})
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(gradient['optimizer_objects_created'], 0)

    def test_future_cached_targets_never_enter_free_forecast(self):
        model, pixels, z, actions = fixture()
        report, first = c.model_checks(model, pixels, z, actions, c.REGISTERED_TOLERANCES)
        corrupt = z.clone(); corrupt[:, 3:] += 1000
        second_report, second = c.model_checks(model, pixels, corrupt, actions, c.REGISTERED_TOLERANCES)
        self.assertEqual(report['status'], 'PASS'); self.assertEqual(second_report['status'], 'FAIL')
        np.testing.assert_array_equal(first['production_cached_rollout_last5'], second['production_cached_rollout_last5'])
        self.assertGreater(second_report['checks']['encoder_cache']['max_abs_difference'], 999)

    def test_nonfinite_and_wrong_shapes_do_not_pass_comparison(self):
        result = c.compare(torch.tensor([float('nan')]), torch.ones(1), {'atol': 1e-6, 'rtol': 1e-6})
        self.assertEqual(result['status'], 'FAIL'); self.assertFalse(result['finite'])
        self.assertIsNone(result['max_abs_difference'])
        self.assertEqual(c.compare(torch.ones(2), torch.ones(3), {'atol': 0, 'rtol': 0})['reason'], 'SHAPE_DIFFERENCE')
        self.assertEqual(c.compare(torch.ones(1), torch.ones(1))['status'], 'DIAGNOSTIC_ONLY')

    def test_illegal_cache_input_gradients_rejected(self):
        model, pixels, z, actions = fixture(); z.requires_grad_()
        with self.assertRaises(ValueError): c.model_checks(model, pixels, z, actions, c.REGISTERED_TOLERANCES)

    def test_wrapper_action_shift_is_detected(self):
        model, pixels, z, actions = fixture()
        original = c.cached_rollout
        def shifted(m, initial, a, horizon): return original(m, initial, a.flip(1), horizon)
        with patch.object(c, 'cached_rollout', side_effect=shifted):
            report, _ = c.model_checks(model, pixels, z, actions, c.REGISTERED_TOLERANCES)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['checks']['rollout_wrapper']['status'], 'FAIL')

    def test_nonfinite_production_rollout_cannot_hide_in_diagnostic_column(self):
        model, pixels, z, actions = fixture(); original = c.cached_rollout; calls = []
        def result(m, initial, a, horizon):
            calls.append(1); output = original(m, initial, a, horizon)
            return output if len(calls) == 1 else output*float('nan')
        with patch.object(c, 'cached_rollout', side_effect=result):
            report, _ = c.model_checks(model, pixels, z, actions, c.REGISTERED_TOLERANCES)
        self.assertEqual(report['checks']['rollout_wrapper']['status'], 'PASS')
        self.assertEqual(report['checks']['all_saved_model_arrays_finite']['status'], 'FAIL')
        self.assertEqual(report['status'], 'FAIL')

    def test_raw_reader_only_opens_pixels_actions_for_the_exact_two_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'synthetic_source'; path.write_bytes(b'no real source observations')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = {'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest}
            actual_roles = roles()
            for index, row in enumerate(actual_roles['episodes']):
                row.update(source_asset_sha256=digest, source_episode_idx=index)
            actions = np.arange(160, dtype=np.float32).reshape(80, 2)
            cache = {'tech': {'z': np.zeros((80, 4), dtype=np.float32), 'actions': actions,
                             'raw_indices': np.arange(80), 'stride': np.asarray(5), 'legal_starts': np.asarray([1, 4])}}
            calls = []; instances = []
            class FakeRawH5:
                def __init__(self, opened_path, keys):
                    self.lengths = np.asarray([80, 80]); self.closed = False
                    calls.append(('open', str(opened_path), keys)); instances.append(self)
                def array(self, ep, key, start, end):
                    if key not in ('pixels', 'action') or ep != 0: raise AssertionError('Forbidden source role/key')
                    calls.append((ep, key, start, end))
                    if key == 'action': return actions[start:end].copy()
                    return np.full((end-start, 224, 224, 3), start, dtype=np.uint8)
                def close(self): self.closed = True
            model = types.SimpleNamespace(r3_contract={'history_size': 3, 'macro_action_dim': 10, 'latent_dim': 4})
            with patch.object(c, '_selected_cache', return_value=(cache, {'tech': record})), \
                 patch.object(c.data, 'verified_file', return_value=path), \
                 patch.object(c.data, 'action_processor', return_value=types.SimpleNamespace(transform=lambda a:a.copy())), \
                 patch.object(c.data, 'image_transform', return_value=lambda x:x.float()/255), \
                 patch.object(c.data, 'RawH5', FakeRawH5):
                pixels, z, macro, audit = c.collect_inputs('pusht', model, actual_roles, {}, {'assets': {digest: record}})
            self.assertEqual(pixels.shape, (2, 8, 3, 224, 224)); self.assertEqual(z.shape, (2, 8, 4))
            self.assertEqual(calls[0], ('open', str(path), ['pixels', 'action']))
            self.assertTrue(all(x.closed for x in instances))
            self.assertEqual([(x[2], x[3]) for x in calls[1:] if x[1] == 'action'], [(1, 36), (4, 39)])
            self.assertEqual([x[2] for x in calls[1:] if x[1] == 'pixels'], [1,6,11,16,21,26,31,36,4,9,14,19,24,29,34,39])
            np.testing.assert_array_equal(macro[0].numpy().reshape(35, 2), actions[1:36])
            self.assertEqual(audit['source_state_reward_goal_labels_read'], 0)
            self.assertEqual(audit['other_role_cache_arrays_read'], 0)


if __name__ == '__main__': unittest.main(verbosity=2)
