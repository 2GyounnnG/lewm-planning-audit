"""Small synthetic R3 statistical contracts; no scientific EVAL data or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from r3 import statistics as s


def rows_for(matrix, families=None):
    if families is None:
        families = [f'f{i}' for i in range(len(matrix))]
    return [{'case_id': f'c{i:03}', 'family_id': families[i], 'arm': arm,
             'success': int(matrix[i][j]), 'evaluation_kind': 'CLOSED_LOOP_CEM'}
            for i in range(len(matrix)) for j, arm in enumerate(s.ARMS)]


class StatisticsContracts(unittest.TestCase):
    def test_fixed_seed_derivation_and_pairing(self):
        want = int.from_bytes(hashlib.sha256(b'R3_BOOTSTRAP_20261002/pusht').digest()[:8], 'big')
        self.assertEqual(s.bootstrap_seed('pusht'), want)
        self.assertNotEqual(want, s.bootstrap_seed('pusht', True))
        self.assertNotEqual(want, s.bootstrap_seed('reacher'))
        indices = s.bootstrap_indices('pusht', 4)
        self.assertEqual(indices.shape, (5000, 4))
        np.testing.assert_array_equal(indices, s.bootstrap_indices('pusht', 4))
        self.assertTrue(((indices >= 0) & (indices < 4)).all())

    def test_direction_counts_and_fixed_seed_mean_not_pseudoreplication(self):
        matrix = [[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 0]]
        result = s.analyze_task('pusht', rows_for(matrix))
        self.assertEqual(result['cases'], 4)
        self.assertAlmostEqual(result['main_effect_pp'], 25.)
        counts = result['paired_counts'][0]
        self.assertEqual(counts['s01_H0_failure_REFIT_success'], 1)
        self.assertEqual(counts['s10_H0_success_REFIT_failure'], 0)
        self.assertEqual(counts['s11_both_success'], 2)
        self.assertEqual(counts['s00_both_failure'], 1)
        self.assertAlmostEqual(result['main_effect_pp'], np.mean([x['delta_success_pp'] for x in result['paired_counts']]))

    def test_all_identical_arms_have_zero_paired_intervals(self):
        result = s.analyze_task('reacher', rows_for([[0]*4, [1]*4, [1]*4]))
        for scheme in ('CASE', 'FAMILY'):
            for name, metric in result['schemes'][scheme]['metrics'].items():
                if 'minus_H0' in name:
                    self.assertEqual(metric['estimate'], 0.)
                    self.assertEqual(metric['ci95'], [0., 0.])
                    self.assertEqual(metric['ci97_5_two_task_bonferroni_approximation'], [0., 0.])

    def test_unequal_family_ratio_preserves_case_weight(self):
        success = np.asarray([[0, 1, 1, 1]]*3 + [[1, 0, 0, 0]], dtype=float)
        families = ['large']*3 + ['small']
        indices = np.asarray([[0, 1], [0, 0], [1, 1]], dtype=np.int64)
        means, sizes = s.family_resampled_means(success, families, ['large', 'small'], indices)
        self.assertEqual(sizes.tolist(), [3, 1])
        np.testing.assert_allclose(s.success_values(means)[:, -1], [50, 100, -100])
        result = s.analyze_task('pusht', rows_for(success, families))
        self.assertEqual(result['main_effect_pp'], 50)
        self.assertEqual(result['schemes']['FAMILY']['metrics'][s.BOOTSTRAP_COLUMNS[-1]]['estimate'], 50)

    def test_percentile_linear_exact_and_two_task_interval(self):
        matrix = np.asarray([[0, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]], dtype=float)
        result = s.analyze_task('pusht', rows_for(matrix))
        draws = s.success_values(matrix[s.bootstrap_indices('pusht', 3)].mean(axis=1))
        expected = np.quantile(draws, [.025, .975, .0125, .9875], axis=0, method='linear')
        for i, name in enumerate(s.BOOTSTRAP_COLUMNS):
            metric = result['schemes']['CASE']['metrics'][name]
            np.testing.assert_array_equal(metric['ci95'], expected[:2, i])
            np.testing.assert_array_equal(metric['ci97_5_two_task_bonferroni_approximation'], expected[2:, i])

    def test_missing_duplicate_nonbinary_and_identity_are_rejected(self):
        rows = rows_for([[0, 1, 0, 1], [1, 0, 1, 1]])
        cases = [rows[:-1], rows + [rows[0]]]
        for key, value in [('success', .5), ('success', float('nan')), ('arm', 'BEST_SEED'),
                           ('success', 1+0j),
                           ('family_id', 'changed'), ('task', 'reacher'),
                           ('evaluation_kind', 'OPEN_LOOP'), ('evaluation_complete', False),
                           ('phase', 'TECH'),
                           ('failure_category', 'INFRASTRUCTURE')]:
            bad = copy.deepcopy(rows); bad[0][key] = value; cases.append(bad)
        for bad in cases:
            with self.subTest(rows=bad[:1]):
                with self.assertRaises(ValueError):
                    s.analyze_task('pusht', bad)

    def test_formal_case_cap(self):
        with self.assertRaisesRegex(ValueError, 'at most100'):
            s.analyze_task('pusht', rows_for([[0]*4]*101))

    def test_duplicate_episode_cannot_inflate_cases(self):
        rows = rows_for([[0, 1, 0, 1], [1, 0, 1, 1]])
        for row in rows:
            row['episode_id'] = 'same_episode'
        with self.assertRaisesRegex(ValueError, 'same episode'):
            s.analyze_task('pusht', rows)

    def test_unknown_family_does_not_become_fake_independent_families(self):
        result = s.analyze_task('pusht', rows_for([[0, 1, 0, 1], [1, 0, 1, 1]], [None, 'known']))
        self.assertEqual(result['schemes']['FAMILY']['status'], 'UNAVAILABLE_FAMILY_UNKNOWN')
        self.assertEqual(result['schemes']['CASE']['status'], 'COMPLETE')
        single = s.analyze_task('pusht', rows_for([[0]*4, [1]*4], ['one', 'one']))
        self.assertIn('DEGENERATE', single['schemes']['FAMILY']['cluster_warning'])

    def test_algorithm_failure_preserved_and_auxiliary_not_silently_dropped(self):
        rows = rows_for([[0, 1, 0, 1], [1, 0, 1, 1]])
        rows[0].update(status='ALGORITHM_FAILURE', failure_category='ALGORITHM',
                       failure_reason='nonfinite planner', final_goal_error=float('inf'))
        rows[4]['final_goal_error'] = 2.
        result = s.analyze_task('pusht', rows)
        desc = result['auxiliary_descriptions']['H0']['final_goal_error']
        self.assertEqual(result['cases'], 2)
        self.assertEqual(desc['nonfinite_cases'], 1)
        self.assertIsNone(desc['all_case_mean'])
        self.assertEqual(desc['finite_subset_mean_DESCRIPTIVE_ONLY'], 2.)

    def test_saved_indices_raw_values_idempotence_and_tamper_refusal(self):
        rows = rows_for([[0, 1, 0, 1], [1, 0, 1, 1]])
        rows[0]['final_goal_error'] = float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); result = s.analyze_task('pusht', rows, p)
            receipt = json.loads((p/'STATISTICS_RECEIPT.json').read_text())
            self.assertEqual(receipt['cases'], 2)
            for name, record in receipt['files'].items():
                self.assertEqual(s.file_sha256(p/name), record['sha256'])
            with np.load(p/'BOOTSTRAP_INDICES.npz', allow_pickle=False) as f:
                self.assertEqual(f['case'].shape, (5000, 2))
                self.assertEqual(s.array_sha256(f['case']), result['bootstrap']['case_indices_sha256'])
            with np.load(p/'BOOTSTRAP_DISTRIBUTIONS.npz', allow_pickle=False) as f:
                self.assertEqual(f['case'].shape, (5000, 8))
            saved = json.loads((p/'RAW_CASE_ROWS.json').read_text())
            self.assertEqual(saved[0]['final_goal_error'], {'__nonfinite_float__': 'NaN'})
            self.assertEqual(result, s.analyze_task('pusht', s.restore_json_value(saved), p))
            self.assertEqual(result, s.analyze_task('pusht', list(reversed(rows)), p))
            changed = copy.deepcopy(rows); changed[0]['success'] = 1
            with self.assertRaisesRegex(ValueError, 'changed input'):
                s.analyze_task('pusht', changed, p)
            (p/'PAIRED_COUNTS.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'changed'):
                s.analyze_task('pusht', rows, p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
