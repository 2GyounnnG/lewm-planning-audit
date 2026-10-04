"""R3 fixed-task CEM paired inference. Pure NumPy; no model or data loader.

Input: one completed row per case and arm, with case_id, family_id (or null),
arm and binary success. Optional metrics/provenance are retained in full.
Open-loop errors belong to a different analysis and are explicitly excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path

import numpy as np

VERSION = 'R3_PAIRED_CEM_STATISTICS_V1'
TASKS = ('pusht', 'reacher')
SEEDS = (103201, 103202, 103203)
ARMS = ('H0',) + tuple(f'REFIT_{s}' for s in SEEDS)
REPLICATES = 5000
BOOTSTRAP_COLUMNS = tuple(a + '_success_percent' for a in ARMS) + tuple(
    a + '_minus_H0_pp' for a in ARMS[1:]) + ('FIXED_THREE_REFIT_MEAN_minus_H0_pp',)
AUXILIARY_FIELDS = ('final_goal_error', 'wall_seconds', 'planning_seconds',
                    'environment_seconds', 'executed_raw_steps', 'replan_count',
                    'trajectory_wall_seconds', 'planning_synchronized_wall_seconds',
                    'environment_step_seconds', 'replan_calls')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def json_value(value):
    """Strict JSON retaining nonfinite auxiliary observations as tagged values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite_float__': 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('Row metadata keys must be strings')
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Unsupported row metadata type: ' + type(value).__name__)


def canonical_bytes(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def restore_json_value(value):
    """Restore this module's lossless strict-JSON auxiliary nonfinite encoding."""
    if isinstance(value, dict):
        if set(value) == {'__nonfinite_float__'}:
            return {'NaN': float('nan'), '+Infinity': float('inf'), '-Infinity': -float('inf')}[value['__nonfinite_float__']]
        return {k: restore_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_json_value(v) for v in value]
    return value


def bootstrap_seed(task, family=False):
    if task not in TASKS:
        raise ValueError('Task must be pusht or reacher')
    salt = f'R3_BOOTSTRAP_20261002/{task}' + ('/family' if family else '')
    return int.from_bytes(hashlib.sha256(salt.encode('utf-8')).digest()[:8], 'big')


def bootstrap_indices(task, count, family=False):
    if type(count) is not int or count < 1:
        raise ValueError('A positive integer resampling population is required')
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed(task, family)))
    return rng.integers(0, count, size=(REPLICATES, count), dtype=np.int64)


def array_sha256(array):
    a = np.ascontiguousarray(array, dtype='<i8')
    return hashlib.sha256(canonical_bytes({'shape': list(a.shape), 'dtype': '<i8'}) + a.tobytes()).hexdigest()


def normalize_rows(task, rows):
    if task not in TASKS:
        raise ValueError('Task must be pusht or reacher')
    rows = list(rows)
    if not rows:
        raise ValueError('Empty paired case table')
    by = {}
    for row in rows:
        if not isinstance(row, dict) or any(k not in row for k in ('case_id', 'family_id', 'arm', 'success')):
            raise ValueError('Each row requires case_id, family_id, arm, success')
        case, arm, family, success = (row[k] for k in ('case_id', 'arm', 'family_id', 'success'))
        if not isinstance(case, str) or not case.strip() or case != case.strip():
            raise ValueError('case_id must be nonempty canonical text')
        if arm not in ARMS:
            raise ValueError('Unexpected arm: ' + str(arm))
        if family is not None and (not isinstance(family, str) or not family.strip() or family != family.strip()):
            raise ValueError('family_id must be canonical text or null')
        if row.get('task', task) != task or row.get('evaluation_kind', 'CLOSED_LOOP_CEM') != 'CLOSED_LOOP_CEM':
            raise ValueError('Mixed task or non-CEM/open-loop input')
        if row.get('phase', 'FORMAL') != 'FORMAL':
            raise ValueError('TECH trajectories cannot enter formal statistics')
        if row.get('evaluation_complete', True) is not True or row.get('failure_category') == 'INFRASTRUCTURE':
            raise ValueError('Incomplete infrastructure attempt is not a scored case')
        if row.get('status') in ('PENDING', 'RUNNING', 'INFRASTRUCTURE_ERROR', 'INFRASTRUCTURE_FAILURE'):
            raise ValueError('Unfinished case cannot be silently scored as failure')
        if not isinstance(success, (numbers.Real, np.bool_)) or not np.isfinite(success) or success not in (0, 1):
            raise ValueError('Success must explicitly be binary 0 or 1; never impute a missing result')
        if (case, arm) in by:
            raise ValueError('Duplicate case/arm: ' + case + '/' + arm)
        by[case, arm] = row
        json_value(row)  # Reject opaque metadata before any output is written.
    cases = sorted({case for case, _ in by})
    if len(cases) > 100:
        raise ValueError('R3 allows at most100 formal cases per task')
    if len(by) != len(cases) * len(ARMS):
        raise ValueError('Missing paired arm(s) for a case')
    ordered, families, episodes = [], [], {}
    for case in cases:
        part = [by[case, arm] for arm in ARMS]
        if len({r['family_id'] for r in part}) != 1:
            raise ValueError('Family identity differs across arms: ' + case)
        if any('episode_id' in r for r in part):
            ids = [r.get('episode_id') for r in part]
            if not isinstance(ids[0], str) or not ids[0] or any(x != ids[0] for x in ids):
                raise ValueError('Episode identity differs or is incomplete across arms')
            if ids[0] in episodes:
                raise ValueError('Two cases use the same episode: ' + ids[0])
            episodes[ids[0]] = case
        ordered.extend(part)
        families.append(part[0]['family_id'])
    matrix = np.asarray([[by[c, a]['success'] for a in ARMS] for c in cases], dtype=np.float64)
    return ordered, cases, families, matrix


def success_values(arm_means):
    """Absolute success percentages and paired differences, never seed pooling."""
    a = np.asarray(arm_means, dtype=np.float64)
    diff = a[..., 1:] - a[..., :1]
    return 100 * np.concatenate((a, diff, diff.mean(axis=-1, keepdims=True)), axis=-1)


def family_resampled_means(success, families, family_ids, indices):
    """Ratio of sampled family outcome totals to sampled family case counts."""
    groups = [np.flatnonzero(np.asarray(families) == f) for f in family_ids]
    sizes = np.asarray([len(g) for g in groups], dtype=np.int64)
    totals = np.asarray([success[g].sum(axis=0) for g in groups], dtype=np.float64)
    if np.any(sizes == 0):
        raise ValueError('Empty family in resampling population')
    return totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None], sizes


def summarize_scheme(point, samples, clusters=None):
    q = np.quantile(samples, [0.025, 0.975, 0.0125, 0.9875], axis=0, method='linear')
    return {'status': 'COMPLETE', 'replicates': REPLICATES,
            'interval_method': 'percentile_numpy_quantile_linear',
            'conditional_on': 'fixed official model, TRAIN data and three fixed refit models; no resampling of training seeds',
            'formal_FWER_guarantee': False, 'finite_sample_guarantee': False,
            'cluster_count': clusters,
            'cluster_warning': ('ONE_CLUSTER_BOOTSTRAP_IS_DEGENERATE_NOT_INFORMATIVE' if clusters == 1 else
                                'FEW_CLUSTERS_SENSITIVITY_UNSTABLE' if clusters is not None and clusters < 20 else None),
            'metrics': {name: {'estimate': float(point[k]), 'ci95': [float(q[0, k]), float(q[1, k])],
                               'ci97_5_two_task_bonferroni_approximation': [float(q[2, k]), float(q[3, k])]}
                        for k, name in enumerate(BOOTSTRAP_COLUMNS)}}


def paired_counts(success):
    h0 = success[:, 0]
    result = []
    for i, arm in enumerate(ARMS[1:], 1):
        refit = success[:, i]
        result.append({'arm': arm, 'cases': len(success),
                       's01_H0_failure_REFIT_success': int(((h0 == 0) & (refit == 1)).sum()),
                       's10_H0_success_REFIT_failure': int(((h0 == 1) & (refit == 0)).sum()),
                       's11_both_success': int(((h0 == 1) & (refit == 1)).sum()),
                       's00_both_failure': int(((h0 == 0) & (refit == 0)).sum()),
                       'H0_successes': int(h0.sum()), 'REFIT_successes': int(refit.sum()),
                       'delta_success_pp': float(100 * (refit - h0).mean())})
    return result


def auxiliary_descriptions(rows):
    result = {}
    for arm in ARMS:
        part = [r for r in rows if r['arm'] == arm]
        metrics = {}
        for name in AUXILIARY_FIELDS:
            observed = [r.get(name) for r in part]
            if not any(v is not None for v in observed):
                continue
            finite, missing, nonfinite = [], 0, 0
            for v in observed:
                if v is None:
                    missing += 1
                elif not isinstance(v, numbers.Real):
                    raise ValueError('Auxiliary numeric metric has a nonnumeric value: ' + name)
                elif not np.isfinite(v):
                    nonfinite += 1
                else:
                    finite.append(float(v))
            metrics[name] = {'total_cases': len(part), 'finite_cases': len(finite), 'missing_cases': missing,
                             'nonfinite_cases': nonfinite, 'all_case_mean': float(np.mean(finite)) if len(finite) == len(part) else None,
                             'finite_subset_mean_DESCRIPTIVE_ONLY': float(np.mean(finite)) if finite else None,
                             'finite_subset_median_DESCRIPTIVE_ONLY': float(np.median(finite)) if finite else None,
                             'no_auxiliary_inferential_CI': True}
        result[arm] = metrics
    return result


def _atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def _csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _npz_bytes(**arrays):
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def analyze_task(task, rows, output_dir=None):
    """Analyze complete paired CEM outcomes, optionally save all evidence.

    This never loads weights, simulates an environment, reads state labels,
    trains, or treats open-loop prediction errors as control outcomes.
    """
    rows = list(rows)
    ordered, cases, families, success = normalize_rows(task, rows)
    identity = hashlib.sha256(canonical_bytes({'version': VERSION, 'task': task, 'rows': ordered})).hexdigest()
    source_hash = file_sha256(__file__)
    directory = Path(output_dir) if output_dir is not None else None
    if directory is not None and (directory / 'STATISTICS_RECEIPT.json').exists():
        prior = json.loads((directory / 'STATISTICS_RECEIPT.json').read_text())
        if prior['input_sha256'] != identity or prior['source_sha256'] != source_hash:
            raise ValueError('Refusing to overwrite statistics with changed input or implementation')
        for name, record in prior['files'].items():
            path = directory / name
            if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
                raise ValueError('Existing statistical output changed: ' + name)
        return json.loads((directory / 'NUMERIC_RESULTS.json').read_text())
    indices = bootstrap_indices(task, len(cases))
    point = success_values(success.mean(axis=0))
    case_draws = success_values(success[indices].mean(axis=1))
    schemes = {'CASE': summarize_scheme(point, case_draws)}
    family_ids, family_sizes, family_indices = [], np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.int64)
    family_draws = np.empty((0, len(BOOTSTRAP_COLUMNS)), dtype=np.float64)
    if any(f is None or f == 'FAMILY_UNKNOWN' for f in families):
        schemes['FAMILY'] = {'status': 'UNAVAILABLE_FAMILY_UNKNOWN', 'case_estimator_preserved': True,
                             'reason': 'At least one true/inferred family identity is unknown; episode IDs are not asserted independent families.'}
    else:
        family_ids = sorted(set(families))
        family_indices = bootstrap_indices(task, len(family_ids), family=True)
        means, family_sizes = family_resampled_means(success, families, family_ids, family_indices)
        family_draws = success_values(means)
        schemes['FAMILY'] = summarize_scheme(point, family_draws, len(family_ids))
    counts = paired_counts(success)
    result = {'version': VERSION, 'status': 'COMPLETE', 'task': task, 'evaluation_kind': 'CLOSED_LOOP_CEM',
              'input_sha256': identity, 'source_sha256': source_hash, 'numpy_version': np.__version__,
              'cases': len(cases), 'arm_order': list(ARMS), 'case_order': cases, 'family_by_case': families,
              'mean_fixed_refit_success_percent_NOT_ENSEMBLE': float(100 * success[:, 1:].mean()),
              'family_order': family_ids, 'family_sizes': family_sizes.tolist(),
              'family_evidence': 'Identifiers may be inferred; no independent-family proof is implied. Preserve source provenance in raw rows.',
              'sample_scope': 'BELOW_PROTOCOL_MINIMUM_20_ANALYTIC_ONLY' if len(cases) < 20 else
                              'LIMITED_EVALUATION_SAMPLE' if len(cases) < 100 else 'FIXED_CASE_SAMPLE',
              'bootstrap': {'replicates': REPLICATES, 'rng': 'NumPy.Generator.PCG64',
                            'seed_salt': f'R3_BOOTSTRAP_20261002/{task}', 'case_seed': bootstrap_seed(task),
                            'family_seed_salt': f'R3_BOOTSTRAP_20261002/{task}/family', 'family_seed': bootstrap_seed(task, True),
                            'case_indices_sha256': array_sha256(indices),
                            'family_indices_sha256': array_sha256(family_indices) if family_ids else None,
                            'array_hash_encoding': 'canonical JSON shape/dtype then C-contiguous little-endian int64 bytes'},
              'main_effect_pp': float(point[-1]), 'paired_counts': counts, 'schemes': schemes,
              'auxiliary_descriptions': auxiliary_descriptions(ordered),
              'limitations': ['Intervals are finite-sample bootstrap approximations conditional on the fixed models.',
                             'The three refits are not independent pretrained worlds and do not triple the case count.',
                             'Both CASE and FAMILY sensitivity are retained; never select a narrower interval after seeing outcomes.',
                             '97.5% per-task intervals are an approximate Bonferroni presentation for two-task statements, not a strict FWER guarantee.',
                             'Unknown or inferred families do not establish independent novel-task generalization.',
                             'No max-stat method, adaptive stopping, extra fitting, or additional evaluation is used.'],
              'new_optimizer_updates': 0, 'new_CEM_trajectories': 0}
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        paired = []
        for i, case in enumerate(cases):
            row = {'case_id': case, 'family_id': families[i]}
            row.update({arm: int(success[i, j]) for j, arm in enumerate(ARMS)})
            row.update({arm + '_minus_H0': float(success[i, j] - success[i, 0]) for j, arm in enumerate(ARMS[1:], 1)})
            row['fixed_refit_mean_minus_H0'] = float(success[i, 1:].mean() - success[i, 0])
            paired.append(row)
        payloads = {'RAW_CASE_ROWS.json': canonical_bytes(rows), 'PAIRED_CASES.csv': _csv_bytes(paired),
                    'PAIRED_COUNTS.csv': _csv_bytes(counts), 'NUMERIC_RESULTS.json': canonical_bytes(result),
                    'BOOTSTRAP_INDICES.npz': _npz_bytes(case=indices, family=family_indices,
                                                       case_ids=np.asarray(cases), family_ids=np.asarray(family_ids, dtype=str)),
                    'BOOTSTRAP_DISTRIBUTIONS.npz': _npz_bytes(case=case_draws, family=family_draws,
                                                             columns=np.asarray(BOOTSTRAP_COLUMNS), point=point)}
        for name, payload in payloads.items():
            _atomic(directory / name, payload)
        receipt = {'version': VERSION, 'status': 'COMPLETE', 'task': task, 'input_sha256': identity,
                   'source_sha256': source_hash, 'numpy_version': np.__version__,
                   'files': {n: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for n, b in payloads.items()},
                   'cases': len(cases), 'arms': list(ARMS), 'bootstrap': result['bootstrap'],
                   'family_status': schemes['FAMILY']['status'],
                   'raw_nonfinite_json_encoding': 'Tagged __nonfinite_float__ object; no observations are dropped',
                   'new_optimizer_updates': 0, 'new_CEM_trajectories': 0}
        _atomic(directory / 'STATISTICS_RECEIPT.json', canonical_bytes(receipt))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--rows', type=Path, required=True, help='JSON array or JSONL of completed paired CEM rows')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.rows.read_text()
    rows = restore_json_value(json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(line) for line in raw.splitlines() if line.strip()])
    result = analyze_task(a.task, rows, a.output)
    print(json.dumps({'status': result['status'], 'task': a.task, 'cases': result['cases'],
                      'main_effect_pp': result['main_effect_pp'], 'output': str(a.output)}))


if __name__ == '__main__':
    main()
