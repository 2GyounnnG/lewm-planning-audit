"""Validate X1 repeated measurements and R4 secondary coverage on local CPU."""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from deliver import atomic, record

ARMS = ('H0', 'REFIT_103201', 'REFIT_103202', 'REFIT_103203')
STREAMS = ('R3_ORIGINAL', 'R4_ALT_CEM_1', 'R4_ALT_CEM_2')


def read(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def check(value, message):
    if not value:
        raise ValueError(message)


def compare(observed, expected):
    # csv serializes both None and a missing value as an empty cell.
    def value(v):
        return '' if v is None else str(v)
    left = sorted(json.dumps({k: value(v) for k, v in row.items()}, sort_keys=True) for row in observed)
    right = sorted(json.dumps({k: value(v) for k, v in row.items()}, sort_keys=True) for row in expected)
    check(left == right, 'Frozen CPU reconstruction differs')


def x1(root, task):
    folder = root / f'x1/reports/{task}_RANDOMNESS'
    status = json.loads((folder / 'MODULE_STATUS.json').read_text())
    limited = task == 'cube'
    check(status['status'] == ('COMPLETE_WITH_TECHNICAL_LIMITATIONS' if limited else 'COMPLETE'), 'Module incomplete')
    files = [record(folder / 'MODULE_STATUS.json')]
    for name, entry in status['tables'].items():
        actual = record(folder / name)
        check(all(actual[k] == entry[k] for k in ('bytes', 'sha256')), 'Recovered table hash mismatch')
        files.append(actual)
    raw = read(folder / 'ALL_RAW_VALUES.csv')
    cases = sorted({row['case_id'] for row in read(root / f'x1/reports/{task}_MAIN/ALL_RAW_VALUES.csv')})
    keys = [(r['case_id'], r['arm'], r['stream']) for r in raw]
    check(len(cases) == 100 and len(keys) == len(set(keys)) == 1200, 'Fixed100 x4 x3 coverage')
    check(set(keys) == {(c, a, s) for c in cases for a in ARMS for s in STREAMS}, 'Case/arm/stream matrix differs')
    random = read(folder / 'PLANNING_RANDOMNESS_TABLE.csv')
    learning = read(folder / 'S3_LEARNING_MAIN_TABLE.csv')
    check(len(random) == 12 and len(learning) == 15, 'Summary shape mismatch')
    if limited:
        check(status.get('reason') == 'RESET_FALLBACK_EXACT_CHECK_FAILED', 'Wrong technical gate')
        check(status['observed_closed_loop_trajectories'] == 0 and status['observed_TECH_P90_seconds'] is None, 'Unobserved Cube timing/outcomes invented')
        check(all(r['observation_status'] == 'MISSING' and r['missing_reason'] == status['reason'] and not r['success'] for r in raw), 'Cube missingness not explicit')
        for row in random + learning:
            check(row['status'] == 'TECHNICALLY_UNEVALUABLE' and int(row['cases']) == 0 and int(row['universe_cases']) == 100, 'Technical summary denominator')
            check(all(not row[k] for k in ('estimate', 'conditional95_low', 'conditional95_high', 'difference_vs_reference', 'difference95_low', 'difference95_high')), 'Unobserved technical estimate invented')
    else:
        from analysis.aggregate import summarize
        check(status['original_reused'] == 400 and status['additional'] == 800 and status['new_optimizer_updates'] == 0, 'X1 secondary budget differs')
        values = {(r['case_id'], r['arm'], r['stream']): float(r['success']) for r in raw}
        check(all(v in (0, 1) for v in values.values()), 'Nonboolean success outcome')
        main = read(root / 'x1/reports/tworoom_MAIN/ALL_RAW_VALUES.csv')
        original = [r for r in main if r['evaluation_kind'] == 'CLOSED_LOOP_CEM' and r['arm'] in ARMS]
        check(len(original) == 400 and all(values[r['case_id'], r['arm'], 'R3_ORIGINAL'] == float(r['success']) for r in original), 'Original400 changed')
        family = {c: None for c in cases}
        expected_random, expected_learning = [], []
        for arm in ARMS:
            v = {(c, s): [values[c, arm, s]] for c in cases for s in STREAMS}
            expected_random.extend(summarize(task, cases, family, v, STREAMS, 'success_fraction', {'policy': arm, 'scope': 'SAME_MODEL_DIFFERENT_PLANNER_STREAM'}))
        for stream in STREAMS:
            v = {(c, a): [values[c, a, stream]] for c in cases for a in ARMS}
            expected_learning.extend(summarize(task, cases, family, v, ARMS, 'success_fraction', {'scope': 'LEARNING_POLICIES_ALL100', 'stream': stream}))
        compare(random, expected_random)
        compare(learning, expected_learning)
    return {'status': 'COMPLETE_WITH_TECHNICAL_LIMITATIONS' if limited else 'COMPLETE',
            'reason': status.get('reason') if limited else None, 'task': task,
            'fixed_cases': 100, 'logical_rows': 1200, 'observed_rows': 0 if limited else 1200,
            'randomness_rows': 12, 'learning_rows': 15,
            'frozen_statistics_reproduced_from_raw': not limited,
            'explicit_missingness_verified': limited, 'inputs': files}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--task', required=True, choices=('tworoom', 'cube'))
    a = p.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root))
    result = x1(root, a.task)
    atomic(root / f'manifests/secondary_validation/x1_{a.task}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}))
