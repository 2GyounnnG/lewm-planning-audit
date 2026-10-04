#!/usr/bin/env python3
"""Validate recovered secondary evidence without simulation or model inference.

python -B ops/validate_r4_secondary.py --task pusht --r3-root /path/to/r3/recovery
Run separately for reacher. Frozen statistics are reconstructed in a fresh
ROOT/recomputed/secondary_<task>_<UTC>/ directory. Only the dedicated
manifests/secondary_validation/r4_<task>.json receipt is written outside it.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from recompute_delivery import FROZEN, atomic, check, load_helper, read_csv, read_json, record, same_csv


def truth(value):
    return str(value).lower() in ('true', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--r3-root', type=Path, required=True)
    parser.add_argument('--task', choices=('pusht', 'reacher'), required=True)
    args = parser.parse_args()
    root, r3, task = args.root.resolve(), args.r3_root.resolve(), args.task
    folder = root / f'r4/reports/secondary/{task}'
    recovery = folder / 'recovery'
    out = root / 'recomputed' / f'secondary_{task}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")}'
    out.mkdir(parents=True, exist_ok=False)
    gatepath = root / f'manifests/secondary_validation/r4_{task}.json'
    result = {'status': 'STARTED', 'task': task, 'batch': 'SECONDARY_ONLY_AFTER_PRIMARY_DELIVERY',
              'reason': None, 'output': str(out), 'new_simulator_calls': 0,
              'new_model_inference_calls': 0, 'optimizer_updates': 0,
              'scientific_definition_changes': 0, 'inputs': []}
    inputs = {}

    def verify(path, expected=None):
        path = Path(path).resolve()
        value = record(path)
        if expected:
            for key in ('bytes', 'sha256'):
                if key in expected:
                    check(value[key] == expected[key], f'{key} differs: {path}')
        inputs[str(path)] = value
        return value

    def exact_file_map(base, manifest):
        for relative, expected in manifest['files'].items():
            path = (base / relative).resolve()
            check(path.is_relative_to(base.resolve()), 'Manifest member escapes recovery root')
            verify(path, expected)

    try:
        import numpy as np
        verify(__file__)
        verify(root / 'ops/recompute_delivery.py')
        freeze_path = root / 'manifests/statistics_full_v3/STATISTICS_FREEZE.json'
        verify(freeze_path)
        freeze = read_json(freeze_path)
        check(freeze['files'] == FROZEN and freeze['replicates'] == 5000, 'Unexpected frozen statistics contract')
        for name, digest in FROZEN.items():
            verify(freeze_path.parent / 'source' / name, {'sha256': digest})
        package = types.ModuleType('_r4_secondary_frozen')
        package.__path__ = [str(freeze_path.parent / 'source')]
        sys.modules[package.__name__] = package
        aggregate = load_helper(freeze_path.parent / 'source/aggregate.py', package.__name__ + '.aggregate')
        roles_path = r3 / f'manifests/{task}_data_roles.json'
        verify(roles_path)
        roles = {row['case_id']: row for row in read_json(roles_path)['cases']['EVAL']}
        subset = set(aggregate.fixed_subset(r3, task))
        check(len(subset) == 20, 'Expected fixed metadata subset20')
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        ranked = sorted(roles.values(), key=lambda c: (digest(['R4_CASE_SUBSET_V1', task, c]), c['case_id']))
        if task == 'pusht':
            families = defaultdict(list)
            for case in ranked:
                families[case['family_id']].append(case)
            family_order = sorted(families, key=lambda f: digest(['R4_CASE_SUBSET_V1', task, 'family', f]))
            ordered = [families[f][i] for i in range(max(map(len, families.values()))) for f in family_order if i < len(families[f])][:20]
        else:
            ordered = ranked[:20]
        check({c['case_id'] for c in ordered} == subset, 'Ordered protocol subset differs from frozen source')
        for name in ('MODULE_STATUS.json', 'SECONDARY_TABLES_MANIFEST.json', 'SECONDARY_RECOVERY_MANIFEST.json',
                     'MIDDLE_FORK_AVAILABILITY_RAW.csv', 'MIDDLE_FORK_AVAILABILITY_COMPLETE.json',
                     'S2_SECONDARY_TABLE.csv', 'ALL_CANDIDATE_RAW.csv', 'SIM_CEM_ESTIMATE.json',
                     'SIM_CEM_SKIP_EVIDENCE.json', 'CPU/RECOVERY_CHECK.json'):
            verify(folder / name)
        module = read_json(folder / 'MODULE_STATUS.json')
        check(module['status'] in ('COMPLETE', 'COMPLETE_WITH_TECHNICAL_LIMITATIONS'), 'Secondary module incomplete')
        check(module['task'] == task and module['batch'] == 'SECONDARY_ONLY_AFTER_PRIMARY_DELIVERY', 'Wrong task or batch')
        check(module['replacement_cases'] == 0 and module['new_neural_training'] == 0, 'Replacement or training not allowed')
        primary = root / 'r4/PRIMARY_DELIVERY_RECEIPT.json'
        verify(primary, module['primary_delivery'])
        check(read_json(primary)['status'] == 'MAIN_RESULTS_DELIVERED', 'Secondary preceded actual primary delivery')
        table_manifest = read_json(folder / 'SECONDARY_TABLES_MANIFEST.json')
        check(table_manifest['task'] == task and table_manifest['batch'] == 'SECONDARY_ONLY', 'Wrong tables manifest')
        exact_file_map(folder, table_manifest)
        recovery_manifest = read_json(folder / 'SECONDARY_RECOVERY_MANIFEST.json')
        verify(recovery / 'SECONDARY_RECOVERY_MANIFEST.json', record(folder / 'SECONDARY_RECOVERY_MANIFEST.json'))
        check(recovery_manifest['status'] == 'SEALED_SECONDARY_ORIGINAL_OUTPUTS' and recovery_manifest['task'] == task,
              'Original secondary recovery not sealed')
        check(recovery_manifest['fixed_cases'] == 20, 'Recovery subset count differs')
        exact_file_map(recovery, recovery_manifest)
        # The portable top-level tables must be byte-identical to the copies
        # sealed alongside original second-batch outputs.
        for name in table_manifest['files']:
            verify(folder / name, record(recovery / 'secondary_reports' / name))

        available = read_csv(folder / 'MIDDLE_FORK_AVAILABILITY_RAW.csv')
        check(len(available) == 20 and {r['case_id'] for r in available} == subset, 'Availability must retain exactly fixed20 cases')
        availability_receipt = read_json(folder / 'MIDDLE_FORK_AVAILABILITY_COMPLETE.json')
        check(availability_receipt['status'] == 'COMPLETE' and availability_receipt['observed_results'] == 20,
              'Availability ledger incomplete')
        verify(folder / 'MIDDLE_FORK_AVAILABILITY_RAW.csv', availability_receipt['raw'])
        verify(roles_path, availability_receipt['source_role_manifest'])
        stop_path = root / f'r4/reports/remote_{task}_source_stop_flags/S1_SOURCE_STOP_FLAGS_RAW.csv'
        verify(stop_path)
        stops = {r['case_id']: r for r in read_csv(stop_path) if r['source_arm'] == 'H0' and r['stream'] == 'R3_ORIGINAL'}
        candidate_rows = read_csv(folder / 'ALL_CANDIDATE_RAW.csv')
        grouped = defaultdict(list)
        for row in candidate_rows:
            check(row['case_id'] in subset and row['task'] == task and row['fork'] == 'middle_raw10', 'Candidate outside fixed middle subset')
            grouped[row['case_id']].append(row)
        verified_rows = []
        for row in available:
            case = row['case_id']
            check(row['task'] == task and row['fork'] == 'middle_raw10' and truth(row['fixed_metadata_case_no_replacement']), 'Availability scope differs')
            check(row['family_id'] == (roles[case].get('family_id') or ''), 'Metadata family label differs')
            source = stops[case]
            for left, right in (('source_executed_raw_steps', 'source_executed_raw_steps'),
                                ('source_world_terminated_raw', 'first_world_terminated_raw'),
                                ('source_world_truncated_raw', 'first_world_truncated_raw'),
                                ('source_trajectory_sha256', 'source_trajectory_sha256')):
                check(row[left] == source[right], 'Original H0 prefix/stop provenance differs')
            casefolder = recovery / 's2_middle' / case
            verify(casefolder / 'result.json', {'sha256': row['middle_result_sha256']})
            complete = read_json(casefolder / 'COMPLETE.json')
            verify(casefolder / 'COMPLETE.json')
            exact_file_map(casefolder, complete)
            raw = read_json(casefolder / 'result.json')
            check(raw['status'] == row['status'] and raw['case_id'] == case, 'Availability result differs')
            check(not raw['replacement_case'], 'Unavailable fixed case was replaced')
            if raw['status'] == 'UNAVAILABLE_MID_FORK':
                check(not truth(row['fork_available']) and bool(row['missing_reason']) and not grouped[case], 'Missing fork gained scores or lacks reason')
                check(not (casefolder / 'candidate_raw.csv').exists(), 'Unavailable fork must not have a candidate measurement table')
                if task == 'reacher':
                    check(row['missing_reason'] == 'TASK_FIXED25_TECHNICALLY_UNEVALUABLE_TRUE_DMC_LAST', 'Wrong frozen true-LAST reason')
                    check(all(raw[k] == 0 for k in ('new_simulator_calls', 'new_CEM_calls', 'optimizer_updates')), 'Reacher technical branch made new calls')
                    verify(folder / 'SIM_TECHNICAL_LIMITATION.json', raw['technical_gate'])
                else:
                    check(row['missing_reason'] == 'INSUFFICIENT_EXECUTED_PREFIX' and int(row['source_executed_raw_steps']) < 10,
                          'Unavailable PushT fork is inconsistent with actual prefix')
                continue
            check(task == 'pusht' and truth(row['fork_available']), 'Unexpected measured Reacher middle fork')
            check(int(row['source_executed_raw_steps']) >= 10, 'No original raw10 prefix')
            rows = read_csv(casefolder / 'candidate_raw.csv')
            check(len(rows) == len(grouped[case]) == 64 and len({r['id'] for r in rows}) == 64, 'Each observed fork requires all64 logical candidates')
            check({int(r['logical_index']) for r in rows} == set(range(64)), 'Logical candidate index coverage differs')
            check(sum(r['source'] == 'INITIAL_DISTRIBUTION' for r in rows) == 16, 'Initialization candidates were selected or omitted')
            check(sorted(json.dumps(r, sort_keys=True) for r in rows) == sorted(json.dumps(r, sort_keys=True) for r in grouped[case]), 'Combined original values differ')
            lock = read_json(casefolder / 'MENU_LOCK.json')
            check(lock['locked_before_truth'] and lock['identity']['fork_raw'] == 10, 'Menu not locked at fixed raw10')
            check(lock['identity']['source_trajectory']['sha256'] == source['source_trajectory_sha256'], 'Menu identity source differs')
            with np.load(casefolder / 'menu.npz', allow_pickle=False) as menu:
                check(menu['raw_actions'].shape[0] == 64 and len(menu['prefix']) == 10, 'Recovered menu/prefix shape differs')
                for r in rows:
                    digest = hashlib.sha256(menu['raw_actions'][int(r['logical_index'])].tobytes(order='C')).hexdigest()
                    check(digest == r['raw_action_sha256'], 'Candidate action SHA differs')
            valid = sum(truth(r['valid_fixed_horizon']) for r in rows)
            check(valid == int(row['valid_fixed_horizon_candidates']) == raw['valid_fixed_horizon_candidates'], 'Candidate valid count differs')
            check(int(row['logical_candidates_observed']) == raw['logical_candidates'] == 64, 'Logical denominator differs')
            for r in rows:
                if truth(r['valid_fixed_horizon']):
                    check(int(r['candidate_raw_steps']) == 25 and all(math.isfinite(float(r[k])) for k in ('J_sim_lat', 'J_sim_task', 'J_sim_task_terminal')), 'Invalid fixed25 truth')
                else:
                    check(bool(r['missing_reason']) and all(r[k] == '' for k in ('J_sim_lat', 'J_sim_task', 'J_sim_task_terminal')), 'Missing future was filled')
            verified_rows.extend(rows)
        if task == 'reacher':
            check(not candidate_rows and not read_csv(folder / 'S2_SECONDARY_TABLE.csv'), 'Reacher technical task must contain no measured or zero-filled rows')
            check(module['recorded_candidate_rows'] is None, 'Technical candidate count must be null, not a measured zero')
        else:
            check(module['recorded_candidate_rows'] == len(candidate_rows), 'Module candidate count differs')

        aggregate.s2(recovery / 's2_middle', r3, task, out)
        same_csv(out / 'S2_SECONDARY_TABLE.csv', folder / 'S2_SECONDARY_TABLE.csv')
        check(not read_csv(out / 'S2_MAIN_TABLE.csv'), 'Middle forks must not produce primary estimates')
        if task == 'pusht':
            goal_values = defaultdict(list)
            for row in read_csv(out / 'S2_FORK_RAW_METRICS.csv'):
                goal_values[row['case_id'], row['task_time_reduction']].append(row['D_goal'])
            check(all(len(values) == 4 and len(set(values)) == 1 for values in goal_values.values()), 'D_goal should be common across model rankings on the same menu')
            result['D_goal_numeric_interpretation'] = {
                'all_four_original_model_values_exactly_equal_per_case_and_time': True,
                'model_difference_by_definition': 0,
                'fixed3_difference_and_interval_are_preserved_unmodified': True,
                'note': 'Tiny fixed-three-minus-H0 differences arise from floating-point summation of three identical values. A tiny CI excluding zero is not evidence of a model D_goal effect.'}
        result['auxiliary_initial_scope_status'] = 'Generic frozen s2 emits empty S2_MAIN_TABLE/PARTIAL S2_STATUS when only middle forks are supplied; these do not govern secondary completion.'

        estimate = read_json(folder / 'SIM_CEM_ESTIMATE.json')
        skip = read_json(folder / 'SIM_CEM_SKIP_EVIDENCE.json')
        verify(folder / 'SIM_CEM_ESTIMATE.json', skip['original_estimate'])
        original_estimate = root / ('r4/reports/remote_pusht_reproduction/SIM_CEM_ESTIMATE.json' if task == 'pusht' else 'r4/reports/remote_reacher_tech/SIM_CEM_ESTIMATE.json')
        verify(original_estimate, skip['original_estimate'])
        check(skip['status'] == 'NOT_EXECUTED' and skip['new_full_CEM_simulator_calls'] == 0 and module['SIM_CEM'] == 'ESTIMATE_ONLY_NOT_RUN', 'SIM_CEM unexpected execution')
        if task == 'pusht':
            timings = estimate['raw_timing_rows']
            check(len(timings) == estimate['timing_queries'] == 20, 'Expected20 preserved complete branch probes')
            check(all(r['candidate_raw'] == 25 and r['valid_fixed_horizon'] for r in timings), 'Probe fixed25 scope differs')
            p90 = float(np.quantile([r['end_to_end_branch_render_encode_seconds'] for r in timings], .9))
            close = lambda a, b: math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
            check(close(p90, estimate['p90_end_to_end_seconds']) and close(p90, skip['p90_seconds_per_complete_candidate_branch']), 'P90 does not reproduce raw timing')
            check((skip['cases'], skip['max_replans_per_case'], skip['total_replans_per_arm'], skip['queries_per_replan'], skip['per_arm_total_branch_queries']) == (20, 2, 40, 9000, 360000), 'SIM_CEM query/replan budget differs')
            serial = 360000 * p90 / 3600
            for key, expected in (('serial_work_hours_per_arm', serial), ('both_arms_serial_work_hours', 2 * serial),
                                  ('two_worker_ideal_wall_hours_per_arm', serial / 2), ('two_worker_ideal_wall_hours_both_arms_total', serial)):
                check(close(skip[key], expected), 'SIM_CEM work/wall estimate differs: ' + key)
            check(serial > 2 and not skip['qualified'] and skip['reason'] == 'ESTIMATED_FULL_SIM_CEM_TIME_EXCEEDS_TWO_HOURS', 'SIM_CEM skip does not satisfy2hour gate')
            result['SIM_CEM_independent_check'] = {'p90_seconds': p90, 'queries_per_arm': 360000, 'both_arms_two_worker_ideal_hours': serial, 'executed': False}
        else:
            check(estimate['status'] == 'TECHNICALLY_UNEVALUABLE_FIXED_HORIZON' and estimate['new_full_CEM_simulator_calls'] == 0,
                  'Reacher SIM_CEM lacks original technical gate')
            check(skip['reason'] == 'TRUE_DMC_LAST_PREVENTS_COMMON_FIXED25_OBJECTIVE', 'Wrong Reacher SIM_CEM reason')
            check('p90_seconds_per_complete_candidate_branch' not in skip, 'Reacher must not invent branch timing')
            result['SIM_CEM_independent_check'] = {'status': 'TECHNICALLY_UNEVALUABLE_FIXED_HORIZON', 'executed': False, 'p90_seconds': None}
        cpu = read_json(folder / 'CPU/RECOVERY_CHECK.json')
        if task == 'pusht':
            check(cpu['status'] == 'COMPLETE_RECOVERY_COMPARISON' and cpu['scope'] == 'FIRST_METADATA_MIDDLE_FORK_64CANDIDATES_4MODELS', 'CPU comparison scope differs')
            check(cpu['case_id'] == ordered[0]['case_id'] and cpu['candidate_model_cells'] == 256 and cpu['new_simulator_steps'] == cpu['optimizer_updates'] == 0,
                  'CPU fixed fork or call count differs')
            check({r['arm'] for r in cpu['model_checks']} == set(aggregate.ARMS) and all(r['parameters_and_buffers_unchanged'] for r in cpu['model_checks']), 'CPU model freeze checks incomplete')
            verify(folder / 'CPU/RECOVERED_MIDDLE_CPU_RAW.csv', cpu['raw_table'])
            check(len(read_csv(folder / 'CPU/RECOVERED_MIDDLE_CPU_RAW.csv')) == 256, 'CPU comparison raw count differs')
        else:
            check(cpu['status'] == 'NOT_APPLICABLE_TECHNICALLY_UNEVALUABLE', 'Reacher must not claim new CPU rescoring')
            check(cpu['fixed_unavailable_cases'] == 20 and cpu['new_simulator_steps'] == cpu['optimizer_updates'] == 0 and cpu['new_prediction_score_cells'] is None,
                  'Reacher technical CPU scope must be empty, not invented scores')
            verify(folder / 'SIM_TECHNICAL_LIMITATION.json', cpu['technical_gate'])
            verify(cpu['primary_model_recovery_receipt']['path'], cpu['primary_model_recovery_receipt'])
        local_receipt = read_json(folder / 'LOCAL_RECOVERY_RECEIPT.json')
        verify(folder / 'LOCAL_RECOVERY_RECEIPT.json')
        check(local_receipt['status'] == 'COMPLETE_LOCAL_RECOVERY_AND_SHA_VALIDATION' and Path(local_receipt['recovery_root']).resolve() == recovery,
              'Local incremental recovery root/status differs')
        verify(folder / 'SECONDARY_RECOVERY_MANIFEST.json', local_receipt['remote_original_manifest'])
        verify(folder / 'CPU/RECOVERY_CHECK.json', local_receipt['CPU_scope_receipt'])
        reasons = sorted({r['missing_reason'] for r in available if r['missing_reason']})
        reasons.append(skip['reason'])
        result.update(status='COMPLETE_WITH_TECHNICAL_LIMITATIONS' if task == 'reacher' or any(not truth(r['fork_available']) for r in available) or any(not truth(r['valid_fixed_horizon']) for r in candidate_rows) else 'COMPLETE',
                      reason='; '.join(reasons), fixed_cases=20, availability_rows=20, measured_forks=sum(bool(rows) for rows in grouped.values()),
                      logical_candidates=len(candidate_rows) if candidate_rows else None, original_recovery_member_files=len(recovery_manifest['files']),
                      frozen_secondary_table_all_fields_equal=True, primary_and_secondary_not_pooled=True,
                      missing_future_not_zero_imputed=True, CPU_scope=cpu.get('scope'), CPU_receipt=record(folder / 'CPU/RECOVERY_CHECK.json'))
        for original in inputs.values():
            check(record(original['path']) == original, 'Input changed during validation: ' + original['path'])
        result['outputs'] = [record(p) for p in sorted(out.glob('*')) if p.is_file()]
        result['source_files_unchanged'] = True
        return 0
    except Exception as error:
        result.update(status='FAILED', reason=f'{type(error).__name__}: {error}')
        return 1
    finally:
        result['inputs'] = list(inputs.values())
        result['verified_utc'] = datetime.now(timezone.utc).isoformat()
        atomic(out / 'VALIDATION_RECEIPT.json', result)
        atomic(gatepath, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('inputs', 'outputs')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
