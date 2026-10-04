"""Verify completed local delivery chains, then seal their explicit recovery scope.

No simulation, training, new statistical estimates, remote deletion, or instance
release. The specification lists required completed line receipts; file hashes
are verified again locally before the root seal can be written.
"""
import argparse
import datetime
import json
from pathlib import Path
from deliver import atomic, record


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected):
    actual = record(path)
    for field in ('bytes', 'sha256'):
        if field in expected:
            require(actual[field] == expected[field], f'{field} mismatch: {path}')
    return actual


def finalize(root, specpath):
    root = Path(root).resolve()
    specpath = Path(specpath).resolve()
    spec = read(specpath)
    receipts = []
    files = {}
    cpu_failures = []
    def add(path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        # A repeated input needs only one byte read, but every expected SHA is
        # checked. Relative paths inside line manifests resolve separately.
        key = str(path.resolve())
        rec = files.get(key)
        if rec is None:
            rec = verify(path, expected or {})
            files[key] = rec
        elif expected:
            for k in ('bytes', 'sha256'):
                if k in expected:
                    require(rec[k] == expected[k], f'Conflicting hash: {path}')
        return rec

    for item in spec['required_receipts']:
        path = root / item['path']
        data = read(path)
        require(data.get('status') in item['accepted_statuses'],
                f'Incomplete required receipt: {item["path"]}')
        receipts.append({'role': item['role'], 'status': data['status'], **add(path)})
        mode = item.get('audit')
        if mode == 'merge':
            require(data.get('source_values_unchanged') is True, 'Nonlossless merge')
            for key in ('inputs', 'gates', 'outputs'):
                for entry in data[key]:
                    add(entry['path'], entry)
        elif mode == 'x1_local_files':
            require(data['new_optimizer_updates'] == 0, 'Recovery trained a model')
            cpu = read(data['CPU_recovery']['path'])
            require(cpu['cases'] == 100 and cpu['arms'] == 4 and cpu['new_optimizer_updates'] == 0,
                    'X1 CPU recovery scope differs from fixed100 x4')
            if data['status'] == 'COMPLETE_WITH_CPU_TOLERANCE_FAILURE':
                require(data.get('strict_cpu_gate_passed') is False and data.get('release_gate') == 'HOLD',
                        'CPU tolerance failure cannot be relabeled as an acceptance pass')
                require(cpu['status'] == 'FAILED_FROZEN_CPU_TOLERANCE', 'Missing original CPU failure receipt')
                cpu_failures.append({'task': data['task'], 'receipt': add(path),
                                     'reason': 'FAILED_ORIGINAL_FROZEN_CPU_REPLAY_TOLERANCE'})
            else:
                require(data['status'] == 'PASS', 'Unknown X1 recovery status')
                require(cpu['status'] == 'PASS', 'An X1 PASS seal requires an actual strict CPU PASS receipt')
            for local, entry in data['local_files'].items():
                add(local, entry)
            add(data['CPU_recovery']['path'], data['CPU_recovery'])
            previous = data.get('previous_seal')
            seen = set()
            while previous:
                oldpath = previous['path']
                require(oldpath not in seen, 'Cycle in X1 recovery revision chain')
                seen.add(oldpath)
                add(oldpath, previous)
                old = read(oldpath)
                require(old['status'] in ('PASS', 'COMPLETE_WITH_CPU_TOLERANCE_FAILURE'),
                        'Unknown prior X1 seal status')
                previous = old.get('previous_seal')
        elif mode == 'r4_inventory':
            require(data['s2_initial_forks'] == 100 and data['s2_logical_candidate_rows'] == 6400,
                    'R4 initial menu count mismatch')
            for rel, expected in data['per_file_SHA_manifests'].items():
                manifest = path.parent / rel
                add(manifest, expected)
                for relfile, entry in read(manifest)['files'].items():
                    add(manifest.parent / relfile, entry)
            for key in ('archive_manifest', 'trajectory_count_table'):
                entry = data[key]
                # R4 file_record may omit a path; the containing inventory gives
                # an unambiguous conventional location for these receipts.
                name = 'RECOVERY_MANIFEST.json' if key == 'archive_manifest' else 'MINIMUM_TRAJECTORY_COUNTS_RAW.csv'
                add(path.parent / name, entry)
            add(data['CPU_restore_receipt']['path'], data['CPU_restore_receipt'])
        elif mode == 'r4_secondary':
            require(data['fixed_cases'] == 20, 'R4 secondary fixed20 changed')
            require(data['batch'] == 'SECONDARY_ONLY_PRIMARY_ARCHIVES_UNCHANGED',
                    'R4 main and secondary evidence must remain separate')
            base = path.parent / 'recovery'
            for relative, expected in data['files'].items():
                local = (base / relative).resolve()
                require(local.is_relative_to(base.resolve()), 'Secondary path escapes recovery directory')
                add(local, expected)
        elif mode == 'x2':
            require(data['completed_models'] == 324 and data['formal_predictor_jobs'] == 648
                    and data['closed_heads_recovered'] == 2592, 'X2 incomplete coverage')
            require(data['encoder_updates'] == 0 and data['readout_fits'] == 0, 'X2 freeze violation')
            for name, digest in data['main_result_sha256'].items():
                add(path.parent / name, {'sha256': digest})
            for arm, digest in data['module_seals'].items():
                modulepath = path.parent / (arm + '_RECOVERY_SEAL.json')
                add(modulepath, {'sha256': digest})
                module = read(modulepath)
                require(module['status'] == 'MODULE_RECOVERY_VERIFIED', 'X2 module unverified')
                add(root / 'x2/recover_outputs.py', {'sha256': module['verification_source_sha256']})
                base = path.parent / 'recovery_compact'
                manifestpath = base / (arm + '_RECOVERY_MANIFEST.json')
                add(manifestpath, {'sha256': module['derived_manifest_sha256']})
                add(base / (arm + '_CPU_RESTORATION.json'), {'sha256': module['CPU_restoration_sha256']})
                for entry in read(manifestpath)['records']:
                    add(base / entry['derived_path'], {'sha256': entry['derived_sha256']})
                    add(base / entry['input_check_path'], {'sha256': entry['input_check_sha256']})

    freeze = root / 'manifests/statistics_full_v3/STATISTICS_FREEZE.json'
    add(freeze)
    for name, digest in read(freeze)['files'].items():
        add(root / 'analysis' / name, {'sha256': digest})
        add(freeze.parent / 'source' / name, {'sha256': digest})
    index = root / 'manifests/DELIVERY_INDEX.json'
    add(index)
    for module in read(index)['modules'].values():
        require(len(module['conclusions']) <= 5, 'Module conclusion length exceeds contract')
        for entry in module['files']:
            add(entry['path'], entry)
    for rel in spec['additional_files']:
        add(root / rel)
    add(specpath)
    add(__file__)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    recovery = {'status': 'FILES_COMPLETE_CPU_GATE_FAILED' if cpu_failures else 'COMPLETE_WITH_EXPLICIT_RECOVERY_SCOPE', 'protocol': 'R4-v2.3',
                'verified_utc': stamp, 'line_receipts': receipts,
                'verified_local_files': list(files.values()), 'file_count': len(files),
                'restoration_scope': spec['restoration_scope'],
                'strict_CPU_acceptance_failures': cpu_failures,
                'remote_originals_retained': True, 'remote_deletion_performed': False}
    atomic(root / 'RECOVERY_MANIFEST.json', recovery)
    gate = {'status': 'HOLD_CPU_RECOVERY_TOLERANCE_FAILURE' if cpu_failures else 'PASS_WITH_DOCUMENTED_TECHNICAL_LIMITATIONS',
            'scientific_delivery_complete': True, 'required_local_file_recovery_complete': True,
            'all_frozen_CPU_acceptance_gates_passed': not cpu_failures,
            'instance_release_ready': not cpu_failures,
            'strict_CPU_acceptance_failures': cpu_failures,
            'technical_limitations': spec['technical_limitations'],
            'CPU_acceptance_scope': spec['restoration_scope'],
            'no_new_neural_updates_during_recovery': True,
            'automatic_release_or_destroy': False, 'instance_release_performed': False,
            'recovery_manifest': record(root / 'RECOVERY_MANIFEST.json'), 'verified_utc': stamp}
    atomic(root / 'RELEASE_GATE.json', gate)
    seal = {'status': 'SEALED_WITH_CPU_RECOVERY_FAILURE' if cpu_failures else 'SEALED_WITH_DOCUMENTED_TECHNICAL_LIMITATIONS', 'protocol': 'R4-v2.3',
            'sealed_utc': stamp, 'recovery_manifest': record(root / 'RECOVERY_MANIFEST.json'),
            'release_gate': record(root / 'RELEASE_GATE.json'),
            'main_and_secondary_are_separate_immutable_deliveries': True,
            'instance_destroyed': False}
    atomic(root / 'SEAL.json', seal)
    return seal


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--spec', required=True)
    a = p.parse_args()
    print(json.dumps(finalize(a.root, a.spec), ensure_ascii=False))
