"""R3 external-recovery CPU acceptance; no optimizer, SSH, GPU or environment run.

All required scientific files are SHA/byte exact. CPU/GPU prediction comparison
uses only the prospectively frozen1e-5 absolute/relative tolerance. Original bulk
HDF5 can remain source-restorable through its exact archive/member SHA chain.
Failed attempts preserve evidence and never create a release gate or seal.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, math, os, stat, sys, time
from pathlib import Path, PurePosixPath
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.dont_write_bytecode = True

DEFAULT_ROOT = Path('/Volumes/MyProj/r3_official_lewm_predictor_refit/recovery')
MOUNT = Path('/Volumes/MyProj')
DEVICE = 16777237
TASKS = ('pusht', 'reacher')
SEEDS = (103201, 103202, 103203)
STEPS = (3000, 10000, 30000)
ATOL = RTOL = 1e-5
REPORTS = ('INTAKE_AND_INSTANCE_REUSE.md', 'OFFICIAL_SOURCE_AND_PROTOCOL_AUDIT.md',
           'FREEZE_AND_CACHE_VALIDATION.md', 'BASELINE_TECHNICAL_REPRODUCTION.md',
           'TRAINING_COMPLETION.md', 'OPEN_LOOP_RESULTS.md', 'CEM_PAIRED_RESULTS.md',
           'COMPUTE_STORAGE_AND_ETA.md', 'FINAL_SCIENTIFIC_REPORT_ZH.md')


def read(path): return json.loads(Path(path).read_text())
def canonical(obj): return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
def digest(obj):
    # Match the frozen worker/open-loop identity encoder, including ASCII escapes.
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def signature(path):
    s = Path(path).stat(); return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_path(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or any(ord(c) < 32 for c in name): raise ValueError('Unsafe evidence path')
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink(): raise ValueError('Symlink in recovered evidence path: ' + name)
    if not current.resolve().is_relative_to(root.resolve()): raise ValueError('Recovered evidence escaped its root')
    return current


def external_guard(root):
    if root != DEFAULT_ROOT or MOUNT.is_symlink() or not MOUNT.is_mount() or MOUNT.stat().st_dev != DEVICE:
        raise RuntimeError('Required actual MyProj device16777237/recovery root unavailable; no internal fallback')
    current = MOUNT
    for part in root.relative_to(MOUNT).parts:
        current = current / part; s = current.lstat()
        if stat.S_ISLNK(s.st_mode) or not stat.S_ISDIR(s.st_mode) or s.st_dev != DEVICE: raise RuntimeError('External recovery ancestor/device changed')
    return {'mount': str(MOUNT), 'root': str(root), 'device': DEVICE, 'no_internal_fallback': True}


def budget_bounds(ledger):
    lower = upper = 0
    for run_id, entry in ledger.get('runs', {}).items():
        reserved = entry['reserved_updates']; actual = entry.get('actual_updates')
        if type(reserved) is not int or not 0 <= reserved <= 1024: raise ValueError('Invalid technical reservation: ' + run_id)
        if actual is None:
            known = entry.get('detail', {}).get('known_successful_updates', 0)
            if type(known) is not int or not 0 <= known <= reserved: raise ValueError('Invalid uncertain technical lower bound')
            lower += known; upper += reserved
        else:
            if type(actual) is not int or not 0 <= actual <= reserved: raise ValueError('Invalid actual technical count')
            lower += actual; upper += actual
    if upper > 1024: raise ValueError('Technical optimizer upper bound exceeds1024')
    occupied = sum(x['status'] != 'INFRASTRUCTURE_FAILURE_INCOMPLETE' for x in ledger.get('trajectories', {}).values())
    if occupied > 16: raise ValueError('Conservative full technical trajectory count exceeds16')
    return {'optimizer_known_lower_bound': lower, 'optimizer_charged_upper_bound': upper,
            'conservative_full_technical_trajectories': occupied, 'limit_updates': 1024, 'limit_full_trajectories': 16}


def trajectory_metrics(result, arrays):
    import numpy as np
    required = {'raw_actions', 'rewards', 'step_success', 'step_truncated', 'goal_error', 'physical_state', 'goal_state', 'environment_step_seconds'}
    if set(arrays) != required: raise ValueError('Trajectory array fields differ')
    n = result['executed_raw_steps']; task = result['task']; failure = result['failure_category']
    if type(n) is not int or not 0 <= n <= 50 or result['replan_calls'] != len(result['replans']) or not 0 <= result['replan_calls'] <= 2: raise ValueError('Trajectory budget/count differs')
    if arrays['raw_actions'].shape != (n, 2): raise ValueError('Raw executed action shape differs')
    for key in required - {'raw_actions', 'physical_state', 'goal_state'}:
        if arrays[key].shape != (n,): raise ValueError('Per-step array length differs: ' + key)
    if any(not np.isfinite(arrays[k]).all() for k in required): raise ValueError('Saved physical/action/timing trajectory has nonfinite entries')
    success = arrays['step_success'].astype(bool); trunc = arrays['step_truncated'].astype(bool)
    if not np.array_equal(arrays['step_success'], success) or not np.array_equal(arrays['step_truncated'], trunc): raise ValueError('Nonbinary success/truncation')
    if result['any_step_success'] != int(success.any()) or result['terminal_success'] != int(bool(n and success[-1])): raise ValueError('Saved success summaries differ from raw arrays')
    if failure is None and result['success'] != int(success.any()): raise ValueError('Official OR success differs from observed termination events')
    if failure == 'METHOD' and result['success'] != 0: raise ValueError('Method-failure policy must retain success0')
    if failure not in (None, 'METHOD'): raise ValueError('Infrastructure attempt cannot be included as a completed score')
    # Pinned SWM PushT's observation state is seven-dimensional (including
    # agent velocity); source eval_state uses the full vector for L2 distance.
    d = 7 if task == 'pusht' else 2
    if n:
        if arrays['physical_state'].shape != (n, d) or arrays['goal_state'].shape != (d,): raise ValueError('Saved physical state dimensions differ')
        # The producer calls scalar norm once per environment step. A vectorized
        # reduction can legitimately use a different floating-point order.
        errors = np.asarray([float(np.linalg.norm(row - arrays['goal_state'])) for row in arrays['physical_state']])
        if not np.array_equal(errors, arrays['goal_error']): raise ValueError('Goal errors do not exactly match recorded physical states')
        if result['final_goal_error'] != float(arrays['goal_error'][-1]): raise ValueError('Final goal error differs')
    elif failure != 'METHOD' or result['final_goal_error'] is not None: raise ValueError('Zero-step result must be an explicit method failure')
    if failure is None and not success.any() and not trunc.any() and n != 50: raise ValueError('Unexplained short unsuccessful trajectory')
    if result['environment_step_seconds'] != float(sum(arrays['environment_step_seconds'].tolist())): raise ValueError('Environment time does not sum saved step times')
    if result['planning_synchronized_wall_seconds'] != sum(r['synchronized_wall_seconds'] for r in result['replans']): raise ValueError('Planning time sum differs')
    return {'steps': n, 'success': result['success'], 'method_failure': failure == 'METHOD'}


def sampling_accounting(counts, steps, identity):
    """Reconstruct transition exposures from saved window counts, without labels."""
    import numpy as np
    fields = {'episode_ids', 'window_offsets', 'legal_starts', 'raw_offsets', 'raw_indices',
              'window_counts', 'episode_counts', 'raw_action_counts', 'macro_transition_counts'}
    if set(counts) != fields: raise ValueError('Sampling ledger fields differ')
    episodes = counts['episode_ids'].tolist(); wo = counts['window_offsets']; ro = counts['raw_offsets']
    for key in fields - {'episode_ids'}:
        if counts[key].ndim != 1 or counts[key].dtype != np.int64: raise ValueError('Sampling integer vector differs: ' + key)
    if episodes != identity['episode_ids'] or episodes != sorted(set(episodes)): raise ValueError('Sampling episode roster differs')
    for offsets in (wo, ro):
        if len(offsets) != len(episodes) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0): raise ValueError('Invalid sampling offsets')
    if len(counts['legal_starts']) != wo[-1] or len(counts['window_counts']) != wo[-1] or len(counts['episode_counts']) != len(episodes): raise ValueError('Sampling window lengths differ')
    if any(len(counts[k]) != ro[-1] for k in ('raw_indices', 'raw_action_counts', 'macro_transition_counts')): raise ValueError('Sampling raw exposure lengths differ')
    for key, multiplier in (('window_counts', 1), ('episode_counts', 1), ('raw_action_counts', 15), ('macro_transition_counts', 3)):
        if int(counts[key].sum()) != steps * 128 * multiplier or np.any(counts[key] < 0): raise ValueError('Sampling total differs: ' + key)
    legal = {}; raw_expected = np.zeros(int(ro[-1]), dtype=np.int64); macro_expected = raw_expected.copy()
    for e, episode in enumerate(episodes):
        starts = counts['legal_starts'][wo[e]:wo[e+1]]; weights = counts['window_counts'][wo[e]:wo[e+1]]; n = int(ro[e+1]-ro[e])
        if not np.array_equal(counts['raw_indices'][ro[e]:ro[e+1]], np.arange(n)) or len(np.unique(starts)) != len(starts) or np.any(starts < 0) or np.any(starts + 20 > n): raise ValueError('Sampling legal raw window identities differ')
        if int(weights.sum()) != counts['episode_counts'][e]: raise ValueError('Sampling per-episode exposure differs')
        legal[episode] = starts.tolist(); origins = ro[e] + starts
        for offset in range(15): np.add.at(raw_expected, origins + offset, weights)
        for offset in (0, 5, 10): np.add.at(macro_expected, origins + offset, weights)
    if digest(legal) != identity['legal_starts_sha256']: raise ValueError('Sampling legal-start identity changed')
    if not np.array_equal(raw_expected, counts['raw_action_counts']) or not np.array_equal(macro_expected, counts['macro_transition_counts']): raise ValueError('Window/transition exposure reconstruction differs')
    return {'sampled_windows': steps * 128, 'raw_action_exposures': steps * 128 * 15, 'predicted_target_tokens': steps * 128 * 3}


class Audit:
    def __init__(self, root):
        self.root = root; self.files = {}; self.verified = {}; self.checks = []; self.manifest_sha = None
        self.roles = {}; self.formal_rows = {}; self.models = {}; self.restorable = {}; self.reset_fallback = {}
    def check(self, name, fn):
        started = time.perf_counter()
        try: detail = fn(); status = 'PASS'
        except FileNotFoundError as error: detail = {'missing': str(error)}; status = 'PENDING'
        except Exception as error: detail = {'type': type(error).__name__, 'reason': str(error)}; status = 'FAIL'
        self.checks.append({'check': name, 'status': status, 'seconds': time.perf_counter() - started, 'detail': detail})
        print(json.dumps({'check': name, 'status': status}), flush=True)
        return status
    def require(self, name):
        path = safe_path(self.root, name)
        if name not in self.files or name not in self.verified: raise FileNotFoundError('Required delivered/hash-verified file: ' + name)
        if signature(path) != self.verified[name]: raise ValueError('File changed after SHA verification: ' + name)
        return path
    def json(self, name): return read(self.require(name))
    def npz(self, name):
        import numpy as np
        with np.load(self.require(name), allow_pickle=False) as f: return {k: f[k].copy() for k in f.files}
    def verify_map(self, files, prefix=''):
        if not isinstance(files, dict) or not files: raise ValueError('Empty nested evidence file map')
        for name, expected in files.items():
            rel = str(PurePosixPath(prefix) / name) if prefix else name
            self.require(rel)
            actual = self.files[rel]
            d = expected if isinstance(expected, dict) else {'sha256': expected}
            if actual['sha256'] != d['sha256'] or ('bytes' in d and actual['bytes'] != d['bytes']): raise ValueError('Nested seal differs: ' + rel)
        return len(files)
    def verify_records(self, records):
        for item in records: self.verify_map({item['path']: item})
        return len(records)
    def verify_or_restorable(self, files):
        for name, value in files.items():
            expected = value if isinstance(value, dict) else {'sha256': value}
            if name in self.files: self.verify_map({name: expected})
            elif name in self.restorable:
                item = self.restorable[name]
                if item['sha256'] != expected['sha256'] or ('bytes' in expected and item['bytes'] != expected['bytes']): raise ValueError('Restorable source differs: ' + name)
            else: raise FileNotFoundError('Evidence not delivered or source-restorable: ' + name)
        return len(files)
    def verify_manifest(self):
        path = safe_path(self.root, 'manifests/RECOVERY_FINAL.json'); doc = read(path); self.manifest_sha = sha(path)
        if doc.get('status') != 'FINAL_SNAPSHOT_READY_FOR_RECOVERY_AND_ACCEPTANCE': raise ValueError('Final recovery snapshot is not ready')
        self.files = doc['files']
        if not isinstance(self.files, dict) or not self.files: raise ValueError('Empty final snapshot')
        missing = []; mismatched = []; total = 0
        for name, item in self.files.items():
            if name in ('manifests/RECOVERY_FINAL.json', 'RELEASE_GATE.json', 'RELEASE_GATE_SEAL.json'): raise ValueError('Circular/self-containing delivery manifest')
            if type(item.get('bytes')) is not int or item['bytes'] < 0 or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64: raise ValueError('Invalid final file identity')
            p = safe_path(self.root, name)
            if not p.is_file(): missing.append(name); continue
            sig = signature(p)
            if sig[0] != DEVICE or sig[2] != item['bytes'] or sha(p) != item['sha256'] or signature(p) != sig: mismatched.append(name); continue
            self.verified[name] = sig; total += item['bytes']
        if mismatched: raise ValueError('Final byte/hash/device mismatch: ' + json.dumps(mismatched))
        if missing: raise FileNotFoundError('Missing final files: ' + json.dumps(missing))
        if sha(__file__) != self.files['scripts/final_acceptance.py']['sha256']: raise ValueError('Executed acceptance helper differs from delivered helper')
        return {'files': len(self.files), 'bytes': total, 'all_delivered_bytes_SHA256_exact': True}
    def runtime_and_prior(self):
        import importlib.metadata as metadata
        versions = {k: metadata.version(k) for k in ('torch', 'transformers', 'numpy')}
        if versions['torch'].split('+')[0] != '2.8.0' or versions['transformers'] != '4.51.3' or versions['numpy'] != '2.3.5': raise RuntimeError('CPU runtime differs from frozen technical versions: ' + str(versions))
        tol = self.json('manifests/NUMERICAL_TOLERANCES.json')
        if tol['status'] != 'PROSPECTIVELY_FROZEN' or tol['comparisons']['CPU_replay'] != {'atol': ATOL, 'rtol': RTOL} or not tol['selected_before_actual_model_GPU_gates']: raise ValueError('CPU numerical tolerance missing/changed')
        prior = self.json('manifests/PRIOR_EVIDENCE.json')
        if prior['status'] != 'R2P1_DELIVERY_CPU_GATE_AND_SEAL_VERIFIED' or prior['R2_gate_status'] != 'R2P1_READY_FOR_USER_RELEASE': raise ValueError('Prior R2.1 release evidence not complete')
        for item in prior['items'].values():
            p = Path(item['path'])
            if not p.is_absolute() or not p.is_relative_to(MOUNT) or p.is_symlink() or p.stat().st_dev != DEVICE or item['external_device'] != DEVICE: raise ValueError('Prior evidence left the actual external device')
            current = MOUNT
            for part in p.relative_to(MOUNT).parts:
                current = current / part
                if current.is_symlink() or current.stat().st_dev != DEVICE: raise ValueError('Prior evidence ancestor/device changed')
            if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']: raise ValueError('Prior gate/seal bytes changed')
        gate_item = prior['items']['RELEASE_GATE.json']; seal = read(prior['items']['RELEASE_GATE_SEAL.json']['path'])
        gate = read(gate_item['path'])
        if gate['status'] != 'R2P1_READY_FOR_USER_RELEASE' or seal['release_gate_sha256'] != gate_item['sha256'] or seal['self_containment'] or seal['delivery_manifest_contains_gate_or_seal']: raise ValueError('Prior gate/seal chain invalid')
        if seal['delivery_manifest_sha256'] != gate['delivery_manifest_sha256'] or seal['acceptance_helper_sha256'] != gate['acceptance_helper_sha256']: raise ValueError('Prior seal identity differs from its gate')
        return {'CPU_versions': versions, 'python': sys.version, 'prior_R2_gate_SHA256': gate_item['sha256'], 'new_prior_science_run': False, 'CPU_replay_tolerance': {'atol': ATOL, 'rtol': RTOL}}
    def source_chains(self):
        for task in TASKS:
            model_api = self.json(f'state/{task}_model_api.json'); model_tree = self.json(f'state/{task}_model_tree.json')
            model_assets = self.json(f'manifests/{task}_model_assets.json')
            if model_assets['repo'] != 'quentinll/lewm-' + task or model_assets['revision'] != model_api['sha'] or model_assets['status'] != 'PINNED_ASSETS_DOWNLOADED_VERIFIED': raise ValueError('Official model provenance differs')
            for name, item in model_assets['files'].items():
                entry = next(x for x in model_tree if x['path'] == name)
                if item['bytes'] != entry['size'] or ('lfs' in entry and item['sha256'] != entry['lfs']['oid']): raise ValueError('Pinned official model hash/size differs')
                self.verify_map({item['path']: item})
            api = self.json(f'state/{task}_data_api.json'); tree = self.json(f'state/{task}_data_tree.json')
            assets = self.json(f'manifests/{task}_data_assets.json'); unpack = self.json(f'manifests/{task}_data_unpacked.json'); source = self.json(f'manifests/{task}_source_map.json')
            if assets['repo'] != 'quentinll/lewm-' + task or assets['revision'] != api['sha'] or source['official_revision'] != api['sha']: raise ValueError('Official source revision chain differs')
            if unpack['source_receipt_sha256'] != self.files[f'manifests/{task}_data_assets.json']['sha256']: raise ValueError('Unpack receipt archive manifest changed')
            for name, item in assets['files'].items():
                entry = next(x for x in tree if x['path'] == name)
                if item['bytes'] != entry['size'] or ('lfs' in entry and item['sha256'] != entry['lfs']['oid']): raise ValueError('Pinned official archive hash/size differs')
                self.restorable[item['path']] = {'sha256': item['sha256'], 'bytes': item['bytes'], 'evidence': f'manifests/{task}_data_assets.json'}
            members = {x['path']: x for x in unpack['files']}
            for source_sha, item in source['assets'].items():
                member = members[item['path']]
                if source_sha != item['sha256'] or any(member[k] != item[k] for k in ('sha256', 'bytes')): raise ValueError('Unpacked source asset chain differs')
                proof = self.json(member['member_receipt'])
                if self.files[member['member_receipt']]['sha256'] != member['member_receipt_sha256'] or proof['status'] != 'COMPLETE' or any(proof[k] != item[k] for k in ('sha256', 'bytes')): raise ValueError('Unpack member verification differs')
                archive = proof['identity']['archive']; expected = self.restorable[archive['path']]
                if any(archive[k] != expected[k] for k in ('sha256', 'bytes')): raise ValueError('Member archive identity differs')
                self.restorable[item['path']] = {'sha256': source_sha, 'bytes': item['bytes'], 'evidence': member['member_receipt']}
        return {'restorable_original_assets': self.restorable, 'absent_bulk_bytes_rehashed_here': False, 'source_identity_hashes_exact': True}
    def locks_and_reports(self):
        lock = self.json('manifests/MODELS_AND_SELECTION_LOCK.json')
        if lock['status'] != 'MODELS_AND_SELECTION_LOCKED': raise ValueError('Final models/selection not locked')
        self.verify_or_restorable(lock['files'])
        outputs = self.json('manifests/ANALYSIS_OUTPUTS.json')
        if outputs['status'] != 'COMPLETE' or set(outputs['tasks']) != set(TASKS): raise ValueError('Two-task actual analysis incomplete')
        self.verify_map(outputs['files']); self.analysis = outputs
        if outputs['models_lock_sha256'] != self.files['manifests/MODELS_AND_SELECTION_LOCK.json']['sha256']: raise ValueError('Analysis used another model/selection lock')
        self.verify_map(outputs['inputs'])
        reports = {}
        for base in REPORTS:
            matches = [name for name in self.files if PurePosixPath(name).name == base]
            if len(matches) != 1: raise FileNotFoundError('Exactly one delivered protocol report required: ' + base)
            p = self.require(matches[0])
            if not p.read_text().strip(): raise ValueError('Empty required report: ' + base)
            reports[base] = matches[0]
        exit_receipt = self.json('state/FINAL_GPU_EXIT.json'); instance = self.json('state/INSTANCE_REUSE.json')
        if exit_receipt['status'] != 'ALL_R3_GPU_JOBS_EXITED' or exit_receipt['hostname'] != instance['hostname'] or exit_receipt['active_compute_pids'] != [] or exit_receipt['R3_processes'] != [] or not exit_receipt['checked_at']: raise ValueError('Final GPU/process exit evidence incomplete')
        return {'reports': reports, 'analysis_outputs': len(outputs['files']), 'model_lock_entries': len(lock['files']), 'remote_exit_record_verified_without_local_GPU_query': True}
    def training(self):
        import numpy as np, torch
        from r3 import model as models
        from r3.train_worker import scheduler_record, lr_at
        jobs = self.json('manifests/JOBS.json')['jobs']
        if {(j['task'], j['refit_seed']) for j in jobs} != {(t, s) for t in TASKS for s in SEEDS} or len(jobs) != 6: raise ValueError('Six fixed jobs differ')
        strict = self.json('state/OFFICIAL_STRICT_LOAD_CPU.json')
        total = 0; details = []
        for task in TASKS:
            base = models.load_official(task, 'cpu'); frozen = models.frozen_hashes(base); whitelist = models.whitelist_manifest(base)
            if base.r3_strict_load['missing_keys'] or base.r3_strict_load['unexpected_keys'] or strict['tasks'][task]['identity'] != base.r3_identity: raise ValueError('Actual strict official reconstruction differs')
            self.models[task] = base
            roles = self.json(f'manifests/{task}_data_roles.json'); self.roles[task] = roles
            episode_ids = [e['episode_id'] for e in roles['episodes']]
            if len(set(episode_ids)) != len(episode_ids) or roles['status'] != 'METADATA_ROLES_FROZEN' or roles['selection_used_model_outputs']: raise ValueError('Frozen role assignment is invalid')
            role_groups = {r: {tuple((e['split_group']['kind'], e['split_group']['id'])) for e in roles['episodes'] if e['role'] == r} for r in ('REFIT_TRAIN', 'MONITOR', 'TECH', 'EVAL')}
            if role_groups['REFIT_TRAIN'] & (role_groups['MONITOR'] | role_groups['TECH'] | role_groups['EVAL']) or role_groups['MONITOR'] & (role_groups['TECH'] | role_groups['EVAL']): raise ValueError('TRAIN/MONITOR group leakage')
            for job in [j for j in jobs if j['task'] == task]:
                prefix = 'artifacts/train/' + job['job_id']; identity = self.json(prefix + '/RUN_IDENTITY.json'); result = self.json(prefix + '/result.json')
                if identity['job'] != job or identity['technical'] is not False or identity['role'] != 'REFIT_TRAIN' or identity['endpoint'] != 30000: raise ValueError('Formal job identity differs')
                if result['status'] != 'REFIT_TRAINING_COMPLETE_UNSCORED' or result['actual_updates'] != 30000 or result['technical'] or result['new_encoder_updates'] or result['state_labels_read'] or result['checkpoint_selection'] or result['formal_initialization_inherits_technical_updates']: raise ValueError('Formal completion/update/freeze contract differs')
                if result['identity_sha256'] != digest(identity) or result['frozen_before'] != frozen['sha256'] or result['frozen_after'] != frozen['sha256'] or self.json(prefix + '/FROZEN_INITIAL.json') != frozen: raise ValueError('Formal base/frozen hashes differ')
                if self.json(prefix + '/PARAMETER_WHITELIST.json') != whitelist: raise ValueError('Exact official initialization/parameter whitelist differs')
                for name, expected in identity['inputs'].items():
                    if self.files[name]['sha256'] != expected: raise ValueError('Formal locked input changed')
                for name, expected in identity['source'].items():
                    if self.files['r3/' + name]['sha256'] != expected: raise ValueError('Formal training source changed')
                rows = [json.loads(line) for line in self.require(prefix + '/updates.jsonl').read_text().splitlines()]
                if [x['step'] for x in rows] != list(range(1, 30001)): raise ValueError('Formal journal must contain exactly1..30000')
                for row in rows:
                    if row['technical'] or row['sampled_windows'] != 128 or row['predicted_tokens'] != 384 or row['raw_action_exposures'] != 1920 or not all(math.isfinite(row[k]) for k in ('loss_raw_MSE', 'lr', 'preclip_gradient_norm')): raise ValueError('Formal per-step budget/numerical record differs')
                    if row['lr'] != lr_at(row['step']): raise ValueError('Fixed learning-rate schedule differs')
                checks = [json.loads(line) for line in self.require(prefix + '/freeze_checks.jsonl').read_text().splitlines()]
                if not set(range(100, 30001, 100)).issubset(x['step'] for x in checks) or any(not x['passed'] or x['frozen_sha256'] != frozen['sha256'] for x in checks): raise ValueError('Required every100-step frozen checks missing/failed')
                monitor = [json.loads(line) for line in self.require(prefix + '/monitor.jsonl').read_text().splitlines()]
                if [x['step'] for x in monitor] != list(range(0, 30001, 1000)) or any(x['role'] != 'MONITOR' or x['checkpoint_selection'] for x in monitor): raise ValueError('Fixed MONITOR curve/selection contract differs')
                windows = roles['monitor_windows']
                if not 1 <= len(windows) <= 256 or identity['monitor_windows_sha256'] != digest(windows) or any(x['window_identity_sha256'] != digest(windows) or x['windows'] != len(windows) or x['state_labels_read'] or any(len(v) != len(windows) for v in x['per_window'].values()) for x in monitor): raise ValueError('Fixed monitor window population differs')
                if prefix + '/OPTIMIZER_INFLIGHT.json' in self.files: raise ValueError('Formal unresolved optimizer intent')
                for name in self.files:
                    if str(PurePosixPath(name).parent) == prefix and PurePosixPath(name).name.startswith('failure_'):
                        failure = self.json(name)
                        if failure.get('uncertain_optimizer_attempt') or failure.get('successful_updates', 0) > 30000: raise ValueError('Uncertain/extra formal optimizer attempt')
                public = self.json(prefix + '/last.json'); resume_ptr = self.json(prefix + '/resume.json')
                if public['path'] != 'checkpoint_30000.pt' or resume_ptr['path'] != 'resume.pt': raise ValueError('Compact final/resume pointer differs')
                for pointer in (public, resume_ptr):
                    if self.files[prefix + '/' + pointer['path']]['sha256'] != pointer['sha256'] or pointer['step'] != 30000: raise ValueError('Checkpoint pointer hash differs')
                resume = torch.load(self.require(prefix + '/resume.pt'), map_location='cpu', weights_only=True)
                if resume['step'] != 30000 or resume['scheduler'] != scheduler_record(30000) or resume['identity_sha256'] != digest(identity): raise ValueError('Resume state endpoint/scheduler differs')
                for key in ('optimizer', 'rng', 'sampler', 'next_batch_sha256'): 
                    if key not in resume: raise ValueError('Resume state missing ' + key)
                counts = self.npz(prefix + '/sampling_counts.npz')
                if counts['episode_ids'].tolist() != sorted(e['episode_id'] for e in roles['episodes'] if e['role'] == 'REFIT_TRAIN'): raise ValueError('Sampling exposed a non-REFIT_TRAIN episode')
                for key, multiplier in (('window_counts', 1), ('episode_counts', 1), ('raw_action_counts', 15), ('macro_transition_counts', 3)):
                    if int(counts[key].sum()) != 30000 * 128 * multiplier or np.any(counts[key] < 0) or not np.array_equal(counts[key], resume['sampler'][key].numpy()): raise ValueError('Actual sampling/exposure counts differ: ' + key)
                offsets = counts['window_offsets']; expected_ep = np.array([counts['window_counts'][a:b].sum() for a, b in zip(offsets[:-1], offsets[1:])])
                if not np.array_equal(expected_ep, counts['episode_counts']): raise ValueError('Per-episode/window accounting mismatch')
                sampling_accounting(counts, 30000, identity)
                rng = np.random.default_rng(); rng.bit_generator.state = resume['sampler']['rng']; indices = rng.integers(int(offsets[-1]), size=128, dtype=np.int64)
                ei = np.searchsorted(offsets[1:], indices, side='right'); cursor = [[str(counts['episode_ids'][e]), int(counts['legal_starts'][i])] for e, i in zip(ei, indices)]
                if digest(cursor) != resume['next_batch_sha256']: raise ValueError('Saved next-batch resume cursor differs')
                for step in STEPS:
                    cp = torch.load(self.require(prefix + f'/checkpoint_{step}.pt'), map_location='cpu', weights_only=True)
                    if set(cp) != {'version', 'step', 'identity_sha256', 'delta', 'frozen', 'sampling_summary'} or cp['step'] != step or cp['identity_sha256'] != digest(identity): raise ValueError('Milestone is not an exact compact parameter delta')
                    if cp['frozen'] != frozen or cp['sampling_summary']['sampled_windows'] != step * 128: raise ValueError('Milestone freeze/token accounting differs')
                    if step == 30000 and (set(cp['delta']['replacement_parameters']) != set(resume['delta']['replacement_parameters']) or any(not torch.equal(v, resume['delta']['replacement_parameters'][k]) for k, v in cp['delta']['replacement_parameters'].items())): raise ValueError('Public final delta differs from actual resumable endpoint')
                    models.apply_delta(base, self.require(prefix + f'/checkpoint_{step}.pt'))
                total += len(rows); details.append({'job_id': job['job_id'], 'updates': len(rows), 'sampled_windows': int(counts['window_counts'].sum()), 'compact_milestones': list(STEPS), 'frozen_sha256': frozen['sha256']})
                del resume
            # Restore original H0 before saved-input CPU replay.
            self.models[task] = models.load_official(task, 'cpu')
        if total != 180000: raise ValueError('Formal total differs from180000')
        technical = budget_bounds(self.json('state/TECHNICAL_LEDGER.json'))
        smoke = self.json('state/GPU_SMOKE.json'); gate = self.json('state/TRAINING_TECHNICAL_GATE.json')
        if smoke['status'] != 'PASS' or gate['status'] != 'TRAINING_TECHNICAL_GATES_PASS' or not smoke['genuine_process_exit_resume'] or smoke['optimizer_updates'] != 32: raise ValueError('Actual training technical gate missing')
        self.verify_records(smoke['files']); self.verify_records([gate[k] for k in ('smoke_receipt', 'profile_receipt', 'profile_table')]); self.verify_map(gate['source_files'])
        auth = self.json('manifests/TRAINING_AUTHORIZATION.json')
        if auth['status'] != 'TECHNICAL_GATES_PASS' or sorted(auth['job_ids']) != sorted(j['job_id'] for j in jobs): raise ValueError('Formal job authorization differs')
        self.verify_or_restorable(auth['source_hashes'])
        technical_checks = {}
        for basename in ('REAL_DATA_MODEL_CHECK.json', 'COST_WRAPPER_CHECK.json'):
            paths = [p for p in auth['source_hashes'] if PurePosixPath(p).name == basename]
            if len(paths) != 2: raise ValueError('Both actual task gates must be bound to formal authorization: ' + basename)
            task_receipts = {}
            for path in paths:
                record = self.json(path); prefix = str(PurePosixPath(path).parent)
                if record['status'] != 'PASS' or record['task'] in task_receipts or record['task'] not in TASKS or record['optimizer_updates']: raise ValueError('Actual zero-update technical gate differs')
                task_receipts[record['task']] = path
                self.verify_map(self.json(prefix + '/SHA256.json')['files'], prefix); self.verify_or_restorable(record['input_hashes'])
                if basename == 'REAL_DATA_MODEL_CHECK.json':
                    if not record['loaded_model_tensors_unchanged'] or record['optimizer_objects_created'] or record['EVAL_arrays_read'] or record['source_state_reward_goal_labels_read'] or record['full_policy_trajectories']: raise ValueError('Actual zero-update input/freeze gate invalid')
                else:
                    if record['complete_CEM_trajectories'] or record['environment_steps'] or record['source_future_actions_read'] or len(record['rows']) != 2: raise ValueError('Cost gate budget/inputs differ')
                    if [r['case_id'] for r in record['rows']] != [c['case_id'] for c in self.roles[record['task']]['cases']['TECH'][:2]]: raise ValueError('Cost wrapper used different TECH cases')
                    arrays = self.npz(prefix + '/COST_ARRAYS.npz'); tol = self.json('manifests/NUMERICAL_TOLERANCES.json')['comparisons']['rollout_wrapper']
                    for i, row in enumerate(record['rows']):
                        if not all(row[k] for k in ('zero_update_wrapped_cost_exact','official_goal_encoding_exact','terminal_squared_sum_cost_matches','all_finite')) or row['manual_comparison_tolerance'] != tol: raise ValueError('Actual cost/goal equivalence did not pass')
                        direct, wrapped, goal, rollout = (arrays[str(i)+'_'+k] for k in ('direct_cost','wrapped_cost','goal','rollout'))
                        manual = np.square(rollout[:,:,-1] - goal[:,None,-1]).sum(-1)
                        if not np.array_equal(direct, wrapped) or not np.allclose(direct, manual, **tol): raise ValueError('Saved official terminal goal cost does not reproduce')
            technical_checks[basename] = task_receipts
        return {'formal_updates': total, 'jobs': details, 'technical': technical, 'real_input_and_goal_cost_gates': technical_checks, 'new_encoder_updates': 0}
    def cached_input_evidence(self):
        """Rehashing is already complete; now validate index and TRAIN moments."""
        import numpy as np
        from r3 import open_loop
        details = {}
        for task in TASKS:
            cm = self.json(f'manifests/{task}_cache.json'); roles = self.roles[task]
            episodes = {e['episode_id']: e for e in roles['episodes'] if e['role'] in ('REFIT_TRAIN', 'MONITOR', 'TECH', 'EVAL')}
            if cm['status'] != 'FROZEN_OBSERVED_CACHE_COMPLETE' or set(cm['episodes']) != set(episodes) or cm['optimizer_updates'] or cm['state_labels_read']: raise ValueError('Cache membership/input-only declaration differs')
            self.verify_map(cm['input_hashes'])
            s1 = np.zeros(192, dtype=np.float64); s2 = s1.copy(); n_train = 0; frames = 0
            cases = {c['episode_id']: (i, c) for i, c in enumerate(roles['cases']['EVAL'])}
            prefix = self.analysis['tasks'][task]['open_loop_dir']
            saved = self.npz(prefix + '/inputs.npz'); target = self.npz(prefix + '/targets.npz')
            for eid in sorted(episodes):
                ep = episodes[eid]; item = cm['episodes'][eid]; self.verify_map({item['path']: item})
                data = self.npz(item['path']); z, a, raw, starts = (data[k] for k in ('z', 'actions', 'raw_indices', 'legal_starts')); length = ep['length']
                if set(data) != {'z', 'actions', 'raw_indices', 'stride', 'legal_starts'} or int(data['stride']) != 5 or z.shape != (length, 192) or z.dtype != np.float32 or a.shape != (length, 2) or a.dtype != np.float32 or not np.array_equal(raw, np.arange(length)) or not np.isfinite(z).all(): raise ValueError('Cached input schema/raw coordinate differs')
                if starts.ndim != 1 or starts.dtype != np.int64 or len(starts) != item['legal_windows'] or len(set(starts.tolist())) != len(starts) or np.any(starts < 0) or np.any(starts + 20 > length): raise ValueError('Cached legal window mapping differs')
                finite_rows = np.isfinite(a).all(1).astype(np.int64); prefix_finite = np.r_[0, finite_rows.cumsum()]
                if np.any(prefix_finite[starts+15]-prefix_finite[starts] != 15): raise ValueError('Declared legal cached window includes a nonfinite action')
                if item['role'] != ep['role'] or item['length'] != length: raise ValueError('Cache source role/length differs')
                side = self.json(str(PurePosixPath(item['path']).with_suffix('.json')))
                if side['identity']['episode'] != ep or side['file'] != item or side['source_state_or_reward_read']: raise ValueError('Cache source episode receipt differs')
                frames += length
                if ep['role'] == 'REFIT_TRAIN':
                    d = z.astype(np.float64); s1 += d.sum(0); s2 += np.square(d).sum(0); n_train += len(d)
                elif eid in cases:
                    i, case = cases[eid]; start = case['open_loop_window_start_raw']
                    if not np.array_equal(saved['initial_z'][i], z[start+np.arange(3)*5]) or not np.array_equal(saved['macro_actions'][i], a[start:start+35].reshape(7,10)) or not np.array_equal(target['target_z'][i], z[start+np.arange(3,8)*5]): raise ValueError('Saved forecast inputs/targets differ from exact original cache slices')
            mean = s1/n_train; coordinate_variance = s2/n_train-np.square(mean); scalar = float(coordinate_variance.mean())
            coords = self.npz(prefix + '/TRAIN_coordinates.npz'); variance = self.json(prefix + '/TRAIN_variance.json')
            if not np.array_equal(coords['coordinate_mean'], mean) or not np.array_equal(coords['coordinate_variance'], coordinate_variance) or variance['scalar'] != scalar or variance['frames'] != n_train or variance['role'] != 'REFIT_TRAIN' or variance['formula'] != open_loop.VARIANCE_FORMULA: raise ValueError('TRAIN-only coordinates/variance reconstruction differs')
            if cm['frozen_hashes'] != self.json('artifacts/train/' + f'R3_{task}_PRED_REFIT_s103201/FROZEN_INITIAL.json'): raise ValueError('Cache encoder frozen identity differs from training base')
            details[task] = {'episodes': len(episodes), 'cached_raw_frames': frames, 'TRAIN_frames': n_train, 'TRAIN_scalar_variance': scalar, 'saved_eval_slices_exact': True}
        return details
    def reset_fallback_evidence(self):
        required = {task: [c for phase in ('TECH', 'EVAL') for c in self.roles[task]['cases'][phase] if c['reset_seed'] is None] for task in TASKS}
        if not any(required.values()): return {'status': 'NOT_NEEDED_SOURCE_SEEDS_PRESENT'}
        name = 'manifests/RESET_FALLBACK_MANIFEST.json'; manifest = self.json(name); lock = self.json('manifests/MODELS_AND_SELECTION_LOCK.json')
        if manifest['status'] != 'VALIDATED_FIXED_RESET_FALLBACK_FROZEN' or name not in lock['files']: raise ValueError('Null source seeds require independently locked validated fallback')
        details = {}
        for task in TASKS:
            if not required[task]: continue
            item = manifest['tasks'][task]; role_path = f'manifests/{task}_data_roles.json'; receipt_record = item['validation_receipt']
            if item['data_roles']['path'] != role_path or item['fallback_seed'] != 0 or item['provenance'] != 'SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED' or item['affected_case_ids'] != sorted(c['case_id'] for c in required[task]): raise ValueError('Fallback changed source identity/affected population')
            self.verify_records([item['data_roles'], receipt_record])
            if receipt_record['path'] not in lock['files']: raise ValueError('Reset validation receipt was not in pre-evaluation model lock')
            receipt = self.json(receipt_record['path'])
            if receipt['status'] != 'PASS_TECH_RESET_FALLBACK' or receipt['routing_eligible'] is not True or receipt['task'] != task or receipt['candidate_reset_seed'] != 0 or receipt['roles_sha256'] != self.files[role_path]['sha256']: raise ValueError('Actual TECH reset evidence did not validate the registered fallback')
            self.verify_map(receipt['files'], str(PurePosixPath(receipt_record['path']).parent))
            self.verify_or_restorable(receipt['input_hashes'])
            for source_sha, source in receipt['source_files'].items():
                if source_sha != source['sha256']: raise ValueError('Reset source SHA identity differs')
                self.verify_or_restorable({source['path']: source})
            if receipt['cases'] != self.roles[task]['cases']['TECH'][:2]: raise ValueError('Reset fallback validation used different technical cases')
            # Import only the pure JSON/NumPy verifier and comparison callbacks;
            # never call its run()/environment/data paths during CPU acceptance.
            import copy, importlib.util
            helper = self.require('scripts/validate_reset_fallback.py' if task == 'pusht' else 'scripts/validate_reacher_reset.py')
            spec = importlib.util.spec_from_file_location('_r3_acceptance_reset_verifier', helper)
            reset = importlib.util.module_from_spec(spec); spec.loader.exec_module(reset)
            for source_sha in {c['source_asset_sha256'] for c in required[task]}:
                reset.verify_receipt(self.require(receipt_record['path']), roles_sha256=self.files[role_path]['sha256'], source_asset_sha256=source_sha)
            prefix = str(PurePosixPath(receipt_record['path']).parent)
            trials = {(r['case_id'], r['seed'], r['repeat'], r['path']): r for r in receipt['trials']}
            if len(trials) != 16: raise ValueError('Duplicate/missing reset diagnostic trial')
            def saved_trial(case, seed, repeat, path):
                row = trials[(case['case_id'], seed, repeat, path)]; trial_id = row['trial_id']
                if self.json(prefix + '/trial_' + trial_id + '.json') != row: raise ValueError('Reset trial row differs from durable original')
                return copy.deepcopy(row), self.npz(prefix + '/trial_' + trial_id + '.npz')
            derived, _ = reset.evaluate_matrix(receipt['cases'], saved_trial)
            if any(derived[k] != receipt[k] for k in derived): raise ValueError('Raw reset diagnostics do not reproduce the recorded16-trial/14-comparison matrix')
            self.reset_fallback[task] = {'reset_seed': 0, 'reset_seed_validation_sha256': receipt_record['sha256'], 'reset_seed_provenance': item['provenance'], 'reset_fallback_manifest_sha256': self.files[name]['sha256']}
            details[task] = {'affected_cases': len(required[task]), 'actual_technical_validation_receipt': receipt_record, 'source_seed_recovered': False, 'scope': 'Limited fixed TECH reproducibility only; not proof of all-EVAL/original-seed equivalence'}
        return details
    def effective_case(self, task, case):
        return dict(case) if case['reset_seed'] is not None else {**case, **self.reset_fallback[task]}
    def planning(self):
        import numpy as np
        from r3 import planning
        planning.verify_official_sources(self.root / 'source/swm_compat')
        complete = self.json('state/PLANNING_FORMAL_COMPLETE.json'); tech_gate = self.json('state/PLANNING_TECH_GATE.json')
        if complete['status'] != 'COMPLETE' or tech_gate['status'] != 'PASS' or tech_gate['clone_exact'] is not True or tech_gate['complete_trajectories'] != 8: raise ValueError('Actual closed-loop gates incomplete')
        self.verify_map(complete['files']); self.verify_map(tech_gate['files'])
        tech_plan = self.json('state/planning/TECH/EXECUTION_PLAN.json')
        formal_plan = self.json('state/planning/FORMAL/EXECUTION_PLAN.json')
        if formal_plan['models_lock_sha256'] != self.files['manifests/MODELS_AND_SELECTION_LOCK.json']['sha256'] or formal_plan['max_workers_per_gpu'] != 1: raise ValueError('Formal planning lock/device schedule differs')
        if len(tech_plan['cases']) != 4 or {t: sum(x['task'] == t for x in tech_plan['cases']) for t in TASKS} != {t: 2 for t in TASKS}: raise ValueError('Actual technical clone population differs')
        technical_ledger = self.json('state/TECHNICAL_LEDGER.json')['trajectories']
        clone_fields = ('raw_actions', 'step_success', 'step_truncated', 'physical_state', 'goal_error', 'goal_state', 'rewards')
        for case_record in tech_plan['cases']:
            task, case = case_record['task'], case_record['case']
            original = next((c for c in self.roles[task]['cases']['TECH'] if c['case_id'] == case['case_id']), None)
            if original is None or case not in (original, self.effective_case(task, original)) or case['episode_id'] in {c['episode_id'] for c in self.roles[task]['cases']['EVAL']}: raise ValueError('TECH/EVAL case overlap')
            case = self.effective_case(task, original)
            pair = []; pair_results = []
            for arm in ('H0', 'H0_CLONE'):
                prefix = f'artifacts/planning/TECH/{task}/{case["case_id"]}/{arm}/attempt_0'
                result = self.json(prefix + '/result.json'); seal = self.json(prefix + '/TRAJECTORY_SHA256.json'); self.verify_map(seal['files'], prefix)
                if result['status'] != 'COMPLETE' or result['failure_category'] is not None or result['identity']['case'] != case or result['phase'] != 'TECH' or result['arm'] != arm: raise ValueError('Full technical clone trajectory incomplete/different')
                rid = f'R3/TECH/{task}/{case["case_id"]}/{arm}/attempt_0'; entry = technical_ledger[rid]
                if entry['status'] != 'COMPLETE' or entry['result_sha256'] != self.files[prefix + '/result.json']['sha256']: raise ValueError('Technical clone ledger differs')
                arrays = self.npz(prefix + '/trajectory.npz'); trajectory_metrics(result, arrays); pair.append(arrays); pair_results.append(result)
            for key in clone_fields:
                if pair[0][key].dtype != pair[1][key].dtype or not np.array_equal(pair[0][key], pair[1][key]): raise ValueError('Actual H0 clone exact comparison fails: ' + key)
            if any(pair_results[0][k] != pair_results[1][k] for k in ('success', 'any_step_success', 'terminal_success', 'executed_raw_steps', 'replan_calls', 'final_goal_error')): raise ValueError('Actual clone summary differs')
        ledger = self.json('state/FORMAL_TRAJECTORY_LEDGER.json')['trajectories']; all_rows = []; counts = {}
        for task in TASKS:
            roles = self.roles[task]; cases = roles['cases']['EVAL']
            if not 20 <= len(cases) <= 100 or len({c['episode_id'] for c in cases}) != len(cases): raise ValueError('Formal case count or uniqueness differs')
            if set(c['episode_id'] for c in cases) != {e['episode_id'] for e in roles['episodes'] if e['role'] == 'EVAL'}: raise ValueError('Formal cases not exact EVAL membership')
            if set(c['episode_id'] for c in cases) & {e['episode_id'] for e in roles['episodes'] if e['role'] in ('REFIT_TRAIN', 'MONITOR', 'TECH')}: raise ValueError('Role leakage')
            plan_cases = [c['case'] for c in formal_plan['cases'] if c['task'] == task]
            if len(plan_cases) != len(cases) or {c['case_id'] for c in plan_cases} != {c['case_id'] for c in cases}: raise ValueError('Formal execution plan case population differs')
            for original in cases:
                selected = next(c for c in plan_cases if c['case_id'] == original['case_id'])
                if selected not in (original, self.effective_case(task, original)): raise ValueError('Planning changed a fixed metadata case')
            task_rows = []
            for case in cases:
                case_prefix = f'artifacts/planning/FORMAL/{task}/' + case['case_id']; case_seal = self.json(case_prefix + '/CASE_SHA256.json'); self.verify_map(case_seal['files'])
                if case_seal['complete_trajectories'] != 4: raise ValueError('Incomplete four-arm planning case')
                if case_seal['execution_plan_sha256'] != digest(formal_plan): raise ValueError('Case seal uses a different execution plan')
                for arm in ('H0',) + tuple(f'REFIT_{s}' for s in SEEDS):
                    prefix = case_prefix + '/' + arm + '/attempt_0'; result = self.json(prefix + '/result.json'); seal = self.json(prefix + '/TRAJECTORY_SHA256.json'); self.verify_map(seal['files'], prefix)
                    if result['status'] != 'COMPLETE' or not result['evaluation_complete'] or (result['task'], result['case_id'], result['arm'], result['phase']) != (task, case['case_id'], arm, 'FORMAL'): raise ValueError('Trajectory identity/status differs')
                    if result['identity']['case'] != self.effective_case(task, case) or result['identity']['plan_config'] != planning.PLAN_CONFIG or result['identity']['cem_config'] != planning.CEM_CONFIG or result['optimizer_updates'] != 0: raise ValueError('Scientific planning budget/source case differs')
                    identity_sha = hashlib.sha256(json.dumps(result['identity'], sort_keys=True, allow_nan=False).encode()).hexdigest()
                    if result['identity_sha256'] != identity_sha or seal['identity_sha256'] != identity_sha: raise ValueError('Trajectory full identity SHA differs')
                    rid = f'R3/FORMAL/{task}/{case["case_id"]}/{arm}/attempt_0'; entry = ledger[rid]
                    if entry['status'] != 'COMPLETE' or entry['result_path'] != prefix + '/result.json' or entry['result_sha256'] != self.files[prefix + '/result.json']['sha256']: raise ValueError('Trajectory ledger does not match the saved full result')
                    for i, replan in enumerate(result['replans']):
                        if replan['replan_index'] != i or replan['seed_uint64'] != planning.replan_seed(task, case['case_id'], i): raise ValueError('Paired per-case CEM RNG differs')
                    trajectory_metrics(result, self.npz(prefix + '/trajectory.npz')); task_rows.append(result)
            self.formal_rows[task] = task_rows; all_rows += task_rows; counts[task] = len(cases)
        if len(all_rows) != complete['complete_trajectories'] or len(all_rows) > 800: raise ValueError('Full formal trajectory count differs')
        occupied = [x for x in ledger.values() if x['status'] != 'INFRASTRUCTURE_FAILURE_INCOMPLETE']
        if len(occupied) != len(all_rows) or any(x['status'] != 'COMPLETE' for x in occupied): raise ValueError('Unresolved or extra formal trajectory budget')
        return {'cases': counts, 'complete_formal_trajectories': len(all_rows), 'maximum': 800, 'technical_clone_pairs_rechecked_exactly': 4, 'metrics_from_saved_arrays': True, 'new_environment_steps': 0}
    def open_loop_and_replay(self):
        import numpy as np, torch
        from r3 import model as models, open_loop
        routes_doc = self.json('manifests/OPEN_LOOP_ROUTING.json'); replay = []; metrics_checked = 0
        for task in TASKS:
            config = self.analysis['tasks'][task]; prefix = config['open_loop_dir']; cases = open_loop.validate_cases(self.roles[task], task); routes = open_loop.validate_routes(routes_doc, task)
            receipt = self.json(prefix + '/OPEN_LOOP_RECEIPT.json'); self.verify_map(receipt['files'])
            if receipt['status'] != 'COMPLETE' or receipt['arms'] != 10 or receipt['cases'] != len(cases) or not receipt['all_future_targets_scoring_only'] or receipt['optimizer_updates'] or receipt['physical_state_labels_read']: raise ValueError('Open-loop completeness/forecast contract differs')
            inputs = self.npz(prefix + '/inputs.npz'); targets = self.npz(prefix + '/targets.npz'); per_case = self.json(prefix + '/per_case.json'); variance = self.json(prefix + '/TRAIN_variance.json')
            identity = receipt['identity']
            if identity['models_lock_sha256'] != self.files['manifests/MODELS_AND_SELECTION_LOCK.json']['sha256'] or identity['routing_sha256'] != self.files['manifests/OPEN_LOOP_ROUTING.json']['sha256'] or identity['case_sha256'] != digest(cases): raise ValueError('Open-loop model/case/routing lock differs')
            ids = [c['case_id'] for c in cases]
            if inputs['case_ids'].tolist() != ids or inputs['episode_ids'].tolist() != [c['episode_id'] for c in cases] or len(per_case) != len(cases): raise ValueError('Saved-input case order differs')
            expected_history = np.asarray([[c['open_loop_anchor_raw'] - 10 + 5 * k for k in range(3)] for c in cases])
            expected_actions = np.asarray([list(range(c['open_loop_anchor_raw'] - 10, c['open_loop_anchor_raw'] + 25)) for c in cases])
            expected_targets = np.asarray([[c['open_loop_anchor_raw'] + 5 * k for k in range(1, 6)] for c in cases])
            if not np.array_equal(inputs['history_raw_indices'], expected_history) or not np.array_equal(inputs['action_raw_indices'], expected_actions) or not np.array_equal(targets['target_raw_indices'], expected_targets): raise ValueError('Raw history/action/target index mapping differs')
            if inputs['initial_z'].shape != (len(cases), 3, 192) or inputs['macro_actions'].shape != (len(cases), 7, 10) or targets['target_z'].shape != (len(cases), 5, 192): raise ValueError('Saved latent/macro-action dimensions differ')
            cm = self.json(f'manifests/{task}_cache.json')
            if variance['scalar'] != receipt['train_variance']['scalar'] or variance['formula'] != open_loop.VARIANCE_FORMULA or not math.isclose(variance['scalar'], cm['train_scalar_variance'], rel_tol=1e-10, abs_tol=1e-12): raise ValueError('Fixed TRAIN variance differs')
            model = self.models[task]; frozen = models.frozen_hashes(model); batch = int(receipt['identity']['batch_size'])
            for route in routes:
                pred_file = prefix + '/predictions/' + route['arm'] + '.npz'; predictions = self.npz(pred_file)
                arm_receipt = self.json(prefix + '/predictions/' + route['arm'] + '.json')
                if arm_receipt['identity_sha256'] != digest(identity) or arm_receipt['route'] != route or arm_receipt['future_targets_read_by_forecast'] or arm_receipt['optimizer_updates']: raise ValueError('Forecast provenance/target separation differs')
                self.verify_map({pred_file: arm_receipt['prediction_file']})
                if route['step']:
                    self.verify_map({route['checkpoint']['path']: route['checkpoint']})
                    cp = torch.load(self.require(route['checkpoint']['path']), map_location='cpu', weights_only=True)
                    open_loop.validate_checkpoint(route, cp, self.json(route['run_identity']), self.json(route['result']), model)
                    del cp
                if predictions['case_ids'].tolist() != ids or predictions['prediction'].shape != targets['target_z'].shape or predictions['prediction'].dtype != np.float32: raise ValueError('Saved forecast shape/identity differs')
                for i, case in enumerate(cases):
                    row = per_case[i]
                    if row['case'] != case or row['evaluation_kind'] != 'OPEN_LOOP' or row['phase'] != 'EVAL' or row['task'] != task or len(row['arms']) != 10: raise ValueError('Per-case raw identity differs')
                    matched = [a for a in row['arms'] if a['arm'] == route['arm']]
                    expected = open_loop.score_case(predictions['prediction'][i], targets['target_z'][i], variance['scalar'], inputs['initial_z'][i])
                    if len(matched) != 1 or matched[0]['metrics'] != expected or matched[0]['checkpoint_step'] != route['step'] or matched[0]['refit_seed'] != route['seed']: raise ValueError('Recomputed saved open-loop metrics differ')
                    metrics_checked += len(expected)
                if route['step'] not in (0, 30000): continue
                if route['step']: models.apply_delta(model, self.require(route['checkpoint']['path']))
                before = digest({k: models.tensor_sha256(v) for k, v in model.state_dict().items()})
                parts = []
                with torch.inference_mode():
                    for i in range(0, len(cases), batch):
                        parts.append(models.cached_rollout(model, torch.from_numpy(inputs['initial_z'][i:i + batch]), torch.from_numpy(inputs['macro_actions'][i:i + batch]), 5).numpy())
                actual = np.concatenate(parts); expected = predictions['prediction']; finite = np.isfinite(actual) & np.isfinite(expected)
                if not np.array_equal(np.isfinite(actual), np.isfinite(expected)) or not np.array_equal(np.isnan(actual), np.isnan(expected)) or not np.array_equal(np.isposinf(actual), np.isposinf(expected)) or not np.array_equal(np.isneginf(actual), np.isneginf(expected)): raise ValueError('CPU/GPU nonfinite pattern differs')
                errors = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64)); bounds = ATOL + RTOL * np.abs(expected[finite].astype(np.float64))
                if np.any(errors > bounds): raise ValueError(f'CPU replay exceeds fixed tolerance: {task}/{route["arm"]}; max_excess={float((errors-bounds).max())}')
                if before != digest({k: models.tensor_sha256(v) for k, v in model.state_dict().items()}): raise ValueError('CPU replay mutated model state')
                models.assert_frozen(model, frozen)
                replay.append({'task': task, 'arm': route['arm'], 'cases': len(cases), 'horizon': 5, 'atol': ATOL, 'rtol': RTOL, 'finite_coordinates': int(finite.sum()), 'max_abs_error': float(errors.max()) if len(errors) else None, 'nonfinite_pattern_exact': True})
        return {'all_saved_metric_records_recomputed': metrics_checked, 'CPU_replayed_models': 8, 'replays': replay, 'new_optimizer_updates': 0, 'new_encoder_forward_calls': 0}
    def statistics(self):
        import numpy as np
        from r3 import statistics
        out = {}
        for task in TASKS:
            prefix = self.analysis['tasks'][task]['statistics_dir']; receipt = self.json(prefix + '/STATISTICS_RECEIPT.json'); self.verify_map(receipt['files'], prefix)
            rows = statistics.restore_json_value(self.json(prefix + '/RAW_CASE_ROWS.json'))
            by = {(r['case_id'], r['arm']): statistics.canonical_bytes(r) for r in rows}
            source_by = {(r['case_id'], r['arm']): statistics.canonical_bytes(r) for r in self.formal_rows[task]}
            if len(by) != len(rows) or by != source_by: raise ValueError('Statistics raw rows are not the exact completed formal trajectories')
            result = statistics.analyze_task(task, rows, output_dir=None)
            if statistics.canonical_bytes(result) != self.require(prefix + '/NUMERIC_RESULTS.json').read_bytes(): raise ValueError('Full numerical/statistical rederivation differs')
            ordered, cases, families, success = statistics.normalize_rows(task, rows)
            saved = self.npz(prefix + '/BOOTSTRAP_INDICES.npz'); draws = self.npz(prefix + '/BOOTSTRAP_DISTRIBUTIONS.npz')
            seed = int.from_bytes(hashlib.sha256(f'R3_BOOTSTRAP_20261002/{task}'.encode()).digest()[:8], 'big')
            ix = np.random.Generator(np.random.PCG64(seed)).integers(0, len(cases), size=(5000, len(cases)), dtype=np.int64)
            def values(means): return 100 * np.concatenate((means, means[..., 1:] - means[..., :1], (means[..., 1:] - means[..., :1]).mean(-1, keepdims=True)), axis=-1)
            case_draws = values(success[ix].mean(1)); point = values(success.mean(0))
            if not np.array_equal(saved['case'], ix) or saved['case_ids'].tolist() != cases or not np.array_equal(draws['case'], case_draws) or not np.array_equal(draws['point'], point) or draws['columns'].tolist() != list(statistics.BOOTSTRAP_COLUMNS): raise ValueError('Independent paired bootstrap indices/draws differ')
            family_ids = sorted(set(families)) if all(f is not None and f != 'FAMILY_UNKNOWN' for f in families) else []
            if family_ids:
                seed = int.from_bytes(hashlib.sha256(f'R3_BOOTSTRAP_20261002/{task}/family'.encode()).digest()[:8], 'big')
                fi = np.random.Generator(np.random.PCG64(seed)).integers(0, len(family_ids), size=(5000, len(family_ids)), dtype=np.int64)
                members = [np.flatnonzero(np.asarray(families) == f) for f in family_ids]; totals = np.asarray([success[x].sum(0) for x in members]); sizes = np.asarray([len(x) for x in members])
                family_draws = values(totals[fi].sum(1) / sizes[fi].sum(1)[:, None])
            else: fi = np.empty((0, 0), dtype=np.int64); family_draws = np.empty((0, 8))
            if not np.array_equal(saved['family'], fi) or saved['family_ids'].tolist() != family_ids or not np.array_equal(draws['family'], family_draws): raise ValueError('Independent family-sensitivity bootstrap differs')
            paired = []
            for i, case in enumerate(cases):
                row = {'case_id': case, 'family_id': families[i]}; row.update({arm: int(success[i, j]) for j, arm in enumerate(statistics.ARMS)})
                row.update({arm + '_minus_H0': float(success[i, j] - success[i, 0]) for j, arm in enumerate(statistics.ARMS[1:], 1)}); row['fixed_refit_mean_minus_H0'] = float(success[i, 1:].mean() - success[i, 0]); paired.append(row)
            if self.require(prefix + '/PAIRED_CASES.csv').read_bytes() != statistics._csv_bytes(paired) or self.require(prefix + '/PAIRED_COUNTS.csv').read_bytes() != statistics._csv_bytes(statistics.paired_counts(success)): raise ValueError('Saved paired raw/count tables differ')
            out[task] = {'cases': len(cases), 'shared_bootstrap_replicates': 5000, 'family_status': result['schemes']['FAMILY']['status'], 'all_raw_rows_and_bootstrap_exact': True}
        return out
    def unchanged(self):
        if sha(self.root / 'manifests/RECOVERY_FINAL.json') != self.manifest_sha: raise ValueError('Final recovery manifest changed during acceptance')
        for name in self.files: self.require(name)
        return {'all_files_unchanged_since_exact_hash': True}


def configure(root):
    os.environ['R3_ROOT'] = str(root); sys.path.insert(0, str(root))
    import torch
    torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
    from r3 import common, model, open_loop, statistics
    for module in (common, model, open_loop, statistics):
        if not Path(module.__file__).resolve().is_relative_to(root): raise RuntimeError('CPU audit imported an unrecovered R3 module')
    if common.ROOT != root: raise RuntimeError('CPU audit imported a different R3 root')
    common.fp32_policy()
    return {'import_root': str(root), 'device': 'cpu', 'threads': torch.get_num_threads(), 'deterministic': True}


def run(root):
    root = Path(root).absolute(); storage = external_guard(root); audit = Audit(root)
    if audit.check('all_recovered_file_SHA256_and_bytes', audit.verify_manifest) == 'PASS':
        if audit.check('CPU_runtime_tolerance_and_prior_release_chain', audit.runtime_and_prior) == 'PASS':
            if audit.check('recovered_CPU_modules_only', lambda: configure(root)) == 'PASS':
                audit.check('pinned_original_source_restorability', audit.source_chains)
                if audit.check('model_selection_lock_reports_and_GPU_exit', audit.locks_and_reports) == 'PASS':
                    if audit.check('official_reconstruction_formal180k_and_technical_budgets', audit.training) == 'PASS':
                        audit.check('exact_cache_inputs_and_TRAIN_coordinates', audit.cached_input_evidence)
                        audit.check('source_seed_or_validated_reset_fallback', audit.reset_fallback_evidence)
                        audit.check('all_complete_paired_CEM_trajectories', audit.planning)
                        audit.check('all_open_loop_metrics_and_H0_threefinal_CPU_replay', audit.open_loop_and_replay)
                        if len(audit.formal_rows) == 2: audit.check('exact_paired_tables_and_shared_bootstrap_rederivation', audit.statistics)
            audit.check('final_evidence_unchanged', audit.unchanged)
    required = {'all_recovered_file_SHA256_and_bytes', 'CPU_runtime_tolerance_and_prior_release_chain', 'recovered_CPU_modules_only', 'pinned_original_source_restorability', 'model_selection_lock_reports_and_GPU_exit', 'official_reconstruction_formal180k_and_technical_budgets', 'exact_cache_inputs_and_TRAIN_coordinates', 'source_seed_or_validated_reset_fallback', 'all_complete_paired_CEM_trajectories', 'all_open_loop_metrics_and_H0_threefinal_CPU_replay', 'exact_paired_tables_and_shared_bootstrap_rederivation', 'final_evidence_unchanged'}
    ready = {x['check'] for x in audit.checks} == required and all(x['status'] == 'PASS' for x in audit.checks)
    result = {'status': 'R3_READY_FOR_USER_RELEASE' if ready else 'ACCEPTANCE_PENDING_OR_FAILED',
              'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'root': str(root),
              'recovery_manifest_sha256': audit.manifest_sha, 'acceptance_helper_sha256': sha(__file__),
              'external_storage': storage, 'checks': audit.checks,
              'optimizer_updates_by_acceptance': 0, 'new_environment_steps': 0, 'new_physical_state_acquisitions': 0,
              'GPU_queries_or_forward_calls': 0, 'SSH_connections': 0, 'instance_destroyed': False,
              'scope': 'Execution/evidence/numerical recovery only; scientific lack of improvement is not a technical failure'}
    external_guard(root)
    if ready:
        gate_path = safe_path(root, 'RELEASE_GATE.json'); seal_path = safe_path(root, 'RELEASE_GATE_SEAL.json')
        if gate_path.exists() or seal_path.exists():
            if not gate_path.exists() or not seal_path.exists(): raise RuntimeError('Partial previous release gate/seal retained; explicitly audit before replacement')
            old = read(gate_path); seal = read(seal_path)
            if old['status'] != 'R3_READY_FOR_USER_RELEASE' or old['recovery_manifest_sha256'] != audit.manifest_sha or old['acceptance_helper_sha256'] != sha(__file__) or seal['release_gate_sha256'] != sha(gate_path): raise RuntimeError('Different previous release gate/seal retained without overwrite')
            return old
        atomic(gate_path, result)
        atomic(seal_path, {'recovery_manifest_sha256': audit.manifest_sha, 'release_gate_sha256': sha(gate_path),
               'acceptance_helper_sha256': sha(__file__), 'self_containment': False, 'recovery_manifest_contains_gate_or_seal': False})
    else:
        name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + f'_{os.getpid()}.json'
        atomic(safe_path(root, 'reports/acceptance_attempts/' + name), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', default=os.environ.get('R3_ROOT', str(DEFAULT_ROOT))); args = parser.parse_args()
    outcome = run(args.root); print(json.dumps({'status': outcome['status'], 'root': outcome['root']}), flush=True)
    raise SystemExit(0 if outcome['status'] == 'R3_READY_FOR_USER_RELEASE' else 1)
