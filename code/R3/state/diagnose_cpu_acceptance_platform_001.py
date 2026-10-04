"""Run unchanged independent V2 checks after the known native-libm LR failure.

Diagnostic only: does not call run(), replace any checks, or write a release gate.
The original failed training check remains failed and is not counted as passed.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
ROOT = Path('/Volumes/MyProj/r3_official_lewm_predictor_refit/recovery')
HELPER = ROOT / 'scripts/final_acceptance_delivery_v2.py'
EXPECTED = 'f66d76d4517494bc189679422ff351a58585325c1cf684f3d0c5c4031083a77e'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == EXPECTED
spec = importlib.util.spec_from_file_location('frozen_v2_diagnostic', HELPER)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
v2.external_guard(ROOT)
out = ROOT / 'recovery_receipts/CPU_PLATFORM_REMAINING_DIAGNOSTIC_001.json'
assert not out.exists()
audit = v2.Audit(ROOT)
for name, fn in [
    ('all_recovered_file_SHA256_and_bytes', audit.verify_manifest),
    ('CPU_runtime_tolerance_and_prior_release_chain', audit.runtime_and_prior),
    ('recovered_CPU_modules_only', lambda: v2.configure(ROOT)),
    ('pinned_original_source_restorability', audit.source_chains),
    ('model_selection_lock_reports_and_GPU_exit', audit.locks_and_reports),
]:
    if audit.check(name, fn) != 'PASS':
        raise RuntimeError('Diagnostic prerequisite failed: ' + name)

# These are explicit diagnostic prerequisites, not replacement training checks.
from r3 import model as models
for task in v2.TASKS:
    audit.roles[task] = audit.json(f'manifests/{task}_data_roles.json')
    audit.models[task] = models.load_official(task, 'cpu')

for name, fn in [
    ('exact_cache_inputs_and_TRAIN_coordinates', audit.cached_input_evidence),
    ('source_seed_or_validated_reset_fallback', audit.reset_fallback_evidence),
    ('all_complete_paired_CEM_trajectories', audit.planning),
    ('all_open_loop_metrics_and_H0_threefinal_CPU_replay', audit.open_loop_and_replay),
]:
    audit.check(name, fn)

# If an earlier platform-exact trajectory check failed, still diagnose the
# independent statistics function using SHA-verified completed result bytes.
# This does not turn the trajectory check into PASS.
for task in v2.TASKS:
    audit.formal_rows[task] = []
    for case in audit.roles[task]['cases']['EVAL']:
        for arm in ('H0',) + tuple(f'REFIT_{s}' for s in v2.SEEDS):
            name = f'artifacts/planning/FORMAL/{task}/{case["case_id"]}/{arm}/attempt_0/result.json'
            audit.formal_rows[task].append(audit.json(name))
audit.check('exact_paired_tables_and_shared_bootstrap_rederivation', audit.statistics)
audit.check('final_evidence_unchanged', audit.unchanged)
v2.external_guard(ROOT)
record = {
    'status': 'DIAGNOSTIC_ONLY_NOT_A_RELEASE_GATE',
    'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'platform': platform.platform(), 'python': sys.version,
    'acceptance_helper_sha256': EXPECTED,
    'diagnostic_helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'recovery_manifest_sha256': audit.manifest_sha,
    'known_training_check': 'FAILED_IN_ORIGINAL_ATTEMPT001_NOT_REPLACED_OR_RECLASSIFIED',
    'checks': audit.checks,
    'new_optimizer_updates': 0, 'new_environment_steps': 0,
    'source_or_tolerance_changes': False, 'creates_release_gate': False,
}
with out.open('x') as f:
    json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n'); f.flush(); os.fsync(f.fileno())
print(json.dumps({'status': record['status'], 'path': str(out),
                  'checks': [{k: c[k] for k in ('check','status')} for c in audit.checks]}), flush=True)
