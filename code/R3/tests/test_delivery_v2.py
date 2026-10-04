"""Delivery-path regression only: synthetic metadata/bytes, no science execution."""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

p = load('publisher_delivery_v2_test', ROOT/'scripts/publish_final_delivery_v2.py')
a = load('acceptance_delivery_v2_test', ROOT/'scripts/final_acceptance_delivery_v2.py')
f = load('publisher_existing_metadata_fixture', ROOT/'tests/test_publish_final.py')
HISTORY = (
    'reports/intake_snapshot_20261003T0010/INTAKE_AND_INSTANCE_REUSE.md',
    'reports/intake_snapshot_20261003T0010/OFFICIAL_SOURCE_AND_PROTOCOL_AUDIT.md',
    'protocol/inputs_readonly/FINAL_SCIENTIFIC_REPORT_ZH.md',
)
AUTOMATIC = ('TRAINING_COMPLETION.md', 'OPEN_LOOP_RESULTS.md',
             'CEM_PAIRED_RESULTS.md', 'COMPUTE_STORAGE_AND_ETA.md')
HELPERS = ('scripts/publish_final.py', 'scripts/final_acceptance.py',
           p.PUBLISHER_PATH, p.ACCEPTANCE_PATH)


def augment(root, put):
    for name in HELPERS:
        put(name, (ROOT/name).read_bytes())
    for name in HISTORY:
        put(name, b'Immutable historical report; not the current R3 report.\n')


def small_audit_fixture(root):
    def put(name, value):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
        return {'sha256': a.sha(path), 'bytes': path.stat().st_size}
    augment(root, put)
    reports = {name: put('reports/'+name, b'Current R3 report.\n') for name in p.REPORTS}
    locked = {HISTORY[-1]: {'sha256': a.sha(root/HISTORY[-1]), 'bytes': (root/HISTORY[-1]).stat().st_size}}
    lock = put(p.LOCK, {'status': 'MODELS_AND_SELECTION_LOCKED', 'files': locked})
    put('manifests/ANALYSIS_OUTPUTS.json', {'status': 'COMPLETE', 'tasks': {t: {} for t in a.TASKS},
        'models_lock_sha256': lock['sha256'], 'inputs': {p.LOCK: lock},
        'files': {'reports/'+name: reports[name] for name in AUTOMATIC}})
    put('state/INSTANCE_REUSE.json', {'hostname': 'metadata_fixture'})
    put('state/FINAL_GPU_EXIT.json', {'status': 'ALL_R3_GPU_JOBS_EXITED', 'hostname': 'metadata_fixture',
                                   'active_compute_pids': [], 'R3_processes': [], 'checked_at': 'fixed'})
    return put


def refresh(audit):
    audit.files = {}; audit.verified = {}
    for path in audit.root.rglob('*'):
        if path.is_file() and path.name != 'RECOVERY_FINAL.json':
            name = str(path.relative_to(audit.root))
            audit.files[name] = {'sha256': a.sha(path), 'bytes': path.stat().st_size}
            audit.verified[name] = a.signature(path)


class DeliveryV2Tests(unittest.TestCase):
    def test_publisher_keeps_three_historical_collisions_and_binds_both_versions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); put, rec = f.fixture(root); augment(root, put)
            out, inv = p.build(root)
            self.assertEqual(out['reports'], {base: 'reports/'+base for base in p.REPORTS})
            for name in HISTORY + HELPERS:
                self.assertEqual(out['files'][name], rec(name))
            self.assertEqual(out['delivery_revision']['publisher'], {'path': p.PUBLISHER_PATH, **rec(p.PUBLISHER_PATH)})
            self.assertEqual(out['delivery_revision']['acceptance'], {'path': p.ACCEPTANCE_PATH, **rec(p.ACCEPTANCE_PATH)})
            _, created = p.publish(root, out, inv); self.assertTrue(created)
            original = (root/p.TARGET).read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'Different existing'):
                p.publish(root, {**out, 'optimizer_updates': 1}, inv)
            self.assertEqual(original, (root/p.TARGET).read_bytes())

    def test_publisher_rejects_missing_and_empty_current_report_despite_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); put, rec = f.fixture(root); augment(root, put)
            # Actual summarizer seals its four automatic reports, not the final narrative.
            output = p.read(root/'manifests/ANALYSIS_OUTPUTS.json')
            output['files'] = {name: value for name, value in output['files'].items()
                               if Path(name).name in AUTOMATIC}
            put('manifests/ANALYSIS_OUTPUTS.json', output)
            for history in HISTORY:
                current = root/'reports'/Path(history).name
                original = current.read_bytes()
                for contents in (None, b' \n\t'):
                    with self.subTest(report=current.name, empty=contents is not None):
                        if contents is None: current.unlink()
                        else: current.write_bytes(contents)
                        with self.assertRaisesRegex(ValueError, 'current R3 canonical report'):
                            p.build(root)
                        self.assertTrue((root/history).is_file())
                        current.write_bytes(original)

    def test_cpu_report_gate_keeps_locked_history_and_rejects_substitution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); small_audit_fixture(root); audit = a.Audit(root); refresh(audit)
            out = audit.locks_and_reports()
            self.assertEqual(out['reports'], {base: 'reports/'+base for base in p.REPORTS})
            self.assertEqual(out['model_lock_entries'], 1)
            locked_sha = a.sha(root/HISTORY[-1])
            for history in HISTORY:
                current = root/'reports'/Path(history).name; original = current.read_bytes()
                for contents in (None, b' \n'):
                    with self.subTest(report=current.name, empty=contents is not None):
                        if contents is None: current.unlink()
                        else: current.write_bytes(contents)
                        refresh(audit)
                        with self.assertRaises((FileNotFoundError, ValueError)):
                            audit.locks_and_reports()
                        self.assertEqual(a.sha(root/HISTORY[-1]), locked_sha)
                        current.write_bytes(original)

    def test_cpu_manifest_requires_actual_v2_self_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); put = small_audit_fixture(root); audit = a.Audit(root); refresh(audit)
            def manifest():
                return {'status': 'FINAL_SNAPSHOT_READY_FOR_RECOVERY_AND_ACCEPTANCE',
                    'version': a.DELIVERY_VERSION, 'files': audit.files,
                    'delivery_revision': {'version': a.DELIVERY_VERSION,
                        'report_namespace': 'reports/<required basename>',
                        'publisher': {'path': a.PUBLISHER_PATH, **audit.files[a.PUBLISHER_PATH]},
                        'acceptance': {'path': a.ACCEPTANCE_PATH, **audit.files[a.ACCEPTANCE_PATH]},
                        'frozen_v1_helpers': {name: audit.files[name] for name in HELPERS[:2]}}}
            put(p.TARGET, manifest())
            with patch.object(a, 'DEVICE', root.stat().st_dev):
                self.assertTrue(audit.verify_manifest()['all_delivered_bytes_SHA256_exact'])
                # Even a manifest whose hashes are internally consistent cannot bind another executable.
                put(a.ACCEPTANCE_PATH, b'not the actual V2 acceptance executable\n'); refresh(audit)
                put(p.TARGET, manifest())
                with self.assertRaisesRegex(ValueError, 'Executed V2 acceptance helper'):
                    a.Audit(root).verify_manifest()

    def test_frozen_v1_bytes_and_all_unmodified_functions_match(self):
        expected = {'publish_final.py': 'be50a41d9f12f1b73946032df5bf4f5456c5db0ef72461fc74996257c2f31890',
                    'final_acceptance.py': '46493cb587ac2959173ed8282d09619fb0e86e53627640b27e87d721296eb3d5'}
        def definitions(path):
            out = {}
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): out[node.name] = ast.dump(node)
                elif isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            out[node.name+'.'+child.name] = ast.dump(child)
            return out
        pairs = [('publish_final.py', 'publish_final_delivery_v2.py', {'build'}),
                 ('final_acceptance.py', 'final_acceptance_delivery_v2.py',
                  {'Audit.verify_manifest', 'Audit.locks_and_reports', 'run'})]
        for old, new, allowed in pairs:
            self.assertEqual(p.sha(ROOT/'scripts'/old), expected[old])
            before = definitions(ROOT/'scripts'/old); after = definitions(ROOT/'scripts'/new)
            self.assertEqual(set(before), set(after))
            self.assertEqual({name for name in before if before[name] != after[name]}, allowed)


if __name__ == '__main__': unittest.main(verbosity=2)
