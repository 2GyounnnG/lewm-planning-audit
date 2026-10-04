"""Small metadata fixtures; no model/environment/data-array execution."""
from pathlib import Path
import json
import tempfile
import unittest
from scripts import build_official_asset_manifest as m


class OfficialAssetManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture()

    def write(self, rel, value):
        p = self.root/rel; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return m.identity_bytes(p.read_bytes())

    def read(self, rel): return json.loads((self.root/rel).read_text())

    def fixture(self):
        self.write('r3/model.py', '# fixture only')
        self.write('scripts/build_official_asset_manifest.py', '# fixture only')
        manifests = {s: {'repo': s, 'revision': 'a'*40, 'archive_sha256': 'c'*64, 'files': {}} for s in ('lewm','spt','swm_compat')}
        source_hashes = {}
        for name, (source, relative) in m.SOURCE_KEYS.items():
            rec = self.write(f'source/{source}/{relative}', '# '+name)
            manifests[source]['files'][relative] = rec; source_hashes[name] = rec['sha256']
        manifests['swm_compat']['files']['environment.py'] = {'sha256': '1'*64, 'bytes': 5}
        for source, d in manifests.items(): self.write(f'state/{source}_source_manifest.json', d)
        self.write('state/ENVIRONMENT_READY.json', {'status': 'ISOLATED_ENVIRONMENT_READY', 'environment': {'python': 'fixture'}})
        cpu = {'status': 'STRICT_LOAD_AND_SYNTHETIC_WRAPPER_PASS', 'tasks': {}}
        for task in m.TASKS:
            config = self.write(f'official/{task}/config.json', {'_target_': 'original.Class'})
            weights = {'path': f'official/{task}/weights.pt', 'sha256': '3'*64, 'bytes': 20, 'lfs_sha_verified': True}
            self.write(f'manifests/{task}_model_assets.json', {'status': 'PINNED_ASSETS_DOWNLOADED_VERIFIED', 'task': task, 'kind': 'model', 'repo': task, 'revision': 'a'*40, 'files': {'weights.pt': weights, 'config.json': {'path': f'official/{task}/config.json', **config}}})
            identity = {'task': task, 'weights_sha256': weights['sha256'], 'config_sha256': config['sha256'], 'source_sha256': source_hashes}
            cpu['tasks'][task] = {'identity': identity, 'strict_load': {'missing_keys': [], 'unexpected_keys': [], 'state_dict_keys': 303}, 'contract': {'checkpoint_constructor_mapping': {'declared_model': 'original.Class', 'implemented_model': 'fixed.Class'}}}
            archive = {'path': f'data/source/{task}/train.zst', 'bytes': 10, 'sha256': '4'*64}
            source_receipt = self.write(f'manifests/{task}_data_assets.json', {'status': 'PINNED_ASSETS_DOWNLOADED_VERIFIED', 'task': task, 'kind': 'data', 'repo': task, 'revision': 'b'*40, 'files': {'train.zst': archive}})
            h5 = {'path': f'data/unpacked/{task}/train.h5', 'bytes': 100, 'sha256': '5'*64}
            member_path = f'state/unpack_members/{task}.json'
            member = {'status': 'COMPLETE', 'bytes': h5['bytes'], 'sha256': h5['sha256'], 'identity': {'archive': archive, 'member_name': 'train.h5', 'output_path': h5['path']}}
            receipt = self.write(member_path, member)
            h5.update(member_receipt=member_path, member_receipt_sha256=receipt['sha256'])
            self.write(f'manifests/{task}_data_unpacked.json', {'status': 'SOURCE_UNPACKED_SHA_VERIFIED', 'source_receipt_sha256': source_receipt['sha256'], 'files': [h5]})
            self.write(f'manifests/{task}_source_map.json', {'assets': {h5['sha256']: h5}, 'official_dataset': task, 'official_revision': 'b'*40})
            norm = self.write(f'manifests/{task}_normalization.json', {'mean': [0, 0], 'scale': [1, 1]})
            role = self.write(f'manifests/{task}_data_roles.json', {'normalization_sha256': norm['sha256'], 'counts': {'EVAL': 1}, 'episodes': [{'role': 'EVAL', 'episode_id': 'e', 'source_seed': None, 'family_id': None, 'family_evidence': 'FAMILY_UNKNOWN', 'exposure': {'OFFICIAL_PRETRAIN_EXPOSURE': 'UNKNOWN_EXACT_EPISODES', 'R3_REFIT_EXPOSURE': False, 'PRIOR_USER_STUDY_EXPOSURE': 'UNKNOWN'}}]})
            self.write(f'manifests/{task}_cache.json', {'status': 'FROZEN_OBSERVED_CACHE_COMPLETE', 'roles_manifest_sha256': role['sha256']})
            inputs = {f'manifests/{task}_{suffix}.json': m.identity_bytes((self.root/f'manifests/{task}_{suffix}.json').read_bytes())['sha256'] for suffix in ('cache','data_roles','source_map','normalization')}
            inputs['r3/model.py'] = m.identity_bytes((self.root/'r3/model.py').read_bytes())['sha256']
            self.write(f'artifacts/technical/{task}_real_model_attempt1/REAL_DATA_MODEL_CHECK.json', {'status': 'PASS', 'loaded_model_tensors_unchanged': True, 'optimizer_updates': 0, 'optimizer_objects_created': 0, 'official_identity': identity, 'input_hashes': inputs})
        self.write('state/OFFICIAL_STRICT_LOAD_CPU.json', cpu)

    def test_complete_metadata_no_binary_or_arrays_needed(self):
        x = m.assemble(self.root)
        self.assertEqual(x['identity_mismatches'], [])
        self.assertEqual(x['status'], 'COMPLETE_METADATA_ASSET_MAPPING')
        self.assertFalse((self.root/'official/pusht/weights.pt').exists())
        self.assertEqual(x['current_invocation']['raw_array_reads'], 0)
        self.assertEqual(x['tasks']['pusht']['exposure']['roles_counts'], {'EVAL': 1})
        self.assertEqual(x['tasks']['reacher']['exposure']['known_family_episodes'], 0)
        self.assertEqual(x['tasks']['reacher']['exposure']['source_seed_missing_episodes'], 1)

    def test_missing_evidence_no_manifest_created(self):
        (self.root/'state/OFFICIAL_STRICT_LOAD_CPU.json').unlink()
        x = m.assemble(self.root)
        self.assertEqual(x['status'], 'PENDING_REQUIRED_EVIDENCE')
        self.assertTrue(any(r['path'] == 'state/OFFICIAL_STRICT_LOAD_CPU.json' for r in x['missing']))
        with self.assertRaises(RuntimeError): m.create_only(self.root, 'manifest.json', x)
        self.assertFalse((self.root/'manifest.json').exists())

    def test_actual_wrong_cpu_weight_identity(self):
        rel = 'state/OFFICIAL_STRICT_LOAD_CPU.json'; x = self.read(rel)
        x['tasks']['pusht']['identity']['weights_sha256'] = 'f'*64; self.write(rel, x)
        x = m.assemble(self.root)
        self.assertTrue(any(r['reason'] == 'STRICT_WEIGHT_IDENTITY_MISMATCH' for r in x['identity_mismatches']))

    def test_wrong_member_archive_is_not_adopted(self):
        rel = 'state/unpack_members/pusht.json'; x = self.read(rel); x['identity']['archive']['sha256'] = 'f'*64
        self.write(rel, x); x = m.assemble(self.root)
        self.assertTrue(any(r['reason'] == 'MEMBER_ARCHIVE_NOT_IN_FIXED_DATA_SOURCE' for r in x['identity_mismatches']))

    def test_create_once_never_overwrites_and_changed_input_rejected(self):
        x = m.assemble(self.root); m.create_only(self.root, 'manifests/OFFICIAL_ASSET_MANIFEST.json', x)
        before = (self.root/'manifests/OFFICIAL_ASSET_MANIFEST.json').read_bytes()
        with self.assertRaises(FileExistsError): m.create_only(self.root, 'manifests/OFFICIAL_ASSET_MANIFEST.json', x)
        self.assertEqual(before, (self.root/'manifests/OFFICIAL_ASSET_MANIFEST.json').read_bytes())
        self.write('r3/model.py', '# changed')
        with self.assertRaises(RuntimeError): m.create_only(self.root, 'another.json', x)

    def test_declared_role_counts_mismatch_rejected(self):
        rel = 'manifests/pusht_data_roles.json'; x = self.read(rel); x['counts']['EVAL'] = 2
        self.write(rel, x); result = m.assemble(self.root)
        self.assertTrue(any(r['reason'] == 'DECLARED_ROLE_COUNTS_MISMATCH' for r in result['identity_mismatches']))

    def test_traversal_symlink_and_binary_read_rejected(self):
        with self.assertRaises(ValueError): m.checked(self.root, '../x')
        (self.root/'linked.json').symlink_to(self.root/'state/OFFICIAL_STRICT_LOAD_CPU.json')
        with self.assertRaises(ValueError): m.Evidence(self.root).read('linked.json')
        (self.root/'bad.npz').write_bytes(b'never interpret this')
        with self.assertRaises(ValueError): m.Evidence(self.root).read('bad.npz')


if __name__ == '__main__': unittest.main()
