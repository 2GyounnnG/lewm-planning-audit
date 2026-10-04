"""Metadata-only preflight fixtures; never import dm_control or create an env."""
import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from r3 import environment_check as e


class EnvironmentMetadataTests(unittest.TestCase):
    def test_mjcf_assets_are_resolved_and_hashed_before_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml = b'<mujoco><include file="./common/materials.xml"/></mujoco>'
            assets = {'common/materials.xml': b'<mujoco><asset><texture file="texture.png"/></asset></mujoco>',
                      'common/texture.png': b'fixture binary texture'}
            result = e.referenced_assets(xml, assets, root)
            self.assertEqual(result['status'], 'PASS'); self.assertEqual(len(result['required_file_references']), 2)
            self.assertFalse(result['compiled_environment'])
            self.assertEqual(result['provided_assets']['common/texture.png']['sha256'], hashlib.sha256(b'fixture binary texture').hexdigest())

    def test_missing_assets_and_filesystem_escape_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'suite').mkdir(); (root / 'outside.xml').write_text('<mujoco/>')
            result = e.referenced_assets(b'<mujoco><include file="../outside.xml"/><include file="missing.xml"/></mujoco>', {}, root / 'suite')
            self.assertEqual(result['status'], 'MISSING_ASSETS'); self.assertEqual(len(result['missing']), 2)
            self.assertEqual(result['required_file_references'], [])

    def test_dependency_import_failure_writes_precise_zero_execution_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'attempt'
            missing = ModuleNotFoundError("No module named 'example_missing_dependency'", name='example_missing_dependency')
            with patch('r3.planning.verify_official_sources', return_value={'revision': 'fixture'}), \
                 patch('r3.planning.load_official_api', side_effect=missing), \
                 patch.object(e, 'versions', return_value={}), \
                 patch.object(e.ctypes.util, 'find_library', return_value=None), \
                 patch.dict(e.os.environ, {'MUJOCO_GL': 'egl'}):
                result = e.preflight(output, execute=False)
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['import_or_source_error']['missing_module'], 'example_missing_dependency')
            self.assertTrue(all(value == 0 for value in result['counts'].values()))
            receipt = json.loads((output / 'SHA256.json').read_text())
            self.assertIn('ENVIRONMENT_PREFLIGHT.json', receipt['files'])
            with self.assertRaisesRegex(RuntimeError, 'prior check evidence'): e.preflight(output, execute=False)


if __name__ == '__main__': unittest.main()
