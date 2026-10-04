"""Create an official-asset index from existing JSON evidence, without loading assets.

Default is a read-only preview. --write creates the requested manifest once, only
when all required evidence and identity joins pass. Large binary SHA identities
are inherited from named receipts, not freshly rehashed or independently replayed.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse
import datetime
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'R3_OFFICIAL_ASSET_MANIFEST_V1'
TASKS = ('pusht', 'reacher')
SOURCE_KEYS = {
    'lewm_jepa': ('lewm', 'jepa.py'), 'lewm_module': ('lewm', 'module.py'),
    'lewm_train': ('lewm', 'train.py'), 'lewm_utils': ('lewm', 'utils.py'),
    'vit_constructor': ('spt', 'stable_pretraining/backbone/utils.py')}


def checked(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts:
        raise ValueError('Not a root-relative evidence path: '+str(relative))
    out = root
    for part in p.parts:
        out = out / part
        if out.is_symlink(): raise ValueError('Symlink evidence/output is not accepted: '+str(relative))
    return out


def identity_bytes(b):
    return {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.files = {}; self.missing = []; self.mismatches = []

    def read(self, relative):
        p = checked(self.root, relative)
        if not p.is_file():
            self.missing.append({'path': relative, 'reason': 'REQUIRED_METADATA_ABSENT'}); return {}
        if p.suffix not in ('.json', '.py', '.md'):
            raise ValueError('Only metadata/text source can be read: '+relative)
        if p.stat().st_size > 256 * (1 << 20): raise ValueError('Metadata size cap exceeded: '+relative)
        b = p.read_bytes(); self.files[relative] = identity_bytes(b)
        if p.suffix != '.json': return {}
        try:
            d = json.loads(b)
            if not isinstance(d, dict): raise ValueError('Expected JSON object')
            return d
        except (ValueError, UnicodeError) as error:
            self.mismatches.append({'path': relative, 'reason': 'INVALID_METADATA', 'detail': str(error)}); return {}

    def require(self, condition, path, reason):
        if not condition: self.mismatches.append({'path': path, 'reason': reason})

    def same(self, left, right, path, reason):
        self.require(left is not None and left == right, path, reason)

    def ref(self, relative):
        return {'path': relative, **self.files.get(relative, {}), 'present': relative in self.files}

    def stable(self):
        for rel, rec in self.files.items():
            if identity_bytes(checked(self.root, rel).read_bytes()) != rec:
                raise RuntimeError('Evidence changed while assembling: '+rel)


def asset_record(e, record, path):
    if not isinstance(record, dict): record = {}
    p = record.get('path')
    if isinstance(p, str): checked(e.root, p)
    e.require(isinstance(p, str), path, 'MISSING_ASSET_PATH')
    e.require(type(record.get('bytes')) is int and record['bytes'] >= 0, path, 'INVALID_ASSET_BYTE_COUNT')
    h = record.get('sha256')
    e.require(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h), path, 'INVALID_ASSET_SHA256')
    return record


def assemble(root):
    e = Evidence(root)
    strict_path = 'state/OFFICIAL_STRICT_LOAD_CPU.json'; strict = e.read(strict_path)
    env_path = 'state/ENVIRONMENT_READY.json'; environment = e.read(env_path)
    e.require(strict.get('status') == 'STRICT_LOAD_AND_SYNTHETIC_WRAPPER_PASS', strict_path, 'STRICT_CPU_NOT_PASS')
    e.require(environment.get('status') == 'ISOLATED_ENVIRONMENT_READY', env_path, 'ENVIRONMENT_NOT_READY')
    e.require(bool(environment.get('environment')), env_path, 'ENVIRONMENT_VERSIONS_MISSING')
    e.read('r3/model.py'); e.read('scripts/build_official_asset_manifest.py')
    sources = {}
    for name in ('lewm', 'spt', 'swm_compat'):
        path = f'state/{name}_source_manifest.json'; source = e.read(path)
        e.require(all(source.get(k) for k in ('repo', 'revision', 'archive_sha256', 'files')), path, 'INCOMPLETE_FIXED_SOURCE_IDENTITY')
        sources[name] = {'evidence': e.ref(path), **{k: source.get(k) for k in ('repo', 'revision', 'archive_sha256')}, 'files': source.get('files', {})}
    fixed_source_hashes = {}
    for key, (source, relative) in SOURCE_KEYS.items():
        path = f'source/{source}/{relative}'; e.read(path)
        actual = e.files.get(path, {}).get('sha256')
        e.same(actual, sources[source]['files'].get(relative, {}).get('sha256'), path, 'SOURCE_TEXT_MANIFEST_MISMATCH')
        fixed_source_hashes[key] = actual
    tasks = {}
    for task in TASKS:
        model_path = f'manifests/{task}_model_assets.json'; model = e.read(model_path)
        data_path = f'manifests/{task}_data_assets.json'; data = e.read(data_path)
        unpack_path = f'manifests/{task}_data_unpacked.json'; unpack = e.read(unpack_path)
        map_path = f'manifests/{task}_source_map.json'; source_map = e.read(map_path)
        roles_path = f'manifests/{task}_data_roles.json'; roles = e.read(roles_path)
        cache_path = f'manifests/{task}_cache.json'; cache = e.read(cache_path)
        norm_path = f'manifests/{task}_normalization.json'; norm = e.read(norm_path)
        real_path = f'artifacts/technical/{task}_real_model_attempt1/REAL_DATA_MODEL_CHECK.json'; real = e.read(real_path)
        config_path = f'official/{task}/config.json'; config = e.read(config_path)
        for path, record, kind in ((model_path, model, 'model'), (data_path, data, 'data')):
            e.require(record.get('status') == 'PINNED_ASSETS_DOWNLOADED_VERIFIED', path, 'PINNED_DOWNLOAD_NOT_VERIFIED')
            e.require(record.get('task') == task and record.get('kind') == kind and bool(record.get('repo')) and bool(record.get('revision')), path, 'MISSING_OR_DIFFERENT_REPOSITORY_IDENTITY')
            for name, asset in record.get('files', {}).items(): asset_record(e, asset, path+':'+name)
        weights = model.get('files', {}).get('weights.pt', {}); config_rec = model.get('files', {}).get('config.json', {})
        asset_record(e, weights, model_path+':weights.pt'); asset_record(e, config_rec, model_path+':config.json')
        e.require(weights.get('lfs_sha_verified') is True, model_path, 'WEIGHTS_LFS_NOT_VERIFIED')
        e.same(config_rec.get('sha256'), e.files.get(config_path, {}).get('sha256'), model_path, 'ACTUAL_CONFIG_HASH_MISMATCH')
        cpu = strict.get('tasks', {}).get(task, {}); identity = cpu.get('identity', {}); load = cpu.get('strict_load', {})
        e.same(identity.get('weights_sha256'), weights.get('sha256'), strict_path+':'+task, 'STRICT_WEIGHT_IDENTITY_MISMATCH')
        e.same(identity.get('config_sha256'), config_rec.get('sha256'), strict_path+':'+task, 'STRICT_CONFIG_IDENTITY_MISMATCH')
        e.same(identity.get('source_sha256'), fixed_source_hashes, strict_path+':'+task, 'STRICT_SOURCE_IDENTITY_MISMATCH')
        e.require(load.get('missing_keys') == [] and load.get('unexpected_keys') == [] and type(load.get('state_dict_keys')) is int and load['state_dict_keys'] > 0, strict_path+':'+task, 'STRICT_KEYS_NOT_CLEAN')
        mapping = cpu.get('contract', {}).get('checkpoint_constructor_mapping', {})
        e.require(bool(mapping), strict_path+':'+task, 'CONSTRUCTOR_MAPPING_MISSING')
        e.same(mapping.get('declared_model'), config.get('_target_'), config_path, 'DECLARED_MODEL_CONFIG_MISMATCH')
        e.require(real.get('status') == 'PASS' and real.get('loaded_model_tensors_unchanged') is True and real.get('optimizer_updates') == 0 and real.get('optimizer_objects_created') == 0, real_path, 'REAL_ZERO_UPDATE_CHECK_NOT_PASS')
        e.same(real.get('official_identity'), identity, real_path, 'REAL_AND_CPU_MODEL_IDENTITIES_DIFFER')
        for input_path in (cache_path, roles_path, map_path, norm_path, 'r3/model.py'):
            e.same(real.get('input_hashes', {}).get(input_path), e.files.get(input_path, {}).get('sha256'), real_path, 'REAL_INPUT_IDENTITY_DIFFERS:'+input_path)
        e.require(unpack.get('status') == 'SOURCE_UNPACKED_SHA_VERIFIED', unpack_path, 'UNPACK_NOT_VERIFIED')
        e.same(unpack.get('source_receipt_sha256'), e.files.get(data_path, {}).get('sha256'), unpack_path, 'SOURCE_RECEIPT_MISMATCH')
        members = []
        e.require(bool(unpack.get('files')), unpack_path, 'UNPACKED_FILES_MISSING')
        for rec in unpack.get('files', []):
            asset_record(e, rec, unpack_path)
            member_path = rec.get('member_receipt')
            if not member_path:
                e.missing.append({'path': unpack_path, 'reason': 'MEMBER_RECEIPT_MISSING'}); continue
            member = e.read(member_path); member_identity = member.get('identity', {})
            e.same(rec.get('member_receipt_sha256'), e.files.get(member_path, {}).get('sha256'), member_path, 'MEMBER_RECEIPT_SHA_MISMATCH')
            e.require(member.get('status') == 'COMPLETE', member_path, 'MEMBER_NOT_COMPLETE')
            for key in ('bytes', 'sha256'): e.same(member.get(key), rec.get(key), member_path, 'MEMBER_'+key+'_MISMATCH')
            e.same(member_identity.get('output_path'), rec.get('path'), member_path, 'MEMBER_OUTPUT_MISMATCH')
            archive = member_identity.get('archive', {})
            e.require(any(archive == {k: a.get(k) for k in ('path', 'bytes', 'sha256')} for a in data.get('files', {}).values()), member_path, 'MEMBER_ARCHIVE_NOT_IN_FIXED_DATA_SOURCE')
            e.require(bool(member_identity.get('member_name')), member_path, 'MEMBER_NAME_MISSING')
            members.append({'file': rec, 'evidence': e.ref(member_path), 'identity': member_identity})
        h5_files = {r.get('sha256'): r for r in unpack.get('files', []) if str(r.get('path', '')).endswith(('.h5', '.hdf5'))}
        e.same(source_map.get('assets'), h5_files, map_path, 'SOURCE_MAP_UNPACKED_IDENTITY_MISMATCH')
        e.same(source_map.get('official_dataset'), data.get('repo'), map_path, 'DATA_REPO_MISMATCH')
        e.same(source_map.get('official_revision'), data.get('revision'), map_path, 'DATA_REVISION_MISMATCH')
        e.same(cache.get('roles_manifest_sha256'), e.files.get(roles_path, {}).get('sha256'), cache_path, 'CACHE_ROLES_MISMATCH')
        e.same(roles.get('normalization_sha256'), e.files.get(norm_path, {}).get('sha256'), roles_path, 'ROLES_NORMALIZATION_MISMATCH')
        e.require(cache.get('status') == 'FROZEN_OBSERVED_CACHE_COMPLETE', cache_path, 'CACHE_NOT_COMPLETE')
        episodes = roles.get('episodes', [])
        e.require(bool(episodes), roles_path, 'EXPOSURE_EPISODE_RECORDS_MISSING')
        exposure_counts = {}
        for key in ('OFFICIAL_PRETRAIN_EXPOSURE', 'R3_REFIT_EXPOSURE', 'PRIOR_USER_STUDY_EXPOSURE'):
            e.require(all(key in row.get('exposure', {}) for row in episodes), roles_path, 'EXPOSURE_BOOK_MISSING:'+key)
            # Keep every distinct disclosure, but not the large per-episode reset metadata.
            counts = Counter(json.dumps(row.get('exposure', {}).get(key), sort_keys=True) for row in episodes)
            exposure_counts[key] = [{'value': json.loads(value), 'episodes': n} for value, n in sorted(counts.items())]
        family_counts = Counter(row.get('family_evidence', 'UNRECORDED') for row in episodes)
        role_counts = dict(Counter(row['role'] for row in episodes))
        if 'counts' in roles:
            declared = roles['counts']
            e.require(isinstance(declared, dict) and all(type(v) is int and v >= 0 for v in declared.values()), roles_path, 'INVALID_DECLARED_ROLE_COUNTS')
            e.require(isinstance(declared, dict) and {k: v for k, v in declared.items() if v != 0} == role_counts, roles_path, 'DECLARED_ROLE_COUNTS_MISMATCH')
        tasks[task] = {
            'original_checkpoint': {'evidence': e.ref(model_path), 'repository': model.get('repo'), 'revision': model.get('revision'), 'weights': weights, 'config': config_rec},
            'conversion_result': {'status': 'NO_WEIGHT_SERIALIZATION_OR_KEY_CONVERSION', 'constructor_mapping': mapping,
                                  'basis': 'Original state_dict loaded strict with unchanged keys; only constructor configuration mapping is used.', 'implementation': e.ref('r3/model.py')},
            'strict_CPU_load': {'evidence': e.ref(strict_path), **cpu},
            'real_zero_update_check': {'evidence': e.ref(real_path), 'status': real.get('status'), 'official_identity': real.get('official_identity'),
                                      'optimizer_updates': real.get('optimizer_updates'), 'loaded_model_tensors_unchanged': real.get('loaded_model_tensors_unchanged'), 'checks': real.get('model_checks', {}).get('checks')},
            'data': {'evidence': e.ref(data_path), 'repository': data.get('repo'), 'revision': data.get('revision'), 'source_files': data.get('files'),
                     'unpack_evidence': e.ref(unpack_path), 'members': members, 'source_map_evidence': e.ref(map_path), 'source_map': source_map},
            'exposure': {'episode_ledger_evidence': e.ref(roles_path), 'books': exposure_counts,
                         'family_evidence_episode_counts': dict(family_counts), 'known_family_episodes': sum(r.get('family_id') is not None for r in episodes),
                         'source_seed_missing_episodes': sum(r.get('source_seed') is None for r in episodes),
                         'roles_counts': role_counts,
                         'pretraining_exposure_note': roles.get('pretraining_exposure_note'),
                         'prior_user_study_source_alignment': roles.get('prior_user_study_source_alignment'),
                         'interpretation': 'R3 split is refit-held-out, not evidence of official-pretraining-unseen episodes. Missing source seeds/families are not inferred as known.'},
            'action_normalization': {'evidence': e.ref(norm_path), 'metadata': norm},
            'observed_cache': {'evidence': e.ref(cache_path), 'status': cache.get('status'), 'roles_manifest_sha256': cache.get('roles_manifest_sha256')}}
    e.stable()
    return {'version': VERSION, 'status': 'COMPLETE_METADATA_ASSET_MAPPING' if not e.missing and not e.mismatches else 'PENDING_REQUIRED_EVIDENCE',
            'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tasks': tasks, 'sources': sources,
            'environment': {'evidence': e.ref(env_path), **environment}, 'files': e.files,
            'missing': e.missing, 'identity_mismatches': e.mismatches,
            'verification_scope': 'Metadata joins and hashes only. Binary identities are inherited from explicitly identified prior download/unpack/strict-load/real-check receipts; no fresh binary read or scientific result selection.',
            'current_invocation': {'GPU_calls': 0, 'environment_calls': 0, 'optimizer_updates': 0, 'raw_array_reads': 0, 'model_weight_loads': 0}}


def create_only(root, output, manifest):
    if manifest.get('status') != 'COMPLETE_METADATA_ASSET_MAPPING': raise RuntimeError('Required evidence is incomplete; no official manifest written')
    root = Path(root).resolve(); p = checked(root, output)
    for rel, rec in manifest['files'].items():
        if identity_bytes(checked(root, rel).read_bytes()) != rec: raise RuntimeError('Input changed before publication: '+rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    # Exclusive creation also refuses an existing partial. Never replace prior evidence.
    with p.open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return {'path': output, **identity_bytes(payload)}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', default='manifests/OFFICIAL_ASSET_MANIFEST.json'); p.add_argument('--write', action='store_true')
    a = p.parse_args(); result = assemble(a.root)
    if a.write and result['status'] == 'COMPLETE_METADATA_ASSET_MAPPING': result['published_manifest'] = create_only(a.root, a.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'COMPLETE_METADATA_ASSET_MAPPING' else 2)


if __name__ == '__main__': main()
