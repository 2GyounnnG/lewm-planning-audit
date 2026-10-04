"""Publish verified, completed task cache for direct external recovery; no GPU work."""
from pathlib import Path
import argparse, fcntl, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3 import common


def publish(task):
    common.require_authorization({}, technical=True)
    common.ensure_space()
    root = common.ROOT
    output = root / f'manifests/RECOVERY_{task.upper()}_INPUTS_V1.json'
    if output.exists():
        raise RuntimeError('Existing publication remains immutable; recover its exact manifest')
    cache = common.read_json(f'manifests/{task}_cache.json')
    if cache['status'] != 'FROZEN_OBSERVED_CACHE_COMPLETE' or cache['task'] != task:
        raise RuntimeError('Only a completed official cache may be published')
    for name, expected in cache['input_hashes'].items():
        if common.sha256(root / name) != expected:
            raise RuntimeError('Frozen cache input changed: ' + name)
    roles = common.read_json(f'manifests/{task}_data_roles.json')
    expected_ids = {e['episode_id'] for e in roles['episodes'] if e['role'] in ('REFIT_TRAIN', 'MONITOR', 'TECH', 'EVAL')}
    if set(cache['episodes']) != expected_ids:
        raise RuntimeError('Completed cache episode identities differ')
    names = [f'manifests/{task}_{suffix}.json' for suffix in
             ('cache', 'data_roles', 'source_map', 'normalization', 'model_assets', 'data_assets', 'data_unpacked')]
    names += ['state/ROLE_SELECTION_CONTRACT.json']
    names += (['state/REACHER_METADATA_AUDIT.json'] if task == 'reacher' else
              ['state/PUSHT_PRIOR_SOURCE_ALIGNMENT.json', 'state/PUSHT_PRIOR_SOURCE_ALIGNMENT_V2.json', 'state/PUSHT_ENCODER_TIMING.json'])
    for record in cache['episodes'].values():
        path = root / record['path']
        if not record['path'].startswith(f'data/cache/{task}/') or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Cache file outside its fixed task directory')
        if path.stat().st_size != record['bytes'] or common.sha256(path) != record['sha256']:
            raise RuntimeError('Completed cache bytes differ: ' + record['path'])
        names += [record['path'], str(Path(record['path']).with_suffix('.json'))]
    source = common.read_json(f'manifests/{task}_source_map.json')
    asset_hashes = set(source['assets'])
    covered = set()
    for path in sorted((root / 'state/unpack_members').glob('*.json')):
        # The receipt itself must reference an exact source asset identity.
        raw = path.read_text()
        if any(value in raw for value in asset_hashes):
            names.append(str(path.relative_to(root)))
            covered.update(value for value in asset_hashes if value in raw)
    if covered != asset_hashes:
        raise RuntimeError('Missing original full-SHA unpack member receipt')
    files = {}
    for name in sorted(set(names)):
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError('Publication path is not an ordinary R3 file: ' + name)
        files[name] = {'bytes': path.stat().st_size, 'sha256': common.sha256(path)}
    publication = {'status': 'COMPLETED_INPUTS_SHARED_OFFICIAL_CACHE', 'task': task,
                   'published_at': common.now(), 'source_HDF5_restorable_by_official_SHA_not_copied': True,
                   'optimizer_updates': 0, 'files': files}
    common.atomic_json(output, publication)
    return {'manifest': str(output.relative_to(root)), 'sha256': common.sha256(output),
            'files': len(files), 'bytes': sum(x['bytes'] for x in files.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=common.TASKS)
    args = parser.parse_args()
    with (common.ROOT / 'state/input_publication.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(publish(args.task)), flush=True)
