"""Fetch pinned public author assets into one task-shared copy, without loading pickle."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, time
from huggingface_hub import hf_hub_download

ROOT = Path(os.environ.get('R3_ROOT', Path(__file__).resolve().parents[1]))
FLOOR = 30 * (1 << 30)

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def atomic(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(d, indent=2) + '\n'); os.replace(q, p)

def run(task, kind):
    meta = json.loads((ROOT / 'state' / f'{task}_{kind}_api.json').read_text())
    tree = json.loads((ROOT / 'state' / f'{task}_{kind}_tree.json').read_text())
    repo = 'quentinll/lewm-' + task
    wanted = ('config.json', 'weights.pt', 'README.md') if kind == 'model' else tuple(r['path'] for r in tree if r['path'].endswith('.zst')) + ('README.md',)
    destination = ROOT / ('official' if kind == 'model' else 'data/source') / task
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    start = time.time()
    for name in wanted:
        expected = next(r for r in tree if r['path'] == name)
        if shutil.disk_usage(ROOT).free < FLOOR + expected['size']:
            raise RuntimeError('Insufficient free space above reserved 30 GiB')
        print(json.dumps({'task': task, 'kind': kind, 'file': name, 'status': 'FETCH', 'bytes': expected['size']}), flush=True)
        p = Path(hf_hub_download(repo_id=repo, repo_type='dataset' if kind == 'data' else 'model',
                                filename=name, revision=meta['sha'], local_dir=destination, token=False))
        actual = digest(p)
        if p.stat().st_size != expected['size']: raise RuntimeError('Source size mismatch')
        if 'lfs' in expected and actual != expected['lfs']['oid']: raise RuntimeError('Source LFS SHA mismatch')
        records[name] = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': actual,
                         'lfs_sha_verified': 'lfs' in expected}
        atomic(ROOT / 'state' / f'{task}_{kind}_download_progress.json', {'files': records, 'revision': meta['sha']})
        print(json.dumps({'task': task, 'kind': kind, 'file': name, 'status': 'SHA_VERIFIED', 'bytes': expected['size']}), flush=True)
    receipt = {'status': 'PINNED_ASSETS_DOWNLOADED_VERIFIED', 'task': task, 'kind': kind, 'repo': repo,
               'revision': meta['sha'], 'files': records, 'seconds': time.time()-start,
               'unknown_pickle_executed': False, 'source_files_modified': False}
    atomic(ROOT / 'manifests' / f'{task}_{kind}_assets.json', receipt)
    return receipt

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('task', choices=['pusht', 'reacher']); p.add_argument('kind', choices=['model', 'data'])
    a = p.parse_args(); print(json.dumps(run(a.task, a.kind)), flush=True)
