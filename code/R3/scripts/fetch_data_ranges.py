"""Resumable public pinned-data HTTP ranges; no model/data deserialization.

One sparse partial is written in-place. Verified chunks are never overwritten;
failed or unregistered files are retained. The CLI makes no URL/token log output.
"""
from __future__ import annotations
import argparse, concurrent.futures, contextlib, fcntl, hashlib, http.client, json, os, re, shutil
import threading, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(os.environ.get('R3_ROOT', Path(__file__).resolve().parents[1])).resolve()
FLOOR = 30 * (1 << 30)
CONTROL_RESERVE = 16 << 20  # Atomic chunk receipts and filesystem metadata.
CHUNK = 8 << 20
WORKERS = 8
VERSION = 'R3_PINNED_HTTP_RANGES_V1'

class DownloadError(RuntimeError):
    """Deliberately sanitized: never retain request URLs or HTTP headers."""


def checked(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    try: rel = path.relative_to(ROOT)
    except ValueError: raise DownloadError('Download path escaped the independent R3 root') from None
    if not rel.parts or '..' in rel.parts: raise DownloadError('Unsafe download path')
    current = ROOT
    for part in rel.parts:
        current = current / part
        if current.is_symlink(): raise DownloadError('Download path contains a symlink')
    return path


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic(path, data):
    path = checked(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = checked(path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp'))
    with temp.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path); sync_directory(path.parent)


def read_json(path): return json.loads(checked(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with checked(path).open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''): h.update(block)
    return h.hexdigest()


def ensure_space(additional=0):
    if shutil.disk_usage(ROOT).free < FLOOR + additional + CONTROL_RESERVE:
        raise DownloadError('Insufficient free space above reserved 30 GiB')


@contextlib.contextmanager
def file_lock(path, nonblocking=False):
    path = checked(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a+') as handle:
        flag = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try: fcntl.flock(handle, flag)
        except BlockingIOError: raise DownloadError('Another range downloader holds this task lock') from None
        yield


def pinned_assets(task):
    if task not in ('pusht', 'reacher'): raise ValueError('Unknown official task')
    api_path = checked(ROOT / 'state' / f'{task}_data_api.json')
    tree_path = checked(ROOT / 'state' / f'{task}_data_tree.json')
    api, tree = read_json(api_path), read_json(tree_path)
    repo = 'quentinll/lewm-' + task
    if api.get('id') != repo or not re.fullmatch(r'[0-9a-f]{40}', api.get('sha', '')):
        raise DownloadError('Pinned official repository/revision mismatch')
    if api.get('private') or api.get('gated') or api.get('disabled'):
        raise DownloadError('Only the pinned public ungated data is authorized')
    names = [row['path'] for row in tree if row.get('type') == 'file' and row['path'].endswith('.zst')] + ['README.md']
    if len(names) != len(set(names)) or len(names) != 2: raise DownloadError('Unexpected pinned official asset list')
    out = []
    for name in names:
        if PurePosixPath(name).name != name: raise DownloadError('Unexpected data asset path')
        rows = [row for row in tree if row.get('path') == name and row.get('type') == 'file']
        if len(rows) != 1: raise DownloadError('Pinned asset missing or duplicated')
        row = rows[0]
        if type(row.get('size')) is not int or row['size'] <= 0: raise DownloadError('Invalid pinned size')
        lfs = row.get('lfs')
        if name.endswith('.zst') and (not isinstance(lfs, dict) or lfs.get('size') != row['size'] or not re.fullmatch(r'[0-9a-f]{64}', lfs.get('oid', ''))):
            raise DownloadError('Archive must have exact pinned LFS SHA256 and size')
        if not lfs and (row['size'] > (1 << 20) or not re.fullmatch(r'[0-9a-f]{40}', row.get('oid', ''))):
            raise DownloadError('Small metadata requires pinned Git blob identity')
        out.append({'task': task, 'repo': repo, 'revision': api['sha'], 'name': name, 'bytes': row['size'],
                    'lfs_sha256': lfs['oid'] if lfs else None, 'git_blob_sha1': None if lfs else row['oid'],
                    'api_sha256': file_sha(api_path), 'tree_sha256': file_sha(tree_path)})
    return out


def verify_final(path, identity):
    path = checked(path)
    if not path.is_file() or path.stat().st_size != identity['bytes']: raise DownloadError('Final pinned asset size differs')
    h = hashlib.sha256(); git = hashlib.sha1(f'blob {identity["bytes"]}\0'.encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''): h.update(block); git.update(block)
    digest = h.hexdigest()
    if identity.get('lfs_sha256') and digest != identity['lfs_sha256']: raise DownloadError('Final pinned LFS SHA256 differs; partial retained')
    if identity.get('git_blob_sha1') and git.hexdigest() != identity['git_blob_sha1']: raise DownloadError('Final pinned Git blob SHA1 differs; partial retained')
    return digest


def url_for(identity):
    repo = urllib.parse.quote(identity['repo'], safe='/')
    name = urllib.parse.quote(identity['name'], safe='')
    return f'https://huggingface.co/datasets/{repo}/resolve/{identity["revision"]}/{name}'


def request_chunk(identity, start, count, *, retries=4, timeout=45, opener=urllib.request.urlopen):
    end = start + count - 1
    # Distinct range query avoids a CDN reusing a redirect cached for another range.
    url = url_for(identity) + f'?download=true&r3_range={start}-{end}'
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity', 'User-Agent': 'R3-Pinned-Data-Range/1'})
    last = 'REQUEST_FAILED'
    for attempt in range(retries):
        try:
            with opener(request, timeout=timeout) as response:
                status = response.status
                whole_small = not identity.get('lfs_sha256') and start == 0 and count == identity['bytes']
                expected = f'bytes {start}-{end}/{identity["bytes"]}'
                if status != 206 and not (whole_small and status == 200): raise DownloadError('Unexpected HTTP response status')
                if status == 206 and response.headers.get('Content-Range') != expected: raise DownloadError('HTTP Content-Range differs from exact requested interval')
                if response.headers.get('Content-Encoding', 'identity').lower() not in ('identity', ''): raise DownloadError('HTTP content encoding changed pinned byte representation')
                length = response.headers.get('Content-Length')
                if length is not None and (not length.isdigit() or int(length) != count): raise DownloadError('HTTP Content-Length differs from exact requested bytes')
                parts = []; received = 0
                while received <= count:
                    block = response.read(min(1 << 20, count + 1 - received))
                    if not block: break
                    parts.append(block); received += len(block)
                if received != count: raise DownloadError('HTTP body differs from exact requested byte count')
                return b''.join(parts)
        except urllib.error.HTTPError as error:
            last = f'HTTP_{int(error.code)}'
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            last = type(error).__name__
        except DownloadError as error:
            last = str(error)  # Our messages contain no response content or URLs.
        if attempt + 1 < retries: time.sleep(min(2 ** attempt, 8))
    raise DownloadError(f'Range retries exhausted ({last})') from None


def _pread_exact(fd, size, offset):
    parts = []; n = 0
    while n < size:
        block = os.pread(fd, size - n, offset + n)
        if not block: raise DownloadError('Registered chunk truncated')
        parts.append(block); n += len(block)
    return b''.join(parts)


def _pwrite_all(fd, data, offset):
    view = memoryview(data); n = 0
    while n < len(view):
        amount = os.pwrite(fd, view[n:], offset + n)
        if amount <= 0: raise DownloadError('Sparse partial write made no progress')
        n += amount


def download_asset(identity, *, chunk_bytes=CHUNK, workers=WORKERS, retries=4, timeout=45, opener=urllib.request.urlopen):
    if not 1 <= workers <= 8 or not 1 <= retries <= 8 or chunk_bytes <= 0 or not 0 < timeout <= 600: raise ValueError('Invalid bounded downloader settings')
    destination = checked(ROOT / 'data/source' / identity['task'] / identity['name'])
    partial = checked(destination.with_name(destination.name + '.ranges.partial'))
    receipt = checked(ROOT / 'state/range_downloads' / identity['task'] / (identity['name'] + '.json'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    layout = {'version': VERSION, 'asset': identity, 'chunk_bytes': chunk_bytes,
              'final_path': str(destination.relative_to(ROOT)), 'partial_path': str(partial.relative_to(ROOT))}
    total = identity['bytes']; chunks = [(i, start, min(chunk_bytes, total - start)) for i, start in enumerate(range(0, total, chunk_bytes))]
    if receipt.exists():
        record = read_json(receipt)
        if record.get('identity') != layout: raise DownloadError('Registered range receipt identity differs')
    else:
        if partial.exists(): raise DownloadError('Unregistered range partial retained; refusing to adopt or overwrite it')
        record = {'identity': layout, 'status': 'REGISTERED', 'chunks': {}}
        if destination.exists():
            digest = verify_final(destination, identity)
            record.update(status='COMPLETE', sha256=digest, adopted_verified_canonical=True)
        atomic(receipt, record)
    if record['status'] in ('READY_TO_INSTALL', 'COMPLETE'):
        if destination.exists() and partial.exists(): raise DownloadError('Both final and sparse partial exist; retained for inspection')
        path = destination if destination.exists() else partial
        if record['status'] == 'COMPLETE' and path != destination: raise DownloadError('Completed canonical asset is missing')
        digest = verify_final(path, identity)
        if record.get('sha256') != digest: raise DownloadError('Completed range receipt differs from bytes')
        if path == partial:
            os.rename(partial, destination); sync_directory(destination.parent)
        if record['status'] != 'COMPLETE': record['status'] = 'COMPLETE'; atomic(receipt, record)
        return {'path': str(destination.relative_to(ROOT)), 'bytes': total, 'sha256': digest, 'lfs_sha_verified': bool(identity.get('lfs_sha256'))}
    if record['status'] not in ('REGISTERED', 'DOWNLOADING', 'INCOMPLETE'): raise DownloadError('Unsupported range recovery status')
    if destination.exists(): raise DownloadError('Canonical asset appeared before verified install; retained unchanged')
    if partial.exists():
        if not partial.is_file(): raise DownloadError('Registered sparse partial is not a regular file')
        size = partial.stat().st_size
        if size != total and not (size == 0 and not record['chunks'] and record['status'] == 'REGISTERED'):
            raise DownloadError('Registered sparse partial size differs')
        fd = os.open(partial, os.O_RDWR | os.O_NOFOLLOW)
        if size == 0:
            try:
                ensure_space(total); os.ftruncate(fd, total); os.fsync(fd)
            except BaseException:
                os.close(fd); raise
    else:
        if record['chunks']: raise DownloadError('Registered verified chunks are missing; refusing silent restart')
        ensure_space(total)
        fd = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.ftruncate(fd, total); os.fsync(fd); sync_directory(partial.parent)
        except BaseException:
            os.close(fd); raise
    write_lock = threading.Lock(); stop = threading.Event()
    try:
        chunk_map = {str(index): (start, count) for index, start, count in chunks}
        if not isinstance(record['chunks'], dict) or set(record['chunks']) - set(chunk_map): raise DownloadError('Unexpected retained chunk IDs')
        for key, saved in record['chunks'].items():
            start, count = chunk_map[key]
            if saved.get('start') != start or saved.get('bytes') != count or hashlib.sha256(_pread_exact(fd, count, start)).hexdigest() != saved.get('sha256'):
                raise DownloadError('Registered chunk content differs; retained unchanged')
        missing = [(i, start, count) for i, start, count in chunks if str(i) not in record['chunks']]
        ensure_space(sum(count for _, _, count in missing))
        record['status'] = 'DOWNLOADING'; record.pop('last_error', None); atomic(receipt, record)
        def one(index, start, count):
            if stop.is_set(): return
            block = request_chunk(identity, start, count, retries=retries, timeout=timeout, opener=opener)
            digest = hashlib.sha256(block).hexdigest()
            with write_lock:
                if stop.is_set(): return
                checked(partial)
                on_disk, opened = partial.stat(), os.fstat(fd)
                if (on_disk.st_dev, on_disk.st_ino) != (opened.st_dev, opened.st_ino): raise DownloadError('Sparse partial was replaced during download')
                with file_lock(ROOT / 'state/range_downloads/disk_write.lock'):
                    ensure_space(count)
                    _pwrite_all(fd, block, start); os.fsync(fd)
                    record['chunks'][str(index)] = {'start': start, 'bytes': count, 'sha256': digest}
                    atomic(receipt, record)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix='pinned-range') as pool:
            futures = [pool.submit(one, *args) for args in missing]
            try:
                for future in concurrent.futures.as_completed(futures): future.result()
            except BaseException:
                stop.set()
                for future in futures: future.cancel()
                raise
        if len(record['chunks']) != len(chunks): raise DownloadError('Chunk completion count differs')
        os.fsync(fd)
        digest = verify_final(partial, identity)
        record.update(status='READY_TO_INSTALL', sha256=digest); atomic(receipt, record)
    except BaseException as error:
        record.update(status='INCOMPLETE', last_error={'type': type(error).__name__, 'message': str(error) if isinstance(error, DownloadError) else 'Failure retained; consult sanitized process status'})
        atomic(receipt, record)
        raise
    finally: os.close(fd)
    if destination.exists(): raise DownloadError('Canonical asset appeared at install boundary; retained unchanged')
    os.rename(partial, destination); sync_directory(destination.parent)
    record['status'] = 'COMPLETE'; atomic(receipt, record)
    return {'path': str(destination.relative_to(ROOT)), 'bytes': total, 'sha256': digest, 'lfs_sha_verified': bool(identity.get('lfs_sha256'))}


def run(task, *, retries=4, timeout=45):
    started = time.perf_counter(); assets = pinned_assets(task)
    with file_lock(ROOT / 'state/range_downloads' / (task + '.lock'), nonblocking=True):
        records = {}
        for identity in assets:
            print(json.dumps({'task': task, 'file': identity['name'], 'status': 'RANGE_FETCH_OR_VERIFIED_RESUME', 'bytes': identity['bytes']}), flush=True)
            records[identity['name']] = download_asset(identity, retries=retries, timeout=timeout)
            print(json.dumps({'task': task, 'file': identity['name'], 'status': 'SHA_VERIFIED', 'bytes': identity['bytes']}), flush=True)
        result = {'status': 'PINNED_ASSETS_DOWNLOADED_VERIFIED', 'task': task, 'kind': 'data', 'repo': assets[0]['repo'],
                  'revision': assets[0]['revision'], 'files': records, 'seconds': time.perf_counter() - started,
                  'unknown_pickle_executed': False, 'source_files_modified': False,
                  'download_method': VERSION, 'range_chunk_bytes': CHUNK, 'range_workers': WORKERS}
        target = checked(ROOT / 'manifests' / (task + '_data_assets.json'))
        if target.exists():
            old = read_json(target)
            if any(old.get(k) != result[k] for k in ('status', 'task', 'kind', 'repo', 'revision', 'files')):
                raise DownloadError('Existing data asset manifest differs; preserved without overwrite')
            return old
        atomic(target, result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('task', choices=['pusht', 'reacher'])
    parser.add_argument('--retries', type=int, default=4); parser.add_argument('--timeout', type=float, default=45)
    args = parser.parse_args()
    try: print(json.dumps(run(args.task, retries=args.retries, timeout=args.timeout)), flush=True)
    except BaseException as error:
        print(json.dumps({'status': 'DOWNLOAD_FAILED_PARTIAL_PRESERVED', 'type': type(error).__name__,
                          'reason': str(error) if isinstance(error, (DownloadError, ValueError)) else 'See preserved chunk receipt; raw network details suppressed'}), flush=True)
        raise SystemExit(1) from None
