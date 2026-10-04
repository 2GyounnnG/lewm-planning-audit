"""Read-only verification of SHA-anchored component seals and their local bytes.

API: validate_component_seals(config) -> sorted list of {path, bytes, sha256}.
Each config['module_recovery_seals'][name] requires local_path and
expected_sha256. Remote-named records require explicit path_mappings:
[{source_prefix, local_prefix}], or file_mappings: {source_path: local_path}.
The same mappings may be provided globally as component_path_mappings and
component_file_mappings. Longest prefix wins; conflicting equal mappings fail.

INITIAL_STAGE and H2 automatically expand their anchored recovery_manifest /
manifest local-file collections. H3X local_files and inherited dependencies
are verified directly. C1/C2 recovery_manifest.files are expanded using their
original record paths and the explicit bundle prefix mapping. Optional
manifest_bindings entries {seal_field, local_path?, member_root?,
member_field?, record_field?} support explicit relative-member layouts.

No file writes, model/statistical execution, downloads, or replacement hashes.
Remote-only originals are not falsely asserted to be recovered locally.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")
_REMOTE_ONLY = frozenset({
    'remote_only_files', 'remote_only_collections', 'remote_only_TRAIN_cache',
    'remote_required_files', 'remote_files', 'remote_to_local',
    'remote_module_seal_paths', 'V1_archives_retained_primary',
})
_LOCAL_COLLECTIONS = ('verified_local_files', 'local_files',
                      'inherited_local_readonly_dependencies')


class ComponentSealError(ValueError):
    """A pinned seal, mapping, expected identity or current byte check failed."""


def _fail(message):
    raise ComponentSealError(message)


def _field(obj, dotted):
    try:
        for key in dotted.split('.'):
            obj = obj[key]
    except (KeyError, TypeError):
        _fail('Missing anchored field: ' + dotted)
    return obj


def _identity(record):
    if not isinstance(record, dict) or not {'bytes', 'sha256'} <= record.keys():
        _fail('Expected an original bytes/SHA record: ' + repr(record)[:200])
    size, digest = record['bytes'], record['sha256']
    if type(size) is not int or size < 0 or not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        _fail('Malformed bytes/SHA record: ' + repr(record)[:200])
    return size, digest.lower()


def _file_records(value):
    """Walk seal metadata only; never follow arbitrary JSON provenance files."""
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            _identity(value)
            yield value
        else:
            for key, child in value.items():
                if key not in _REMOTE_ONLY:
                    yield from _file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _file_records(child)


def _rows(value):
    if isinstance(value, dict):
        return list(value.items())
    if isinstance(value, list):
        return [(r.get('relative_path'), r) for r in value]
    _fail('Local file collection must be a list or dictionary')


def _check_file(row):
    path = Path(row['path'])
    if not path.is_file():
        _fail('Missing sealed local file: ' + str(path))
    before = path.stat()
    if before.st_size != row['bytes']:
        _fail('Sealed local size changed: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        _fail('Sealed file changed while checking: ' + str(path))
    if digest.hexdigest() != row['sha256']:
        _fail('Sealed local SHA changed: ' + str(path))


def validate_component_seals(config):
    """Verify every supplied component, returning original expected identities.

    The caller remains responsible for requiring its complete module set and
    for the semantic release/CPU gates. This function accepts a subset for
    read-only early validation of already finished components.
    """
    components = config.get('module_recovery_seals')
    if not isinstance(components, dict) or not components:
        _fail('No component recovery seals configured')
    expected = {}

    def add(path, original):
        size, digest = _identity(original)
        path = str(Path(path).expanduser().resolve())
        row = {'path': path, 'bytes': size, 'sha256': digest}
        old = expected.get(path)
        if old is not None and old != row:
            _fail('Conflicting sealed identities for local path: ' + path)
        expected[path] = row
        return row

    def checked_json(path, original):
        row = add(path, original)
        data = Path(row['path']).read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            _fail('Anchored seal/manifest changed: ' + row['path'])
        try:
            return json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail('Invalid anchored JSON: ' + row['path'] + ': ' + str(exc))

    for component, spec in components.items():
        seal_path = Path(spec['local_path']).expanduser().resolve()
        pinned = spec.get('expected_sha256')
        if not isinstance(pinned, str) or not _DIGEST.fullmatch(pinned):
            _fail(component + ': expected_sha256 is required, never inferred')
        if not seal_path.is_file():
            _fail(component + ': seal not present: ' + str(seal_path))
        seal = checked_json(seal_path, {'bytes': seal_path.stat().st_size, 'sha256': pinned})
        exact = dict(config.get('component_file_mappings', {}))
        for key, target in spec.get('file_mappings', {}).items():
            if key in exact and Path(exact[key]).resolve() != Path(target).resolve():
                _fail(component + ': conflicting exact file mapping: ' + key)
            exact[key] = target
        prefixes = list(config.get('component_path_mappings', [])) + list(spec.get('path_mappings', []))
        bindings = {b['seal_field']: b for b in spec.get('manifest_bindings', [])}
        for field, binding in bindings.items():
            reference = _field(seal, field)
            if binding.get('local_path'):
                source = reference['path']; target = binding['local_path']
                if source in exact and Path(exact[source]).resolve() != Path(target).resolve():
                    _fail(component + ': conflicting manifest mapping: ' + source)
                exact[source] = target

        def mapped(source):
            if source in exact:
                return Path(exact[source]).expanduser().resolve()
            found = []
            source_path = PurePosixPath(source)
            for mapping in prefixes:
                prefix = PurePosixPath(mapping['source_prefix'])
                try:
                    relative = source_path.relative_to(prefix)
                except ValueError:
                    continue
                target = (Path(mapping['local_prefix']).expanduser().resolve() / str(relative)).resolve()
                found.append((len(prefix.parts), target))
            if found:
                longest = max(n for n, _ in found)
                choices = {p for n, p in found if n == longest}
                if len(choices) != 1:
                    _fail(component + ': ambiguous prefix mapping: ' + source)
                return choices.pop()
            if not Path(source).is_absolute() or source.startswith('/workspace/'):
                _fail(component + ': missing explicit local mapping: ' + source)
            return Path(source).expanduser().resolve()

        # Seal-level receipts, code, archives and H3X local/dependency lists.
        for reference in _file_records(seal):
            add(mapped(reference['path']), reference)

        manifest_fields = {name for name in ('manifest', 'recovery_manifest')
                           if isinstance(seal.get(name), dict) and 'path' in seal[name]}
        manifest_fields.update(bindings)
        expanded = 0
        for field in sorted(manifest_fields):
            reference = _field(seal, field)
            manifest = checked_json(mapped(reference['path']), reference)
            binding = bindings.get(field, {})
            collections = [name for name in _LOCAL_COLLECTIONS if name in manifest]
            if binding.get('member_field'):
                collections = [binding['member_field']]
            elif not collections and 'files' in manifest:
                collections = ['files']
            if not collections:
                _fail(component + ': anchored manifest has no recognized local collection: ' + field)
            for collection in collections:
                values = _field(manifest, collection)
                if not values:
                    _fail(component + ': empty anchored local collection: ' + collection)
                for relative, original in _rows(values):
                    if binding.get('record_field'):
                        original = _field(original, binding['record_field'])
                    if binding.get('member_root'):
                        if not relative or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
                            _fail(component + ': invalid relative manifest member: ' + str(relative))
                        root = Path(binding['member_root']).expanduser().resolve()
                        target = (root / relative).resolve()
                        if not target.is_relative_to(root):
                            _fail(component + ': local member escaped its explicit root: ' + relative)
                    else:
                        target = mapped(original['path'])
                    add(target, original); expanded += 1
        if not expanded and not any(seal.get(k) for k in _LOCAL_COLLECTIONS):
            _fail(component + ': no sealed local-file collection was expanded')

    rows = [expected[p] for p in sorted(expected)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_check_file, rows))
    return rows
