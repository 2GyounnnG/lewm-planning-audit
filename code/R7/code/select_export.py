from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

R3 = Path('/workspace/shared_data/r3')
OUT = Path('/workspace/r7')

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def record(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}

def rank(namespace, task, episode_id):
    return hashlib.sha256((namespace + task + '/' + episode_id).encode()).hexdigest()

def r3_start(task, row, offset=50):
    parts = (row['episode_id'], row['source_asset_sha256'], row['source_episode_idx'])
    legal = [int(i) for i in row['planning_starts'] if int(i) + offset < int(row['length'])]
    if not legal:
        raise ValueError(f'no legal {offset}-step start for {row["episode_id"]}')
    def h(i):
        s = json.dumps([*parts, i], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(('R3_CASE_START_20261002' + task + s).encode()).hexdigest()
    return min(legal, key=lambda i: (h(i), i))

def known_case_ids(task):
    roles = json.loads((R3 / 'manifests' / f'{task}_data_roles.json').read_text())
    ids = {c['case_id'] for kind in ('TECH', 'EVAL') for c in roles['cases'][kind]}
    # Bind the exclusion audit to any stored R4/R5 CSVs present on the host.
    roots = [Path('/workspace/r4_reacher'), Path('/workspace/r4_pusht'), Path('/workspace/r5/H2')]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.csv'):
            try:
                with path.open(newline='') as f:
                    for row in csv.DictReader(f):
                        cid = row.get('case_id') or row.get('case')
                        if cid:
                            ids.add(cid)
            except (OSError, UnicodeDecodeError):
                continue
    if task == 'reacher':
        p = Path('/workspace/r6/ops/R6_CASE_SELECTION.json')
        if p.exists():
            d = json.loads(p.read_text())
            ids.update(c['case_id'] for c in d.get('cases', []))
    return ids, roles

def select(task):
    roles_path = R3 / 'manifests' / f'{task}_data_roles.json'
    roles = json.loads(roles_path.read_text())
    used_case_ids, _ = known_case_ids(task)
    used_episode_ids = set()
    role_cases = roles['cases']['TECH'] + roles['cases']['EVAL']
    by_case = {c['case_id']: c for c in role_cases}
    for cid in used_case_ids:
        if cid in by_case:
            used_episode_ids.add(by_case[cid]['episode_id'])
    if task == 'reacher':
        p = Path('/workspace/r6/ops/R6_CASE_SELECTION.json')
        if p.exists():
            used_episode_ids.update(c['episode_id'] for c in json.loads(p.read_text()).get('cases', []))
    fallback = json.loads((R3 / 'manifests' / 'RESET_FALLBACK_MANIFEST.json').read_text())['tasks'][task]
    candidates = []
    for row in roles['episodes']:
        if row.get('role') not in ('MONITOR', 'EVAL_POOL_UNUSED'):
            continue
        if row.get('episode_id') in used_episode_ids or not row.get('metadata_eligible'):
            continue
        legal = [int(i) for i in row.get('planning_starts', []) if int(i) + 50 < int(row['length'])]
        if not legal:
            continue
        start = r3_start(task, row, 50)
        goal = start + 50
        namespace = 'R7_LONG/'
        sh = rank(namespace, task, row['episode_id'])
        cid = f'R7_{task}_{sh[:24]}'
        seed = row.get('source_seed')
        case = {
            'task': task, 'case_id': cid, 'role': 'R7_LONG',
            'episode_id': row['episode_id'], 'source_episode_idx': int(row['source_episode_idx']),
            'source_asset_sha256': row['source_asset_sha256'], 'episode_sha256': row['episode_sha256'],
            'family_id': row.get('family_id'), 'length': int(row['length']),
            'start_raw_index': int(start), 'goal_raw_index': int(goal), 'goal_offset_raw': 50,
            'source_seed': seed, 'reset_seed': int(seed) if seed is not None else 0,
            'requires_validated_reset_fallback': seed is None,
            'reset_metadata': row.get('reset_metadata', {}),
            'selection_sha256': sh, 'selection_rule': f'sha256("R7_LONG/{task}/"+episode_id) ascending',
            'construction_rule': 'R3 metadata case start hash, target offset 50 raw, legal source window',
            'selection_used_model_outputs': False,
            'reset_seed_validation_sha256': fallback['validation_receipt']['sha256'] if seed is None else None,
            'reset_seed_provenance': fallback['provenance'] if seed is None else 'SOURCE_SEED',
        }
        candidates.append((sh, row['episode_id'], case))
    candidates.sort(key=lambda x: (x[0], x[1]))
    if len(candidates) < 50:
        status = 'STOP_BELOW_MINIMUM_50'
    else:
        status = 'SEALED_BEFORE_CLOSED_LOOP'
    selected = candidates[:min(104, len(candidates))]
    tech = [x[2] | {'role': 'TECH'} for x in selected[:4]]
    formal = [x[2] | {'role': 'EVAL'} for x in selected[4:104]]
    result = {
        'version': 'R7_LONG_FRESH_CASE_SELECTION_V1',
        'evidence_label': 'PREREGISTERED_STRESS_TEST_FRESH_CASES',
        'status': status, 'task': task,
        'selection_rule': f'sha256("R7_LONG/{task}/"+episode_id) ascending',
        'offset_raw': 50, 'budget_raw': 100, 'max_replans': 3,
        'source_roles_manifest': record(roles_path),
        'exclusion': {'excluded_case_ids': len(used_case_ids), 'excluded_episode_ids': len(used_episode_ids),
                      'candidate_roles': ['MONITOR', 'EVAL_POOL_UNUSED'], 'REFIT_TRAIN_excluded': True,
                      'selection_used_model_outputs': False, 'prior_case_sets': 'R3-R6 audited by case ids and R6 selection'},
        'candidate_count': len(candidates), 'tech_count': len(tech), 'formal_count': len(formal),
        'minimum_case_count': 50 if len(candidates) >= 50 else len(candidates),
        'tech_cases': tech, 'formal_cases': formal,
        'candidate_hash_order': [{'selection_sha256': x[0], 'episode_id': x[1]} for x in candidates],
        'planned_formal_trajectories': len(formal) * 24,
    }
    out = OUT / 'ops' / f'R7_{task.upper()}_CASE_SELECTION.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result

def export(task, selection):
    import h5py, hdf5plugin
    sm_path = R3 / 'manifests' / f'{task}_source_map.json'
    sm = json.loads(sm_path.read_text())
    assets = {c['source_asset_sha256']: sm['assets'][c['source_asset_sha256']]
              for c in selection['tech_cases'] + selection['formal_cases']}
    out = OUT / 'assets' / task; out.mkdir(parents=True, exist_ok=True)
    entries = []
    handles = {}
    try:
        for ah, a in assets.items():
            src = R3 / a['path'];
            if src.stat().st_size != a['bytes']:
                raise RuntimeError('source bytes differ')
            handles[ah] = h5py.File(src, 'r', swmr=True, rdcc_nbytes=512 * 1024**2)
        for case in selection['tech_cases'] + selection['formal_cases']:
            a = assets[case['source_asset_sha256']]; h = handles[case['source_asset_sha256']]
            off = int(h['ep_offset'][case['source_episode_idx']]); length = int(h['ep_len'][case['source_episode_idx']])
            start, end = case['start_raw_index'], case['goal_raw_index'] + 1
            if not 0 <= start < end <= length: raise RuntimeError('illegal source window')
            keys = ['pixels', 'action', 'state'] if task == 'pusht' else ['pixels', 'action', 'qpos', 'qvel']
            arr = {k: np.asarray(h[k][off + start:off + end]) for k in keys}
            if arr['pixels'].shape[-1] == 3: arr['pixels'] = arr['pixels'].transpose(0, 3, 1, 2)
            arr.update(source_start_raw=np.int64(start), source_episode_idx=np.int64(case['source_episode_idx']))
            target = out / (case['case_id'] + '.npz')
            if target.exists():
                with np.load(target, allow_pickle=False) as f:
                    if set(f.files) != set(arr) or any(not np.array_equal(f[k], v) for k, v in arr.items()):
                        raise RuntimeError('existing export differs')
            else:
                with target.open('wb') as f: np.savez_compressed(f, **arr)
            entries.append({'case': case, 'file': target.name, **record(target), 'source_asset': a})
    finally:
        for h in handles.values(): h.close()
    manifest = {'version': 'R7_LONG_CASE_WINDOWS_V1', 'evidence_label': 'PREREGISTERED_STRESS_TEST_FRESH_CASES',
                'task': task, 'source_roles_manifest': record(R3 / 'manifests' / f'{task}_data_roles.json'),
                'source_map': record(sm_path), 'cases': entries}
    p = OUT / 'assets' / f'CASE_WINDOWS_{task.upper()}.json'; p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return {'task': task, 'cases': len(entries), 'sha256': sha256_file(p)}

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    results=[]
    for task in ('reacher', 'pusht'):
        sel = select(task); print(json.dumps({'selected': task, 'candidates': sel['candidate_count'], 'tech': sel['tech_count'], 'formal': sel['formal_count']}), flush=True)
        results.append(export(task, sel)); print(json.dumps(results[-1]), flush=True)
