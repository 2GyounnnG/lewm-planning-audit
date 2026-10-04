from __future__ import annotations
import csv, hashlib, json, pathlib, time
from collections import defaultdict, Counter
import numpy as np

ROOT = pathlib.Path('/workspace/r8')
REPORT = ROOT / 'reports'
REPORT.mkdir(parents=True, exist_ok=True)
ARMS = ('H0', 'REFIT_103201', 'REFIT_103202', 'REFIT_103203')
STREAMS = ('R3_ORIGINAL', 'R4_ALT_CEM_1', 'R4_ALT_CEM_2')
HISTORIES = ('H_POLICY', 'H_REAL3_REPLAN')
B = 5000


def sha_file(path):
    h = hashlib.sha256(); n = 0
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b); n += len(b)
    return h.hexdigest(), n


def seed_for(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'big')


def bootstrap_ci(values, label):
    x = np.asarray(values, dtype=np.float64)
    seed = seed_for(label)
    if len(x) == 0:
        return {'point_estimate': None, 'ci95': [None, None], 'n_cases': 0, 'bootstrap': B,
                'bootstrap_unit': 'CASE', 'bootstrap_seed_rule': label, 'bootstrap_seed_uint64': seed}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return {'point_estimate': float(x.mean()),
            'ci95': [float(np.quantile(means, 0.025, method='linear')), float(np.quantile(means, 0.975, method='linear'))],
            'n_cases': int(len(x)), 'bootstrap': B, 'bootstrap_unit': 'CASE',
            'bootstrap_seed_rule': label, 'bootstrap_seed_uint64': seed}


def read_dataset(name, rootrel, task, expected_cases=100, expected_histories=HISTORIES):
    root = ROOT / rootrel
    rows = []
    for p in sorted(root.glob('FORMAL/*/*/*/*/result.json')):
        rel = p.relative_to(root).parts
        if len(rel) != 6:
            continue
        _, stream, case, arm, hist, _ = rel
        d = json.loads(p.read_text())
        fsha, fbytes = sha_file(p)
        rows.append({
            'dataset': name, 'task': task, 'unit': name, 'case_id': d.get('case_id', case),
            'arm': d.get('arm', arm), 'stream': d.get('stream', stream), 'history': d.get('history', hist),
            'status': d.get('status'), 'success': int(d.get('success', 0)),
            'method_failure': d.get('method_failure'), 'executed_raw_steps': int(d.get('executed_raw_steps', -1)),
            'replan_calls': int(d.get('replan_calls', -1)), 'entered_replanning': int(bool(d.get('entered_replanning', False))),
            'attempts': int(d.get('attempts', -1)), 'optimizer_updates': int(d.get('optimizer_updates', -1)),
            'paired_first_plan_checked': int(bool(d.get('paired_first_plan_checked', False))),
            'cross_unit_first_plan_checked': int(bool(d.get('cross_unit_first_plan_checked', False))),
            'result_path': str(p), 'result_sha256': fsha, 'result_bytes': fbytes,
        })
    expected = expected_cases * len(ARMS) * len(STREAMS) * len(expected_histories)
    keys = [(r['case_id'], r['arm'], r['stream'], r['history']) for r in rows]
    duplicates = [k for k, c in Counter(keys).items() if c > 1]
    cases = sorted(set(r['case_id'] for r in rows))
    if len(rows) != expected or len(cases) != expected_cases or duplicates:
        raise RuntimeError(f'{name}: rows={len(rows)} expected={expected}; cases={len(cases)} expected={expected_cases}; duplicates={len(duplicates)}')
    bad = [r for r in rows if r['status'] != 'COMPLETE' or r['method_failure'] is not None or r['success'] not in (0, 1)]
    if bad:
        raise RuntimeError(f'{name}: bad outcomes={len(bad)}')
    return rows


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def condition_table(rows, label):
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r['arm'], r['stream'], r['history'])].append(r)
    for (arm, stream, hist), rr in sorted(by.items()):
        bycase = defaultdict(list)
        for r in rr:
            bycase[r['case_id']].append(r['success'])
        casevals = np.array([np.mean(bycase[c]) for c in sorted(bycase)], dtype=float)
        ci = bootstrap_ci(casevals, f'R8_CASE_BOOTSTRAP_20261004/{label}/condition/{arm}/{stream}/{hist}')
        out.append({
            'dataset': label, 'arm': arm, 'stream': stream, 'history': hist,
            'rows': len(rr), 'cases': len(bycase), 'successes': sum(r['success'] for r in rr),
            'success_rate': float(np.mean([r['success'] for r in rr])),
            'entered_replanning': sum(r['entered_replanning'] for r in rr),
            'entered_replanning_rate': float(np.mean([r['entered_replanning'] for r in rr])),
            'mean_replan_calls': float(np.mean([r['replan_calls'] for r in rr])),
            'mean_executed_raw_steps': float(np.mean([r['executed_raw_steps'] for r in rr])),
            'case_bootstrap_ci_low': ci['ci95'][0], 'case_bootstrap_ci_high': ci['ci95'][1],
            'bootstrap_seed_uint64': ci['bootstrap_seed_uint64'],
        })
    return out


def case_contrasts(rows, label):
    data = {(r['case_id'], r['arm'], r['stream'], r['history']): r for r in rows}
    cases = sorted(set(r['case_id'] for r in rows))
    available = sorted(set(r['history'] for r in rows))
    out = []
    def add(name, vals):
        ci = bootstrap_ci(np.asarray(vals, dtype=float), f'R8_CASE_BOOTSTRAP_20261004/{label}/{name}')
        out.append({'dataset': label, 'contrast': name, 'n_cases': len(cases), 'point_estimate': ci['point_estimate'],
                    'ci95_low': ci['ci95'][0], 'ci95_high': ci['ci95'][1], 'bootstrap': B,
                    'bootstrap_seed_rule': ci['bootstrap_seed_rule'], 'bootstrap_seed_uint64': ci['bootstrap_seed_uint64']})
    def hist_vec(hist, arms=ARMS, streams=STREAMS):
        return np.array([np.mean([data[(c, a, s, hist)]['success'] for a in arms for s in streams]) for c in cases], dtype=float)
    if set(HISTORIES).issubset(available):
        add('H_REAL3_REPLAN_minus_H_POLICY_pooled', hist_vec('H_REAL3_REPLAN') - hist_vec('H_POLICY'))
        for arm in ARMS:
            add(f'{arm}:H_REAL3_REPLAN_minus_H_POLICY', hist_vec('H_REAL3_REPLAN', arms=(arm,)) - hist_vec('H_POLICY', arms=(arm,)))
        for stream in STREAMS:
            add(f'{stream}:H_REAL3_REPLAN_minus_H_POLICY', hist_vec('H_REAL3_REPLAN', streams=(stream,)) - hist_vec('H_POLICY', streams=(stream,)))
        for hist in HISTORIES:
            add(f'REFIT_3mean_minus_H0:{hist}', hist_vec(hist, arms=ARMS[1:]) - hist_vec(hist, arms=('H0',)))
    # Always provide H_POLICY pooled rate and fixed refit-minus-H0 when H_POLICY is available.
    if 'H_POLICY' in available:
        add('H_POLICY_success_pooled', hist_vec('H_POLICY'))
        if 'H_REAL3_REPLAN' not in available:
            add('REFIT_3mean_minus_H0:H_POLICY', hist_vec('H_POLICY', arms=ARMS[1:]) - hist_vec('H_POLICY', arms=('H0',)))
    if 'H_REAL3_REPLAN' in available:
        vals = np.array([np.mean([data[(c, a, s, 'H_REAL3_REPLAN')]['entered_replanning'] for a in ARMS for s in STREAMS]) for c in cases], dtype=float)
        add('entered_replanning:H_REAL3_REPLAN', vals)
    return out

DATASETS = [
    ('M1_U1', 'raw/U1', 'reacher', 100, HISTORIES),
    ('M1_U2', 'raw/U2', 'reacher', 100, HISTORIES),
    ('M2_REACHER', 'raw/M2_reacher', 'reacher', 100, ('H_POLICY',)),
    ('M2_PUSHT', 'raw/M2_pusht', 'pusht', 100, ('H_POLICY',)),
]
allmeta = []
for label, rootrel, task, ncase, expected_histories in DATASETS:
    rows = read_dataset(label, rootrel, task, ncase, expected_histories)
    raw_fields = ['dataset','task','unit','case_id','arm','stream','history','status','success','method_failure','executed_raw_steps','replan_calls','entered_replanning','attempts','optimizer_updates','paired_first_plan_checked','cross_unit_first_plan_checked','result_path','result_sha256','result_bytes']
    raw_path = REPORT / f'R8_{label}_RAW_VALUES.csv'; write_csv(raw_path, rows, raw_fields)
    cond = condition_table(rows, label)
    cond_path = REPORT / f'R8_{label}_CONDITION_TABLE.csv'; write_csv(cond_path, cond, list(cond[0]))
    contr = case_contrasts(rows, label)
    contr_path = REPORT / f'R8_{label}_CASE_BOOTSTRAP_TABLE.csv'; write_csv(contr_path, contr, list(contr[0]))
    primary_name = 'H_REAL3_REPLAN_minus_H_POLICY_pooled' if set(HISTORIES).issubset(set(r['history'] for r in rows)) else 'H_POLICY_success_pooled'
    pooled = next(x for x in contr if x['contrast'] == primary_name)
    hp = [r['success'] for r in rows if r['history'] == 'H_POLICY']
    hr = [r['success'] for r in rows if r['history'] == 'H_REAL3_REPLAN']
    ep = [r['entered_replanning'] for r in rows if r['history'] == 'H_POLICY']
    er = [r['entered_replanning'] for r in rows if r['history'] == 'H_REAL3_REPLAN']
    main = {
        'dataset': label, 'task': task, 'status': 'COMPLETE', 'rows': len(rows), 'cases': len(set(r['case_id'] for r in rows)),
        'expected_rows': ncase * 24 if set(HISTORIES).issubset(set(expected_histories)) else ncase * 12,
        'available_histories': sorted(set(r['history'] for r in rows)), 'primary_contrast': pooled['contrast'], 'point_estimate': pooled['point_estimate'],
        'ci95_low': pooled['ci95_low'], 'ci95_high': pooled['ci95_high'], 'bootstrap': B,
        'bootstrap_seed_rule': pooled['bootstrap_seed_rule'], 'bootstrap_seed_uint64': pooled['bootstrap_seed_uint64'],
        'H_POLICY_success_rate': float(np.mean(hp)) if hp else None, 'H_REAL3_REPLAN_success_rate': float(np.mean(hr)) if hr else None,
        'H_POLICY_entered_replanning_rate': float(np.mean(ep)) if ep else None, 'H_REAL3_REPLAN_entered_replanning_rate': float(np.mean(er)) if er else None,
        'paired_first_plan_checked_H_REAL3_rows': sum(r['paired_first_plan_checked'] for r in rows if r['history'] == 'H_REAL3_REPLAN'),
        'cross_unit_first_plan_checked_rows': sum(r['cross_unit_first_plan_checked'] for r in rows),
    }
    main_path = REPORT / f'R8_{label}_MAIN_TABLE.json'; main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + '\n')
    outs = []
    for p in (raw_path, cond_path, contr_path, main_path):
        h, b = sha_file(p); outs.append({'path': str(p), 'bytes': b, 'sha256': h})
    allmeta.append({'dataset': label, 'task': task, 'status': 'COMPLETE', 'rows': len(rows), 'cases': len(set(r['case_id'] for r in rows)), 'outputs': outs, 'main': main})

manifest = {
    'version': 'R8_FIXED_CASE_BOOTSTRAP_SUMMARY_V1',
    'evidence_label': 'R8_ORIGINAL_VALUE_SUMMARY',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'read_only_inputs': True,
    'source_roots': ['/workspace/r8/raw/U1','/workspace/r8/raw/U2','/workspace/r8/raw/M2_reacher','/workspace/r8/raw/M2_pusht'],
    'bootstrap': {'B': B, 'unit': 'CASE', 'seed_rule': 'SHA256("R8_CASE_BOOTSTRAP_20261004/<dataset>/<contrast>") first 8 bytes big-endian', 'quantile': 'linear 2.5%,97.5%'},
    'datasets': allmeta, 'R4_R5_R6_R7_modified': False,
}
manifest_path = REPORT / 'R8_STATS_MANIFEST.json'; manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
# Include manifest itself in a final inventory receipt (one pass after all files are written).
receipt = {'version':'R8_STATS_RECEIPT_V1','status':'COMPLETE','manifest_path':str(manifest_path),'outputs':[]}
for p in sorted(REPORT.glob('R8_*.csv')) + sorted(REPORT.glob('R8_*.json')):
    h,b=sha_file(p); receipt['outputs'].append({'path':str(p),'bytes':b,'sha256':h})
receipt_path = REPORT/'R8_STATS_RECEIPT.json'; receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'COMPLETE','datasets':[(x['dataset'],x['rows'],x['cases']) for x in allmeta],'manifest':str(manifest_path),'receipt':str(receipt_path)},ensure_ascii=False))
