"""Prospectively fixed R3 role/case selection from supplied metadata only.

planning_starts/open_loop_starts are CURRENT raw-frame anchors (last history
frame), not history-window starts. The reader must supply actual schema/action
admissibility; this module never invents legal starts or reset seeds.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

VERSION = 'R3_METADATA_ROLES_V1'
TASKS = ('pusht', 'reacher')
GOAL_OFFSET = 25
TECH_CASES = 4
EVAL_CAP = 100
MONITOR_WINDOW_CAP = 256
NAMESPACES = {
    'group': 'R3_GROUP_SPLIT_20261002',
    'monitor': 'R3_MONITOR_SPLIT_20261002',
    'episode': 'R3_EPISODE_ORDER_20261002',
    'start': 'R3_CASE_START_20261002',
    'offline': 'R3_OPEN_LOOP_ANCHOR_20261002',
    'case_id': 'R3_CASE_ID_20261002',
    'monitor_window': 'R3_MONITOR_WINDOW_20261002',
}
FORBIDDEN_SCORE_KEYS = {'success', 'model_success', 'risk', 'score', 'model_error',
                        'prediction_error', 'loss', 'final_goal_error'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def rank(namespace, *parts):
    return hashlib.sha256(namespace.encode('utf-8') + b'\n' + canonical(list(parts))).hexdigest()


def contract():
    return {
        'version': VERSION, 'namespaces': NAMESPACES,
        'hash_encoding': 'UTF8(namespace) + newline + canonical JSON positional-parts array; sort_keys=True, ensure_ascii=False, compact separators; lexicographic SHA256 hex order; IDs break a hash tie',
        'eval_group_count': 'ceil(G/5); take LAST groups under group namespace ordering',
        'monitor_group_count': 'ceil(FIT_GROUPS/10); take FIRST groups under separate monitor namespace ordering; require at least one remaining REFIT_TRAIN group, never silently alter rounding',
        'unknown_family': 'Each null/FAMILY_UNKNOWN episode gets an episode-only split group; family_id remains null and independence is not asserted',
        'within_eval_pool': 'Round-robin groups in their fixed group-SHA order, episodes ordered within group by episode namespace; first4 unique episodes TECH, next up to100 EVAL, remainder EVAL_POOL_UNUSED',
        'family_separation': 'REFIT_TRAIN/MONITOR/EVAL_POOL split groups are disjoint; TECH/EVAL may share a known family and its exact intersection is disclosed',
        'case_choice': 'One metadata-hashed legal planning raw anchor per selected episode; goal=start+25. No success/error-based replacement.',
        'offline_choice': 'Use planning anchor if in supplied open_loop_starts; otherwise minimum offline-namespace hash legal anchor, with OPEN_LOOP_ANCHOR_NOT_PLANNING_START flag',
        'offline_legality': 'anchor >= (history_size-1)*frameskip; conservative source clip anchor+(5+1)*frameskip <= length; actual finite actions/schema supplied by reader',
        'monitor_windows': f'Minimum {NAMESPACES["monitor_window"]} hashes over all supplied legal offline anchors in MONITOR/TECH, independently per role, up to{MONITOR_WINDOW_CAP}; output history-window starts for train_worker',
        'index_units': 'All start/goal/anchor indices are raw frames; arbitrary raw phases retained without rounding to frameskip grid',
        'case_limit': EVAL_CAP, 'minimum_eval_cases': 20, 'technical_cases': TECH_CASES,
        'reset_seed': 'Use only explicit source_seed; null stays null and requires separately validated reset fallback',
        'cem_replan_seed': "int.from_bytes(SHA256(f'R3_CEM_CASE_20261002/{task}/{case_id}/{replan_index}').digest()[:8], 'big')",
        'source_scope': 'Default protocol3.3 original-source split. If an independently verified official evaluation source exists, do not silently apply this single-pool builder; register that separate-source allocation explicitly.',
    }


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' must be a lowercase SHA256')


def _indices(value, name):
    if not isinstance(value, list) or any(type(x) is not int for x in value) or len(set(value)) != len(value):
        raise ValueError(name + ' must be an explicit unique integer list')
    return sorted(value)


def _group(row):
    return ('family', row['family_id']) if row['family_id'] is not None else ('episode', row['episode_id'])


def _episode_parts(row):
    return row['episode_id'], row['source_asset_sha256'], row['source_episode_idx']


def validate_records(records, history_size, frameskip):
    required = {'episode_id', 'source_episode_idx', 'source_asset_sha256', 'episode_sha256',
                'length', 'family_id', 'planning_starts', 'open_loop_starts'}
    normalized, ids, identities = [], set(), set()
    for original in records:
        if not isinstance(original, dict) or required - set(original):
            raise ValueError('Missing required real episode metadata')
        if FORBIDDEN_SCORE_KEYS & set(original):
            raise ValueError('Model outcomes/errors are forbidden in role selection metadata')
        row = copy.deepcopy(original)
        ep = row['episode_id']
        if not isinstance(ep, str) or not ep or ep != ep.strip() or ep in ids:
            raise ValueError('Episode IDs must be unique nonempty canonical strings')
        ids.add(ep)
        for name in ('source_asset_sha256', 'episode_sha256'):
            _sha(row[name], name)
        if type(row['source_episode_idx']) is not int or row['source_episode_idx'] < 0:
            raise ValueError('Invalid source episode index')
        identity = (row['source_asset_sha256'], row['source_episode_idx'])
        if identity in identities:
            raise ValueError('Duplicate source asset/episode identity')
        identities.add(identity)
        if type(row['length']) is not int or row['length'] < 1:
            raise ValueError('Actual positive episode length required')
        family = row['family_id']
        if family == 'FAMILY_UNKNOWN':
            row['family_id'] = None
        elif family is not None and (not isinstance(family, str) or not family or family != family.strip()):
            raise ValueError('family_id must be canonical text or null')
        seed = row.get('source_seed')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('source_seed must be explicitly recorded uint32 or null')
        row['source_seed'] = seed
        row['planning_starts'] = _indices(row['planning_starts'], 'planning_starts')
        row['open_loop_starts'] = _indices(row['open_loop_starts'], 'open_loop_starts')
        if any(i < 0 or i + GOAL_OFFSET >= row['length'] for i in row['planning_starts']):
            raise ValueError('Supplied planning anchor lacks its exact25-raw-step goal')
        if any(i < (history_size-1)*frameskip or i + 6*frameskip > row['length'] for i in row['open_loop_starts']):
            raise ValueError('Supplied offline anchor lacks full history or conservative H5 source span')
        canonical(row)  # Real metadata must be serializable without NaN guesses.
        row['metadata_eligible'] = bool(row['planning_starts'] and row['open_loop_starts'])
        row['ineligibility_reason'] = None if row['metadata_eligible'] else 'NO_SUPPLIED_LEGAL_PLANNING_OR_OFFLINE_ANCHOR'
        normalized.append(row)
    return sorted(normalized, key=lambda r: r['episode_id'])


def _round_robin(task, group_order, groups):
    ordered = {g: sorted(groups[g], key=lambda r: (rank(NAMESPACES['episode'], task, *g, *_episode_parts(r)), r['episode_id']))
               for g in group_order}
    maximum = max((len(x) for x in ordered.values()), default=0)
    return [ordered[g][i] for i in range(maximum) for g in group_order if i < len(ordered[g])]


def make_case(task, row, role, history_size, frameskip):
    parts = _episode_parts(row)
    case_id = 'R3_' + task + '_' + rank(NAMESPACES['case_id'], task, *parts)[:24]
    start = min(row['planning_starts'], key=lambda i: (rank(NAMESPACES['start'], task, *parts, i), i))
    same = start in row['open_loop_starts']
    offline = start if same else min(row['open_loop_starts'], key=lambda i: (rank(NAMESPACES['offline'], task, *parts, i), i))
    seed_salt = f'R3_CEM_CASE_20261002/{task}/{case_id}'
    return {'task': task, 'case_id': case_id, 'role': role, 'episode_id': row['episode_id'],
            'source_episode_idx': row['source_episode_idx'], 'source_asset_sha256': row['source_asset_sha256'],
            'episode_sha256': row['episode_sha256'], 'family_id': row['family_id'], 'length': row['length'],
            'start_raw_index': start, 'goal_raw_index': start + GOAL_OFFSET, 'goal_offset_raw': GOAL_OFFSET,
            'open_loop_anchor_raw': offline, 'open_loop_window_start_raw': offline-(history_size-1)*frameskip,
            'open_loop_target_raw': {str(h): offline+h*frameskip for h in (1, 2, 5)},
            'open_loop_anchor_relation': 'SAME_AS_PLANNING_START' if same else 'OPEN_LOOP_ANCHOR_NOT_PLANNING_START',
            'source_seed': row['source_seed'], 'reset_seed': row['source_seed'],
            'requires_validated_reset_fallback': row['source_seed'] is None,
            'reset_metadata': copy.deepcopy(row.get('reset_metadata', {})),
            'cem_case_seed_uint64': int.from_bytes(hashlib.sha256(seed_salt.encode()).digest()[:8], 'big'),
            'cem_replan_seed_rule': contract()['cem_replan_seed'],
            'selection_used_model_outputs': False}


def _monitor_windows(task, rows, role, history_size, frameskip):
    candidates = [(r, anchor) for r in rows if r['role'] == role for anchor in r['open_loop_starts']]
    candidates.sort(key=lambda pair: (rank(NAMESPACES['monitor_window'], task, role, *_episode_parts(pair[0]), pair[1]),
                                      pair[0]['episode_id'], pair[1]))
    return [[r['episode_id'], anchor-(history_size-1)*frameskip] for r, anchor in candidates[:MONITOR_WINDOW_CAP]]


def build_roles(task, episode_records, *, history_size, frameskip):
    if task not in TASKS or type(history_size) is not int or history_size < 1 or type(frameskip) is not int or frameskip < 1:
        raise ValueError('Explicit official task/history_size/frameskip required')
    rows = validate_records(list(episode_records), history_size, frameskip)
    input_hash = hashlib.sha256(canonical(rows)).hexdigest()
    groups = {}
    for row in rows:
        if row['metadata_eligible']:
            groups.setdefault(_group(row), []).append(row)
    ordered = sorted(groups, key=lambda g: (rank(NAMESPACES['group'], task, *g), g))
    eval_n = math.ceil(len(ordered)/5)
    eval_groups = ordered[len(ordered)-eval_n:] if eval_n else []
    fit_groups = ordered[:len(ordered)-eval_n] if eval_n else []
    monitor_n = math.ceil(len(fit_groups)/10)
    fit_order = sorted(fit_groups, key=lambda g: (rank(NAMESPACES['monitor'], task, *g), g))
    monitor_groups = fit_order[:monitor_n]
    train_groups = fit_order[monitor_n:]
    pool = _round_robin(task, eval_groups, groups)
    tech, evaluation = pool[:TECH_CASES], pool[TECH_CASES:TECH_CASES+EVAL_CAP]
    tech_ids = {r['episode_id'] for r in tech};eval_ids = {r['episode_id'] for r in evaluation}
    train_set, monitor_set = set(train_groups), set(monitor_groups)
    for row in rows:
        g = _group(row);ep = row['episode_id']
        row['split_group'] = {'kind': g[0], 'id': g[1]}
        row['family_evidence'] = row.get('family_evidence', 'FAMILY_UNKNOWN' if row['family_id'] is None else 'PROVIDED_ID_INDEPENDENCE_UNPROVEN')
        row['role'] = ('EXCLUDED_METADATA_INELIGIBLE' if not row['metadata_eligible'] else
                       'REFIT_TRAIN' if g in train_set else 'MONITOR' if g in monitor_set else
                       'TECH' if ep in tech_ids else 'EVAL' if ep in eval_ids else 'EVAL_POOL_UNUSED')
    cases = {'TECH': [make_case(task, r, 'TECH', history_size, frameskip) for r in tech],
             'EVAL': [make_case(task, r, 'EVAL', history_size, frameskip) for r in evaluation]}
    if len({c['case_id'] for part in cases.values() for c in part}) != len(tech)+len(evaluation):
        raise RuntimeError('Case hash collision; stop rather than select replacement')
    blockers = []
    if not train_groups:
        blockers.append('EMPTY_REFIT_TRAIN_AFTER_FIXED_GROUP_SPLIT')
    if len(evaluation) < 20:
        blockers.append('FEWER_THAN_20_LEGAL_EVAL_EPISODES_AFTER_RESERVED_TECH')
    if len(tech) < TECH_CASES:
        blockers.append('FEWER_THAN_FOUR_DISTINCT_TECH_EPISODES')
    known_tech = {r['family_id'] for r in tech if r['family_id'] is not None}
    known_eval = {r['family_id'] for r in evaluation if r['family_id'] is not None}
    return {'version': VERSION, 'task': task, 'status': 'BLOCKED_DATA' if blockers else 'METADATA_ROLES_FROZEN',
            'contract': contract(), 'input_metadata_sha256': input_hash,
            'history_size': history_size, 'frameskip': frameskip, 'episodes': rows, 'cases': cases,
            'roles': {role: [r['episode_id'] for r in rows if r['role'] == role]
                      for role in ('REFIT_TRAIN','MONITOR','TECH','EVAL','EVAL_POOL_UNUSED','EXCLUDED_METADATA_INELIGIBLE')},
            'groups': {'all_count': len(groups), 'eval_pool_count': len(eval_groups),
                       'fit_pool_count': len(fit_groups), 'monitor_count': len(monitor_groups),
                       'refit_train_count': len(train_groups),
                       'REFIT_TRAIN': [list(g) for g in train_groups], 'MONITOR': [list(g) for g in monitor_groups],
                       'EVAL_POOL': [list(g) for g in eval_groups]},
            'technical_eval_known_family_overlap': sorted(known_tech & known_eval),
            'technical_eval_episode_overlap': sorted(tech_ids & eval_ids),
            'sample_scope': 'BLOCKED_DATA' if blockers else 'LIMITED_EVALUATION_SAMPLE' if len(evaluation)<100 else 'TARGET_100_CASES',
            'blocked_reasons': blockers, 'planned_formal_trajectories': 4*len(evaluation),
            'planned_technical_clone_trajectories': 2*len(tech),
            'reset_validation_pending_case_ids': [c['case_id'] for part in cases.values() for c in part if c['requires_validated_reset_fallback']],
            'monitor_windows': _monitor_windows(task, rows, 'MONITOR', history_size, frameskip),
            'technical_monitor_windows': _monitor_windows(task, rows, 'TECH', history_size, frameskip),
            'actual_optimizer_updates': 0, 'actual_CEM_trajectories': 0,
            'selection_used_model_outputs': False,
            'pretraining_exposure_note': 'R3 role exclusion does not establish official pretraining-unseen or no prior G1/R2 inspection.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, required=True);p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--history-size', type=int, required=True);p.add_argument('--frameskip', type=int, required=True)
    p.add_argument('--output', type=Path, required=True);a=p.parse_args()
    result=build_roles(a.task,json.loads(a.episodes.read_text()),history_size=a.history_size,frameskip=a.frameskip)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists() and json.loads(a.output.read_text()) != result:
        raise RuntimeError('Refusing to replace a different frozen role selection')
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'task':a.task,'formal_cases':len(result['cases']['EVAL']),
                      'blocked_reasons':result['blocked_reasons']}))


if __name__=='__main__':
    main()
