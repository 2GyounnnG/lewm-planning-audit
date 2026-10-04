"""Seal the completed, preselected R3 matrix before any EVAL effects are read.

CPU metadata/tensor validation only. No model construction, forecast, environment,
optimizer or hardware query. Explicit --run is required to publish immutable
routing and lock files. Source HDF5 bytes are hashed, never interpreted here.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3 import common, reset_fallback
from scripts import run_formal_training as controller

ROOT = common.ROOT
VERSION = 'R3_FIXED_MODEL_LOCK_V1'
ROUTING = 'manifests/OPEN_LOOP_ROUTING.json'
LOCK = 'manifests/MODELS_AND_SELECTION_LOCK.json'
STEPS = (3000, 10000, 30000)
ROOT_PREFIXES = {'artifacts', 'state', 'data', 'manifests', 'official', 'source', 'scripts', 'r3', 'tests', 'protocol', 'support', 'reports'}
FORBIDDEN = ('artifacts/open_loop/', 'artifacts/planning/FORMAL/', 'artifacts/statistics/', 'tables/')
SOURCE_TREES = ('lewm', 'spt', 'swm_compat')


def local(relative):
    p = Path(relative)
    if p.is_absolute() or not p.parts or '..' in p.parts: raise ValueError('ROOT-relative path required')
    q = ROOT/p
    if not q.resolve().is_relative_to(ROOT.resolve()): raise ValueError('Path escapes R3')
    return q


def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def json_bytes(value): return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
def read(relative): return common.read_json(local(relative))
def signature(path):
    s = path.stat(); return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class Inventory:
    """One full SHA per file; stable stat identities checked again before publish."""
    def __init__(self): self.files = {}; self.signatures = {}; self.expanded = set(); self.restorable = {}; self.historical_smoke_receipts = {}
    def add(self, name, expected=None):
        name = str(Path(name))
        if name in (LOCK, ROUTING): raise ValueError('Generated lock/routing cannot enter their own input closure')
        if name.startswith(FORBIDDEN): raise PermissionError('No EVAL effects may enter model selection/lock')
        p = local(name)
        if not p.is_file(): raise FileNotFoundError('Required frozen input missing: '+name)
        if name not in self.files:
            before = signature(p); value = {'sha256': common.sha256(p), 'bytes': p.stat().st_size}
            if signature(p) != before: raise RuntimeError('Input changed while hashing: '+name)
            self.files[name] = value; self.signatures[name] = before
        value = self.files[name]
        if expected is not None:
            expected = {'sha256': expected} if isinstance(expected, str) else expected
            if value['sha256'] != expected['sha256'] or ('bytes' in expected and value['bytes'] != expected['bytes']):
                raise RuntimeError('Frozen input SHA/bytes differ: '+name)
        return p
    def json(self, name, expected=None, expand=False):
        p = self.add(name, expected); value = json.loads(p.read_text())
        if expand and name not in self.expanded:
            self.expanded.add(name)
            if name in self.historical_smoke_receipts:
                self.add(name, self.historical_smoke_receipts[name])
            else: self.references(value, name)
        return value
    def reference_path(self, name, owner):
        p = Path(name)
        if p.is_absolute():
            if not p.resolve().is_relative_to(ROOT.resolve()): raise ValueError('External absolute receipt input is not a ROOT artifact')
            return str(p.resolve().relative_to(ROOT.resolve()))
        if '..' in p.parts or not p.parts: raise ValueError('Unsafe receipt reference')
        if owner.startswith('state/') and Path(owner).name.endswith('_source_manifest.json'):
            tree = Path(owner).name.removesuffix('_source_manifest.json')
            if tree not in SOURCE_TREES + ('swm',): raise ValueError('Unknown official source tree')
            return str(Path('source')/tree/p)
        return str(p if p.parts[0] in ROOT_PREFIXES else Path(owner).parent/p)
    def referenced(self, name, expected, owner):
        name = self.reference_path(name, owner)
        self.add(name, expected)
        if name.endswith('.json') and name not in self.expanded: self.json(name, expand=True)
    def references(self, value, owner):
        if isinstance(value, list):
            for item in value: self.references(item, owner)
        elif isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.referenced(value['path'], value, owner)
            if isinstance(value.get('result_path'), str) and isinstance(value.get('result_sha256'), str):
                self.referenced(value['result_path'], value['result_sha256'], owner)
            for key, item in value.items():
                if key in ('files', 'input_hashes', 'source_hashes', 'source_files') and isinstance(item, dict):
                    swm_source_map = (key == 'files' and not Path(owner).name.endswith('_source_manifest.json') and 'revision' in value and item
                                      and all(str(name).startswith('stable_worldmodel/') for name in item))
                    if swm_source_map:
                        pinned = self.json('state/swm_compat_source_manifest.json')
                        if value['revision'] != pinned['revision']: raise RuntimeError('Nested official SWM source revision differs')
                    for name, expected in item.items():
                        if swm_source_map: name = 'source/swm_compat/'+name
                        if isinstance(expected, str) and len(expected) == 64:
                            self.referenced(name, expected, owner)
                        elif isinstance(expected, dict) and isinstance(expected.get('sha256'), str):
                            self.referenced(expected.get('path', name), expected, owner)
                        else: self.references(expected, owner)
                else: self.references(item, owner)
    def unchanged(self):
        for name, before in self.signatures.items():
            if signature(local(name)) != before: raise RuntimeError('Frozen input changed before publication: '+name)


def route_matrix(jobs, files):
    controller.validate_jobs(jobs)
    out = {'version': 'R3_OPEN_LOOP_ROUTING_V1', 'selection': 'H0_AND_EVERY_FIXED_SEED_AT_3000_10000_30000_NO_RISK_SELECTION', 'tasks': {}}
    for task in common.TASKS:
        arms = [{'arm': 'H0', 'seed': None, 'step': 0}]
        for seed in common.SEEDS:
            job = next(j for j in jobs if j['task'] == task and j['refit_seed'] == seed); folder = 'artifacts/train/'+job['job_id']
            for step in STEPS:
                path = folder+f'/checkpoint_{step}.pt'; value = files[path]
                arms.append({'arm': f'REFIT_{seed}_{step}', 'seed': seed, 'step': step,
                             'checkpoint': {'path': path, **value}, 'run_identity': folder+'/RUN_IDENTITY.json', 'result': folder+'/result.json'})
        out['tasks'][task] = {'arms': arms, 'planning_steps': [0, 30000]}
    return out


def tensor_equal(a, b):
    import torch
    return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def audit_rng(rng):
    import numpy as np
    import torch
    if set(rng) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}: raise RuntimeError('Incomplete resume RNG state')
    random.Random().setstate(rng['python'])
    n = rng['numpy']; np.random.RandomState().set_state((n['bit_generator'], np.asarray(n['keys'], dtype=np.uint32), n['pos'], n['has_gauss'], n['cached_gaussian']))
    if not torch.is_tensor(rng['torch_cpu']) or rng['torch_cpu'].dtype != torch.uint8: raise RuntimeError('CPU RNG state dtype differs')
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    if not isinstance(rng['torch_cuda'], list) or len(rng['torch_cuda']) != 1:
        raise RuntimeError('Exactly one bound logical CUDA RNG state required')
    if any(not torch.is_tensor(t) or t.dtype != torch.uint8 or t.ndim != 1 or not t.numel() or t.device.type != 'cpu' for t in rng['torch_cuda']):
        raise RuntimeError('Malformed saved CUDA RNG bytes')
    return {'python_numpy_cpu_RNG_parse': 'PASS', 'CUDA_RNG_bytes_present': True, 'CUDA_runtime_called': False}


def audit_payload(payload, step, identity, frozen, whitelist, *, full=False):
    import torch
    from r3.train_worker import scheduler_record
    from r3.model import DELTA_FORMAT
    required = {'version', 'step', 'identity_sha256', 'delta', 'frozen', 'sampling_summary'}
    if (not full and set(payload) != required) or not required.issubset(payload): raise RuntimeError('Checkpoint compact schema differs')
    if payload['version'] != identity['version'] or payload['step'] != step or type(payload['step']) is not int or payload['identity_sha256'] != digest(identity) or payload['frozen'] != frozen:
        raise RuntimeError('Checkpoint step/identity/frozen mismatch')
    delta = payload['delta']; parameters = {r['name']: r for r in whitelist['parameters'] if r['trainable'] and r['name'] == r['shared_parameter_identity']}
    if (delta.get('format') != DELTA_FORMAT or delta['base_identity'] != identity['base_identity'] or delta['contract'] != identity['contract']
        or delta['frozen_sha256'] != frozen['sha256'] or set(delta['replacement_parameters']) != set(parameters)):
        raise RuntimeError('Delta official-coordinate/whitelist identity differs')
    for name, t in delta['replacement_parameters'].items():
        expected = parameters[name]
        if (not torch.is_tensor(t) or t.device.type != 'cpu' or list(t.shape) != expected['shape'] or str(t.dtype) != expected['dtype'] or not torch.isfinite(t).all()):
            raise RuntimeError('Invalid exact replacement parameter: '+name)
    summary = payload['sampling_summary']
    for key, factor in [('sampled_windows', 128), ('sampled_episode_draws', 128), ('predicted_target_tokens', 384), ('raw_action_exposures', 1920)]:
        if summary[key] != step*factor: raise RuntimeError('Checkpoint sampling/token budget differs')
    if full:
        if payload.get('scheduler') != scheduler_record(30000) or payload.get('optimizer_state_is_new_post_refit') is not True or payload.get('new_encoder_updates') != 0 or payload.get('state_labels_read') != 0:
            raise RuntimeError('Resume scheduler/optimizer provenance differs')
        audit_rng(payload['rng'])
        optimizer = payload['optimizer']; groups = optimizer['param_groups']
        if len(groups) != 1: raise RuntimeError('Expected one fixed AdamW parameter group')
        group = groups[0]
        if group['lr'] != scheduler_record(30000)['lr_last'] or group['weight_decay'] != .001 or tuple(group['betas']) != (.9, .999) or group['eps'] != 1e-8:
            raise RuntimeError('Fixed AdamW endpoint configuration differs')
        ids = group['params']
        if len(ids) != len(parameters) or len(set(ids)) != len(ids) or set(optimizer['state']) != set(ids): raise RuntimeError('Optimizer parameter/state set differs')
        for ident, parameter in zip(ids, delta['replacement_parameters'].values()):
            state = optimizer['state'][ident]
            if float(state['step']) != 30000: raise RuntimeError('AdamW update count differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                value = state[key]
                if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all(): raise RuntimeError('Invalid resumable AdamW moment')
            if (state['exp_avg_sq'] < 0).any(): raise RuntimeError('Negative AdamW squared moment')
        if not isinstance(payload.get('next_batch_sha256'), str) or len(payload['next_batch_sha256']) != 64 or 'sampler' not in payload:
            raise RuntimeError('Missing exact next-batch sampler cursor')
    return delta


def audit_job(job, inv):
    import numpy as np
    import torch
    from r3.train_worker import lr_at
    folder = 'artifacts/train/'+job['job_id']; publication = inv.json(folder+'/JOB_SHA256.json', expand=True)
    inv.json('state/recovery_queue/'+job['job_id']+'.json', expand=True)
    identity = inv.json(folder+'/RUN_IDENTITY.json'); result = inv.json(folder+'/result.json')
    frozen = inv.json(folder+'/FROZEN_INITIAL.json'); whitelist = inv.json(folder+'/PARAMETER_WHITELIST.json')
    strict = inv.json('state/OFFICIAL_STRICT_LOAD_CPU.json')
    if strict.get('status') != 'STRICT_LOAD_AND_SYNTHETIC_WRAPPER_PASS': raise RuntimeError('Actual official strict-load receipt missing')
    base = strict['tasks'][job['task']]
    if (base['identity'] != identity['base_identity'] or base['contract'] != identity['contract']
        or base['frozen_hashes'] != frozen or base['whitelist'] != whitelist):
        raise RuntimeError('Training base/frozen/whitelist differs from actual official strict load')
    if identity['role'] != 'REFIT_TRAIN' or result['frozen_before'] != frozen['sha256'] or result['frozen_after'] != frozen['sha256']:
        raise RuntimeError('Frozen initial/result/role identity differs')
    if result.get('new_encoder_updates') != 0 or result.get('state_labels_read') != 0 or result.get('checkpoint_selection') is not False or result.get('formal_initialization_inherits_technical_updates') is not False:
        raise RuntimeError('Formal budget/exposure/initialization contract differs')
    assets = inv.json(f"manifests/{job['task']}_model_assets.json")
    for name in ('weights', 'config'):
        if identity['base_identity'][name+'_sha256'] != assets['files'][name+('.pt' if name == 'weights' else '.json')]['sha256']:
            raise RuntimeError('Run base checkpoint/config identity differs')
    official_sources = {'lewm_jepa':'source/lewm/jepa.py','lewm_module':'source/lewm/module.py',
                        'lewm_train':'source/lewm/train.py','lewm_utils':'source/lewm/utils.py',
                        'vit_constructor':'source/spt/stable_pretraining/backbone/utils.py'}
    if identity['base_identity']['source_sha256'] != {key: inv.files[name]['sha256'] for key,name in official_sources.items()}:
        raise RuntimeError('Official construction source identity differs')
    rows = [json.loads(line) for line in inv.add(folder+'/updates.jsonl').read_text().splitlines()]
    for row in rows:
        if row['predicted_tokens'] != 384 or row['raw_action_exposures'] != 1920 or row['lr'] != lr_at(row['step']) or any(not math.isfinite(float(row[k])) for k in ('loss_raw_MSE', 'lr', 'preclip_gradient_norm')):
            raise RuntimeError('Formal raw update/token/LR record differs')
    checks = [json.loads(line) for line in inv.add(folder+'/freeze_checks.jsonl').read_text().splitlines()]
    if not set(range(100, 30001, 100)).issubset(x['step'] for x in checks) or any(x['passed'] is not True or x['frozen_sha256'] != frozen['sha256'] for x in checks):
        raise RuntimeError('Required every100-step freeze checks missing/failed')
    resume = torch.load(inv.add(folder+'/resume.pt'), map_location='cpu', weights_only=True)
    audit_payload(resume, 30000, identity, frozen, whitelist, full=True)
    with np.load(inv.add(folder+'/sampling_counts.npz'), allow_pickle=False) as f: counts = {k: f[k] for k in f.files}
    roles = inv.json(f"manifests/{job['task']}_data_roles.json")
    expected_ids = sorted(x['episode_id'] for x in roles['episodes'] if x['role'] == 'REFIT_TRAIN')
    if counts['episode_ids'].tolist() != expected_ids or identity['episode_ids'] != expected_ids: raise RuntimeError('Sampler crossed frozen data role')
    for key, factor in [('window_counts', 128), ('episode_counts', 128), ('raw_action_counts', 1920), ('macro_transition_counts', 384)]:
        a = counts[key]
        if int(a.sum()) != 30000*factor or np.any(a < 0) or not np.array_equal(a, resume['sampler'][key].numpy()): raise RuntimeError('Sampler count/result mismatch')
    offsets = counts['window_offsets']; legal = counts['legal_starts']
    if len(offsets) != len(expected_ids)+1 or offsets[0] != 0 or offsets[-1] != len(legal) or np.any(np.diff(offsets) < 0): raise RuntimeError('Invalid sampler offsets')
    if not np.array_equal([counts['window_counts'][a:b].sum() for a,b in zip(offsets[:-1], offsets[1:])], counts['episode_counts']): raise RuntimeError('Episode/window accounting differs')
    legal_identity = {episode: legal[a:b].tolist() for episode,a,b in zip(expected_ids,offsets[:-1],offsets[1:])}
    if digest(legal_identity) != identity['legal_starts_sha256']: raise RuntimeError('Saved sampler legal-window identity differs')
    rng = np.random.default_rng(); rng.bit_generator.state = resume['sampler']['rng']; indices = rng.integers(int(offsets[-1]), size=128, dtype=np.int64)
    ei = np.searchsorted(offsets[1:], indices, side='right'); cursor = [[str(counts['episode_ids'][e]), int(legal[i])] for e,i in zip(ei,indices)]
    if digest(cursor) != resume['next_batch_sha256']: raise RuntimeError('Resume next-batch identity differs')
    for step in STEPS:
        cp = torch.load(inv.add(folder+f'/checkpoint_{step}.pt'), map_location='cpu', weights_only=True)
        delta = audit_payload(cp, step, identity, frozen, whitelist)
        if step == 30000 and any(not tensor_equal(value, resume['delta']['replacement_parameters'][name]) for name,value in delta['replacement_parameters'].items()):
            raise RuntimeError('Public final delta differs from actual resumable endpoint')
        del cp
    return {'job_id': job['job_id'], 'updates': 30000, 'checkpoints': list(STEPS), 'frozen_sha256': frozen['sha256'],
            'exact_sampler_cursor': True, 'all_RNG_present': True, 'CPU_tensor_validation_only': True}


def audit_smoke_history(inv):
    """Only the declared split4 stage is historical; independently recheck final8."""
    from scripts import run_training_technical as technical
    if technical.ROOT != ROOT: raise RuntimeError('Smoke verifier ROOT differs')
    smoke = inv.json('state/GPU_SMOKE.json')
    if smoke.get('status') != 'PASS' or smoke.get('optimizer_updates') != 32 or smoke.get('genuine_process_exit_resume') is not True:
        raise RuntimeError('Actual final smoke gate missing')
    partial = [item for item in smoke['files'] if Path(item['path']).name == 'smoke_split4.json']
    if len(partial) != 1: raise RuntimeError('Exactly one declared historical split4 receipt required')
    entry = partial[0]; name = inv.reference_path(entry['path'], 'state/GPU_SMOKE.json'); parts = Path(name).parts
    if len(parts) != 4 or parts[:2] != ('state','technical_training'):
        raise RuntimeError('Unexpected historical smoke stage location')
    p = inv.json(name, entry)
    if p.get('status') != 'PASS' or p.get('stage') != 'smoke_split4' or p.get('identity_sha256') != smoke['identity_sha256'] or p.get('total_successful_updates') != 8:
        raise RuntimeError('Declared partial smoke identity/count differs')
    base = str(Path(name).parent)
    continuous = inv.json(base+'/smoke_continuous8.json')
    resumed = inv.json(base+'/smoke_resume4.json')
    for value, stage, count in ((continuous,'smoke_continuous8',16),(resumed,'smoke_resume4',8)):
        if value.get('status') != 'PASS' or value.get('stage') != stage or value.get('identity_sha256') != smoke['identity_sha256'] or value.get('total_successful_updates') != count:
            raise RuntimeError('Final smoke stage identity/count differs')
    specs = lambda value: {x['job']['task']: x for x in value['specs']}
    a,b,c = specs(continuous),specs(p),specs(resumed)
    if any(set(v) != set(common.TASKS) or len(value['specs']) != 2 for v,value in ((a,continuous),(b,p),(c,resumed))):
        raise RuntimeError('Smoke task matrix differs')
    checks=[]
    for task in common.TASKS:
        if b[task].get('stop_after') != 4 or b[task].get('endpoint') != 8 or {**b[task],'stop_after':None} != c[task]:
            raise RuntimeError('Historical split4 is not exact same run resumed to8')
        if a[task].get('endpoint') != 8 or a[task].get('stop_after') is not None:
            raise RuntimeError('Continuous final8 spec differs')
        check=technical.compare_smoke(a[task],c[task])  # CPU tensors only; exact RNG/optimizer/journal equality and two distinct PIDs.
        saved=[v for v in smoke['tasks'] if v['task']==task]
        if len(saved)!=1 or saved[0]!=check: raise RuntimeError('Current final8 state differs from sealed smoke equality proof')
        partial_process=[v['pid'] for v in p['processes'] if v['run_id']==b[task]['run_id']]
        resume_process=[v['pid'] for v in resumed['processes'] if v['run_id']==c[task]['run_id']]
        if partial_process != check['split_process_ids'][:1] or resume_process != check['split_process_ids'][1:]:
            raise RuntimeError('Historical actual4+4 process chain differs')
        checks.append(check)
    # No generic historical exemption: this exact file was SHA-bound by final smoke,
    # and its replaced endpoints have just passed independent final8 CPU verification.
    inv.historical_smoke_receipts[name]=entry
    inv.json('state/GPU_SMOKE.json',expand=True)
    inv.json(base+'/smoke_continuous8.json',expand=True)
    inv.json(base+'/smoke_resume4.json',expand=True)
    return {'status':'PASS','historical_receipt':name,'historical_receipt_sha256':inv.files[name]['sha256'],
            'superseded_endpoint_scope':'Only this declared split4 receipt nested files are a retained historical snapshot; final8 gate files verified at current hashes.',
            'final8_exact_state_and_rng_recomputed':True,'distinct_process_4_plus_4_verified':True,'tasks':[x['task'] for x in checks]}


def static_inventory(inv, auth):
    # The official-asset manifest is a metadata mapping. Its top-level files
    # are ROOT-relative evidence; nested source file maps keep their original
    # repository namespaces and are not an additional relative-path closure.
    official = inv.json('manifests/OFFICIAL_ASSET_MANIFEST.json')
    if (official.get('version') != 'R3_OFFICIAL_ASSET_MANIFEST_V1'
        or official.get('status') != 'COMPLETE_METADATA_ASSET_MAPPING'
        or set(official.get('tasks', {})) != set(common.TASKS)
        or not official.get('files')):
        raise RuntimeError('Complete official-asset metadata mapping required before model lock')
    for name, value in official['files'].items(): inv.add(name, value)
    for directory, suffix in [('r3', '*.py'), ('scripts', '*.py'), ('tests', '*.py'), ('protocol', '*')]:
        for p in sorted((ROOT/directory).rglob(suffix)):
            if p.is_file() and '__pycache__' not in p.parts: inv.add(str(p.relative_to(ROOT)))
    for name in controller.REQUIRED_HASHES | {'state/FORMAL_TRAINING_COMPLETE.json', 'manifests/TRAINING_AUTHORIZATION.json', 'state/TECHNICAL_LEDGER.json', 'state/OFFICIAL_STRICT_LOAD_CPU.json', 'state/INSTANCE_REUSE.json', 'state/ENVIRONMENT_READY.json', 'manifests/NUMERICAL_TOLERANCES.json'}:
        inv.json(name, expand=True) if name.endswith('.json') else inv.add(name)
    for name, value in auth['source_hashes'].items():
        inv.json(name, value, expand=True) if name.endswith('.json') else inv.add(name, value)
        if Path(name).name in ('REAL_DATA_MODEL_CHECK.json','COST_WRAPPER_CHECK.json'):
            inv.json(str(Path(name).parent/'SHA256.json'), expand=True)
    for tree in SOURCE_TREES + (('swm',) if local('state/swm_source_manifest.json').exists() else ()):
        name = f'state/{tree}_source_manifest.json'; manifest = inv.json(name)
        for member, value in manifest['files'].items(): inv.add(f'source/{tree}/{member}', value)
    for task in common.TASKS:
        names = {suffix: f'manifests/{task}_{suffix}.json' for suffix in ('data_roles','cache','normalization','model_assets','source_map','data_assets','data_unpacked')}
        for suffix, name in names.items(): inv.json(name, expand=suffix not in ('data_roles',))
        for kind in ('model','data'):
            inv.json(f'state/{task}_{kind}_api.json'); inv.json(f'state/{task}_{kind}_tree.json')
        roles = inv.json(names['data_roles'])
        for name in reset_fallback.evidence_paths(task, roles):
            inv.json(name, expand=True) if name.endswith('.json') else inv.add(name)
        cm = inv.json(names['cache'])
        if cm['status'] != 'FROZEN_OBSERVED_CACHE_COMPLETE': raise RuntimeError('Observed cache is incomplete')
        for ep in cm['episodes'].values():
            inv.add(ep['path'], ep); inv.json(str(Path(ep['path']).with_suffix('.json')))
        source = inv.json(names['source_map'])
        archives = inv.json(names['data_assets']); unpacked = inv.json(names['data_unpacked'])
        if unpacked['source_receipt_sha256'] != inv.files[names['data_assets']]['sha256']:
            raise RuntimeError('Unpacked source/archive receipt identity differs')
        for value in archives['files'].values():
            inv.add(value['path'], value)
            inv.restorable[value['path']] = {**value, 'source_manifest': names['data_assets'],
                'policy': 'Original pinned archive, retained once on host; final transfer may restore by revision and exact SHA'}
        members = {item['path']: item for item in unpacked['files']}
        for key, value in source['assets'].items():
            if key != value['sha256']: raise RuntimeError('Actual HDF5 source ID differs')
            member = members[value['path']]
            if any(member[k] != value[k] for k in ('sha256','bytes')): raise RuntimeError('Unpacked member/source map differs')
            proof = inv.json(member['member_receipt'], member['member_receipt_sha256'], expand=True)
            if proof['status'] != 'COMPLETE' or proof['identity']['output_path'] != value['path'] or any(proof[k] != value[k] for k in ('sha256','bytes')):
                raise RuntimeError('Actual complete unpack receipt differs')
            inv.add(value['path'], value)
            inv.restorable[value['path']] = {**value, 'source_map': names['source_map'],
                'policy': 'Verified original source; required on execution host, final transfer may restore from pinned source instead of copying HDF5'}
    ledger = inv.json('state/TECHNICAL_LEDGER.json'); runs = ledger.get('runs', {})
    charged = sum(v['reserved_updates'] if v.get('actual_updates') is None else v['actual_updates'] for v in runs.values())
    trajectories = sum(v['status'] != 'INFRASTRUCTURE_FAILURE_INCOMPLETE' for v in ledger.get('trajectories', {}).values())
    if charged > 1024 or trajectories > 16: raise RuntimeError('Technical ledger exceeds fixed authorization')
    return {'formal_updates': 180000, 'formal_jobs': 6, 'technical_optimizer_charged': charged, 'technical_optimizer_limit': 1024,
            'technical_trajectory_charged': trajectories, 'technical_trajectory_limit': 16, 'lock_optimizer_updates': 0, 'lock_trajectories': 0}


def immutable_write(path, payload):
    """Atomic create without replacing a different existing file."""
    path = Path(path); data = json_bytes(payload); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data: raise RuntimeError('Existing immutable file differs: '+str(path))
        return
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        try: os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data: raise RuntimeError('Concurrent immutable publication differs')
    finally:
        if tmp.exists(): tmp.unlink()


def run():
    directory = local('state'); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'model_freeze.lock').open('a+') as own, (directory/'execution_phase.lock').open('a+') as phase:
        fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(phase, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = controller.build_plan()  # Metadata gates only; never queries GPUs.
        completion = read('state/FORMAL_TRAINING_COMPLETE.json'); jobs = plan['jobs']
        if completion.get('status') != 'COMPLETE' or completion.get('complete_jobs') != 6 or completion.get('actual_updates') != 180000 or completion.get('execution_plan_sha256') != digest(plan):
            raise RuntimeError('Actual fixed six-job formal completion is required')
        verified = {job['job_id']: controller.verify_complete(job) for job in jobs}
        if any(v is None for v in verified.values()) or completion.get('jobs') != verified: raise RuntimeError('Formal completion differs from current exact published jobs')
        inv = Inventory(); smoke_audit = audit_smoke_history(inv); auth = inv.json('manifests/TRAINING_AUTHORIZATION.json'); budget = static_inventory(inv, auth)
        audits = [audit_job(job, inv) for job in jobs]
        routing = route_matrix(jobs, inv.files)
        route_bytes = json_bytes(routing); files = dict(sorted(inv.files.items()))
        files[ROUTING] = {'sha256': hashlib.sha256(route_bytes).hexdigest(), 'bytes': len(route_bytes)}
        lock = {'version': VERSION, 'status': 'MODELS_AND_SELECTION_LOCKED', 'selection_policy': routing['selection'],
                'files': dict(sorted(files.items())), 'restorable_input_files': inv.restorable, 'budget': budget, 'job_checks': audits, 'smoke_history_audit': smoke_audit,
                'formal_training_complete_sha256': inv.files['state/FORMAL_TRAINING_COMPLETE.json']['sha256'],
                'EVAL_effects_read': 0, 'new_optimizer_updates': 0, 'new_trajectories': 0,
                'scientific_effects_are_not_a_gate': True, 'source_archive_roots': ['r3','scripts','tests','protocol','source']}
        inv.unchanged(); controller.assert_plan(plan)
        # A differing existing lock blocks even creation of a new routing file.
        if local(LOCK).exists() and local(LOCK).read_bytes() != json_bytes(lock): raise RuntimeError('Existing model lock differs; it will not be overwritten')
        immutable_write(local(ROUTING), routing); inv.unchanged(); immutable_write(local(LOCK), lock)
        return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    if not args.run:
        print(json.dumps({'status': 'NOT_EXECUTED', 'command': 'python scripts/lock_models.py --run', 'required_formal_jobs': 6, 'required_formal_updates': 180000})); return
    result = run(); print(json.dumps({'status': result['status'], 'files': len(result['files']), 'formal_updates': result['budget']['formal_updates']}), flush=True)


if __name__ == '__main__': main()
