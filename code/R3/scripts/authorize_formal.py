"""Explicit metadata-only R3 formal authorization, never a training launcher.

The caller names the exact two data-check and two cost-check receipts. No attempt
is selected automatically, and no scientific outcome enters this gate. File
contents are SHA-verified; tensors, HDF5 arrays and environments are never opened.
"""
from __future__ import annotations
import argparse, contextlib, datetime, fcntl, hashlib, importlib, json, os, sys
from pathlib import Path, PurePosixPath

VERSION = 'R3_EXPLICIT_FORMAL_AUTHORIZATION_V1'
TASKS = ('pusht', 'reacher')
SEEDS = (103201, 103202, 103203)
AUTH = 'manifests/TRAINING_AUTHORIZATION.json'
FIXED_INPUTS = (
    'scripts/authorize_formal.py', 'scripts/run_formal_training.py', 'scripts/run_training_technical.py',
    'scripts/run_planning.py', 'scripts/build_cache.py', 'scripts/check_real_data_model.py',
    'scripts/check_cost_wrapper.py', 'r3/common.py', 'r3/model.py', 'r3/train_worker.py', 'r3/data.py',
    'r3/roles.py', 'r3/planning.py', 'r3/open_loop.py', 'r3/env_compat.py', 'r3/reset_fallback.py',
    'manifests/JOBS.json', 'manifests/EXPERIMENT_LOCK.json', 'manifests/NUMERICAL_TOLERANCES.json',
    'state/INSTANCE_REUSE.json', 'state/ENVIRONMENT_READY.json', 'state/OFFICIAL_STRICT_LOAD_CPU.json',
    'state/TRAINING_TECHNICAL_GATE.json', 'state/GPU_SMOKE.json', 'state/GPU_PROFILE.json',
    'state/PLANNING_TECH_GATE.json', 'state/TECHNICAL_LEDGER.json')
SOURCE_TREES = ('lewm', 'spt', 'swm_compat')
FORBIDDEN = ('artifacts/train/', 'artifacts/open_loop/', 'artifacts/planning/FORMAL/', 'artifacts/statistics/')
RUNTIME_ARTIFACT_JSON = frozenset(('REAL_DATA_MODEL_CHECK.json','COST_WRAPPER_CHECK.json','RESET_VALIDATION_RECEIPT.json',
    'SHA256.json','CASE_SHA256.json','TRAJECTORY_SHA256.json','RUN_IDENTITY.json','PARAMETER_WHITELIST.json','FROZEN_INITIAL.json'))


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20), b''): h.update(chunk)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def signature(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def local(root,name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p)!=name or any(ord(c)<32 for c in name): raise ValueError('Canonical ROOT-relative path required: '+str(name))
    current=root
    for part in p.parts:
        current=current/part
        if current.is_symlink(): raise ValueError('Symlink in authorization evidence: '+name)
    if not current.resolve().is_relative_to(root.resolve()): raise ValueError('Evidence escaped ROOT')
    return current


class Evidence:
    """Memoized full hashes; signatures rechecked before the atomic publication."""
    def __init__(self,root): self.root=Path(root).resolve();self.files={};self.signatures={}
    def bind(self,name,expected=None):
        if name==AUTH or name.startswith(FORBIDDEN): raise PermissionError('Formal outcomes/authorization cannot enter a pre-training technical gate: '+name)
        p=local(self.root,name)
        if not p.is_file(): raise FileNotFoundError(name)
        before=signature(p)
        if name not in self.files:
            self.files[name]={'sha256':sha(p),'bytes':before[2]};self.signatures[name]=before
            if signature(p)!=before: raise RuntimeError('Evidence changed while hashing: '+name)
        elif self.signatures[name]!=before: raise RuntimeError('Evidence changed after verification: '+name)
        if expected is not None:
            e={'sha256':expected} if isinstance(expected,str) else expected
            if self.files[name]['sha256']!=e['sha256'] or ('bytes' in e and self.files[name]['bytes']!=e['bytes']): raise RuntimeError('Evidence SHA/bytes mismatch: '+name)
        return p
    def json(self,name,expected=None): return read(self.bind(name,expected))
    def file_map(self,files,prefix=''):
        if not isinstance(files,dict) or not files: raise ValueError('Required receipt file map is empty')
        for name,expected in files.items(): self.bind(str(PurePosixPath(prefix)/name) if prefix else name,expected)
    def records(self,records):
        if not isinstance(records,list) or not records: raise ValueError('Required receipt file list is empty')
        for r in records:self.bind(r['path'],r)
    def unchanged(self):
        for name,sig in self.signatures.items():
            if signature(local(self.root,name))!=sig: raise RuntimeError('Evidence changed before publication: '+name)


def validate_budget(ledger):
    lower=upper=0
    for run_id,r in ledger.get('runs',{}).items():
        reserve=r['reserved_updates'];actual=r.get('actual_updates')
        if type(reserve) is not int or not 0<=reserve<=1024: raise ValueError('Invalid technical reservation')
        if actual is None:
            known=r.get('detail',{}).get('known_successful_updates',0)
            if type(known) is not int or not 0<=known<=reserve: raise ValueError('Invalid technical lower bound')
            # Uncertain optimizer calls remain fully charged; active reservations
            # without an explicit failure record may still be running and block.
            if r.get('status')!='UNCERTAIN_OPTIMIZER_ATTEMPT': raise RuntimeError('Unfinished active technical reservation: '+run_id)
            lower+=known;upper+=reserve
        else:
            if type(actual) is not int or not 0<=actual<=reserve: raise ValueError('Invalid actual technical count')
            lower+=actual;upper+=actual
    occupied=0
    for rid,r in ledger.get('trajectories',{}).items():
        if r['status'] not in ('COMPLETE','INFRASTRUCTURE_FAILURE_INCOMPLETE'): raise RuntimeError('Unfinished technical CEM trajectory: '+rid)
        occupied+=r['status']=='COMPLETE'
    if upper>1024 or occupied>16: raise ValueError('Technical budget exceeded')
    return {'known_optimizer_updates':lower,'charged_optimizer_upper_bound':upper,'complete_technical_trajectories':occupied,'optimizer_limit':1024,'trajectory_limit':16}


def formal_zero(root):
    folder=local(root,'artifacts/train')
    if folder.exists():
        files=[str(p.relative_to(root)) for p in folder.rglob('*') if p.is_file()]
        if files: raise RuntimeError('A prior formal attempt exists; preserve and audit instead of recreating authorization: '+str(files[:5]))
    for name in ('state/FORMAL_TRAJECTORY_LEDGER.json','state/FORMAL_TRAINING_COMPLETE.json'):
        p=local(root,name)
        if p.exists() and (name.endswith('COMPLETE.json') or read(p).get('trajectories')): raise RuntimeError('Formal execution already has evidence')
    return {'formal_optimizer_updates':0,'formal_CEM_trajectories':0,'prior_formal_attempts':0}


def validate_jobs(doc):
    jobs=doc['jobs'];expected={(t,s) for t in TASKS for s in SEEDS}
    if len(jobs)!=6 or {(j['task'],j['refit_seed']) for j in jobs}!=expected: raise ValueError('Exactly six fixed task/seed jobs required')
    for j in jobs:
        if j['job_id']!=f'R3_{j["task"]}_PRED_REFIT_s{j["refit_seed"]}' or j['updates']!=30000 or j['effective_batch']!=128 or j['microbatch']!=128: raise ValueError('Formal job recipe differs')
    if len({j['job_id'] for j in jobs})!=6: raise ValueError('Duplicate formal job')
    return jobs


def runtime_hashes(files):
    """Strict runtime whitelist; full original evidence remains in verified_files.

    Each worker already SHA-verifies the latent files it actually consumes.
    Rehashing original HDF5 and technical optimizer tensors on every worker
    authorization adds no new check of the formal training inputs.
    """
    selected={}
    for name,record in files.items():
        p=PurePosixPath(name)
        is_code=p.suffix=='.py' and p.parts[0] in ('r3','scripts','source')
        is_identity=p.suffix=='.json' and p.parts[0] in ('manifests','state')
        is_base=len(p.parts)==3 and p.parts[0]=='official' and p.parts[1] in TASKS and p.name in ('weights.pt','config.json')
        is_receipt=p.parts[0]=='artifacts' and p.name in RUNTIME_ARTIFACT_JSON
        if is_code or is_identity or is_base or is_receipt:selected[name]=record['sha256']
    return dict(sorted(selected.items()))


def select_profile(gate,smoke,profile):
    if gate['status']!='TRAINING_TECHNICAL_GATES_PASS' or smoke['status']!='PASS' or profile['status']!='PROFILE_COMPLETE': raise RuntimeError('Actual training smoke/profile incomplete')
    if smoke.get('optimizer_updates')!=32 or smoke.get('genuine_process_exit_resume') is not True: raise RuntimeError('Genuine two-task exit/resume smoke absent')
    for field in ('identity_sha256','selected_workers','workers_per_gpu','microbatch'):
        if gate[field]!=profile[field]: raise RuntimeError('Training gate/profile selection differs: '+field)
    if gate['identity_sha256']!=smoke['identity_sha256'] or gate['microbatch']!=128 or gate['selected_workers'] not in (2,4) or gate['workers_per_gpu']!=gate['selected_workers']//2: raise RuntimeError('Unregistered concurrency/microbatch')
    if gate.get('scientific_outcomes_used') or profile.get('scientific_outcomes_used'): raise ValueError('Scientific improvement cannot gate scheduling')
    A,B=profile['A'],profile['B']
    if A['status']!='PASS': raise RuntimeError('Lower-concurrency profile failed')
    if B['status']=='PASS':
        end=B['end_to_end_updates_per_second']/A['end_to_end_updates_per_second'];sa=A['steady_update_span'];sb=B['steady_update_span']
        steady=sb['aggregate_updates_per_second_lower']/sa['aggregate_updates_per_second_upper'] if sa['status']==sb['status']=='BOUNDED_OBSERVED_INTERVAL' else None
        expected=4 if end>=1.1 and steady is not None and steady>=1.1 and not A['telemetry_errors'] and not B['telemetry_errors'] else 2
    elif B['status']=='FAILED' and B.get('higher_concurrency_rejected')=='AUDITED_OUT_OF_MEMORY': expected=2
    else: raise RuntimeError('Unexplained failed higher-concurrency profile')
    if gate['selected_workers']!=expected: raise RuntimeError('Selection differs from the fixed comparable10% rule')
    return {'selected_workers':expected,'workers_per_gpu':expected//2,'microbatch':128,'scientific_outcomes_used':False}


def fallback_inputs(e,task,roles):
    if not any(c['reset_seed'] is None for phase in ('TECH','EVAL') for c in roles['cases'][phase]):return
    # These helpers verify only JSON/hashes; their environment entry points are
    # never called. Task-specific validation rules are already independently fixed.
    os.environ['R3_ROOT']=str(e.root);sys.path.insert(0,str(e.root))
    module=importlib.import_module('r3.reset_fallback');common=importlib.import_module('r3.common')
    if common.ROOT!=e.root or not Path(module.__file__).resolve().is_relative_to(e.root): raise RuntimeError('Reset verifier imported a different project')
    manifest,entry,actual_roles,paths=module.evidence(task)
    if actual_roles!=roles: raise RuntimeError('Fallback used different raw roles')
    for path in paths:e.bind(path)
    receipt=e.json(entry['validation_receipt']['path'],entry['validation_receipt'])
    e.file_map(receipt['input_hashes']);e.file_map(receipt['files'],str(PurePosixPath(entry['validation_receipt']['path']).parent))
    for item in receipt['source_files'].values():e.bind(item['path'],item)


def collect_task(e,task):
    docs={key:e.json(f'manifests/{task}_{key}.json') for key in ('data_roles','cache','source_map','normalization','model_assets')}
    roles,cache=docs['data_roles'],docs['cache']
    if roles.get('task')!=task or roles.get('status')!='METADATA_ROLES_FROZEN' or roles.get('blocked_reasons') or roles.get('selection_used_model_outputs'): raise RuntimeError('Actual task roles are blocked/unfrozen')
    eps=roles['episodes'];roster={r['episode_id']:r for r in eps if r['role'] in ('REFIT_TRAIN','MONITOR','TECH','EVAL')}
    if len({r['episode_id'] for r in eps})!=len(eps) or not 20<=len(roles['cases']['EVAL'])<=100 or len(roles['cases']['TECH'])<2 or not roles['monitor_windows'] or not roles['technical_monitor_windows']: raise RuntimeError('Incomplete authorized role populations')
    if cache['status']!='FROZEN_OBSERVED_CACHE_COMPLETE' or cache['task']!=task or set(cache['episodes'])!=set(roster) or cache['optimizer_updates'] or cache['state_labels_read']: raise RuntimeError('Actual official observed cache incomplete')
    e.file_map(cache['input_hashes'])
    for eid,item in cache['episodes'].items():
        if item['role']!=roster[eid]['role'] or item['length']!=roster[eid]['length']: raise RuntimeError('Cache role/source length differs')
        e.bind(item['path'],item);side=e.json(str(PurePosixPath(item['path']).with_suffix('.json')))
        if side['file']!=item or side['identity']['episode']!=roster[eid] or side['source_state_or_reward_read']: raise RuntimeError('Cache provenance differs')
    assets=docs['model_assets'];api=e.json(f'state/{task}_model_api.json');tree=e.json(f'state/{task}_model_tree.json')
    if assets['repo']!='quentinll/lewm-'+task or assets['revision']!=api['sha'] or assets['status']!='PINNED_ASSETS_DOWNLOADED_VERIFIED': raise RuntimeError('Official model source identity differs')
    for name,item in assets['files'].items():
        original=next(x for x in tree if x['path']==name)
        if original['size']!=item['bytes'] or ('lfs' in original and original['lfs']['oid']!=item['sha256']): raise RuntimeError('Official asset pinned identity differs')
        e.bind(item['path'],item)
    source=docs['source_map'];data_assets=e.json(f'manifests/{task}_data_assets.json');data_api=e.json(f'state/{task}_data_api.json');data_tree=e.json(f'state/{task}_data_tree.json')
    unpack=e.json(f'manifests/{task}_data_unpacked.json')
    if data_assets['repo']!='quentinll/lewm-'+task or source['official_revision']!=data_api['sha'] or data_assets['revision']!=data_api['sha'] or source['selection_based_on_model_outputs'] or data_assets['status']!='PINNED_ASSETS_DOWNLOADED_VERIFIED': raise RuntimeError('Official data provenance differs')
    if unpack['status']!='SOURCE_UNPACKED_SHA_VERIFIED' or unpack['source_receipt_sha256']!=e.files[f'manifests/{task}_data_assets.json']['sha256']:raise RuntimeError('Official unpack/archive receipt chain differs')
    for name,item in data_assets['files'].items():
        entry=next(x for x in data_tree if x['path']==name)
        if item['bytes']!=entry['size'] or ('lfs' in entry and item['sha256']!=entry['lfs']['oid']):raise RuntimeError('Pinned original archive SHA/size differs')
    for key,item in source['assets'].items():
        if key!=item['sha256']:raise RuntimeError('Raw H5 source content identity differs')
        member=next(r for r in unpack['files'] if r['path']==item['path'])
        proof=e.json(member['member_receipt'],member['member_receipt_sha256'])
        if proof['status']!='COMPLETE' or any(proof[k]!=item[k] or member[k]!=item[k] for k in ('bytes','sha256')):raise RuntimeError('Unpacked original H5 member identity differs')
        archive=proof['identity']['archive'];original=next(v for v in data_assets['files'].values() if v['path']==archive['path'])
        if any(archive[k]!=original[k] for k in ('sha256','bytes')):raise RuntimeError('H5 source did not derive from the pinned original archive')
        e.bind(item['path'],item)
    for item in docs['normalization']['source_assets'].values():e.bind(item['path'],item)
    fallback_inputs(e,task,roles)
    return roles


def explicit_check(e,path,task,kind,roles):
    expected='REAL_DATA_MODEL_CHECK.json' if kind=='data' else 'COST_WRAPPER_CHECK.json'
    if PurePosixPath(path).name!=expected:raise ValueError('Exact explicitly selected check receipt required')
    d=e.json(path);prefix=str(PurePosixPath(path).parent);seal=e.json(prefix+'/SHA256.json')
    if d['task']!=task or d['status']!='PASS' or seal['status']!='PASS' or d['optimizer_updates']: raise RuntimeError('Selected actual task check did not PASS')
    if expected not in seal['files']: raise RuntimeError('Check receipt absent from its own sibling seal')
    e.file_map(seal['files'],prefix);e.file_map(d['input_hashes'])
    if kind=='data':
        if not d['loaded_model_tensors_unchanged'] or any(d[k] for k in ('optimizer_objects_created','full_policy_trajectories','EVAL_arrays_read','source_state_reward_goal_labels_read')): raise RuntimeError('Real data/freeze check violated its zero-update/input boundary')
    else:
        if any(d[k] for k in ('complete_CEM_trajectories','environment_steps','source_future_actions_read')) or len(d['rows'])!=2 or [r['case_id'] for r in d['rows']]!=[c['case_id'] for c in roles['cases']['TECH'][:2]]: raise RuntimeError('Actual goal/cost check used wrong inputs')
        if not all(all(r[k] for k in ('zero_update_wrapped_cost_exact','official_goal_encoding_exact','terminal_squared_sum_cost_matches','all_finite')) for r in d['rows']): raise RuntimeError('Goal/cost hard comparison failed')
    return path


def build_authorization(root,data_checks,cost_checks):
    if set(data_checks)!=set(TASKS) or set(cost_checks)!=set(TASKS) or len(set(data_checks.values())|set(cost_checks.values()))!=4: raise ValueError('Name exactly one data and one cost receipt per task')
    root=Path(root).resolve();zero=formal_zero(root);e=Evidence(root)
    for path in FIXED_INPUTS:e.bind(path)
    jobs=validate_jobs(e.json('manifests/JOBS.json'));experiment=e.json('manifests/EXPERIMENT_LOCK.json')
    if experiment['status']!='EXPERIMENT_PROTOCOL_FROZEN_BEFORE_OPTIMIZER_AND_OUTCOME' or experiment['formal_updates']!=180000:raise RuntimeError('Frozen experiment contract differs')
    e.file_map(experiment['protocol_files'])
    strict=e.json('state/OFFICIAL_STRICT_LOAD_CPU.json')
    if strict['status']!='STRICT_LOAD_AND_SYNTHETIC_WRAPPER_PASS' or set(strict['tasks'])!=set(TASKS):raise RuntimeError('Actual official strict-load evidence incomplete')
    for t in TASKS:
        r=strict['tasks'][t]['strict_load']
        if r['missing_keys'] or r['unexpected_keys']:raise RuntimeError('Official state_dict strict loading failed')
    for tree in SOURCE_TREES:
        doc=e.json(f'state/{tree}_source_manifest.json');e.file_map(doc['files'],'source/'+tree)
    roles={t:collect_task(e,t) for t in TASKS}
    gate=e.json('state/TRAINING_TECHNICAL_GATE.json');smoke=e.json('state/GPU_SMOKE.json');profile=e.json('state/GPU_PROFILE.json')
    selection=select_profile(gate,smoke,profile)
    e.file_map(gate['source_files']);e.records([gate[k] for k in ('smoke_receipt','profile_receipt','profile_table')]);e.records(smoke['files'])
    # Terminal evidence only. In particular, smoke's recorded split4-stage JSON
    # is historical: its old resume4 hashes were legitimately superseded by8.
    # Hash that immutable stage receipt, but do not interpret its old file map as
    # a new promise that resume4 must still equal the final resume8 bytes.
    for part in ('A','B'):e.records(profile[part]['files'])
    plan=e.json('state/planning/TECH/EXECUTION_PLAN.json');cem=e.json('state/PLANNING_TECH_GATE.json')
    if cem['status']!='PASS' or cem['complete_trajectories']!=8 or cem['clone_exact'] is not True or cem['execution_plan_sha256']!=digest(plan):raise RuntimeError('Actual eight-trajectory planning clone gate incomplete')
    if plan['phase']!='TECH' or len(plan['cases'])!=4 or {t:sum(x['task']==t for x in plan['cases']) for t in TASKS}!={t:2 for t in TASKS}:raise RuntimeError('Wrong fixed TECH planning population')
    e.file_map(plan['files']);e.file_map(cem['files'])
    for comparison in cem['clone_comparisons']:
        if comparison['status']!='PASS':raise RuntimeError('Actual CEM clone equality failed')
    if len(cem['clone_comparisons'])!=4:raise RuntimeError('Four actual clone pairs required')
    ledger=e.json('state/TECHNICAL_LEDGER.json');budget=validate_budget(ledger)
    if budget['complete_technical_trajectories']<8:raise RuntimeError('Actual clone trajectories absent from technical ledger')
    for spec in plan['cases']:
        task,case=spec['task'],spec['case']
        if case['case_id'] not in {c['case_id'] for c in roles[task]['cases']['TECH'][:2]}:raise RuntimeError('CEM used a different fixed TECH case')
        for arm in ('H0','H0_CLONE'):
            prefix=f'artifacts/planning/TECH/{task}/{case["case_id"]}/{arm}/attempt_0';path=prefix+'/result.json'
            if path not in cem['files']:raise RuntimeError('Full actual technical trajectory absent from gate seal')
            result=e.json(path,cem['files'][path]);rid=f'R3/TECH/{task}/{case["case_id"]}/{arm}/attempt_0';entry=ledger['trajectories'][rid]
            if result['status']!='COMPLETE' or not result['evaluation_complete'] or result['failure_category'] is not None or (result['task'],result['case_id'],result['arm'],result['phase'])!=(task,case['case_id'],arm,'TECH'):raise RuntimeError('Actual TECH trajectory incomplete/different')
            if entry['status']!='COMPLETE' or entry['result_path']!=path or entry['result_sha256']!=e.files[path]['sha256']:raise RuntimeError('Technical CEM ledger and sealed raw result differ')
    selected={t:{'data':explicit_check(e,data_checks[t],t,'data',roles[t]),'cost':explicit_check(e,cost_checks[t],t,'cost',roles[t])} for t in TASKS}
    e.unchanged();formal_zero(root)
    result={'version':VERSION,'status':'TECHNICAL_GATES_PASS','job_ids':[j['job_id'] for j in jobs],
            'source_hashes':runtime_hashes(e.files),'verified_files':dict(sorted(e.files.items())),
            'source_hash_policy':'Explicit code/identity/base/gate-JSON whitelist for per-job checks. verified_files retains the complete once-verified HDF5/cache/technical-byte closure; formal load_cache independently rehashes each consumed latent file.',
            'explicit_task_receipts':selected,'selection':selection,'technical_budget':budget,'preauthorization_formal_state':zero,
            'scientific_improvement_required':False,'scientific_results_inspected':False,
            'optimizer_updates_by_authorizer':0,'environment_steps_by_authorizer':0,'GPU_queries_or_calls':0,'SSH_connections':0,
            'training_launched':False,'authorization_scope':'Fixed six30000-step jobs; no extension, no initialization inherited from technical updates'}
    return result,e


def publish(root,payload):
    target=local(root,AUTH);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        old=read(target);comparable={k:v for k,v in old.items() if k!='authorized_at'}
        if comparable!=payload:raise RuntimeError('Different existing authorization preserved; never overwrite it')
        return old,False
    data={**payload,'authorized_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    # Atomic create without replacing a concurrent or previously issued authority.
    os.link(tmp,target);tmp.unlink()
    fd=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return data,True


def parse_receipts(values):
    out={}
    for value in values:
        task,sep,path=value.partition('=')
        if not sep or task not in TASKS or task in out:raise ValueError('Use exactly task=ROOT-relative-receipt for each task')
        out[task]=path
    if set(out)!=set(TASKS):raise ValueError('Both tasks must have explicitly selected receipts')
    return out


def run(root,data_checks,cost_checks):
    root=Path(root).resolve();identity=read(local(root,'state/INSTANCE_REUSE.json'))
    if os.uname().nodename!=identity['hostname']:raise RuntimeError('Authorizer must run on the already authorized instance')
    with contextlib.ExitStack() as stack:
        for name in ('state/authorize_formal.lock','state/training_technical_pipeline.lock','state/execution_phase.lock','state/BUDGET.lock'):
            path=local(root,name);path.parent.mkdir(parents=True,exist_ok=True);handle=stack.enter_context(path.open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        payload,e=build_authorization(root,data_checks,cost_checks);e.unchanged();return publish(root,payload)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--root',default=os.environ.get('R3_ROOT',str(Path(__file__).resolve().parents[1])))
    p.add_argument('--data-check',action='append',required=True);p.add_argument('--cost-check',action='append',required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required; this helper authorizes but never launches training')
    result,created=run(args.root,parse_receipts(args.data_check),parse_receipts(args.cost_check));print(json.dumps({'status':result['status'],'created':created,'jobs':len(result['job_ids']),'source_files':len(result['source_hashes']),'training_launched':False}),flush=True)
