"""Append-only initial-stage R5 recovery seal; no inference or R4 writes.

--check may run before H1b recovery is complete and writes no output artifacts.
--seal requires the H1b module's actual final SHA supplied after completion.
--verify-remote checks all remote-bound source bytes from the frozen manifest.
"""
import argparse,datetime,hashlib,json
from pathlib import Path

OPS=Path(__file__).resolve().parent
R5=OPS.parent
TASK_COUNTS={'pusht':36,'reacher':34,'tworoom':32,'cube':34}
PINS={'C0':'11f6611f7f4b9bb60498dcf8bfa1f159cb64c89b6cf630c1759d9242198adafc',
      'H1a':'690d7ea8a2926e5f1865b2f51b706b13dcb303ae78afe7a992871a4b62b71785',
      'R4_BOUNDARY':'ce2211fd9cd61490089d1278d5b8304112b451a14ff7457fcd53ee0b4e5594f5'}
LOCAL={'C0':R5/'c0/recovery/RECOVERY_SEAL.json','H1a':R5/'h1a/recovery/RECOVERY_SEAL.json',
       'H1b':R5/'h1b/reports/H1B_RECOVERY_SEAL.json'}
REMOTE={'C0':'/workspace/r5/C0/RECOVERY_SEAL.json','H1a':'/workspace/r5/H1a/RECOVERY_SEAL.json',
        'H1b':'/workspace/r5/H1b/H1B_RECOVERY_SEAL.json'}
NAMES=['INITIAL_STAGE_DELIVERY_INDEX.json','INITIAL_STAGE_RECOVERY_MANIFEST.json','INITIAL_STAGE_SEAL.json']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def record(p):
    p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def verify(r,path=None):
    p=Path(path or r.get('local_mirror',r['path']))
    if not p.is_file():raise RuntimeError('Missing required recovered file: '+str(p))
    actual=record(p)
    if any(actual[k]!=r[k] for k in ['bytes','sha256']):raise RuntimeError('Recovered bytes differ: '+str(p))
    return actual
def record_values(value):return list(value.values()) if isinstance(value,dict) else value
def freeze_bytes(p,data):
    if p.exists():
        if p.read_bytes()!=data:raise RuntimeError('Refuse to replace immutable stage file: '+str(p))
    else:
        with p.open('xb') as f:f.write(data)
def freeze(p,value):freeze_bytes(p,(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode())

def inspect(h1b_sha=None):
    local={};remote={};modules={};pending=[]
    def add_local(r,path=None):
        actual=verify(r,path);old=local.get(actual['path'])
        if old and old!=actual:raise RuntimeError('Conflicting local identity: '+actual['path'])
        local[actual['path']]=actual
        return actual
    def add_remote(r,path=None):
        entry={k:r[k] for k in ['bytes','sha256']};entry['path']=str(path or r['path'])
        if not entry['path'].startswith('/workspace/'):raise RuntimeError('Remote binding must be a workspace path')
        old=remote.get(entry['path'])
        if old and old!=entry:raise RuntimeError('Conflicting remote identity: '+entry['path'])
        remote[entry['path']]=entry
    for name,path in LOCAL.items():
        if not path.exists():pending.append(name+'_RECOVERY_SEAL_MISSING');continue
        seal_record=record(path)
        expected=PINS.get(name,h1b_sha)
        if expected and seal_record['sha256']!=expected:raise RuntimeError(name+' module seal SHA differs')
        if name=='H1b' and not h1b_sha:pending.append('H1b_FINAL_SEAL_SHA_NOT_YET_SUPPLIED')
        add_local(seal_record);add_remote(seal_record,REMOTE[name]);d=read(path)
        modules[name]={'seal':seal_record,'remote_path':REMOTE[name],'status':d['status']}
        if name=='C0':
            assert d['status']=='SEALED_C0_PASS' and d['scientific_gate']=='PASS_C_A_A'
            assert not d['C1_C2_started'] and d['world_model_updates']==d['probe_updates']==0
            for r in list(d['files'].values())+[d[k] for k in ['archive','local_recovery_check','recovery_manifest']]:add_local(r);add_remote(r)
            check=read(d['local_recovery_check']['local_mirror'])
            assert check['status']=='PASS' and check['different_fields']==0 and check['recomputed_field_comparisons']==8204
            assert check['inventory']['raw_state_render_snapshots']==208
            modules[name]['scope']=d['local_recovery_scope']
        elif name=='H1a':
            assert d['status']=='COMPLETE_WITH_CPU_TOLERANCE_FAILURE' and d['release_gate']=='HOLD' and d['strict_cpu_gate_passed'] is False
            for r in d['local_files'].values():add_local(r)
            for r in list(d['remote_files'].values())+list(d['remote_readonly_dependencies'].values()):add_remote(r)
            cpu=read(d['CPU_recovery']['path']);assert cpu['outliers']==12 and cpu['coordinates']==1536000
            modules[name].update(scope=d['CPU_scope'],strict_cpu_gate_passed=False,CPU_outliers=12,release_gate='HOLD')
        else:
            assert d['status']=='COMPLETE_LOCAL_RECOVERY_AND_CPU_REEVALUATION'
            assert d['expected_fit_counts']==TASK_COUNTS and d['total_probes']==136 and d['saved_EVAL_case_model_cells']==13600 and d['world_model_updates']==0
            for r in record_values(d['local_files']):add_local(r)
            reevaluations=[r for r in d['local_files'] if Path(r['path']).name=='LOCAL_CPU_REEVALUATION.npz']
            assert len(reevaluations)==136
            source_hashes={r['sha256'] for r in d['local_files']}
            scopes={}
            for task,count in TASK_COUNTS.items():
                rr=d['recoveries'][task];r=rr['receipt'];add_local(r);cpu=read(r['path'])
                assert cpu['task']==task and cpu['probes_reevaluated']==count and cpu['case_model_cells']==count*100
                assert len(cpu['checks'])==count and cpu['world_model_updates']==cpu['probe_optimizer_updates']==0
                for row in cpu['checks']:
                    assert row['cases']==100 and row['optimizer_updates']==row['world_model_updates']==0
                    assert row['parameter_file_sha256'] in source_hashes and row['eval_input_file_sha256'] in source_hashes
                scopes[task]={'receipt':r,'probes':count,'saved_EVAL_case_model_cells':count*100,'scope':cpu['scope'],
                    'max_prediction_abs_difference':cpu['max_prediction_abs_difference'],'max_error_abs_difference':cpu['max_error_abs_difference']}
                archive_receipt=R5/'h1b/archives'/f'{task}_ARCHIVE.json';ar=read(archive_receipt);add_local(record(archive_receipt))
                for key in ['archive','manifest']:add_remote(ar[key])
                manifest_path=R5/'h1b/archives'/f'{task}_RECOVERY_MANIFEST.json'
                add_local(ar['manifest'],manifest_path)
                for member in read(manifest_path)['members']:add_remote(member)
            modules[name].update(scope='All136 recovered selected probes on all100 saved EVAL anchors per fit; cross-platform differences reported, no new numerical tolerance or PASS criterion.',
                recoveries=scopes,total_probes=136,saved_EVAL_case_model_cells=13600,new_numerical_acceptance_gate=False,
                remote_only_scope={'files':d.get('remote_only_files',[]),'collections':d.get('remote_only_collections',[])})
    boundary=OPS/'R4_READONLY_POST_R5_CHECK.json';br=record(boundary)
    if br['sha256']!=PINS['R4_BOUNDARY']:raise RuntimeError('Pinned R4 readonly boundary report changed')
    b=read(boundary);assert b['status']=='PASS_ALL_ORIGINAL_SEALED_BYTES_UNCHANGED' and b['records_checked']==53730 and b['difference_count']==0
    add_local(br);add_remote(br,'/workspace/r5/ops/R4_READONLY_POST_R5_CHECK.json')
    index_path=R5/'DELIVERY_INDEX.json';index_bytes=index_path.read_bytes();index=json.loads(index_bytes)
    assert index['stage_scope']==['C0','H1a','H1b'] and index['initial_scientific_stage_complete'] and not index['all_R5_complete']
    assert index['world_model_new_updates']==0 and set(index['not_started'])=={'C1','H2','C2'} and set(index['not_triggered'])=={'H3','H2_TwoRoom'}
    for name,module in index['modules'].items():
        for r in module['files']:add_local(r);add_remote(r)
    if 'RUNNING' in index['modules']['H1b']['status']:pending.append('DELIVERY_INDEX_STILL_REPORTS_H1b_RECOVERY_RUNNING')
    ledger_path=R5/'h1b/reports/main/MEASUREMENT_UPDATE_LEDGER.json';ledger=read(ledger_path);add_local(record(ledger_path))
    assert ledger['formal_MLP_updates']==187200 and ledger['CPU_TECH_MLP_updates']==200 and ledger['total_measurement_updates']==187400 and ledger['world_model_updates']==0
    fits=ledger['formal_fit_receipts'];mlp=[r for r in fits if r['model']=='MLP'];ridge=[r for r in fits if r['model']=='RIDGE']
    assert len(fits)==136 and len(mlp)==96 and len(ridge)==40
    assert sum(r['updates'] for r in mlp)==187200 and all(r['updates']==0 for r in ridge)
    h3=read(R5/'h1b/reports/main/H3_MEASUREMENT_ONLY.json')
    assert h3['at_least_two_horizons_meet_measurement_condition'] is False and h3['H3_started'] is False
    h2_path=R5/'h1a/recovery/bundle/H1a/combined/H2_TWOROOM_CONDITION.json';h2=read(h2_path);add_local(record(h2_path))
    assert h2['condition_met'] is False
    readiness_path=OPS/'NEXT_MODULE_READINESS.json';readiness=read(readiness_path);ready_record=record(readiness_path)
    assert readiness['current_authorized_launch_stage']=='C0_H1a_H1b_ONLY'
    assert readiness['current_new_closed_loop_runs']==readiness['world_model_new_updates']==0
    assert readiness['remaining_closed_loop_count_if_continued_under_original_rules']==3600
    assert readiness['conditions']['H2_tworoom']['condition_met'] is False and readiness['conditions']['H3']['condition_met'] is False
    for condition in readiness['conditions'].values():
        if 'source' in condition:add_local(condition['source'])
    add_local(ready_record);add_remote(ready_record,'/workspace/r5/ops/NEXT_MODULE_READINESS.json')
    provenance={}
    for name,relative in [('package_acceptance','PACKAGE_ACCEPTANCE.json'),('execution_ledger','ops/EXECUTION_LEDGER.json'),('progress_events','ops/PROGRESS_EVENTS.jsonl')]:
        r=record(R5/relative);add_local(r);add_remote(r,'/workspace/r5/'+relative);provenance[name]=r
    add_local(record(Path(__file__)))
    return {'pending':pending,'modules':modules,'local':local,'remote':remote,'boundary':br,'index_bytes':index_bytes,
        'index_source':record(index_path),'ledger':record(ledger_path),'readiness':ready_record,'provenance':provenance,'ledger_values':{'formal_probe_updates':187200,'TECH_probe_updates':200,'total_probe_updates':187400,'world_model_updates':0}}

def seal(h1b_sha):
    if not h1b_sha:raise RuntimeError('An actual completed H1b module seal SHA is required')
    result=inspect(h1b_sha)
    if result['pending']:raise RuntimeError('Initial stage not ready: '+','.join(result['pending']))
    snapshot=OPS/NAMES[0];manifest_path=OPS/NAMES[1];seal_path=OPS/NAMES[2]
    if (R5/'DELIVERY_INDEX.json').read_bytes()!=result['index_bytes']:raise RuntimeError('Dynamic delivery index changed during verification; snapshot not created')
    freeze_bytes(snapshot,result['index_bytes'])
    created=read(manifest_path)['created_at_utc'] if manifest_path.exists() else datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['local'][str(snapshot)]=record(snapshot)
    manifest={'version':'R5_INITIAL_STAGE_RECOVERY_V1','status':'INITIAL_STAGE_FILES_COMPLETE_WITH_H1a_CPU_GATE_FAILURE','created_at_utc':created,
        'stage_scope':['C0','H1a','H1b'],'all_R5_complete':False,'primary_remote_root':'/workspace/r5','module_recovery_seals':result['modules'],
        'verified_local_files':[result['local'][p] for p in sorted(result['local'])],'remote_required_files':[result['remote'][p] for p in sorted(result['remote'])],
        'file_count':len(result['local']),'remote_file_count':len(result['remote']),'delivery_index_snapshot':record(snapshot),
        'delivery_index_source_at_snapshot':result['index_source'],'R4_readonly_boundary':result['boundary'],'next_module_readiness':result['readiness'],'measurement_update_ledger':result['ledger'],
        'initial_stage_provenance':result['provenance'],
        'snapshot_scope':'Historical state of the completed C0/H1a/H1b stage; readiness and not-started fields are not restrictions on separately authorized later R5 work.',
        'updates':result['ledger_values'],'not_started':['C1','H2','C2'],'conditions_not_met':['H3','H2_TwoRoom'],
        'strict_CPU_acceptance_failure':{'module':'H1a','outlier_coordinates':12,'unchanged_atol':1e-5,'unchanged_rtol':1e-5,'release_gate':'HOLD'},
        'H1b_CPU_scope':'136 selected recovered ridge/MLP fits x100 saved EVAL anchors; reports cross-platform differences without a new post-result tolerance. No encoder/simulator/training rerun.',
        'full_training_resume_or_TRAIN_cache_recovery_claimed':False,'R4_files_modified':False,'full_R5_root_SEAL_written':False,
        'producer':record(Path(__file__))}
    freeze(manifest_path,manifest)
    stage_seal={'version':'R5_INITIAL_STAGE_SEAL_V1','status':'SEALED_INITIAL_STAGE_WITH_H1a_CPU_RECOVERY_FAILURE','sealed_utc':created,
        'stage_scope':['C0','H1a','H1b'],'initial_scientific_stage_complete':True,'initial_stage_recovery_complete':True,'all_R5_complete':False,
        'release_gate':'HOLD','strict_cpu_gate_passed':False,'strict_cpu_gate_failure_modules':['H1a'],'recovery_manifest':record(manifest_path),
        'delivery_index_snapshot':record(snapshot),'module_seals':{name:entry['seal'] for name,entry in result['modules'].items()},
        'remote_module_seal_paths':REMOTE,'R4_readonly_boundary':result['boundary'],'next_module_readiness':result['readiness'],'updates':result['ledger_values'],
        'initial_stage_provenance':result['provenance'],
        'snapshot_scope':'Historical initial-stage evidence only; not a restriction on subsequently authorized R5 modules.',
        'not_started':['C1','H2','C2'],'conditions_not_met':['H3','H2_TwoRoom'],'R4_unchanged':True,'future_full_R5_SEAL_untouched':True}
    freeze(seal_path,stage_seal)
    print(json.dumps({'status':stage_seal['status'],'release_gate':'HOLD','outputs':{p.name:record(p) for p in [snapshot,manifest_path,seal_path]}},ensure_ascii=False))

def verify_remote(path):
    d=read(path)
    for r in d['remote_required_files']:verify(r,r['path'])
    print(json.dumps({'status':'ALL_INITIAL_STAGE_REMOTE_REFERENCES_VERIFIED','files':len(d['remote_required_files']),'manifest':record(path)}))

def install_remote(staging,expected_seal):
    if not expected_seal:raise RuntimeError('Expected frozen stage seal SHA is required for installation')
    staged={name:Path(staging)/name for name in NAMES};seal_doc=read(staged[NAMES[2]])
    if sha(staged[NAMES[2]])!=expected_seal:raise RuntimeError('Transferred initial stage seal SHA differs')
    verify(seal_doc['recovery_manifest'],staged[NAMES[1]]);verify(seal_doc['delivery_index_snapshot'],staged[NAMES[0]])
    assert seal_doc['release_gate']=='HOLD' and seal_doc['all_R5_complete'] is False
    manifest=read(staged[NAMES[1]])
    assert sha(Path(__file__))==manifest['producer']['sha256']
    verify_remote(staged[NAMES[1]])
    root=Path('/workspace/r5')
    # Inspect all final destinations before any creation. No replacement is allowed.
    for name,p in staged.items():
        target=root/name
        if target.exists() and target.read_bytes()!=p.read_bytes():raise RuntimeError('Existing initial stage file differs: '+str(target))
    for name,p in staged.items():freeze_bytes(root/name,p.read_bytes())
    print(json.dumps({'status':'IMMUTABLE_INITIAL_STAGE_INSTALLED','release_gate':'HOLD','all_R5_complete':False,
        'outputs':{name:record(root/name) for name in NAMES},'R4_modified':False,'full_R5_SEAL_written':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check',action='store_true');g.add_argument('--seal',action='store_true');g.add_argument('--verify-remote',type=Path);g.add_argument('--install-remote',type=Path);p.add_argument('--h1b-seal-sha');p.add_argument('--expected-seal-sha');a=p.parse_args()
    if a.install_remote:install_remote(a.install_remote,a.expected_seal_sha)
    elif a.verify_remote:verify_remote(a.verify_remote)
    elif a.seal:seal(a.h1b_seal_sha)
    else:
        r=inspect(a.h1b_seal_sha);print(json.dumps({'status':'WAITING_MODULE_RECOVERY' if r['pending'] else 'READY_TO_SEAL','pending':r['pending'],'verified_local_files':len(r['local']),'bound_remote_files':len(r['remote']),'modules_checked':list(r['modules']),'writes':0},ensure_ascii=False))
