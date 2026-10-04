"""Append-only H2 compact recovery manifest, primary verification and seal.

prepare binds actual recovered bytes; seal requires the independent primary
byte check. Full RGB archives remain primary-only and are explicitly identified.
"""
import argparse,datetime,hashlib,json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def record(p):
    p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def check(p,r):
    q=record(p)
    if (q['bytes'],q['sha256'])!=(r['bytes'],r['sha256']):raise RuntimeError('Byte mismatch: '+str(p))
def write_new(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def prepare(a):
    base=a.root.resolve();task=a.task;out=base/'recovery'/task;manifest=out/'RECOVERY_MANIFEST.json'
    if manifest.exists():raise RuntimeError('Immutable manifest exists')
    data=read(out/'CPU_COMPACT_RECOVERY_CHECK.json');cpu=read(out/'CPU_COST/CPU_COST_RECOVERY_CHECK.json')
    assert data['status']=='PASS_ALL_RECOVERED_MEMBERS_AND_ARRAY_IDENTITIES' and data['tasks']=={task:1200}
    assert cpu['candidate_costs']==1200 and cpu['cases_model_stream_replans']==4 and cpu['atol']==cpu['rtol']==1e-5
    assert cpu['status'] in ['PASS_FROZEN_CPU_COST_TOLERANCE','FAIL_FROZEN_CPU_COST_TOLERANCE']
    local={};remote={};mapping={};remote_only=[];archives=[]
    def add(p,r=None):
        p=Path(p).resolve();rec=record(p);local[str(p)]=rec
        if r:
            if r in remote and remote[r]['sha256']!=rec['sha256']:raise RuntimeError('Conflicting primary mapping '+r)
            remote[r]={**rec,'path':r};mapping[r]=str(p)
    def tree(path,remote_root=None):
        for p in sorted(Path(path).rglob('*')):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name!='case.lock':
                add(p,str(Path(remote_root)/p.relative_to(path)) if remote_root else None)
    for receipt_path in sorted((base/'recovery/archives').glob(task+'_*_COMPACT_V1_ARCHIVE.json')):
        receipt=read(receipt_path);arm=receipt['arm'];m=base/'recovery/archives'/Path(receipt['manifest']['path']).name
        ar=base/'recovery/archives'/Path(receipt['archive']['path']).name
        check(m,receipt['manifest']);check(ar,receipt['archive'])
        add(receipt_path,'/workspace/r5/H2/compact_archives/'+receipt_path.name);add(m,receipt['manifest']['path']);add(ar,receipt['archive']['path'])
        restored=base/'recovery/restored'/(task+'_'+arm)
        stage='/workspace/r5/H2/compact_archives/'+task+'_'+arm+'_COMPACT_V1'
        for item in read(m)['members']:check(restored/item['relative_path'],item);add(restored/item['relative_path'],stage+'/'+item['relative_path'])
        archives.append({'archive':record(ar),'manifest':record(m),'receipt':record(receipt_path)})
    assert len(archives)==4
    full=base/'recovery/full_archive_metadata'
    for p in sorted(full.glob(task+'_*_RECOVERY_V1_ARCHIVE.json')):
        d=read(p);m=full/Path(d['manifest']['path']).name;check(m,d['manifest']);add(p,'/workspace/r5/H2/recovery_archives/'+p.name);add(m,d['manifest']['path'])
        remote_only.append({**d['archive'],'reason':'Original complete raw RGB and full trajectories remain primary-only; compact actual planning inputs and all numeric arrays recovered locally.'})
        remote[d['archive']['path']]=d['archive']
    assert len(remote_only)==4
    tree(base/'reports'/f'{task}_MAIN',f'/workspace/r5/H2/{task}/reports')
    tree(base/'reports'/f'{task}_TECH')
    tree(base/'reports/TECH_recovery'/task,f'/workspace/r5/H2/{task}')
    # New local CPU evidence is installed in a separate primary subtree.
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ['RECOVERY_MANIFEST.json','RECOVERY_SEAL.json','PRIMARY_BYTE_CHECK.json']:
            add(p,'/workspace/r5/H2/'+task+'/local_cpu_recovery/'+str(p.relative_to(out)))
    shared=['CPU_COST_SELECTION_V1.json','CPU_COST_CODE_LOCK.json','H2_CONTRACT.json','TECH_COVERAGE_AMENDMENT_V1.json','FORMAL_LAUNCH_V1.json']
    for n in shared:add(base/n,'/workspace/r5/H2/'+n)
    for n in ['CPU_ENVIRONMENT_V1.json','GPU_REFERENCE_ENVIRONMENT_V1.json','DEPENDENCY_INSTALL.log']:
        add(base/'recovery'/n,'/workspace/r5/H2/recovery/'+n)
    for p in sorted(base.glob('*.py')):add(p,'/workspace/r5/code/h2/'+p.name)
    checker=base.parent/'ops/check_h2_tables.py';add(checker,'/workspace/r5/ops/check_h2_tables.py')
    r3=a.r3_root.resolve()
    for p in sorted((r3/'r3').glob('*.py')):add(p,'/workspace/shared_data/r3/'+str(p.relative_to(r3)))
    sources=['source/lewm/jepa.py','source/lewm/module.py','source/lewm/train.py','source/lewm/utils.py','source/spt/stable_pretraining/backbone/utils.py','source/swm_compat/stable_worldmodel/policy.py','state/lewm_source_manifest.json','state/spt_source_manifest.json','state/swm_compat_source_manifest.json',f'official/{task}/weights.pt',f'official/{task}/config.json',f'manifests/{task}_normalization.json',f'manifests/{task}_model_assets.json',f'manifests/{task}_data_roles.json','manifests/OPEN_LOOP_ROUTING.json']
    for n in sources:add(r3/n,'/workspace/shared_data/r3/'+n)
    for item in read(base/'CPU_COST_SELECTION_V1.json')['rows']:
        if item['task']==task and 'checkpoint' in item['provenance']:
            r=item['provenance']['checkpoint'];check(r3/r['path'],r);add(r3/r['path'],'/workspace/shared_data/r3/'+r['path'])
    r4=base.parents[1]/'r4_v23_execution'
    for n in ['r4/common.py','r4/export_cases.py','analysis/statistics.py']:add(r4/n,'/workspace/r4_v23_execution/'+n)
    case=r4/'r4/remote_results/final_recovery'/task/'menu_evidence/source_window_maps/CASE_WINDOWS.json'
    add(case,('/workspace/shared_data/r4_assets' if task=='pusht' else '/workspace/shared_data/r4_assets_reacher')+'/CASE_WINDOWS.json')
    d={'schema':'R5_H2_COMPACT_RECOVERY_MANIFEST_V1','created_utc':now(),'task':task,'status':'LOCAL_RECOVERY_COMPLETE_PRIMARY_VERIFICATION_PENDING',
       'local_files':[local[k] for k in sorted(local)],'remote_files':[remote[k] for k in sorted(remote)],'remote_to_local':mapping,'archives':archives,'remote_only_files':remote_only,
       'scientific_cells':1200,'independent_cases':100,'streams':3,'actual_models':4,'world_model_updates':0,'new_recovery_optimizer_updates':0,'new_recovery_simulator_steps':0,
       'scope':'All1200 compact trajectories, all numeric original arrays and every actual planning RGB/goal image; all original result/control receipts, fixed models and code. Full per-raw-step RGB remains primary-only; original full archive SHAs retained. CPU model validation is only four fixed actual replans,1200 saved last-generation candidate costs, not full trajectory or all CEM replay.',
       'CPU_array_recovery':record(out/'CPU_COMPACT_RECOVERY_CHECK.json'),'CPU_cost_recovery':record(out/'CPU_COST/CPU_COST_RECOVERY_CHECK.json'),'independent_table_check':record(base/'reports'/f'{task}_MAIN/INDEPENDENT_TABLE_CHECK.json'),
       'strict_cpu_cost_gate_passed':cpu['strict_cpu_cost_gate_passed'],'CPU_cost_outliers':cpu['outliers'],'full_raw_RGB_local_recovery_claimed':False,'prior_H1a_strict_CPU_HOLD_unaffected':True,'producer':record(__file__)}
    write_new(manifest,d);print(json.dumps({'manifest':record(manifest),'local_files':len(local),'primary_files':len(remote),'remote_only_archives':4}),flush=True)

def seal(a):
    base=a.root.resolve();out=base/'recovery'/a.task;mp=out/'RECOVERY_MANIFEST.json';m=read(mp);pr=out/'PRIMARY_BYTE_CHECK.json';p=read(pr)
    assert p['status']=='PASS_ALL_PRIMARY_REFERENCED_BYTES' and p['manifest_sha256']==sha(mp) and p['files_checked']==len(m['remote_files'])
    for r in m['local_files']:check(r['path'],r)
    cp=read(out/'CPU_COST/CPU_COST_RECOVERY_CHECK.json')
    d={'schema':'R5_H2_RECOVERY_SEAL_V1','created_utc':now(),'task':a.task,
       'status':'COMPLETE_COMPACT_RECOVERY_WITH_NARROW_CPU_COST_PASS' if cp['strict_cpu_cost_gate_passed'] else 'COMPLETE_COMPACT_RECOVERY_WITH_CPU_COST_TOLERANCE_FAILURE',
       'manifest':record(mp),'primary_byte_check':record(pr),'strict_cpu_cost_gate_passed':cp['strict_cpu_cost_gate_passed'],'CPU_cost_outliers':cp['outliers'],
       'module_narrow_CPU_release_gate':'PASS' if cp['strict_cpu_cost_gate_passed'] else 'HOLD','global_release_gate':'HOLD','global_hold_reason':'Previously sealed H1a strict CPU tolerance failure remains; H2 does not replace or relax it.',
       'scientific_cells':1200,'local_files':len(m['local_files']),'primary_files_verified':p['files_checked'],'scope':m['scope'],'full_raw_RGB_local_recovery_claimed':False,'new_optimizer_updates':0,'new_simulator_steps':0,'producer':record(__file__)}
    write_new(out/'RECOVERY_SEAL.json',d);print(json.dumps({'seal':record(out/'RECOVERY_SEAL.json'),'manifest':record(mp),'status':d['status']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--r3-root',type=Path);p.add_argument('mode',choices=['prepare','seal']);a=p.parse_args();(prepare if a.mode=='prepare' else seal)(a)
