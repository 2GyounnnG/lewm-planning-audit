"""Append-only correction: macOS AppleDouble sidecars are local metadata only."""
import argparse,json
from pathlib import Path
from seal_recovery import read,record,write_new,now,check,sha

def amend(a):
    base=a.root.resolve();out=base/'recovery'/a.task;prior=out/'RECOVERY_MANIFEST.json';d=read(prior)
    removed=[r for r in d['remote_files'] if Path(r['path']).name.startswith('._')]
    assert len(removed)==12 and all(r['path'].startswith('/workspace/shared_data/r3/r3/._') for r in removed)
    d['remote_files']=[r for r in d['remote_files'] if r not in removed]
    for r in removed:d['remote_to_local'].pop(r['path'])
    d['schema']='R5_H2_COMPACT_RECOVERY_MANIFEST_V2';d['created_utc']=now()
    d['append_only_correction']={'reason':'Twelve macOS AppleDouble ._*.py files were erroneously included by a *.py glob in the primary mapping. They are non-executable filesystem metadata, retained and SHA-bound locally; the original Python source files are still fully checked. No model/scientific/source bytes or numeric tolerance changed.',
       'prior_manifest':record(prior),'excluded_primary_metadata_only':removed,'scientific_values_changed':False,'prior_manifest_preserved':True}
    def add(p,remote):
        rec=record(p)
        old=next((r for r in d['local_files'] if r['path']==rec['path']),None)
        if old is None:d['local_files'].append(rec)
        else:assert old==rec
        old=next((r for r in d['remote_files'] if r['path']==remote),None)
        if old is None:d['remote_files'].append({**rec,'path':remote})
        else:assert old=={**rec,'path':remote}
        d['remote_to_local'][remote]=str(Path(p).resolve())
    add(prior,f'/workspace/r5/H2/{a.task}/RECOVERY_MANIFEST.json')
    fail=out/'PRIMARY_BYTE_CHECK.json'
    if fail.exists():
        f=read(fail);assert f['status']=='FAIL_PRIMARY_REFERENCED_BYTES' and len(f['failures'])==12
        assert {r['path'] for r in f['failures']}=={r['path'] for r in removed}
        add(fail,f'/workspace/r5/H2/{a.task}/PRIMARY_BYTE_CHECK.json');d['append_only_correction']['prior_failed_check']=record(fail)
    add(__file__,'/workspace/r5/code/h2/amend_primary_mapping.py')
    write_new(out/'RECOVERY_MANIFEST_V2.json',d);print(json.dumps({'manifest':record(out/'RECOVERY_MANIFEST_V2.json'),'primary_files':len(d['remote_files']),'local_metadata_sidecars':len(removed)}),flush=True)

def seal(a):
    out=a.root.resolve()/'recovery'/a.task;mp=out/'RECOVERY_MANIFEST_V2.json';m=read(mp);pr=out/'PRIMARY_BYTE_CHECK_V2.json';p=read(pr)
    assert p['status']=='PASS_ALL_PRIMARY_REFERENCED_BYTES' and p['manifest_sha256']==sha(mp) and p['files_checked']==len(m['remote_files'])
    for r in m['local_files']:check(r['path'],r)
    cp=read(out/'CPU_COST/CPU_COST_RECOVERY_CHECK.json')
    d={'schema':'R5_H2_RECOVERY_SEAL_V1','created_utc':now(),'task':a.task,
       'status':'COMPLETE_COMPACT_RECOVERY_WITH_NARROW_CPU_COST_PASS' if cp['strict_cpu_cost_gate_passed'] else 'COMPLETE_COMPACT_RECOVERY_WITH_CPU_COST_TOLERANCE_FAILURE',
       'manifest':record(mp),'prior_manifest':record(out/'RECOVERY_MANIFEST.json'),'primary_byte_check':record(pr),'strict_cpu_cost_gate_passed':cp['strict_cpu_cost_gate_passed'],'CPU_cost_outliers':cp['outliers'],
       'module_narrow_CPU_release_gate':'PASS' if cp['strict_cpu_cost_gate_passed'] else 'HOLD','global_release_gate':'HOLD','global_hold_reason':'Previously sealed H1a strict CPU tolerance failure remains; H2 does not replace or relax it.',
       'scientific_cells':1200,'local_files':len(m['local_files']),'primary_files_verified':p['files_checked'],'scope':m['scope'],'full_raw_RGB_local_recovery_claimed':False,'new_optimizer_updates':0,'new_simulator_steps':0,'producer':record(__file__)}
    write_new(out/'RECOVERY_SEAL.json',d);print(json.dumps({'seal':record(out/'RECOVERY_SEAL.json'),'manifest':record(mp),'status':d['status']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('mode',choices=['amend','seal']);a=p.parse_args();(amend if a.mode=='amend' else seal)(a)
