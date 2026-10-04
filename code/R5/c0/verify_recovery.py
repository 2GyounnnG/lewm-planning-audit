"""Local CPU evidence reconstruction; no simulator calls and no model calls."""
import csv, datetime, hashlib, json, tarfile
from pathlib import Path
import numpy as np
from .common import sha,read,record,atomic,comparisons

def main():
    base=Path(__file__).parent;recovery=base/'recovery';receipt=read(recovery/'ARCHIVE_RECEIPT.json');archive=recovery/'C0_v1.tar.gz'
    assert archive.stat().st_size==receipt['archive']['bytes'] and sha(archive)==receipt['archive']['sha256']
    dest=recovery/'extracted';dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        members=t.getmembers();assert len(members)==receipt['member_count']
        for m in members:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        t.extractall(dest,filter='data')
    manifest=read(dest/'C0/RECOVERY_MANIFEST.json');assert sha(dest/'C0/RECOVERY_MANIFEST.json')==receipt['manifest']['sha256']
    for relative,r in manifest['files'].items():
        p=dest/relative;assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],relative
    contract=read(dest/'C0/C0_CONTRACT.json');gate=read(dest/'C0/C0_GATE.json');assert gate['status']=='PASS_C_A_A'
    assert sha(dest/'C0/C0_CONTRACT.json')==gate['contract']['sha256']
    points=[];fields=[];raw_steps=0;trials=0
    for item in contract['inputs']:
        cid=item['case']['case_id'];folder=dest/'C0/trials'/cid
        for repeat in (0,1):
            p=folder/str(repeat);complete=read(p/'COMPLETE.json')
            for r in complete['files'].values():assert sha(p/Path(r['path']).name)==r['sha256']
            result=read(p/'result.json');assert result['status']=='COMPLETE';raw_steps+=result['executed_test_raw_steps'];trials+=1
        with np.load(folder/'0/states.npz') as a,np.load(folder/'1/states.npz') as b:
            for step in range(26):
                prefix=f'{step:03d}:'
                comparison=comparisons({k[len(prefix):]:a[k] for k in a.files if k.startswith(prefix)},
                                       {k[len(prefix):]:b[k] for k in b.files if k.startswith(prefix)})
                for r in comparison:fields.append({'case_id':cid,'raw_step':step,**r})
                points.append((cid,step,len(comparison),all(r['bitwise_equal'] for r in comparison)))
    expected=list(csv.DictReader((dest/'C0/C_A_A_FIELD_RAW.csv').open()))
    assert len(expected)==len(fields)==8204
    for actual,old in zip(fields,expected,strict=True):
        for k,v in actual.items():assert str(v)==old[k],(k,v,old[k])
    summary=list(csv.DictReader((dest/'C0/C_A_A_RAW_TABLE.csv').open()))
    assert len(points)==len(summary)==104 and all(p[-1] for p in points)
    for point,row in zip(points,summary,strict=True):
        assert (row['case_id'],int(row['raw_step']),int(row['fields']),row['all_equal']=='True')==point
    assert trials==8 and raw_steps==200
    out={'status':'PASS','scope':'CPU recovery of all original arrays and exact comparison tables; no new MuJoCo simulation/model inference',
       'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':record(archive),
       'recovery_manifest':record(dest/'C0/RECOVERY_MANIFEST.json'),'members_verified':len(members),'inventory':manifest['inventory'],
       'recomputed_field_comparisons':8204,'table_fields_identical':True,'different_fields':0,'raw_simulation_calls':0,'optimizer_updates':0,
       'verification_code':record(__file__),'gate':record(dest/'C0/C0_GATE.json')}
    atomic(base/'reports/LOCAL_RECOVERY_CHECK.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':main()
