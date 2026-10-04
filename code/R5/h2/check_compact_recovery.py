"""CPU-only verification of every recovered H2 compact array and source byte.

No neural inference, planning, simulator, new case selection or optimizer update.
Does not assert a cross-platform model tolerance gate or full-RGB recovery.
"""
import argparse,hashlib,json,tarfile,csv,platform,sys
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def check(p,r):
    if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise RuntimeError('Recovered member differs: '+str(p))
def array_record(v):
    a=np.ascontiguousarray(v);return {'shape':list(a.shape),'dtype':str(a.dtype),'array_bytes_sha256':hashlib.sha256(a.tobytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--archives',required=True,type=Path);p.add_argument('--restored',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--task',choices=['reacher','pusht']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[];members_checked=0;arrays_checked=0;frames_checked=0;receipts=[];bytask={}
    for receipt_path in sorted(a.archives.glob('*_COMPACT_V1_ARCHIVE.json')):
        receipt=json.loads(receipt_path.read_text())
        if a.task and receipt['task']!=a.task:continue
        manifest_path=a.archives/Path(receipt['manifest']['path']).name;archive=a.archives/Path(receipt['archive']['path']).name
        check(manifest_path,receipt['manifest']);check(archive,receipt['archive']);manifest=json.loads(manifest_path.read_text());assert manifest['case_streams']==300
        destination=a.restored/(receipt['task']+'_'+receipt['arm']);destination.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive) as tar:tar.extractall(destination,filter='data')
        for member in manifest['members']:check(destination/member['relative_path'],member);members_checked+=1
        for case in manifest['cases']:
            path=destination/case['compact_trajectory']['relative_path'];check(path,case['compact_trajectory'])
            with np.load(path,allow_pickle=False) as f:compact={k:f[k] for k in f.files}
            original=case['original_arrays'];kept=0
            for name,r in original.items():
                if name=='raw_pixels':continue
                assert array_record(compact[name])==r;arrays_checked+=1;kept+=1
            frame_count=0
            if 'raw_pixels' in original:
                indexes=compact['planner_raw_indices'];assert indexes.tolist()==case['retained_planning_raw_indices']
                wanted=[0,15,20,25] if len(compact['replan_raw'])>1 else [0];assert indexes.tolist()==wanted
                for i,pixels in zip(indexes,compact['planner_pixels']):
                    assert hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()==case['original_raw_pixel_per_frame_sha256'][i]
                    frame_count+=1;frames_checked+=1
            relative=Path(case['relative_directory']);result=destination/relative/'result.json';control=destination/'controls'/relative/'result.json'
            check(control,case['control_original_result'])
            d=json.loads(result.read_text());b=json.loads(control.read_text());assert d['optimizer_updates']==0
            assert d['replans'][0]['control_first_plan_check']['status']=='BITWISE_EQUAL'
            rows.append({'task':receipt['task'],'case_id':case['case_id'],'arm':case['arm'],'stream':case['stream'],'kept_original_arrays_checked':kept,
                'actual_planning_RGB_frames_checked':frame_count,'control_success':b['success'],'intervention_success':d['success'],'control_replan':b['replan_calls']>1,'intervention_replan':d['replan_calls']>1,
                'original_trajectory_sha256':case['original_trajectory']['sha256'],'compact_trajectory_sha256':case['compact_trajectory']['sha256'],'intervention_result_sha256':sha(result),
                'control_result_sha256':sha(control),'byte_and_array_match':True,'full_raw_RGB_recovered':False})
            bytask[receipt['task']]=bytask.get(receipt['task'],0)+1
        receipts.append({'path':str(receipt_path.resolve()),'sha256':sha(receipt_path)})
    expected={a.task:1200} if a.task else {'pusht':1200,'reacher':1200}
    if len(receipts)!=4*len(expected) or len(rows)!=1200*len(expected) or bytask!=expected:raise RuntimeError('Incomplete fixed compact recovery scope')
    cells={(r['task'],r['case_id'],r['arm'],r['stream']) for r in rows}
    assert len(cells)==len(rows)
    for task in expected:
        task_rows=[r for r in rows if r['task']==task];ids={r['case_id'] for r in task_rows};assert len(ids)==100
        assert all(sum(r['case_id']==c for r in task_rows)==12 for c in ids)
    with (a.out/'COMPACT_RECOVERY_RAW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'PASS_ALL_RECOVERED_MEMBERS_AND_ARRAY_IDENTITIES','archives':len(receipts),'case_stream_model_cells':len(rows),'tasks':bytask,'members_checked':members_checked,
        'original_numeric_arrays_checked':arrays_checked,'actual_planning_RGB_frames_checked':frames_checked,'source_receipts':receipts,
        'scope':'All original member SHA and original non-raw-RGB array identities; every retained actual planning image checked against original frame SHA. Exact original percase success/replan values recovered.',
        'new_neural_inference':0,'new_CEM_plans':0,'new_simulator_steps':0,'new_optimizer_updates':0,'full_raw_RGB_local_recovery_claimed':False,
        'CPU_fingerprint':{'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,'numpy':np.__version__},
        'cross_platform_model_tolerance_gate_executed':False,'strict_cpu_model_gate_status':'NOT_RETESTED_BY_THIS_DATA_RECOVERY_CHECK','original_full_archives_retained_remote':True,
        'raw':{'path':str((a.out/'COMPACT_RECOVERY_RAW.csv').resolve()),'sha256':sha(a.out/'COMPACT_RECOVERY_RAW.csv')},'producer_sha256':sha(__file__)}
    with (a.out/'CPU_COMPACT_RECOVERY_CHECK.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
