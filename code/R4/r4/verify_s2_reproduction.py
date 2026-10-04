"""Complete deferred S2 first-plan comparisons after local original logs arrive."""
import argparse,csv,json
from pathlib import Path
from .common import ARMS,atomic_json,file_record

def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--s2',required=True);p.add_argument('--trajectories',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=[]
    for folder in sorted(Path(a.s2).iterdir()):
        if not (folder/'COMPLETE.json').exists():continue
        complete=json.loads((folder/'COMPLETE.json').read_text());result=json.loads((folder/'result.json').read_text());path=folder/'initial_proposals.npz'
        if file_record(path)!=complete['files']['initial_proposals.npz']:raise RuntimeError('S2 original proposal source differs')
        with np.load(path) as proposal:
            local=[]
            for arm in ARMS:
                original=Path(a.trajectories)/'FORMAL/R3_ORIGINAL'/result['case_id']/arm;receipt=json.loads((original/'COMPLETE.json').read_text())
                if file_record(original/'trajectory.npz')!=receipt['files']['trajectory.npz']:raise RuntimeError('Original trajectory seal differs')
                with np.load(original/'trajectory.npz') as trajectory:
                    returned=proposal[arm+'_returned'];generation=proposal[arm+'_last_generation'];r=trajectory['returned_plans_normalized'][0];g=trajectory['last_generation_normalized'][0]
                    same_return=np.array_equal(returned,r);same_generation=np.array_equal(generation,g)
                    row={'task':result['task'],'case_id':result['case_id'],'arm':arm,'status':'PASS' if same_return and same_generation else 'FAIL','returned_plan_exact':bool(same_return),'last_generation_exact':bool(same_generation),'max_return_abs':float(np.max(np.abs(returned-r))),'max_generation_abs':float(np.max(np.abs(generation-g))),'source_proposals_sha256':complete['files']['initial_proposals.npz']['sha256'],'source_original_trajectory_sha256':receipt['files']['trajectory.npz']['sha256']};rows.append(row);local.append(row)
            atomic_json(folder/'INITIAL_CEM_REPRODUCTION_FINAL.json',{'status':'PASS' if all(r['status']=='PASS' for r in local) else 'FAIL','rows':local,'original_start_time_comparison_preserved':True})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'S2_INITIAL_CEM_REPRODUCTION_FINAL_RAW.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'PASS' if len(rows)==400 and all(r['status']=='PASS' for r in rows) else 'FAIL_OR_INCOMPLETE','rows':len(rows),'passed':sum(r['status']=='PASS' for r in rows),'scientific_outputs_unchanged':True}
    atomic_json(out/'S2_INITIAL_CEM_REPRODUCTION_FINAL.json',result);print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise RuntimeError('Initial CEM reproduction unresolved')

if __name__=='__main__':main()
