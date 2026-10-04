"""Independent delivery hashes and sequential case bootstrap audit; no simulation reruns."""
import argparse, collections, csv, hashlib, json
from pathlib import Path
import numpy as np

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(root, module):
    folder=root/'modules'/module
    pr_path=root/'prereg'/f'MODULE_{module}_FORMAL_PREREG.json'
    pr=json.loads(pr_path.read_text())
    seal=json.loads((folder/'SEAL.json').read_text())
    assert seal['prereg_sha256']==sha(pr_path)
    for name,record in seal['files'].items():
        p=folder/name
        assert sha(p)==record['sha256'] and p.stat().st_size==record['bytes'], p
    raw_count=0
    first_plan_pairs=0
    for receipt in (root/'raw'/module).rglob('COMPLETE.json'):
        d=json.loads(receipt.read_text())
        for name,record in d['files'].items():
            p=receipt.parent/name
            assert sha(p)==record['sha256'] and p.stat().st_size==record['bytes'], p
        if receipt.parent.name in ('HISTORY3','H_REAL3_REPLAN','REAL3_NULLACT','REPEAT3_REALACT'):
            control='HISTORY1' if module=='A' else 'H_POLICY'
            with np.load(receipt.parent/'trajectory.npz',allow_pickle=False) as z:
                history_plan=z['returned_plans_normalized'][0]
            with np.load(receipt.parent.parent/control/'trajectory.npz',allow_pickle=False) as z:
                control_plan=z['returned_plans_normalized'][0]
            assert history_plan.shape==control_plan.shape and history_plan.dtype==control_plan.dtype
            assert history_plan.tobytes()==control_plan.tobytes(),receipt
            first_plan_pairs+=1
        raw_count+=1
    rows=list(csv.DictReader((folder/'RAW_VALUES.csv').open()))
    if module=='A':
        return dict(module=module,status='PASS',raw_complete_receipts=raw_count,
                    first_plan_pairs_bitwise_equal=first_plan_pairs,report_files_verified=len(seal['files']),statistics_check='See A_INDEPENDENT_ANALYSIS_CHECK.json')
    values={(r['case_id'],r['arm'],r['stream'],r['mode']):r for r in rows}
    assert len(values)==len(rows)
    arms=['H0','REFIT_103201','REFIT_103202','REFIT_103203']
    streams=['R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2']
    endpoints={'E_D1':'REAL3_NULLACT','E_D2':'REPEAT3_REALACT'} if module=='D' else {f'E_{module}':'H_REAL3_REPLAN'}
    reported={r['endpoint']:r for r in json.loads((folder/'STATS.json').read_text())['endpoints']}
    checks=[]
    for endpoint,mode in endpoints.items():
        case_values=[]
        for case in pr['case_order']:
            pairs=[(values[case,arm,stream,'H_POLICY'],values[case,arm,stream,mode]) for arm in arms for stream in streams]
            if all(b['success']!='' and h['status']=='COMPLETE' for b,h in pairs):
                case_values.append(sum(int(h['success'])-int(b['success']) for b,h in pairs)/12)
        x=np.asarray(case_values,dtype=float)
        target=reported[endpoint]
        assert len(x)==target['complete_cases']
        if len(x):
            seed=int.from_bytes(hashlib.sha256(f'R10_CASE_BOOTSTRAP/{module}/{endpoint}'.encode()).digest()[:8],'big')
            rng=np.random.default_rng(seed)
            # Draw sequentially, independently of report_old's vectorized implementation.
            draws=[float(x[rng.integers(len(x),size=len(x))].mean()) for _ in range(5000)]
            ci=np.percentile(draws,[2.5,97.5],method='linear')
            assert abs(float(x.mean())-target['point'])<1e-14
            assert np.allclose(ci,[target['ci95_low'],target['ci95_high']],rtol=0,atol=1e-14)
            label='提高' if ci[0]>0 else '降低' if ci[1]<0 else '未检出'
            assert label==target['label']
            checks.append(dict(endpoint=endpoint,cases=len(x),point=float(x.mean()),ci95=ci.tolist(),label=label))
        else:
            assert target['point'] is None
            checks.append(dict(endpoint=endpoint,cases=0,point=None))
    return dict(module=module,status='PASS',raw_complete_receipts=raw_count,
                first_plan_pairs_bitwise_equal=first_plan_pairs,
                report_files_verified=len(seal['files']),rows=len(rows),endpoints=checks,
                success_counts={m:sum(int(r['success']) for r in rows if r['mode']==m and r['success']!='') for m in ['H_POLICY',*endpoints.values()]},
                status_counts=dict(collections.Counter(r['status'] for r in rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--modules',nargs='+',required=True);a=p.parse_args()
    for module in a.modules:
        result=audit(a.root,module)
        (a.root/'ops'/f'{module}_DELIVERY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False))
