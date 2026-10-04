"""Frozen Module A analysis; case bootstrap, no missing-as-failure conversion."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

STREAMS = ['R3_ORIGINAL', 'R4_ALT_CEM_1', 'R4_ALT_CEM_2']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def csv_out(p, rows):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def bootstrap(endpoint, values, descriptive=False):
    seed = int.from_bytes(hashlib.sha256(f'R10_CASE_BOOTSTRAP/A/{endpoint}'.encode()).digest()[:8], 'big')
    values = np.asarray(values, dtype=float)
    if len(values):
        indices = np.random.default_rng(seed).integers(0, len(values), size=(5000, len(values)))
        lo, hi = np.quantile(values[indices].mean(1), [.025, .975], method='linear')
        point = float(values.mean())
        label = '提高' if lo > 0 else '降低' if hi < 0 else '未检出'
    else:
        point, lo, hi, label = None, None, None, '不可估计'
    return dict(endpoint=endpoint, point=point, ci95_low=None if lo is None else float(lo),
                ci95_high=None if hi is None else float(hi), label=label, cases=len(values),
                bootstrap_seed_uint64=seed, descriptive_only=descriptive)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw',required=True); p.add_argument('--prereg',required=True)
    p.add_argument('--baseline',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    prereg=json.loads(Path(a.prereg).read_text())
    assert sha(__file__) == prereg['analysis_code_sha256']
    assert sha(a.baseline) == prereg['baseline_raw_sha256']
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    baseline={(r['case_id'],r['stream']):int(r['success']) for r in csv.DictReader(open(a.baseline))
              if r['arm']=='H0' and r['history']=='H_POLICY'}
    rows=[]; by={}; valid_pairs=0
    for case in prereg['case_order']:
        for stream in STREAMS:
            pair={}
            for hist in (1,3):
                folder=Path(a.raw)/'FORMAL'/stream/case/'H0'/f'HISTORY{hist}'
                row=dict(case_id=case,arm='H0',stream=stream,history_len=hist,status='TECHNICAL_MISSING',
                         success=None,entered_replanning=None,first_plan_equal=None,result_sha256='',trajectory_sha256='')
                if (folder/'COMPLETE.json').exists():
                    complete=json.loads((folder/'COMPLETE.json').read_text())
                    for name, record in complete['files'].items():
                        assert sha(folder/name)==record['sha256'] and (folder/name).stat().st_size==record['bytes']
                    result=json.loads((folder/'result.json').read_text())
                    assert result['identity_sha256']==complete['identity_sha256']
                    row.update(status='COMPLETE',success=result['success'],entered_replanning=result['entered_replanning'],
                               result_sha256=sha(folder/'result.json'),trajectory_sha256=sha(folder/'trajectory.npz'))
                    with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
                        pair[hist]=z['returned_plans_normalized'][0].copy() if len(z['returned_plans_normalized']) else None
                rows.append(row);by[case,stream,hist]=row
            if all(h in pair and pair[h] is not None for h in (1,3)):
                equal=pair[1].dtype==pair[3].dtype and pair[1].tobytes()==pair[3].tobytes()
                by[case,stream,3]['first_plan_equal']=equal
                if equal:valid_pairs+=1
                else:by[case,stream,3].update(status='INVALID_FIRST_PLAN',success=None)
    effects=[]; e2=[]
    for case in prereg['case_order']:
        if all(by[case,s,h]['status']=='COMPLETE' for s in STREAMS for h in (1,3)):
            effects.append(np.mean([by[case,s,3]['success']-by[case,s,1]['success'] for s in STREAMS]))
        if all(by[case,s,1]['status']=='COMPLETE' and (case,s) in baseline for s in STREAMS):
            e2.append(np.mean([by[case,s,1]['success']-baseline[case,s] for s in STREAMS]))
    main_rows=[bootstrap('E_A1',effects),bootstrap('E_A2',e2,True)]
    csv_out(out/'RAW_VALUES.csv',rows);csv_out(out/'MAIN_TABLE.csv',main_rows)
    delivered=sum(r['status']=='COMPLETE' for r in rows);primary=main_rows[0]
    predicted=primary['ci95_low'] is not None and primary['label']=='提高' and primary['ci95_low']<=.169 and primary['ci95_high']>=.106
    stats=dict(module='A',planned=600,delivered=delivered,delivery_rate=delivered/600,
               first_plan_pairs_equal=valid_pairs,planned_first_plan_pairs=300,endpoints=main_rows,
               prediction='提高；区间与+13.7[10.6,16.9]个百分点重叠',prediction_consistent=predicted,
               missing_policy='Complete paired cases only; no imputation; missing case count reported',training_updates=0)
    write_json(out/'STATS.json',stats)
    pct=lambda x:'不可估计' if x is None else f'{100*x:.2f}'
    lines=[f"模块 A：E_A1 = {pct(primary['point'])} 个百分点，95% CI [{pct(primary['ci95_low'])}, {pct(primary['ci95_high'])}]；{primary['label']}。",
           f"交付 {delivered}/600；首计划门 {valid_pairs}/300 对逐位相等；主终点完整 case {len(effects)}/100。",
           f"与预注册预测一致：{predicted}。E_A2 仅描述版本差异，不作推断性结论。",
           '影响论文关于“库自带 history_len=3 能否复现自制历史适配器增益”的表述。',
           '原生639版本单环境第一次只用一个真实帧且无历史动作，第二次为真实三帧与两个已执行动作块。']
    (out/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'SEAL.json',dict(module='A',prereg_sha256=sha(a.prereg),analysis_sha256=sha(__file__),
               files={x.name:dict(sha256=sha(x),bytes=x.stat().st_size) for x in sorted(out.iterdir()) if x.is_file() and x.name!='SEAL.json'}))
    print(json.dumps(stats,ensure_ascii=False))

if __name__=='__main__':main()
