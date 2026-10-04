"""C1 closed-loop X1 schema, frozen final R4-v2.3 paired case statistics."""
from pathlib import Path
import numpy as np
from .common import *
from analysis import statistics as st
MEAN='FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE'
METRICS=('success','final_goal_error','executed_raw_steps','replan_calls','trajectory_wall_seconds')

def main():
    c=read(ROOT/'C1_CONTRACT.json');verify(c['statistics_source'])
    if sha(st.__file__)!=c['statistics_source']['sha256']:raise RuntimeError('Statistics implementation changed')
    cases=sorted(c['cases']['EVAL'],key=lambda v:v['case_id']);raw=[];sources=[]
    for case in cases:
        for arm in ARMS:
            folder=ROOT/'closed_loop/EVAL/R3_ORIGINAL'/arm/case['case_id'];receipt=read(folder/'COMPLETE.json')
            for r in receipt['files'].values():verify(r)
            v=read(folder/'result.json');src=record(folder/'result.json');sources.append(src)
            if v['status']!='COMPLETE' or v['case_id']!=case['case_id'] or v['arm']!=arm:raise RuntimeError('Incomplete matrix')
            raw.append(dict(task='cube',evaluation_kind='CLOSED_LOOP_CEM',case_id=case['case_id'],family_id=case['family_id'],arm=arm,derived_within_case_refit_mean=False,horizon_macro=0,horizon_raw=0,reset_label=LABEL,evidence_label=EVIDENCE,stream='R3_ORIGINAL',source_sha256=src['sha256'],**{k:v[k] for k in METRICS}))
        part=raw[-3:]
        raw.append(dict(task='cube',evaluation_kind='CLOSED_LOOP_CEM',case_id=case['case_id'],family_id=case['family_id'],arm=MEAN,derived_within_case_refit_mean=True,horizon_macro=0,horizon_raw=0,reset_label=LABEL,evidence_label=EVIDENCE,stream='R3_ORIGINAL',source_sha256=digest([r['source_sha256'] for r in part]),**{k:float(np.mean([r[k] for r in part])) if all(r[k] is not None for r in part) else None for k in METRICS}))
    ids=[r['case_id'] for r in cases];arms=ARMS+(MEAN,)
    matrix=np.array([[next(r['success'] for r in raw if r['case_id']==cid and r['arm']==arm) for arm in arms] for cid in ids])*100
    ix=st.case_indices('cube',ids);paired=st.paired_summary('cube',ids,matrix);main=[];flips=[]
    for i,arm in enumerate(arms):
        part=[r for r in raw if r['arm']==arm];ci=st.interval(matrix[ix,i].mean(1));estimate=paired[i]
        row=dict(task='cube',evaluation_kind='CLOSED_LOOP_CEM',arm=arm,horizon_macro=0,horizon_raw=0,cases=100,training_updates_per_seed=0 if arm=='H0' else 30000,new_training_updates=0,refit_seed_count=0 if arm=='H0' else 3 if arm==MEAN else 1,**{k:float(np.mean([r[k] for r in part])) if all(r[k] is not None for r in part) else None for k in METRICS},success_percent=float(matrix[:,i].mean()),success_case_ci95_low=ci['low'],success_case_ci95_high=ci['high'],delta_success_pp=estimate['difference'] if i else None,delta_case_ci95_low=estimate['conditional95']['low'] if i else None,delta_case_ci95_high=estimate['conditional95']['high'] if i else None,reset_label=LABEL,evidence_label=EVIDENCE,stream='R3_ORIGINAL',bootstrap_replicates=5000,interval='UNADJUSTED_POINTWISE_CONDITIONAL95',source_sha256=digest([r['source_sha256'] for r in part]),report_source_sha256=sha(__file__),statistics_source_sha256=sha(st.__file__))
        main.append(row)
        if i in (1,2,3):
            h0=matrix[:,0]/100;y=matrix[:,i]/100
            flips.append(dict(task='cube',arm=arm,cases=100,s01_H0_failure_REFIT_success=int(((h0==0)&(y==1)).sum()),s10_H0_success_REFIT_failure=int(((h0==1)&(y==0)).sum()),s11_both_success=int(((h0==1)&(y==1)).sum()),s00_both_failure=int(((h0==0)&(y==0)).sum()),reset_label=LABEL,source_sha256=row['source_sha256']))
    out=ROOT/'reports';csv_write(out/'MAIN_TABLE.csv',main);csv_write(out/'ALL_RAW_VALUES.csv',raw);csv_write(out/'PAIRED_FLIPS.csv',flips)
    save_npz(out/'BOOTSTRAP_INDICES.npz',case_ids=np.asarray(ids),indices=ix)
    atomic(out/'NUMERIC_RESULTS.json',{'paired':paired,'independent_unit':'CASE','cases':100,'observed_trajectories':400,'derived_case_rows':100,'family_status':'UNKNOWN_NOT_FABRICATED','bootstrap':record(out/'BOOTSTRAP_INDICES.npz'),'sources':sources})
    lines=[f'Cube：100个固定case × 4个固定模型共400条闭环完成，复位标签为 {LABEL}。',f'H0成功率 {main[0]["success_percent"]:.1f}%；三个固定30k后训练模型的case内均值 {main[-1]["success_percent"]:.1f}%。',f'均值−H0：{main[-1]["delta_success_pp"]:.2f}个百分点，5000次配对case条件95%区间 [{main[-1]["delta_case_ci95_low"]:.2f}, {main[-1]["delta_case_ci95_high"]:.2f}]。','三seed是case内重复测量，不是ensemble；全部C1新增训练更新为0。','本结果为已知EVAL案例上的R4后补充，区间未作多重比较校正。']
    (out/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    atomic(out/'MODULE_STATUS.json',{'status':'COMPLETE','module':'C1','observed_trajectories':400,'derived_rows':100,'raw_rows':500,'reset_label':LABEL,'contract':record(ROOT/'C1_CONTRACT.json'),'sources':sources,'files':{p.name:record(p) for p in out.iterdir() if p.is_file() and p.name!='MODULE_STATUS.json'}})
    print({'status':'COMPLETE','successes_percent':[r['success_percent'] for r in main]},flush=True)
if __name__=='__main__':main()
