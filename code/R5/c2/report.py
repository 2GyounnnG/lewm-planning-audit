"""Same-model stream and same-stream model comparisons, shared case bootstrap."""
import numpy as np
from pathlib import Path
from analysis import statistics as st
from .common import *
MEAN='FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE'
ALL_STREAMS=('R3_ORIGINAL',)+STREAMS

def main():
    c=read(ROOT/'C2_CONTRACT.json');verify(c['statistics_source'])
    if sha(st.__file__)!=c['statistics_source']['sha256']:raise RuntimeError('Statistic source changed')
    cases=sorted(c['cases']['EVAL'],key=lambda c:c['case_id']);ids=[x['case_id'] for x in cases];arms=ARMS+(MEAN,);raw=[];sources=[]
    x=np.empty((100,3,5))
    for i,case in enumerate(cases):
        for j,stream in enumerate(ALL_STREAMS):
            base=Path('/workspace/r5/C1') if j==0 else ROOT
            for k,arm in enumerate(ARMS):
                folder=base/'closed_loop/EVAL'/stream/arm/case['case_id'];receipt=read(folder/'COMPLETE.json')
                for f in receipt['files'].values():verify(f)
                r=read(folder/'result.json');src=record(folder/'result.json');sources.append(src)
                if r['status']!='COMPLETE' or r['stream']!=stream or r['arm']!=arm:raise RuntimeError('Missing/mixed stream')
                x[i,j,k]=r['success']
                raw.append({'task':'cube','case_id':case['case_id'],'arm':arm,'stream':stream,'success':r['success'],'derived_within_case_refit_mean':False,'reused_C1':j==0,'reset_label':LABEL,'source_sha256':src['sha256']})
            x[i,j,4]=x[i,j,1:4].mean()
            raw.append({'task':'cube','case_id':case['case_id'],'arm':MEAN,'stream':stream,'success':x[i,j,4],'derived_within_case_refit_mean':True,'reused_C1':j==0,'reset_label':LABEL,'source_sha256':digest([r['source_sha256'] for r in raw[-3:]])})
    ix=st.case_indices('cube',ids);table=[]
    def add(kind,arm,stream,reference,values,base):
        d=100*(values-base);interval=st.interval(d[ix].mean(1));absolute=st.interval(100*values[ix].mean(1))
        table.append(dict(task='cube',comparison=kind,arm=arm,stream=stream,reference=reference,cases=100,success_percent=float(100*values.mean()),success_ci95_low=absolute['low'],success_ci95_high=absolute['high'],delta_success_pp=float(d.mean()),delta_ci95_low=interval['low'],delta_ci95_high=interval['high'],failure_to_success=int(((base==0)&(values==1)).sum()) if arm!=MEAN and kind!='THREE_STREAM_CASE_MEAN' else None,success_to_failure=int(((base==1)&(values==0)).sum()) if arm!=MEAN and kind!='THREE_STREAM_CASE_MEAN' else None,reset_label=LABEL,evidence_label=EVIDENCE,bootstrap_replicates=5000,independent_unit='CASE',source_sha256=digest(sources),statistics_source_sha256=sha(st.__file__),report_source_sha256=sha(__file__)))
    for j,stream in enumerate(ALL_STREAMS):
        for k,arm in enumerate(arms):add('SAME_STREAM_MODEL',arm,stream,'H0',x[:,j,k],x[:,j,0])
    for k,arm in enumerate(arms):
        for a,b in ((0,1),(0,2),(1,2)):add('SAME_MODEL_STREAM',arm,ALL_STREAMS[b],ALL_STREAMS[a],x[:,b,k],x[:,a,k])
        add('THREE_STREAM_CASE_MEAN',arm,'MEAN_OF_FIXED_THREE_STREAMS','H0',x[:,:,k].mean(1),x[:,:,0].mean(1))
    out=ROOT/'reports';csv_write(out/'CUBE_PLANNER_RANDOMNESS_TABLE.csv',table);csv_write(out/'ALL_RAW_VALUES.csv',raw)
    save_npz(out/'BOOTSTRAP_INDICES.npz',case_ids=np.asarray(ids),indices=ix)
    lines=['Cube C2：两个固定ALT流 × 100case × 4模型，800条新增闭环完成。','原流400条直接复用C1；总计1200条observed，派生均值不增加独立样本量。','同流换模型、同模型换流及三流case内均值的配对区间与翻转见原值表。','全部采用封存C0对称复位与官方CEM；新增训练为0。','5000次case bootstrap，条件95%区间未校正；不以随机流或后训练seed作为独立case。']
    (out/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    atomic(out/'MODULE_STATUS.json',{'status':'COMPLETE','module':'C2','new_trajectories':800,'reused_C1':400,'cases':100,'table_rows':len(table),'raw_rows':len(raw),'sources':sources,'contract':record(ROOT/'C2_CONTRACT.json'),'files':{p.name:record(p) for p in out.iterdir() if p.is_file() and p.name!='MODULE_STATUS.json'}})
if __name__=='__main__':main()
