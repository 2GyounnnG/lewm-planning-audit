"""Four-task H1a merge; original source cells preserved, paired case bootstrap."""
import argparse, csv, importlib.util, json
from pathlib import Path
import numpy as np
from run import ARMS,LABEL,atomic,record,read,sha,writecsv
STATS_SHA='e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b'
def table(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();spec=read(a.spec)
    if sha(spec['statistics'])!=STATS_SHA:raise RuntimeError('Inherited frozen statistics source changed')
    s=importlib.util.spec_from_file_location('r5_inherited_statistics',spec['statistics']);st=importlib.util.module_from_spec(s);s.loader.exec_module(st)
    merged=[];summary=[];receipts=[];trigger={}
    for task,source in spec['tasks'].items():
        raw=table(source['raw']);src=record(source['raw']);roles=read(source['roles']);cases=sorted(roles['cases']['EVAL'],key=lambda c:c['case_id']);ids=[c['case_id'] for c in cases]
        if src['sha256']!=source['expected_raw_sha256']:raise RuntimeError('Source differs from accepted raw values')
        by={}
        for r in raw:
            key=(r['case_id'],r['model'],r['history_kind'],int(r['horizon_raw']))
            if key in by:raise RuntimeError('Duplicate source cell')
            if r['source_policy']!='OFFLINE_RECORDED' or str(r['valid']).lower() not in ('true','1'):raise RuntimeError('H1a requires lawful common offline anchors')
            by[key]=float(r['latent_mse']);merged.append(dict(r,source_table_sha256=src['sha256'],evidence_label=LABEL,source_origin=source['origin']))
        if len(by)!=2400 or set(r['case_id'] for r in raw)!=set(ids):raise RuntimeError('Incomplete fixed100 all4 bothhistory coverage')
        ix=st.case_indices(task,ids);weights=st.family_weights(task,ids,[c['family_id'] for c in cases]) if task=='pusht' else None
        context=dict(task=task,cases=100,independent_unit='CASE',bootstrap_replicates=5000,multiplicity='UNADJUSTED_POINTWISE_CONDITIONAL',source_table_sha256=src['sha256'],evidence_label=LABEL)
        def add(metric,arm,history,h,estimate,bs,fbs=None):
            ci=st.interval(bs);fci=st.interval(fbs) if fbs is not None else {}
            summary.append(dict(context,metric=metric,arm=arm,history_kind=history,horizon_macro=h,estimate=float(estimate),conditional95_low=ci['low'],conditional95_high=ci['high'],family95_low=fci.get('low'),family95_high=fci.get('high')))
        for h in (1,2,5):
            x=np.array([[[by[c,arm,k,h*5] for k in ('H_POLICY','H_REAL3')] for arm in ARMS] for c in ids],dtype=np.float64)
            x=np.concatenate([x,x[:,1:].mean(1,keepdims=True)],axis=1);arms=ARMS+['FIXED3_REFIT_MEAN_NOT_ENSEMBLE']
            for j,arm in enumerate(arms):
                for k,kind in enumerate(('H_POLICY','H_REAL3')):
                    values=x[:,j,k];add('latent_mse',arm,kind,h,values.mean(),values[ix].mean(1),None if weights is None else weights@values/weights.sum(1))
                values=x[:,j,1]-x[:,j,0];add('latent_mse_REAL3_minus_POLICY',arm,'PAIRED_REAL3_MINUS_POLICY',h,values.mean(),values[ix].mean(1),None if weights is None else weights@values/weights.sum(1))
                if j:
                    for k,kind in enumerate(('H_POLICY','H_REAL3')):
                        base=x[:,0,k];v=x[:,j,k];base_bs=base[ix].mean(1);v_bs=v[ix].mean(1)
                        estimate=(base.mean()-v.mean())/base.mean();bs=(base_bs-v_bs)/base_bs
                        fbs=None if weights is None else (weights@(base-v))/(weights@base)
                        add('relative_refit_improvement_ratio_of_means',arm,kind,h,estimate,bs,fbs)
            if task=='tworoom' and h==5:
                policy=float(x[:,0,0].mean());real=float(x[:,0,1].mean());trigger=dict(task=task,arm='H0',horizon_macro=5,H_POLICY=policy,H_REAL3=real,ratio_REAL3_to_POLICY=real/policy,predeclared_threshold=0.5,condition_met=real<=0.5*policy,execution_status='NOT_STARTED_INITIAL_STAGE_ONLY',source_table_sha256=src['sha256'])
        receipts.append(dict(task=task,raw=src,roles=record(source['roles']),rows=2400))
    writecsv(a.out/'HISTORY_CONDITION_4TASK_RAW.csv',merged);writecsv(a.out/'HISTORY_CONDITION_4TASK_TABLE.csv',summary);atomic(a.out/'H2_TWOROOM_CONDITION.json',trigger)
    atomic(a.out/'COMPLETE.json',dict(status='COMPLETE',module='H1a',evidence_label=LABEL,inputs=receipts,rows=len(merged),summary_rows=len(summary),world_model_new_updates=0,statistical_source=record(spec['statistics']),spec=record(a.spec),code=record(__file__),outputs=[record(a.out/n) for n in ('HISTORY_CONDITION_4TASK_RAW.csv','HISTORY_CONDITION_4TASK_TABLE.csv','H2_TWOROOM_CONDITION.json')]))
    print(json.dumps({'rows':len(merged),'summary_rows':len(summary),'H2_tworoom_condition':trigger}))
if __name__=='__main__':main()
