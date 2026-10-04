"""Validate recovered main-table identities, coverage and arithmetic.

No models, statistics, simulator calls or optimizers are executed here.
"""
import argparse,csv,json,math
from pathlib import Path
from deliver import record,atomic

ARMS=('H0','REFIT_103201','REFIT_103202','REFIT_103203')
MEAN='FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE'
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def same(a,b):
    a,b=float(a),float(b)
    return a==b or math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def check(condition,message):
    if not condition:raise ValueError(message)

def canonical(folder,tasks):
    folder=Path(folder);state=json.loads((folder/'MODULE_STATUS.json').read_text())
    limited=tasks==['cube'] and state['status']=='COMPLETE_WITH_TECHNICAL_LIMITATIONS' and bool(state.get('reason'))
    check(state['status']=='COMPLETE' or limited,'Canonical source status incomplete')
    files={name:record(folder/name) for name in ('MAIN_TABLE.csv','ALL_RAW_VALUES.csv','MODULE_STATUS.json')}
    if 'tables' in state:
        for name,expected in state['tables'].items():
            check(all(files[name][k]==expected[k] for k in ('bytes','sha256')),'Canonical table source seal mismatch')
    raw=read(folder/'ALL_RAW_VALUES.csv');main=read(folder/'MAIN_TABLE.csv')
    check(len(raw)==2000*len(tasks),'Canonical raw row count');check(len(main)==20*len(tasks),'Canonical main row count')
    keys=[tuple(r[k] for k in ('task','case_id','arm','evaluation_kind','horizon_macro')) for r in raw]
    check(len(keys)==len(set(keys)),'Duplicate canonical raw cell')
    index={k:r for k,r in zip(keys,raw)}
    for task in tasks:
        cases={r['case_id'] for r in raw if r['task']==task};check(len(cases)==100,'Expected fixed100 cases')
        for kind,h in [('CLOSED_LOOP_CEM',0)]+[('OPEN_LOOP',h) for h in (1,2,5)]:
            metric='success' if h==0 else 'latent_raw_MSE'
            for case in cases:
                entries=[index.get((task,case,arm,kind,str(h))) for arm in (*ARMS,MEAN)]
                check(all(r is not None for r in entries),'Missing canonical case/arm/horizon')
                if limited and kind=='CLOSED_LOOP_CEM':
                    check(state.get('closed_loop_status')=='TECHNICALLY_UNEVALUABLE','Cube module must explicitly identify technical closed-loop missingness')
                    check(all(r.get('observation_status') in ('MISSING','TECHNICALLY_UNEVALUABLE') and r.get('missing_reason')=='RESET_FALLBACK_EXACT_CHECK_FAILED' for r in entries),'Unobserved Cube closed-loop rows must explicitly state technical missingness')
                    check(all(not r.get(metric) for r in entries),'Technical missingness cannot be a numeric success outcome')
                    continue
                check(all(r['derived_within_case_refit_mean'].lower()=='false' for r in entries[:4]),'Observed values mislabeled')
                check(entries[4]['derived_within_case_refit_mean'].lower()=='true','Derived mean must be explicit')
                check(same(entries[4][metric],sum(float(r[metric]) for r in entries[1:4])/3),'Three fixed refits do not reproduce derived mean')
            for arm in (*ARMS,MEAN):
                summary=[r for r in main if r['task']==task and r['arm']==arm and r['evaluation_kind']==kind and int(r['horizon_macro'])==h]
                if limited and kind=='CLOSED_LOOP_CEM':
                    check(len(summary)==1 and summary[0].get('status')=='TECHNICALLY_UNEVALUABLE' and int(summary[0]['cases'])==0 and int(summary[0]['universe_cases'])==100,'Missing closed-loop denominator or technical status')
                    check(all(not summary[0].get(k) for k in ('success','success_percent','delta_success_pp','delta_case_ci95_low','delta_case_ci95_high')),'Unobserved closed loop must not contain invented estimates')
                    continue
                check(len(summary)==1 and int(summary[0]['cases'])==100,'Missing main row or denominator')
                check(same(summary[0][metric],sum(float(index[(task,c,arm,kind,str(h))][metric]) for c in cases)/100),'Main estimate does not match recovered raw values')
    return {'status':'COMPLETE_WITH_TECHNICAL_LIMITATIONS' if limited else 'COMPLETE','reason':state['reason'] if limited else None,'tasks':tasks,'raw_rows':len(raw),'main_rows':len(main),'explicitly_missing_closed_loop_rows':500 if limited else 0,'case_coverage_verified':True,'observed_and_derived_rows_distinguished':True,'within_case_and_main_arithmetic_verified':True,'inputs':list(files.values())}

def x2(root):
    root=Path(root);folder=root/'x2/reports';state=json.loads((folder/'X2_STATUS.json').read_text())
    check(state['status']=='COMPLETE' and state['completed']=={'baseline':324,'neural':648,'closed':324},'X2 stages incomplete')
    raw=read(folder/'X2_ALL_RAW_VALUES.csv');main=read(folder/'X2_MAIN_TABLE.csv')
    jobs={r['job_id'] for r in json.loads((root/'x2/inputs/X2_INPUT_MANIFEST.json').read_text())['rows']}
    check(len(jobs)==324,'Original X2 model universe')
    arms={'JOINT_LINUX','PRED_CONT','MLP_POST_H8','S1_SELECTED','LIN','BIL','BIL_CTX','BIL_FULL','BIL_STATIC','BIL_STATIC_FULL','GAIN_CTX','GAIN_CTX_FULL'}
    check(len(raw)==11664,'X2 raw rows');check(len(main)==3888,'X2 main rows')
    keys=[(r['job_id'],r['arm'],int(r['horizon'])) for r in raw]
    check(len(keys)==len(set(keys)) and set(keys)=={(j,a,h) for j in jobs for a in arms for h in (1,8,16)},'X2 full model/arm/horizon coverage')
    groups={}
    for r in raw:
        check(int(r['updates'])==(30000 if r['arm'] in ('PRED_CONT','MLP_POST_H8') else 0),'X2 fixed update budget')
        check(same(r['excess_risk'],float(r['risk'])-float(r['window_bayes_reference'])),'X2 Bayes subtraction')
        if r['arm']=='S1_SELECTED':check(r['selected_head'] in arms-{'JOINT_LINUX','PRED_CONT','MLP_POST_H8','S1_SELECTED'},'Unknown TRAIN-selected head')
        groups.setdefault(tuple(r[k] for k in ('world','method','arm','horizon')),[]).append(r)
    for r in main:
        members=groups[tuple(r[k] for k in ('world','method','arm','horizon'))]
        check(len(members)==3 and int(r['seed_count'])==3,'X2 fixed three seeds')
        for metric in ('risk','excess_risk'):check(same(r['mean_'+metric],sum(float(m[metric]) for m in members)/3),'X2 seed mean arithmetic')
    return {'status':'COMPLETE','models':324,'arms':12,'horizons':[1,8,16],'raw_rows':len(raw),'main_rows':len(main),'coverage_updates_and_arithmetic_verified':True,'inputs':[record(folder/name) for name in ('X2_STATUS.json','X2_ALL_RAW_VALUES.csv','X2_MAIN_TABLE.csv','X2_SCOPE_DIFFERENCES.csv')]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--module',choices=('r3','tworoom','cube','x2'),required=True);a=p.parse_args();root=Path(a.root)
    if a.module=='x2':result=x2(root)
    else:result=canonical(root/('tables/r3_reference' if a.module=='r3' else f'x1/reports/{a.module}_MAIN'),['pusht','reacher'] if a.module=='r3' else [a.module])
    atomic(root/f'manifests/main_validation/{a.module}.json',result);print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))

if __name__=='__main__':main()
