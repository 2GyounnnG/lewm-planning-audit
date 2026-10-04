"""Cube main interface retaining reset-blocked closed-loop values as explicit NA."""
import argparse
from . import core
from .report import OPEN_METRICS,CLOSED_METRICS,MEAN_ARM,average
from .evaluate import write_csv

REASON='RESET_FALLBACK_EXACT_CHECK_FAILED'

def main(task):
    if task!='cube':raise RuntimeError('Only the independently reviewed Cube technical limitation is allowed')
    gatepath=core.ROOT/'reset_audit/RESET_FALLBACK.json';gate=core.read(gatepath)
    if gate['status']!='BLOCKED_RESET_FALLBACK' or gate['raw_calls']!=16:raise RuntimeError('Unchanged failed 16-trial gate required')
    if list((core.ROOT/'closed_loop/EVAL').rglob('COMPLETE.json')):raise RuntimeError('Unexpected scored Cube closed-loop observation')
    roles=core.read(core.ROOT/'manifests/cube_data_roles.json');cases=roles['cases']['EVAL'];arms=['H0']+[f'REFIT_{s}' for s in core.SEEDS]
    opened=core.read(core.ROOT/'open_loop/per_case.json')
    if len(cases)!=100 or len(opened)!=1200 or {(r['case_id'],r['arm'],r['horizon_macro']) for r in opened}!={(c['case_id'],a,h) for c in cases for a in arms for h in (1,2,5)}:
        raise RuntimeError('Complete fixed open-loop matrix required')
    raw=[]
    for r in opened:
        raw.append(dict(task=task,evaluation_kind='OPEN_LOOP',case_id=r['case_id'],family_id=r['family_id'],arm=r['arm'],
            derived_within_case_refit_mean=False,horizon_macro=r['horizon_macro'],horizon_raw=r['horizon_raw'],observation_status='OBSERVED',missing_reason=None,
            **{k:r[k] for k in OPEN_METRICS}))
    for c in cases:
        for h in (1,2,5):
            part=[r for r in opened if r['case_id']==c['case_id'] and r['arm'] in arms[1:] and r['horizon_macro']==h]
            raw.append(dict(task=task,evaluation_kind='OPEN_LOOP',case_id=c['case_id'],family_id=c['family_id'],arm=MEAN_ARM,
                derived_within_case_refit_mean=True,horizon_macro=h,horizon_raw=5*h,observation_status='DERIVED_WITHIN_CASE_REFIT_MEAN',missing_reason=None,
                **{k:average([r[k] for r in part]) for k in OPEN_METRICS}))
        for arm in arms+[MEAN_ARM]:
            raw.append(dict(task=task,evaluation_kind='CLOSED_LOOP_CEM',case_id=c['case_id'],family_id=c['family_id'],arm=arm,
                derived_within_case_refit_mean=arm==MEAN_ARM,horizon_macro=0,horizon_raw=0,observation_status='MISSING',missing_reason=REASON))
    fields=('task','evaluation_kind','case_id','family_id','arm','derived_within_case_refit_mean','horizon_macro','horizon_raw','observation_status','missing_reason',*CLOSED_METRICS,*OPEN_METRICS)
    out=core.ROOT/'reports'/task;write_csv(out/'ALL_RAW_VALUES.csv',[{k:r.get(k) for k in fields} for r in raw]);mainrows=[]
    for arm in arms+[MEAN_ARM]:
        common={'task':task,'arm':arm,'training_updates_per_seed':0 if arm=='H0' else 30000,'refit_seed_count':3 if arm==MEAN_ARM else 0 if arm=='H0' else 1,'universe_cases':100}
        mainrows.append({**common,'evaluation_kind':'CLOSED_LOOP_CEM','horizon_macro':0,'horizon_raw':0,'cases':0,'status':'TECHNICALLY_UNEVALUABLE','reason':REASON,
            'success':None,'success_percent':None,'delta_success_pp':None,'delta_case_ci95_low':None,'delta_case_ci95_high':None,'delta_case_ci97_5_low':None,'delta_case_ci97_5_high':None})
        for h in (1,2,5):
            part=[r for r in raw if r['evaluation_kind']=='OPEN_LOOP' and r['arm']==arm and r['horizon_macro']==h]
            mainrows.append({**common,'evaluation_kind':'OPEN_LOOP','horizon_macro':h,'horizon_raw':5*h,'cases':100,'status':'COMPLETE','reason':None,
                **{k:average([r[k] for r in part]) for k in OPEN_METRICS}})
    columns=list(dict.fromkeys(k for r in mainrows for k in r));write_csv(out/'MAIN_TABLE.csv',[{k:r.get(k) for k in columns} for r in mainrows])
    checkpoints={}
    for seed in core.SEEDS:
        folder=core.ROOT/'train'/str(seed);r=core.read(folder/'result.json');pointer=core.read(folder/'last.json');core.verify(pointer)
        if r['actual_updates']!=30000 or r['technical'] or r['frozen_before']!=r['frozen_after']:raise RuntimeError('Fixed training result invalid')
        checkpoints[str(seed)]={'updates':30000,'delta':pointer,'result':core.file_record(folder/'result.json')}
    artifacts={str(p.relative_to(core.ROOT)):core.file_record(p) for p in [core.ROOT/'manifests'/f'cube_{suffix}.json' for suffix in ('model_assets','source_receipt','data_roles','normalization','cache')]}
    status={'status':'COMPLETE_WITH_TECHNICAL_LIMITATIONS','task':task,'completed_modules':['MODEL_CHECK','CACHE','TRAIN_TECH','FORMAL_TRAIN','OPEN_LOOP'],
        'closed_loop_status':'TECHNICALLY_UNEVALUABLE','reason':REASON,'cases':0,'universe_cases':100,'formal_closed_loop_trajectories':0,'planned_closed_loop_trajectories':400,
        'closed_loop_missing_rows_including_mean':500,'observed_open_loop_rows':1200,'derived_open_loop_rows':300,'combined_raw_rows_including_derived_and_missing':2000,
        'arms':arms,'derived_arm':MEAN_ARM,'technical_optimizer_updates':128,'formal_optimizer_updates_per_seed':30000,'formal_optimizer_updates_total':90000,
        'new_encoder_updates':0,'missing_success_values_imputed':False,'closed_loop_bootstrap_performed':False,'family_sensitivity_status':'UNAVAILABLE_FAMILY_UNKNOWN',
        'reset_gate_evidence':core.file_record(gatepath),'artifacts':artifacts,'checkpoints':checkpoints,'report_source':core.file_record(__file__),
        'tables':{n:core.file_record(out/n) for n in ('MAIN_TABLE.csv','ALL_RAW_VALUES.csv')},'open_loop_raw':core.file_record(core.ROOT/'open_loop/per_case.json')}
    core.atomic(out/'MODULE_STATUS.json',status)
    (out/'CONCLUSION_ZH.txt').write_text('Cube 完成三个固定30k后训练与100个固定case的h1/2/5开环评价。\n正式闭环未执行：原exact复位门槛失败，成功率和区间保留NA。\n原source覆盖后输入qpos/qvel和像素一致，但ctrl/warmstart及一步物理状态存在非零差异。\n所有500条闭环占位行均显式标注缺失，未计作失败或成功。\n未追加复位trial、未放宽容差，技术证据SHA与全部离线原值随表保存。\n')
    print(core.canonical(status).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['cube']);a=p.parse_args();main(a.task)
