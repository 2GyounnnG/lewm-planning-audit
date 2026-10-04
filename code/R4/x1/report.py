"""Complete paired raw tables and unchanged R3 case-bootstrap statistics."""
import math
import numpy as np
from . import core
from .evaluate import write_csv

MEAN_ARM='FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE'
OPEN_METRICS=('latent_raw_MSE','latent_TRAIN_variance_normalized_MSE','prediction_norm_l2','target_norm_l2',
    'prediction_norm2_over_D','target_norm2_over_D','prediction_displacement_from_initial_MSE')
CLOSED_METRICS=('success','final_goal_error','executed_raw_steps','replan_calls','trajectory_wall_seconds')

def numeric(value):
    return float(value) if value is not None else None

def average(values):
    values=[numeric(v) for v in values]
    # Missing values stay missing; method infinity is retained, never dropped.
    if any(v is None for v in values):return None
    value=float(np.mean(values))
    return value if math.isfinite(value) else 'Infinity' if value>0 else '-Infinity' if value<0 else 'NaN'

def combined(task,cases,closed,statistics):
    """Public merge interface: one observed or explicitly derived case/arm/horizon row."""
    folder=core.ROOT/'reports'/task;folder.mkdir(parents=True,exist_ok=True)
    opened=core.read(core.ROOT/'open_loop/per_case.json');arms=['H0']+[f'REFIT_{s}' for s in core.SEEDS]
    expected={(c['case_id'],a,h) for c in cases for a in arms for h in (1,2,5)}
    actual=[(r['case_id'],r['arm'],r['horizon_macro']) for r in opened]
    if len(actual)!=1200 or set(actual)!=expected:raise RuntimeError('Open-loop coverage must be 100 cases x4 arms x3 horizons')
    fields=('task','evaluation_kind','case_id','family_id','arm','derived_within_case_refit_mean','horizon_macro','horizon_raw',
        *CLOSED_METRICS,*OPEN_METRICS)
    raw=[]
    for r in closed:
        raw.append(dict(task=task,evaluation_kind='CLOSED_LOOP_CEM',case_id=r['case_id'],family_id=r['family_id'],arm=r['arm'],
            derived_within_case_refit_mean=False,horizon_macro=0,horizon_raw=0,**{k:r[k] for k in CLOSED_METRICS}))
    for r in opened:
        raw.append(dict(task=task,evaluation_kind='OPEN_LOOP',case_id=r['case_id'],family_id=r['family_id'],arm=r['arm'],
            derived_within_case_refit_mean=False,horizon_macro=r['horizon_macro'],horizon_raw=r['horizon_raw'],**{k:r[k] for k in OPEN_METRICS}))
    for c in cases:
        for kind,h,metrics in [('CLOSED_LOOP_CEM',0,CLOSED_METRICS)]+[('OPEN_LOOP',h,OPEN_METRICS) for h in (1,2,5)]:
            part=[r for r in raw if r['case_id']==c['case_id'] and r['evaluation_kind']==kind and r['horizon_macro']==h and r['arm'] in arms[1:]]
            if len(part)!=3:raise RuntimeError('Three fixed refits required within every case')
            raw.append(dict(task=task,evaluation_kind=kind,case_id=c['case_id'],family_id=c['family_id'],arm=MEAN_ARM,
                derived_within_case_refit_mean=True,horizon_macro=h,horizon_raw=5*h,**{k:average([r[k] for r in part]) for k in metrics}))
    write_csv(folder/'ALL_RAW_VALUES.csv',[{k:r.get(k) for k in fields} for r in raw])
    main=[];case_metrics=statistics['schemes']['CASE']['metrics']
    for arm in arms+[MEAN_ARM]:
        for kind,h,metrics in [('CLOSED_LOOP_CEM',0,CLOSED_METRICS)]+[('OPEN_LOOP',h,OPEN_METRICS) for h in (1,2,5)]:
            part=[r for r in raw if r['arm']==arm and r['evaluation_kind']==kind and r['horizon_macro']==h]
            row=dict(task=task,evaluation_kind=kind,arm=arm,horizon_macro=h,horizon_raw=5*h,cases=len(part),
                training_updates_per_seed=0 if arm=='H0' else 30000,refit_seed_count=3 if arm==MEAN_ARM else 0 if arm=='H0' else 1,
                **{k:average([r[k] for r in part]) for k in metrics})
            if kind=='CLOSED_LOOP_CEM':
                row['success_percent']=100*row['success'];key=('FIXED_THREE_REFIT_MEAN_minus_H0_pp' if arm==MEAN_ARM else arm+'_minus_H0_pp')
                if arm!='H0':
                    estimate=case_metrics[key];row.update(delta_success_pp=estimate['estimate'],delta_case_ci95_low=estimate['ci95'][0],
                        delta_case_ci95_high=estimate['ci95'][1],delta_case_ci97_5_low=estimate['ci97_5_two_task_bonferroni_approximation'][0],
                        delta_case_ci97_5_high=estimate['ci97_5_two_task_bonferroni_approximation'][1])
            main.append(row)
    columns=list(dict.fromkeys(k for r in main for k in r));write_csv(folder/'MAIN_TABLE.csv',[{k:r.get(k) for k in columns} for r in main])
    artifacts={}
    for path in [core.ROOT/'manifests'/f'{task}_{suffix}.json' for suffix in ('model_assets','source_receipt','data_roles','normalization','cache')]:
        artifacts[str(path.relative_to(core.ROOT))]=core.file_record(path)
    checkpoints={}
    for seed in core.SEEDS:
        result=core.read(core.ROOT/'train'/str(seed)/'result.json')
        if result['actual_updates']!=30000 or result['frozen_before']!=result['frozen_after']:raise RuntimeError('Final training ledger mismatch')
        pointer=core.read(core.ROOT/'train'/str(seed)/'last.json');core.verify(pointer)
        checkpoints[str(seed)]={'updates':30000,'delta':pointer,'result':core.file_record(core.ROOT/'train'/str(seed)/'result.json')}
    status={'status':'COMPLETE','task':task,'cases':100,'formal_closed_loop_trajectories':400,'arms':arms,'derived_arm':MEAN_ARM,
        'open_loop_rows':1200,'combined_raw_rows_including_derived':len(raw),'closed_loop_stream':'R3_ORIGINAL','case_weighting':'EQUAL_100_CASES',
        'family_sensitivity_status':statistics['schemes']['FAMILY']['status'],'bootstrap_replicates':5000,
        'technical_optimizer_updates':128,'formal_optimizer_updates_per_seed':30000,'formal_optimizer_updates_total':90000,
        'new_encoder_updates':0,'artifacts':artifacts,'checkpoints':checkpoints,'report_source':core.file_record(__file__),
        'tables':{name:core.file_record(folder/name) for name in ('MAIN_TABLE.csv','ALL_RAW_VALUES.csv')},
        'open_loop_raw':core.file_record(core.ROOT/'open_loop/per_case.json'),'statistics':core.file_record(core.ROOT/'reports/statistics/NUMERIC_RESULTS.json')}
    core.atomic(folder/'MODULE_STATUS.json',status)
    return status

def summarize(task):
    roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json');cases=roles['cases']['EVAL'];rows=[]
    for c in cases:
        for arm in ['H0']+[f'REFIT_{s}' for s in core.SEEDS]:
            folder=core.ROOT/'closed_loop/EVAL/R3_ORIGINAL'/arm/c['case_id'];receipt=core.read(folder/'COMPLETE.json')
            for record in receipt['files'].values():core.verify(record)
            row=core.read(folder/'result.json')
            if row['status']!='COMPLETE' or row['case_id']!=c['case_id'] or row['family_id']!=c['family_id'] or row['arm']!=arm:
                raise RuntimeError('Incomplete/mismatched paired case')
            row=dict(row,phase='FORMAL',source_phase='EVAL',episode_id=c['episode_id'],evaluation_complete=True,
                failure_category='METHOD' if row['failure'] else None)
            rows.append(row)
    if len(cases)!=100 or len(rows)!=400:raise RuntimeError('X1 main matrix must be exactly 100 cases x4 arms')
    folder=core.ROOT/'reports';folder.mkdir(parents=True,exist_ok=True);core.atomic(folder/'closed_loop_raw.json',rows)
    flat=[{k:r[k] for k in ('task','case_id','family_id','arm','success','final_goal_error','executed_raw_steps','replan_calls','trajectory_wall_seconds')} for r in rows]
    write_csv(folder/'closed_loop_raw.csv',flat)
    s=core.r3('statistics');previous=s.TASKS
    try:
        s.TASKS=core.TASKS;result=s.analyze_task(task,rows,folder/'statistics')
    finally:s.TASKS=previous
    table=[]
    for arm in ['H0']+[f'REFIT_{seed}' for seed in core.SEEDS]:
        part=[r for r in rows if r['arm']==arm];table.append({'task':task,'arm':arm,'cases':len(part),'successes':sum(r['success'] for r in part),
            'success_percent':100*sum(r['success'] for r in part)/len(part)})
    write_csv(folder/'main_raw_table.csv',table)
    merged=combined(task,cases,rows,result)
    lines=[f'{task}：100个固定case×4个arm已完成，原值表保留每case成功与动作日志。',
        f'H0成功率{table[0]["success_percent"]:.1f}%；三个固定30k refit均值{result["mean_fixed_refit_success_percent_NOT_ENSEMBLE"]:.1f}%。',
        f'固定refit均值−H0为{result["main_effect_pp"]:.1f}个百分点；5000组配对case重采样区间见statistics。',
        '三个后训练seed不视为独立预训练重复，也不是ensemble policy。',
        '官方预训练逐episode暴露未知；family缺失不以episode ID冒充独立家族。']
    (folder/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    (folder/task/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    return {'status':'COMPLETE','task':task,'cases':100,'arms':4,'main_raw_table':str(folder/task/'MAIN_TABLE.csv'),'main_effect_pp':result['main_effect_pp']}
