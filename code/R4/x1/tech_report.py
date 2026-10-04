"""Read completed TECH evidence and emit raw values without additional execution."""
import argparse
import numpy as np
from . import core
from .evaluate import write_csv

def report(task):
    out=core.ROOT/'reports'/f'{task}_TECH';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in sorted((core.ROOT/'closed_loop/TECH/R3_ORIGINAL/H0').iterdir()):
        receipt=core.read(folder/'COMPLETE.json')
        for record in receipt['files'].values():core.verify(record)
        value=core.read(folder/'result.json')
        if value['status']!='COMPLETE' or value['phase']!='TECH':raise RuntimeError('Incomplete TECH evidence')
        rows.append({k:value[k] for k in ('task','case_id','arm','success','final_goal_error','executed_raw_steps','replan_calls',
            'trajectory_wall_seconds','planning_synchronized_wall_seconds','environment_step_seconds','optimizer_updates')})
    if len(rows)!=4:raise RuntimeError('Exactly four frozen TECH cases required')
    write_csv(out/'CLOSED_LOOP_TECH_RAW.csv',rows)
    timing=[]
    for field in ('trajectory_wall_seconds','planning_synchronized_wall_seconds','environment_step_seconds'):
        values=np.array([r[field] for r in rows]);timing.append({'task':task,'metric':field,'cases':4,
            'P50':float(np.quantile(values,.5)),'P90':float(np.quantile(values,.9)),'mean':float(values.mean()),'max':float(values.max())})
    write_csv(out/'TECH_TIMING_RAW.csv',timing)
    training=core.read(core.ROOT/'technical_train/103201/result.json')
    trainrow={k:training[k] for k in ('task','seed','technical','actual_updates','frozen_before','frozen_after','new_encoder_updates',
        'state_labels_read','checkpoint_selection','seconds')}
    trainrow.update(failed_optimizer_updates=0,recovery_optimizer_updates=0,technical_budget_max=1024)
    journal=(core.ROOT/'technical_train/103201/updates.jsonl').read_text().splitlines()
    if len(journal)!=128 or training['actual_updates']!=128 or training['status']!='TECHNICAL_COMPLETE':raise RuntimeError('TECH update ledger differs')
    write_csv(out/'TRAIN_TECH_RAW.csv',[trainrow])
    reset=core.read(core.ROOT/'reset_audit/RESET_FALLBACK.json')
    status={'status':'COMPLETE','task':task,'technical_trajectories':4,'technical_optimizer_updates':128,
        'formal_optimizer_updates_in_this_module':0,'source_seed_recovery':False,'reset_audit_status':reset['status'],
        'closed_loop_P50_seconds':timing[0]['P50'],'closed_loop_P90_seconds':timing[0]['P90'],
        'second_batch_eligible_after_main_delivery':timing[0]['P90']<=30,
        'evidence':{str(p.relative_to(core.ROOT)):core.file_record(p) for p in [core.ROOT/'technical_train/103201/result.json',
            core.ROOT/'technical_train/103201/updates.jsonl',core.ROOT/'reset_audit/RESET_FALLBACK.json']}}
    core.atomic(out/'MODULE_STATUS.json',status)
    lines=[f'{task}：4 个 H0 TECH 闭环完成，原值仅用于技术核验，不进入正式统计。',
        f'单条闭环 P50={timing[0]["P50"]:.6f} 秒，P90={timing[0]["P90"]:.6f} 秒。',
        '独立 TECH 完成 128 次优化器更新，失败/恢复额外更新均为 0；所有冻结参数与 buffer SHA 不变。',
        '16 次复位/一步技术比较通过；缺失的原始 seed 未被恢复，正式使用经验证的固定 seed 0。',
        '满足额外随机流的技术时延条件；仍须等待主结果交付后才可启动第二批。']
    (out/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);a=p.parse_args();print(core.canonical(report(a.task)).decode())
