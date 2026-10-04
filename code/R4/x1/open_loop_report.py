"""Deliver the completed X1 open-loop module before closed-loop jobs finish."""
import argparse
from . import core
from .report import OPEN_METRICS,average
from .evaluate import write_csv

def main(task):
    source=core.ROOT/'open_loop/per_case.json';raw=core.read(source);arms=['H0']+[f'REFIT_{s}' for s in core.SEEDS]
    roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json');cases={c['case_id'] for c in roles['cases']['EVAL']}
    if len(raw)!=1200 or {(r['case_id'],r['arm'],r['horizon_macro']) for r in raw}!={(c,a,h) for c in cases for a in arms for h in (1,2,5)}:
        raise RuntimeError('Incomplete open-loop module')
    out=core.ROOT/'reports'/f'{task}_OPEN_LOOP';rows=[]
    for arm in arms:
        for h in (1,2,5):
            part=[r for r in raw if r['arm']==arm and r['horizon_macro']==h]
            rows.append({'task':task,'arm':arm,'cases':100,'horizon_macro':h,'horizon_raw':5*h,
                **{m:average([r[m] for r in part]) for m in OPEN_METRICS},
                'nonfinite_prediction_cases':sum(not r['prediction_finite'] for r in part),'optimizer_updates_in_this_module':0})
    write_csv(out/'OPEN_LOOP_RAW_MEANS.csv',rows)
    status={'status':'COMPLETE','task':task,'cases':100,'arms':4,'horizons_macro':[1,2,5],'raw_rows':1200,'raw_source':core.file_record(source),
        'raw_csv':core.file_record(core.ROOT/'open_loop/per_case.csv'),'new_optimizer_updates':0,'new_CEM_trajectories':0,
        'summary':core.file_record(out/'OPEN_LOOP_RAW_MEANS.csv'),'nonfinite_policy':'Retain method Infinity without deleting cases'}
    core.atomic(out/'MODULE_STATUS.json',status)
    (out/'CONCLUSION_ZH.txt').write_text(f'{task} 开环完成：4 个固定模型 × 100 个 case × h=1/2/5。\n原坐标 MSE、TRAIN 方差标准化 MSE 与范数原值见表。\n三个 refit 均使用固定 30,000 更新权重；本模块新增优化器更新和 CEM 轨迹均为 0。\n所有模型预测先落盘，再读取未来 latent 目标评分；不使用 EVAL 拟合归一化。\n开环预测误差单独报告，控制效果仍由完整闭环评价判定。\n')
    print(core.canonical(status).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);a=p.parse_args();main(a.task)
