"""X1 adapter for frozen case-level R4 summaries; no physical margin assumptions."""
import argparse
from collections import defaultdict
import numpy as np
from . import core,data
from .evaluate import write_csv

def frozen():
    from analysis import aggregate,statistics
    roles=core.read(core.ROOT/'manifests/tworoom_data_roles.json');cases=sorted(c['case_id'] for c in roles['cases']['EVAL'])
    if len(cases)!=100 or len(set(cases))!=100:raise RuntimeError('Fixed100 case universe required')
    out=core.ROOT/'reports/tworoom_RANDOMNESS';out.mkdir(parents=True,exist_ok=True)
    ix=statistics.case_indices('tworoom',cases)
    contract={'task':'tworoom','case_ids':cases,'roles_sha256':core.sha(core.ROOT/'manifests/tworoom_data_roles.json'),
        'arms':list(aggregate.ARMS),'streams':list(aggregate.STREAMS),'replicates':5000,'case_seed':statistics.seed('tworoom','case'),
        'sources':{str(p):core.sha(p) for p in (core.Path(__file__),core.Path(aggregate.__file__),core.Path(statistics.__file__))},
        'same_model_stream_comparison':'case x3 streams, reference R3_ORIGINAL','same_stream_model_comparison':'case x4 arms plus within-case equal fixed3 refit mean, reference H0',
        'family':None,'independent_unit':'CASE','multiplicity':'UNADJUSTED_POINTWISE_CONDITIONAL','no_new_case_selection':True,'new_optimizer_updates':0,
        'evidence_label':'POST_R3_MAIN_RESULTS_KNOWN_BEFORE_ALTERNATE_STREAMS_SUPPLEMENT'}
    core.freeze(out/'RANDOMNESS_STATISTICS_FREEZE.json',contract)
    path=out/'CASE_BOOTSTRAP_INDICES.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if not np.array_equal(f['indices'],ix) or f['case_ids'].tolist()!=cases:raise RuntimeError('Shared bootstrap indices changed')
    else:data.save_npz(path,indices=ix,case_ids=np.asarray(cases))
    return cases,out

def report():
    from analysis import aggregate
    cases,out=frozen();family={c:None for c in cases};rows=[]
    for c in cases:
        for arm in aggregate.ARMS:
            for stream in aggregate.STREAMS:
                folder=core.ROOT/'closed_loop/EVAL'/stream/arm/c;receipt=core.read(folder/'COMPLETE.json')
                for rec in receipt['files'].values():core.verify(rec)
                value=core.read(folder/'result.json')
                if value['status']!='COMPLETE' or value['case_id']!=c or value['arm']!=arm or value['stream']!=stream:raise RuntimeError('Incomplete repeated measurement')
                rows.append({k:value[k] for k in ('task','case_id','family_id','arm','stream','success','final_goal_error','executed_raw_steps','replan_calls','trajectory_wall_seconds')})
    randomness=[];table=[]
    for arm in aggregate.ARMS:
        values=defaultdict(list)
        for r in rows:
            if r['arm']==arm:values[r['case_id'],r['stream']].append(float(r['success']))
        randomness.extend(aggregate.summarize('tworoom',cases,family,values,aggregate.STREAMS,'success_fraction',{'policy':arm,'scope':'SAME_MODEL_DIFFERENT_PLANNER_STREAM'}))
    for stream in aggregate.STREAMS:
        values=defaultdict(list)
        for r in rows:
            if r['stream']==stream:values[r['case_id'],r['arm']].append(float(r['success']))
        table.extend(aggregate.summarize('tworoom',cases,family,values,aggregate.ARMS,'success_fraction',{'scope':'LEARNING_POLICIES_ALL100','stream':stream}))
    write_csv(out/'PLANNING_RANDOMNESS_TABLE.csv',randomness);write_csv(out/'S3_LEARNING_MAIN_TABLE.csv',table);write_csv(out/'ALL_RAW_VALUES.csv',rows)
    core.atomic(out/'MODULE_STATUS.json',{'task':'tworoom','status':'COMPLETE','cases':100,'trajectories':1200,'original_reused':400,'additional':800,'new_optimizer_updates':0,
        'source_freeze':core.file_record(out/'RANDOMNESS_STATISTICS_FREEZE.json'),'bootstrap_indices':core.file_record(out/'CASE_BOOTSTRAP_INDICES.npz'),
        'tables':{n:core.file_record(out/n) for n in ('PLANNING_RANDOMNESS_TABLE.csv','S3_LEARNING_MAIN_TABLE.csv','ALL_RAW_VALUES.csv')}})
    (out/'CONCLUSION_ZH.txt').write_text('TwoRoom 固定100个case的4模型×3规划随机流完整配对，原400条复用、新增800条。\n逐流四模型及固定三refit均值见主表；同模型跨流变化单列随机性表。\n全部比较共享5000组case重采样索引，family未知，未增加独立case数。\n区间是固定模型条件下的点对点95%，未经多重性校正。\n这些有限重复不估计全部预训练随机性，也不单独证明噪声主导。\n')
    print('COMPLETE:1200 paired trajectories, 12 randomness rows, 15 model/stream rows')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','report']);a=p.parse_args();frozen() if a.mode=='freeze' else report()
