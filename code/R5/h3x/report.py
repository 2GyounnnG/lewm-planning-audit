"""Case-paired H3X exploratory statistics; final checkpoints; no predefined H3 label."""
import argparse,csv,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
from . import common
from .evaluate import file
SEEDS=(103201,103202,103203)
BASE=['H0']+[f'REFIT_{s}' for s in SEEDS]
NEW=[f'H3X_{s}' for s in SEEDS]
ARMS=BASE+NEW+['FIXED3_R3_MEAN_NOT_ENSEMBLE','FIXED3_H3X_MEAN_NOT_ENSEMBLE']

def readcsv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def writecsv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);root=Path(a.root);r4=Path(a.r4_root)
    statspath=r4/'analysis/statistics.py';lock=common.read_json(Path(__file__).with_name('EVALUATION_CODE_FREEZE.json'));assert common.sha256(statspath)==lock['inherited_r4']['analysis/statistics.py']
    spec=importlib.util.spec_from_file_location('h3x_frozen_statistics',statspath);stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
    roles=common.read_json('manifests/reacher_data_roles.json');cases=roles['cases']['EVAL'];ids=sorted(c['case_id'] for c in cases);assert len(ids)==100;idx=stats.case_indices('reacher',ids)
    refaudit=root/'inputs/REFERENCE_CONTROL_AUDIT.json';aud=common.read_json(refaudit);assert aud['status']=='PASS_ALL_REUSED_MODELS_CASES_AND_ANCHORS' and aud['closed_controls']==1200
    raw=[];sources=[file(refaudit)];summary=[]
    def summarize(scope,metric,values,meta):
        assert set(values)==set(BASE+NEW)
        values=dict(values);values[ARMS[-2]]=np.stack([values[x] for x in BASE[1:]]).mean(0);values[ARMS[-1]]=np.stack([values[x] for x in NEW]).mean(0)
        for arm in ARMS:
            v=values[arm];assert v.shape==(100,) and np.isfinite(v).all();ci=stats.interval(v[idx].mean(1))
            row={'label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','scope':scope,'task':'reacher','metric':metric,'arm':arm,**meta,'mean':float(v.mean()),'conditional95_low':ci['low'],'conditional95_high':ci['high'],'independent_cases':100,'bootstrap_replicates':5000,'multiplicity':'UNADJUSTED_POINTWISE_CONDITIONAL','mean_interpretation':'fixed seed/stream repeated measures; not ensemble predictions/policy'}
            row['refit_seeds_in_group']=3 if arm.startswith('FIXED3_') else 0 if arm=='H0' else 1
            for name,ref in [('H0','H0'),('R3_FIXED3',ARMS[-2])]:
                d=v-values[ref];dc=stats.interval(d[idx].mean(1));row.update({f'difference_vs_{name}':float(d.mean()),f'difference_vs_{name}_low':dc['low'],f'difference_vs_{name}_high':dc['high']})
            summary.append(row)
    old=readcsv(a.lewm);off={}
    for r in old:
        if r['task']!='reacher':continue
        key=(r['case_id'],r['model'],r['history_kind'],int(r['horizon_raw'])//5);assert key not in off;off[key]=float(r['latent_mse'])
    sources.append(file(a.lewm));newrows=[]
    for s in SEEDS:
        folder=root/f'evaluation/open_loop/{s}';complete=common.read_json(folder/'OPEN_LOOP_COMPLETE.json');r=complete['raw'];local_raw=folder/'OPEN_LOOP_RAW.csv';assert common.sha256(local_raw)==r['sha256'];sources.append(file(folder/'OPEN_LOOP_COMPLETE.json'));sources.append(file(local_raw))
        for row in readcsv(local_raw):
            assert row['label']==common.LABEL and int(row['checkpoint_step'])==30000 and row['valid']=='True'
            key=(row['case_id'],row['arm'],row['history_kind'],int(row['horizon_macro']));assert key not in off;off[key]=float(row['latent_mse']);newrows.append(row)
    for hist in ('H_POLICY','H_REAL3'):
        for h in (1,2,5):summarize('OPEN_LOOP','LATENT_MSE_FP32', {arm:np.array([off[c,arm,hist,h] for c in ids]) for arm in BASE+NEW},{'history_kind':hist,'horizon_macro':h,'planner_streams':0,'refit_seeds_in_group':3})
    writecsv(out/'H3X_OPEN_LOOP_RAW.csv',newrows)
    from r4.common import STREAMS
    closed={};controlrows=[];newclosed=[]
    for arm in BASE+NEW:
        origin=Path(a.control) if arm in BASE else root/'evaluation/closed_loop/trajectories'
        for cid in ids:
            for stream in STREAMS:
                folder=origin/'FORMAL'/stream/cid/arm;complete=common.read_json(folder/'COMPLETE.json');rpath=folder/'result.json';r=common.read_json(rpath)
                assert common.sha256(rpath)==complete['files']['result.json']['sha256'];assert r['identity_sha256']==complete['identity_sha256'] and r['status']=='COMPLETE';assert (r['case_id'],r['arm'],r['stream'])==(cid,arm,stream);assert r['success'] in (0,1)
                key=(cid,arm,stream);assert key not in closed;closed[key]=r
                row={'label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','task':'reacher','case_id':cid,'arm':arm,'stream':stream,'success':r['success'],'executed_raw_steps':r['executed_raw_steps'],'replan_calls':r['replan_calls'],'method_failure':json.dumps(r['method_failure'],sort_keys=True),'wall_seconds':r['wall_seconds'],'source_result_sha256':common.sha256(rpath),'source_result_path':str(rpath),'source_complete_sha256':common.sha256(folder/'COMPLETE.json'),'evidence_scope':'POST_R4_EXPLORATORY_ON_KNOWN_EVALUATION_CASES','inherited_runner_evidence_scope':r.get('evidence_scope'),'independent_unit':'CASE'}
                (controlrows if arm in BASE else newclosed).append(row)
    assert len(newclosed)==900 and len(controlrows)==1200
    for metric,key in [('SUCCESS_RATE','success'),('EXECUTED_RAW_STEPS','executed_raw_steps'),('REPLAN_CALLS','replan_calls'),('WALL_SECONDS_DESCRIPTIVE_ONLY','wall_seconds')]:
        vals={arm:np.array([np.mean([closed[c,arm,s][key] for s in STREAMS]) for c in ids]) for arm in BASE+NEW};summarize('CLOSED_LOOP',metric,vals,{'history_kind':'OFFICIAL_DEFAULT','horizon_macro':'','planner_streams':3,'refit_seeds_in_group':3})
    writecsv(out/'H3X_CLOSED_LOOP_RAW.csv',newclosed);writecsv(out/'PAIRED_R4_CONTROL_RAW.csv',controlrows);writecsv(out/'H3X_EXPLORATORY_CONTEXT_MATCHED_REFIT_TABLE.csv',summary)
    # Explicit per-seed/per-stream values remain raw; case means alone enter bootstrap.
    ledger=common.read_json(root/'state/TECHNICAL_LEDGER.json');training=[]
    for s in SEEDS:
        folder=root/f'artifacts/train/H3X_reacher_s{s}';r=common.read_json(folder/'result.json');journal=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()]
        assert r['actual_updates']==30000 and [v['step'] for v in journal]==list(range(1,30001)) and r['frozen_before']==r['frozen_after'];training.append({'seed':s,'actual_updates':30000,'frozen_before':r['frozen_before'],'frozen_after':r['frozen_after'],'result':file(folder/'result.json'),'final_checkpoint':file(folder/'checkpoint_30000.pt'),'update_journal':file(folder/'updates.jsonl')})
    tech=sum(v['actual_updates'] for v in ledger['runs'].values());common.atomic_json(out/'H3X_UPDATE_LEDGER.json',{'label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','formal_updates':90000,'technical_updates':tech,'total_new_predictor_updates':90000+tech,'encoder_updates':0,'projector_updates':0,'action_encoder_updates':0,'formal_closed_trajectories':900,'technical_closed_trajectories':1,'source_technical_ledger':file(root/'state/TECHNICAL_LEDGER.json'),'training':training})
    main_success=next(r for r in summary if r['metric']=='SUCCESS_RATE' and r['arm']==ARMS[-1]);lines=[f'H3X为已知评价case上的探索性追加；预设H3仍NOT_TRIGGERED。',f'3个固定seed各30,000步，编码器/观测投影/action encoder/BN全冻结；正式900条闭环齐。',f'固定3seed×3规划流case内均值成功率={main_success["mean"]:.6f}；H0差={main_success["difference_vs_H0"]:.6f}，条件95%CI=[{main_success["difference_vs_H0_low"]:.6f},{main_success["difference_vs_H0_high"]:.6f}]。',f'对R3固定3seed均值差={main_success["difference_vs_R3_FIXED3"]:.6f}，条件95%CI=[{main_success["difference_vs_R3_FIXED3_low"]:.6f},{main_success["difference_vs_R3_FIXED3_high"]:.6f}]。','seed与规划流为case内重复测量；区间未校正，不应用预设H3结果标签。'];(out/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    common.atomic_json(out/'COMPLETE.json',{'status':'COMPLETE','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','summary_rows':len(summary),'new_openloop_rows':1800,'new_closedloop_rows':900,'independent_cases':100,'sources':sources,'statistics':file(statspath),'code':file(__file__),'training':training,'files':[file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='COMPLETE.json'],'scope':'EXPLORATORY; paired conditional100case inference, no predefined H3 labels; original R4 source evidence_scope retained separately and new report scope explicit'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/workspace/r5/H3X');p.add_argument('--output',default='/workspace/r5/H3X/reports/main');p.add_argument('--r4-root',default='/workspace/r4_v23_execution');p.add_argument('--control',default='/workspace/r4_reacher/trajectories');p.add_argument('--lewm',default='/workspace/r5/H1b/HISTORY_CONDITION_4TASK_RAW.csv');a=p.parse_args();sys.path.insert(0,a.r4_root);main(a)
