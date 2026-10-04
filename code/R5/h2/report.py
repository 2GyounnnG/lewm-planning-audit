"""H2 paired case bootstrap; streams/models are within-case repetitions."""
import argparse,csv,json,importlib.util
from pathlib import Path
import numpy as np
from r4.common import ARMS,STREAMS,file_record,atomic_json,sha256
STATS_SHA='e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b'
LABEL='POST_R4_SUPPLEMENT_ON_KNOWN_EVALUATION_CASES'

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--r3-root',type=Path,required=True);p.add_argument('--statistics',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if sha256(a.statistics)!=STATS_SHA:raise RuntimeError('Frozen R4 statistics source differs')
    spec=importlib.util.spec_from_file_location('h2_frozen_statistics',a.statistics);st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
    roles_path=a.r3_root/f'manifests/{a.task}_data_roles.json';roles=json.loads(roles_path.read_text());cases=sorted(roles['cases']['EVAL'],key=lambda c:c['case_id']);ids=[c['case_id'] for c in cases];assert len(ids)==100
    audit_path=a.root/'CONTROL_REUSE_AUDIT.json';audit=json.loads(audit_path.read_text());assert audit['control_count']==1200
    controls={(r['case_id'],r['arm'],r['stream']):r for r in audit['controls']}
    x=np.full((100,4,3,2),np.nan);replan=np.full_like(x,np.nan);raw=[]
    for i,case in enumerate(cases):
      for j,arm in enumerate(ARMS):
       for k,stream in enumerate(STREAMS):
        c=controls[case['case_id'],arm,stream];cf=Path(c['control_folder']);nf=a.root/'FORMAL'/stream/case['case_id']/arm
        result_path=nf/'result.json';comp=json.loads((nf/'COMPLETE.json').read_text())
        for n,r in comp['files'].items():
            if file_record(nf/n)!=r:raise RuntimeError('H2 completed output differs')
        if file_record(cf/'result.json')!=c['files']['result.json']:raise RuntimeError('R4 control result differs from audit')
        base=json.loads((cf/'result.json').read_text());new=json.loads(result_path.read_text())
        assert base['identity_sha256']==c['identity_sha256'] and new['identity_sha256']==comp['identity_sha256']
        assert new['case_id']==case['case_id'] and new['arm']==arm and new['stream']==stream and new['optimizer_updates']==0
        assert new['replans'][0]['control_first_plan_check']['status']=='BITWISE_EQUAL'
        assert new['replans'][0]['control_first_plan_check']['control_trajectory_sha256']==c['files']['trajectory.npz']['sha256']
        assert (base['replan_calls']>1)==(new['replan_calls']>1)
        if new['replan_calls']==1:assert new['success']==base['success']
        x[i,j,k]=[base['success'],new['success']];replan[i,j,k]=[base['replan_calls']>1,new['replan_calls']>1]
        raw.append({'task':a.task,'case_id':case['case_id'],'family_id':case['family_id'],'arm':arm,'stream':stream,
            'control_history':'H_POLICY','intervention_history':'H_REAL3_REPLAN','control_success':base['success'],'intervention_success':new['success'],
            'difference_REAL3_minus_POLICY':new['success']-base['success'],'control_entered_replan':base['replan_calls']>1,'intervention_entered_replan':new['replan_calls']>1,
            'control_executed_raw_steps':base['executed_raw_steps'],'intervention_executed_raw_steps':new['executed_raw_steps'],
            'intervention_wall_seconds':new['wall_seconds'],'first_plan_bitwise_equal':True,
            'control_result_sha256':c['files']['result.json']['sha256'],'control_trajectory_sha256':c['files']['trajectory.npz']['sha256'],
            'intervention_result_sha256':comp['files']['result.json']['sha256'],'intervention_trajectory_sha256':comp['files']['trajectory.npz']['sha256'],
            'observation_status':'OBSERVED','new_optimizer_updates':0,'evidence_label':LABEL})
    assert np.isfinite(x).all() and len(raw)==1200
    write(a.out/'ALL_RAW_VALUES.csv',raw);raw_sha=sha256(a.out/'ALL_RAW_VALUES.csv')
    # Both forms of repetition are averaged inside each fixed case.
    x=np.concatenate([x,x[:,1:].mean(axis=1,keepdims=True)],axis=1)
    replan=np.concatenate([replan,replan[:,1:].mean(axis=1,keepdims=True)],axis=1)
    x=np.concatenate([x,x.mean(axis=2,keepdims=True)],axis=2)
    replan=np.concatenate([replan,replan.mean(axis=2,keepdims=True)],axis=2)
    arms=(*ARMS,'FIXED3_REFIT_MEAN_NOT_ENSEMBLE');streams=(*STREAMS,'MEAN_OF_THREE_STREAMS_WITHIN_CASE')
    ix=st.case_indices(a.task,ids);weights=st.family_weights(a.task,ids,[c['family_id'] for c in cases]) if a.task=='pusht' else None
    summary=[];case_means=[];transitions=[]
    def add(metric,arm,stream,history,v):
        ci=st.interval(v[ix].mean(1));fci=st.interval(weights@v/weights.sum(1)) if weights is not None else {}
        summary.append({'module':'H2','task':a.task,'metric':metric,'arm':arm,'stream':stream,'history_kind':history,
            'estimate':float(v.mean()),'conditional95_low':ci['low'],'conditional95_high':ci['high'],'family95_low':fci.get('low'),'family95_high':fci.get('high'),
            'cases':100,'universe_cases':100,'independent_unit':'CASE','bootstrap_replicates':5000,'multiplicity':'UNADJUSTED_POINTWISE_CONDITIONAL',
            'planning_streams_averaged_within_case':3 if stream==streams[-1] else 1,'models_averaged_within_case':3 if arm==arms[-1] else 1,
            'source_table_sha256':raw_sha,'control_audit_sha256':sha256(audit_path),'evidence_label':LABEL})
    for j,arm in enumerate(arms):
      for k,stream in enumerate(streams):
        for q,history in enumerate(['H_POLICY','H_REAL3_REPLAN']):
            add('success_fraction',arm,stream,history,x[:,j,k,q]);add('entered_replanning_fraction',arm,stream,history,replan[:,j,k,q])
            if j:add('success_difference_vs_H0',arm,stream,history,x[:,j,k,q]-x[:,0,k,q])
        diff=x[:,j,k,1]-x[:,j,k,0];add('success_difference_REAL3_minus_POLICY',arm,stream,'PAIRED_HISTORY_CONTRAST',diff)
        binary=j<4 and k<3
        tr={'task':a.task,'arm':arm,'stream':stream,'cases':100,'transition_kind':'BINARY_EPISODES' if binary else 'SIGN_OF_WITHIN_CASE_AVERAGED_DIFFERENCE',
            'improved_cases':int((diff>0).sum()),'worsened_cases':int((diff<0).sum()),'unchanged_cases':int((diff==0).sum()),'source_table_sha256':raw_sha}
        if binary:
            b=x[:,j,k,0];n=x[:,j,k,1];tr.update(failure_to_success=int(((b==0)&(n==1)).sum()),success_to_failure=int(((b==1)&(n==0)).sum()),both_success=int(((b==1)&(n==1)).sum()),both_failure=int(((b==0)&(n==0)).sum()))
        transitions.append(tr)
        for i,case in enumerate(cases):case_means.append({'task':a.task,'case_id':case['case_id'],'family_id':case['family_id'],'arm':arm,'stream':stream,
            'control_success_fraction':x[i,j,k,0],'intervention_success_fraction':x[i,j,k,1],'paired_difference':diff[i],
            'control_replan_fraction':replan[i,j,k,0],'intervention_replan_fraction':replan[i,j,k,1],'source_table_sha256':raw_sha})
    write(a.out/'REAL_HISTORY_REPLANNING_TABLE.csv',summary);write(a.out/'CASE_MEANS.csv',case_means);write(a.out/'SUCCESS_TRANSITIONS.csv',transitions)
    primary=[r for r in summary if r['metric']=='success_difference_REAL3_minus_POLICY' and r['stream']==streams[-1]]
    lines=[f"{a.task}：1,200 条真实历史闭环完成；复用同机同案例同流的1,200条R4控制，首段计划全部逐位一致。"]
    for arm in ['H0',arms[-1]]:
        r=next(r for r in primary if r['arm']==arm);lines.append(f"{arm} 三流case内均值配对成功率差 {r['estimate']:.6f}，条件95%区间 [{r['conditional95_low']:.6f}, {r['conditional95_high']:.6f}]。")
    lines.append(f"仅raw25仍存活并重规划者受到干预；H0/固定三模型平均进入比例 {replan[:,0,-1,1].mean():.6f}/{replan[:,-1,-1,1].mean():.6f}。")
    detectable=any(r['conditional95_low']>0 or r['conditional95_high']<0 for r in primary)
    lines.append(('REAL_HISTORY_REPLANNING_CHANGES_CONTROL' if detectable else 'NO_DETECTABLE_CHANGE')+'；未校正条件区间，无效果不等于等效；新增训练0。')
    (a.out/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
    d={'status':'COMPLETE','module':'H2','task':a.task,'formal_new_trajectories':1200,'reused_control_trajectories':1200,'cases':100,'arms':4,'streams':3,'world_model_new_updates':0,
        'first_plan_bitwise_equal_count':1200,'case_means_rows':len(case_means),'raw_rows':len(raw),'summary_rows':len(summary),'control_reuse_audit':{'path':str(audit_path),**file_record(audit_path)},
        'roles':{'path':str(roles_path),**file_record(roles_path)},'statistics':{'path':str(a.statistics),**file_record(a.statistics)},'code':file_record(__file__),
        'outputs':{n:file_record(a.out/n) for n in ['ALL_RAW_VALUES.csv','REAL_HISTORY_REPLANNING_TABLE.csv','CASE_MEANS.csv','SUCCESS_TRANSITIONS.csv','CONCLUSION_ZH.txt']},'evidence_label':LABEL}
    atomic_json(a.out/'MODULE_STATUS.json',d);print(json.dumps(d),flush=True)

if __name__=='__main__':main()
