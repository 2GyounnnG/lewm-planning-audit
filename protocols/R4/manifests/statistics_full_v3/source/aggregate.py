"""R4 raw-table aggregation. Frozen before new simulator outcomes.

Never pools tasks' latent units, candidate rows, source trajectories or planner
streams as independent cases. Partial deliveries explicitly list coverage.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
try:
    from .statistics import paired_summary,menu_metrics,case_indices,interval
except ImportError:
    from statistics import paired_summary,menu_metrics,case_indices,interval

ARMS=('H0','REFIT_103201','REFIT_103202','REFIT_103203')
STREAMS=('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')

def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');fields=list(dict.fromkeys(k for r in rows for k in r))
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    temp.replace(path)
def truth(v):return str(v).lower() in ('true','1')
def numeric(v):return float(v) if v not in ('',None) else float('nan')
def universe(root,task):
    m=json.loads((Path(root)/f'manifests/{task}_data_roles.json').read_text())
    rows=sorted(m['cases']['EVAL'],key=lambda x:x['case_id'])
    return [r['case_id'] for r in rows],{r['case_id']:r.get('family_id') for r in rows}
def summarize(task,cases,family,values,arms,metric,context):
    x=np.full((len(cases),len(arms)),np.nan)
    for i,c in enumerate(cases):
        for j,a in enumerate(arms):
            v=values.get((c,a),[])
            if v:x[i,j]=np.mean(v)
    if tuple(arms)==ARMS:
        complete=np.isfinite(x[:,1:4]).all(1)
        mean=np.where(complete,np.mean(x[:,1:4],axis=1),np.nan)
        x=np.column_stack((x,mean));arms=(*arms,'FIXED3_REFIT_MEAN_NOT_ENSEMBLE')
    fam=[family[c] for c in cases] if task=='pusht' else None
    pairs=paired_summary(task,cases,x,families=fam)
    ix=case_indices(task,cases);rows=[]
    for j,a in enumerate(arms):
        valid=np.isfinite(x[:,j]);sample=x[:,j][ix];n=np.isfinite(sample).sum(1)
        bs=np.divide(np.nansum(sample,1),n,out=np.full(len(ix),np.nan),where=n>0)
        ci=interval(bs);r=pairs[j]
        rows.append(dict(context,task=task,metric=metric,arm=a,estimate=float(np.mean(x[valid,j])) if valid.any() else None,
            cases=int(valid.sum()),universe_cases=len(cases),conditional95_low=ci['low'],conditional95_high=ci['high'],
            paired_cases=r['paired_cases'],reference_arm=arms[0],difference_vs_reference=r['difference'],difference95_low=r['conditional95']['low'],difference95_high=r['conditional95']['high'],
            family_difference95_low=r.get('family_cluster_episode_weighted95',{}).get('low'),family_difference95_high=r.get('family_cluster_episode_weighted95',{}).get('high'),
            multiplicity='UNADJUSTED_POINTWISE',independent_unit='CASE'))
    return rows

def s1(raw,root,task,out):
    rows=read(raw);cases,family=universe(root,task);groups=defaultdict(dict)
    # Pair exactly on source/anchor/history/horizon before any averaging.
    for r in rows:
        k=tuple(r[x] for x in ('case_id','source_policy','anchor_raw','history_kind','horizon_raw'))
        if r['model'] in groups[k]:raise ValueError('Duplicate S1 model cell')
        groups[k][r['model']]=r
    values=defaultdict(lambda:defaultdict(list));coverage=[];relations=[]
    for k,cell in groups.items():
        complete=set(cell)==set(ARMS) and all(truth(r['valid']) for r in cell.values())
        example=next(iter(cell.values()));primary=truth(example['primary_anchor'])
        coverage.append(dict(zip(('case_id','source_policy','anchor_raw','history_kind','horizon_raw'),k),common_valid=complete,
            missing_reason='|'.join(sorted(set(r['missing_reason'] for r in cell.values() if not truth(r['valid'])))),models_present=len(cell)))
        if not complete:continue
        for arm,r in cell.items():
            for metric in ('latent_mse','optimism'):
                v=numeric(r[metric])
                if not np.isfinite(v):continue
                for scope in (('PRIMARY' if primary else 'SECONDARY'),):
                    key=(scope,r['history_kind'],r['horizon_raw'],metric)
                    values[key][(r['case_id'],arm)].append(v)
                if primary:
                    relations.append({'task':task,'case_id':r['case_id'],'model':arm,'source_policy':r['source_policy'],
                        'source_relation':r['source_relation'],'history_kind':r['history_kind'],'horizon_raw':r['horizon_raw'],'metric':metric,'value':v})
    table=[]
    for (scope,history,h,metric),v in sorted(values.items()):
        table.extend(summarize(task,cases,family,v,ARMS,metric,{'scope':scope,'history_kind':history,'horizon_raw':h}))
    # Own/other is descriptive conditioning, not a randomized source effect.
    own=defaultdict(list)
    for r in relations:own[tuple(r[k] for k in ('task','case_id','model','source_relation','history_kind','horizon_raw','metric'))].append(r['value'])
    relrows=[dict(zip(('task','case_id','model','source_relation','history_kind','horizon_raw','metric'),k),value=float(np.mean(v)),sources=len(v)) for k,v in own.items()]
    write(Path(out)/'S1_COMMON_COVERAGE.csv',coverage);write(Path(out)/'S1_MAIN_TABLE.csv',table);write(Path(out)/'S1_FIRST_ANCHOR_OWN_OTHER.csv',relrows)
    relation_values=defaultdict(lambda:defaultdict(list))
    for r in relrows:
        relation_values[r['source_relation'],r['history_kind'],r['horizon_raw'],r['metric']][r['case_id'],r['model']].append(r['value'])
    relation_table=[]
    for (relation,history,h,metric),v in sorted(relation_values.items()):
        relation_table.extend(summarize(task,cases,family,v,ARMS,metric,{'scope':'FIRST_ANCHOR_DESCRIPTIVE_SOURCE_RELATION','source_relation':relation,'history_kind':history,'horizon_raw':h}))
    history_cells=defaultdict(dict)
    for k,cell in groups.items():
        case,source,anchor,history,h=k
        if set(cell)==set(ARMS) and all(truth(r['valid']) for r in cell.values()):history_cells[case,source,anchor,h][history]=cell
    history_values=defaultdict(lambda:defaultdict(list));history_raw=[]
    for (case,source,anchor,h),pair in history_cells.items():
        if not {'H_POLICY','H_REAL3'}.issubset(pair):continue
        for arm in ARMS:
            for metric in ('latent_mse','optimism'):
                a=numeric(pair['H_POLICY'][arm][metric]);b=numeric(pair['H_REAL3'][arm][metric])
                if not np.isfinite([a,b]).all():continue
                scope='PRIMARY' if truth(pair['H_POLICY'][arm]['primary_anchor']) else 'SECONDARY'
                history_raw.append({'task':task,'case_id':case,'source_policy':source,'anchor_raw':anchor,'horizon_raw':h,'arm':arm,'metric':metric,'H_POLICY':a,'H_REAL3':b,'difference_REAL3_minus_POLICY':b-a,'scope':scope})
                history_values[scope,h,metric][case,arm].append(b-a)
    history_table=[]
    for (scope,h,metric),v in sorted(history_values.items()):
        history_table.extend(summarize(task,cases,family,v,ARMS,metric+'_REAL3_minus_POLICY',{'scope':scope,'horizon_raw':h,'pairing':'SAME_SOURCE_ANCHOR_HORIZON_ALL4_MODELS_BOTH_HISTORIES'}))
    write(Path(out)/'S1_SOURCE_RELATION_TABLE.csv',relation_table);write(Path(out)/'S1_HISTORY_PAIRED_RAW.csv',history_raw);write(Path(out)/'S1_HISTORY_PAIRED_TABLE.csv',history_table)

def fixed_subset(root,task):
    original=json.loads((Path(root)/f'manifests/{task}_data_roles.json').read_text())['cases']['EVAL']
    def dh(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    ranked=sorted(original,key=lambda c:(dh(['R4_CASE_SUBSET_V1',task,c]),c['case_id']))
    if task=='pusht':
        groups={}
        for c in ranked:groups.setdefault(c['family_id'],[]).append(c)
        fs=sorted(groups,key=lambda f:dh(['R4_CASE_SUBSET_V1',task,'family',f]))
        selected=[groups[f][i] for i in range(max(map(len,groups.values()))) for f in fs if i<len(groups[f])][:20]
    else:selected=ranked[:20]
    return sorted(c['case_id'] for c in selected)

def s2(directory,root,task,out):
    cases,family=universe(root,task);forkrows=[];cover=[]
    for path in sorted(Path(directory).glob('*/candidate_raw.csv')):
        rows=read(path)
        if len(rows)!=64 or len({r['id'] for r in rows})!=64:raise ValueError('Expected all 64 logical candidates')
        case=rows[0]['case_id'];valid=all(truth(r['valid_fixed_horizon']) for r in rows)
        init=[r for r in rows if r['source']=='INITIAL_DISTRIBUTION']
        if len(init)!=16:raise ValueError('Expected 16 unselected initialization candidates')
        initial_cost=np.asarray([numeric(r['J_sim_lat']) for r in init])
        cover.append({'task':task,'case_id':case,'fork':rows[0]['fork'],'valid_fixed_horizon':valid,'logical_candidates':64,
            'unique_branches':len({r['raw_action_sha256'] for r in rows}),'any_success':any(truth(r['success']) for r in rows),
            'initial16_success_fraction':np.mean([truth(r['success']) for r in init]),
            'initial16_J_sim_lat_mean':float(np.mean([numeric(r['J_sim_lat']) for r in init])) if valid else None,
            **{f'initial16_J_sim_lat_q{q}':float(np.quantile(initial_cost,q/100)) if valid else None for q in (0,10,25,50,75,90,100)},
            'missing_reason':'|'.join(sorted(set(r['missing_reason'] for r in rows if r['missing_reason']))),
            'interpretation':'FIXED_MENU_COVERAGE_ONLY'})
        if not valid:continue
        for arm in ARMS:
            for temporal,column in [('ANYTIME','J_sim_task'),('TERMINAL','J_sim_task_terminal')]:
                metric=menu_metrics([numeric(r['J_'+arm]) for r in rows],[numeric(r['J_sim_lat']) for r in rows],[numeric(r[column]) for r in rows],[r['id'] for r in rows])
                metric.update(latent_mse_mean=float(np.mean([numeric(r['latent_mse_'+arm]) for r in rows])))
                forkrows.append(dict(task=task,case_id=case,family_id=family[case],fork=rows[0]['fork'],arm=arm,task_time_reduction=temporal,**metric))
    # Any real termination invalidates the task's fixed-horizon primary estimand;
    # do not hide successful early branches by reporting complete forks only.
    missing=set(cases)-{r['case_id'] for r in cover if r['fork']=='initial'}
    unavailable=any(not r['valid_fixed_horizon'] for r in cover if r['fork']=='initial')
    tables=[]
    if not unavailable and not missing:
        for temporal in ('ANYTIME','TERMINAL'):
            for metric in ('D_total','D_goal','D_model_signed','latent_finite_menu_loss','spearman_model_sim_lat','pairwise_error_rate','optimism_bias_mean','goal_cost_abs_error_mean','latent_mse_mean','top1_changed','near_tie_pairs_model','near_tie_pairs_sim_lat'):
                v=defaultdict(list)
                for r in forkrows:
                    if r['fork']=='initial' and r['task_time_reduction']==temporal and r[metric] is not None:v[r['case_id'],r['arm']].append(float(r[metric]))
                tables.extend(summarize(task,cases,family,v,ARMS,metric,{'scope':'INITIAL_FIXED64','task_time_reduction':temporal}))
    write(Path(out)/'S2_FORK_RAW_METRICS.csv',forkrows);write(Path(out)/'S2_CANDIDATE_COVERAGE.csv',cover);write(Path(out)/'S2_MAIN_TABLE.csv',tables)
    status={'status':'TECHNICALLY_UNEVALUABLE_FIXED_HORIZON' if unavailable else 'PARTIAL' if missing else 'COMPLETE',
        'missing_cases':sorted(missing),'forks_recorded':len(cover),'case_universe':len(cases),'complete_forks_not_used_as_selected_primary_subset':True}
    (Path(out)/'S2_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    secondary=[];subset=fixed_subset(root,task)
    for fork in sorted({r['fork'] for r in cover if r['fork']!='initial'}):
        part=[r for r in cover if r['fork']==fork]
        if any(not r['valid_fixed_horizon'] for r in part):continue
        for temporal in ('ANYTIME','TERMINAL'):
            for metric in ('D_total','D_goal','D_model_signed','latent_finite_menu_loss','spearman_model_sim_lat','pairwise_error_rate','optimism_bias_mean','goal_cost_abs_error_mean','latent_mse_mean','top1_changed'):
                v=defaultdict(list)
                for r in forkrows:
                    if r['fork']==fork and r['task_time_reduction']==temporal and r[metric] is not None:v[r['case_id'],r['arm']].append(float(r[metric]))
                secondary.extend(summarize(task,subset,family,v,ARMS,metric,{'scope':'SECONDARY_FIXED20_H0_VISIT_AND_SURVIVAL_CONDITIONED','fork':fork,'task_time_reduction':temporal}))
    write(Path(out)/'S2_SECONDARY_TABLE.csv',secondary)

def s3(directory,root,task,out):
    cases,family=universe(root,task);rows=[]
    for p in Path(directory).rglob('result.json'):
        if any(s.startswith('attempt_') for s in p.parts):continue
        r=json.loads(p.read_text())
        if r.get('task')!=task or r.get('case_id') not in cases or r.get('stream') not in STREAMS or r.get('status')!='COMPLETE':continue
        if r.get('arm') not in ARMS+('H0_MENU_RERANK','SIM_LAT_RERANK','SIM_TASK_RERANK'):continue
        row={k:r.get(k) for k in ('task','case_id','family_id','arm','stream','success','executed_raw_steps','replan_calls','wall_seconds','method_failure')}
        for key in ('branch_count','branch_raw_steps','physics_steps_including_prefix','prefix_replay_raw_steps','branch_seconds'):
            row[key]=sum(event.get('rerank',{}).get(key,0) for event in r.get('replans',[]))
        row['margin_missing_reason']='MISSING_TRAJECTORY'
        trajectory=p.parent/'trajectory.npz'
        if trajectory.exists():
            with np.load(trajectory,allow_pickle=False) as f:
                if 'raw_physical_state' in f and 'goal_state' in f:
                    states=f['raw_physical_state'];goal=f['goal_state']
                    if task=='pusht':
                        pos=np.linalg.norm(states[...,:4]-goal[...,:4],axis=-1);angle=np.abs(states[...,4]-goal[...,4]);angle=np.minimum(angle,2*np.pi-angle)
                        margin=np.maximum(pos/20,angle/(np.pi/9))
                    else:margin=np.max(np.abs(states-goal)/.05,axis=-1)
                    row.update(initial_task_margin=float(margin[0]),terminal_task_margin=float(margin[-1]),anytime_executed_task_margin=float(np.min(margin[1:])) if len(margin)>1 else None,margin_missing_reason='' if len(margin)>1 else 'NO_EXECUTED_RAW_STEP')
                else:row['margin_missing_reason']='METHOD_FAILURE_NO_COMPLETE_STATE_LOG' if r.get('method_failure') else 'MISSING_STATE_LOG'
        rows.append(row)
    keys=[(r['case_id'],r['arm'],r['stream']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate S3 logical outcome')
    table=[];randomness=[];menu_table=[];transitions=[]
    for stream in STREAMS:
        part=[r for r in rows if r['stream']==stream and r['arm'] in ARMS];v=defaultdict(list)
        for r in part:v[r['case_id'],r['arm']].append(float(r['success']))
        table.extend(summarize(task,cases,family,v,ARMS,'success_fraction',{'scope':'LEARNING_POLICIES_ALL100','stream':stream}))
    for arm in ARMS:
        v=defaultdict(list)
        for r in rows:
            if r['arm']==arm:v[r['case_id'],r['stream']].append(float(r['success']))
        randomness.extend(summarize(task,cases,family,v,STREAMS,'success_fraction',{'policy':arm,'scope':'SAME_MODEL_DIFFERENT_PLANNER_STREAM'}))
    subset=fixed_subset(root,task)
    contrasts=[('H0','H0_MENU_RERANK'),('H0_MENU_RERANK','SIM_LAT_RERANK'),('SIM_LAT_RERANK','SIM_TASK_RERANK')]
    by={(r['case_id'],r['arm'],r['stream']):r for r in rows}
    for base,treated in contrasts:
        for stream in (*STREAMS,'MEAN_OF_3_STREAMS'):
            v=defaultdict(list)
            for c in subset:
                ss=STREAMS if stream=='MEAN_OF_3_STREAMS' else (stream,)
                # Mean of streams is accepted only when all three paired repeats exist.
                if all((c,a,s) in by for a in (base,treated) for s in ss):
                    for a in (base,treated):v[c,a]=[float(by[c,a,s]['success']) for s in ss]
                    if len(ss)==1:
                        transitions.append({'task':task,'case_id':c,'stream':stream,'reference':base,'treated':treated,'reference_success':v[c,base][0],'treated_success':v[c,treated][0]})
            menu_table.extend(summarize(task,subset,family,v,(base,treated),'success_fraction',{'stream':stream,'scope':'FIXED20_POLICY_INTERVENTION','contrast':treated+' - '+base}))
    write(Path(out)/'S3_ALL_RAW_VALUES.csv',rows);write(Path(out)/'S3_LEARNING_MAIN_TABLE.csv',table)
    write(Path(out)/'PLANNING_RANDOMNESS_TABLE.csv',randomness);write(Path(out)/'S3_MENU_INTERVENTIONS.csv',menu_table);write(Path(out)/'S3_SUCCESS_TRANSITIONS.csv',transitions)
    metric_table=[];learning_transitions=[]
    for stream in STREAMS:
        for metric in ('terminal_task_margin','anytime_executed_task_margin','executed_raw_steps','replan_calls','wall_seconds','branch_count','branch_raw_steps','branch_seconds'):
            v=defaultdict(list)
            for r in rows:
                if r['stream']==stream and r['arm'] in ARMS and np.isfinite(numeric(r.get(metric))):v[r['case_id'],r['arm']].append(numeric(r[metric]))
            metric_table.extend(summarize(task,cases,family,v,ARMS,metric,{'scope':'LEARNING_POLICIES_ALL100','stream':stream}))
            for base,treated in contrasts:
                v=defaultdict(list)
                for c in subset:
                    if all((c,a,stream) in by and np.isfinite(numeric(by[c,a,stream].get(metric))) for a in (base,treated)):
                        for a in (base,treated):v[c,a]=[numeric(by[c,a,stream][metric])]
                metric_table.extend(summarize(task,subset,family,v,(base,treated),metric,{'scope':'FIXED20_POLICY_INTERVENTION','stream':stream,'contrast':treated+' - '+base}))
        for c in cases:
            for arm in ARMS[1:]:
                if (c,'H0',stream) in by and (c,arm,stream) in by:
                    learning_transitions.append({'task':task,'case_id':c,'stream':stream,'reference':'H0','treated':arm,'reference_success':by[c,'H0',stream]['success'],'treated_success':by[c,arm,stream]['success']})
    write(Path(out)/'S3_MARGIN_AND_RESOURCE_TABLE.csv',metric_table);write(Path(out)/'S3_LEARNING_SUCCESS_TRANSITIONS.csv',learning_transitions)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('module',choices=['s1','s2','s3']);p.add_argument('--input',required=True);p.add_argument('--r3-root',required=True);p.add_argument('--task',required=True);p.add_argument('--out',required=True);a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True);globals()[a.module](a.input,a.r3_root,a.task,a.out)
