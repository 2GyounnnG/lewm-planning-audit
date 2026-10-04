"""Preserve original R3 estimates for the eventual four-task R3/X1 table."""
import argparse,csv,gzip,json,math
from pathlib import Path
from deliver import record,atomic

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def canonical(source,out):
    arms=['H0','REFIT_103201','REFIT_103202','REFIT_103203'];mean_arm='FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE'
    closed_metrics=('success','final_goal_error','executed_raw_steps','replan_calls','trajectory_wall_seconds')
    open_metrics=('latent_raw_MSE','latent_TRAIN_variance_normalized_MSE','prediction_norm_l2','target_norm_l2','prediction_norm2_over_D','target_norm2_over_D','prediction_displacement_from_initial_MSE')
    def avg(values):
        v=sum(float(x) for x in values)/len(values)
        return v if math.isfinite(v) else 'Infinity' if v>0 else '-Infinity' if v<0 else 'NaN'
    raw=[];main=[]
    with (source/'tables/formal_planning_raw.csv').open(newline='') as f:
        for r in csv.DictReader(f):raw.append({**{k:r[k] for k in ('task','case_id','family_id','arm',*closed_metrics)},'evaluation_kind':'CLOSED_LOOP_CEM','horizon_macro':0,'horizon_raw':0,'derived_within_case_refit_mean':False})
    with gzip.open(source/'tables/open_loop_per_case.csv.gz','rt',newline='') as f:
        for r in csv.DictReader(f):
            if r['arm']=='H0' or r['checkpoint_step']=='30000':
                raw.append({**{k:r[k] for k in ('task','case_id','family_id',*open_metrics)},'arm':r['arm'].removesuffix('_30000'),'evaluation_kind':'OPEN_LOOP','horizon_macro':int(r['horizon_macro']),'horizon_raw':int(r['horizon_raw']),'derived_within_case_refit_mean':False})
    for task in ('pusht','reacher'):
        stats=json.loads((source/f'artifacts/statistics/{task}/NUMERIC_RESULTS.json').read_text());cases=stats['case_order']
        for case in cases:
            for kind,h,metrics in [('CLOSED_LOOP_CEM',0,closed_metrics)]+[('OPEN_LOOP',h,open_metrics) for h in (1,2,5)]:
                part=[r for r in raw if r['task']==task and r['case_id']==case and r['evaluation_kind']==kind and r['horizon_macro']==h and r['arm'] in arms[1:]]
                if len(part)!=3:raise ValueError('Original three-refit cell incomplete')
                raw.append({'task':task,'case_id':case,'family_id':part[0]['family_id'],'arm':mean_arm,'evaluation_kind':kind,'horizon_macro':h,'horizon_raw':5*h,'derived_within_case_refit_mean':True,**{k:avg([r[k] for r in part]) for k in metrics}})
        for arm in arms+[mean_arm]:
            for kind,h,metrics in [('CLOSED_LOOP_CEM',0,closed_metrics)]+[('OPEN_LOOP',h,open_metrics) for h in (1,2,5)]:
                part=[r for r in raw if r['task']==task and r['arm']==arm and r['evaluation_kind']==kind and r['horizon_macro']==h]
                if len(part)!=100:raise ValueError('Original fixed100 cell incomplete')
                row={'task':task,'source_study':'ORIGINAL_R3_UNCHANGED','evaluation_kind':kind,'arm':arm,'horizon_macro':h,'horizon_raw':5*h,'cases':100,
                     'training_updates_per_seed':0 if arm=='H0' else 30000,'refit_seed_count':3 if arm==mean_arm else 0 if arm=='H0' else 1,**{k:avg([r[k] for r in part]) for k in metrics}}
                if kind=='CLOSED_LOOP_CEM':
                    row['success_percent']=100*row['success']
                    if arm!='H0':
                        key='FIXED_THREE_REFIT_MEAN_minus_H0_pp' if arm==mean_arm else arm+'_minus_H0_pp'
                        for scheme,label in [('CASE','case'),('FAMILY','family')]:
                            data=stats['schemes'][scheme]
                            if data['status']=='COMPLETE':
                                cell=data['metrics'][key];row.update(delta_success_pp=cell['estimate'],**{f'delta_{label}_ci95_low':cell['ci95'][0],f'delta_{label}_ci95_high':cell['ci95'][1]})
                main.append(row)
    for r in raw:r['source_study']='ORIGINAL_R3_UNCHANGED'
    write(out/'MAIN_TABLE.csv',main);write(out/'ALL_RAW_VALUES.csv',raw)
    return len(main),len(raw)

def export(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];inputs=[];raw=[]
    for task in ('pusht','reacher'):
        path=source/f'artifacts/statistics/{task}/NUMERIC_RESULTS.json';value=json.loads(path.read_text());inputs.append(record(path))
        if value['status']!='COMPLETE' or value['cases']!=100:raise ValueError('Original R3 source incomplete')
        for scheme,data in value['schemes'].items():
            if data['status']!='COMPLETE':continue
            for metric,cell in data['metrics'].items():
                rows.append({'task':task,'source_study':'ORIGINAL_R3_UNCHANGED','evaluation_kind':'CLOSED_LOOP_CEM','metric':metric,'scheme':scheme,'estimate':cell['estimate'],'conditional95_low':cell['ci95'][0],'conditional95_high':cell['ci95'][1],'cases':100,'precision':'FP32','source_sha256':inputs[-1]['sha256']})
    path=source/'tables/open_loop_summary.csv';inputs.append(record(path))
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['arm']=='H0' or r['checkpoint_step']=='30000':
                rows.append(dict(r,source_study='ORIGINAL_R3_UNCHANGED',evaluation_kind='OPEN_LOOP',source_sha256=inputs[-1]['sha256']))
    path=source/'tables/formal_planning_raw.csv';inputs.append(record(path))
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            raw.append({**{k:r[k] for k in ('task','case_id','family_id','arm','success','executed_raw_steps','replan_calls','final_goal_error')},'source_study':'ORIGINAL_R3_UNCHANGED','evaluation_kind':'CLOSED_LOOP_CEM','source_sha256':inputs[-1]['sha256']})
    path=source/'tables/open_loop_per_case.csv.gz';inputs.append(record(path))
    with gzip.open(path,'rt',newline='') as f:
        for r in csv.DictReader(f):
            if r['arm']=='H0' or str(r.get('checkpoint_step'))=='30000':raw.append(dict(r,source_study='ORIGINAL_R3_UNCHANGED',evaluation_kind='OPEN_LOOP',source_sha256=inputs[-1]['sha256']))
    for name in ('FINAL_SCIENTIFIC_REPORT_ZH_ERRATUM_001.md','FINAL_SCIENTIFIC_REPORT_ZH_CORRECTED_001.md'):
        inputs.append(record(source/'reports'/name))
    write(out/'ORIGINAL_METRICS_LONG.csv',rows);write(out/'ORIGINAL_OBSERVED_RAW_VALUES.csv',raw)
    nmain,nraw=canonical(source,out)
    atomic(out/'MODULE_STATUS.json',{'status':'COMPLETE','scope':'ORIGINAL_R3_REFERENCE_ONLY_NO_NEW_EXPERIMENT','inputs':inputs,'outputs':[record(out/name) for name in ('MAIN_TABLE.csv','ALL_RAW_VALUES.csv','ORIGINAL_METRICS_LONG.csv','ORIGINAL_OBSERVED_RAW_VALUES.csv')],'rows':nmain,'raw_rows_including_derived_means':nraw,'new_training_updates':0,'new_closed_loop_runs':0,'intervals':'Original pointwise conditional95 retained, no four-task FWER claim'})
    print(json.dumps({'status':'COMPLETE','summary_rows':nmain,'raw_rows_including_derived':nraw}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();export(a.source,a.out)
