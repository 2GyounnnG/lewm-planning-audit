"""Conditional R5 H1b statistics from immutable per-case/per-seed measurements."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from common import *
import inherited_statistics as st

STATS_SHA='e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b'
def csv_read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csv_write(p,rows):
    names=list(dict.fromkeys(k for r in rows for k in r));p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def values(weights,errors,truth=None):
    if truth is None:return weights@errors/weights.sum(1)
    yy=truth-truth[0];n=weights.sum(1);sst=weights@(yy*yy)-(weights@yy)**2/n
    return 1-np.divide(weights@errors,sst,out=np.full(len(weights),np.nan),where=sst>0)
def metrics(errors,truth,weights):
    if truth is None:point=float(np.mean(errors))
    else:
        sst=float(np.sum((truth-truth.mean())**2));point=None if sst==0 else float(1-errors.sum()/sst)
    return point,values(weights,errors,truth)

def main(a):
    assert sha(Path(__file__).with_name('inherited_statistics.py'))==STATS_SHA
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);protocol=read(Path(__file__).with_name('FIT_PROTOCOL.json'));lewm=csv_read(a.lewm);lewmsha=sha(a.lewm);summary=[];seed_summary=[];raw=[];comp=[];trigger=[];sources=[];all_fit_steps=[]
    for task in a.tasks:
        expected=(len([v for v in protocol['state_targets'][task].values() if v is not None])+3)*2+24
        complete=list((root/'fits'/task).glob('**/COMPLETE.json'));assert len(complete)==expected,(task,'INCOMPLETE_FITS',len(complete),expected)
        by={};case_meta={};fit_steps=[]
        for p in sorted(complete):
            d=read(p);assert d['status']=='COMPLETE' and d['kind']=='FORMAL';rows=read(p.parent/'case_values.json');source=sha(p.parent/'case_values.json');sources.append(file(p.parent/'case_values.json'))
            fit_steps.append({'task':task,'model':d['model'],'target':d['target'],'history':d['history'],'seed':d['seed'],'updates':d['optimizer_updates'],'best_step':d.get('best_step'),'train_n':d['train_n'],'holdout_n':d.get('holdout_n'),'source_receipt_sha256':sha(p)})
            for r in rows:
                cid=r['case_id'];case_meta[cid]={'family':r['family_id'],'anchor':r['anchor_raw']};component=r.get('component','LATENT_ALL_192');target='latent_'+r['target'].split('_')[1] if r['target'].startswith('latent_') else 'state'
                key=(r['model'],r['history'],target,component,r['seed'] if r['model']=='MLP' else 0);cell=by.setdefault(key,{})
                assert cid not in cell;cell[cid]={'error':r.get('mse',r.get('squared_error')),'truth':r.get('truth'),'source':source};raw.append({**r,'source_table_sha256':source,'evidence_label':protocol['evidence_label']})
        ids=sorted(case_meta);assert len(ids)==100;ix=st.case_indices(task,ids);w=np.stack([np.bincount(v,minlength=100) for v in ix]);fw=st.family_weights(task,ids,[case_meta[c]['family'] for c in ids]) if task=='pusht' else None
        arrays={};primary={};families={};source_map={}
        for key,cell in sorted(by.items()):
            assert set(cell)==set(ids);model,hist,target,component,seed=key;err=np.array([cell[c]['error'] for c in ids],dtype=float);truth=None if target.startswith('latent_') else np.array([cell[c]['truth'] for c in ids],dtype=float);arrays[key]=(err,truth);point,bs=metrics(err,truth,w);ci=st.interval(bs)
            seed_summary.append({'task':task,'model':model,'history':hist,'target':target,'component':component,'seed':seed if model=='MLP' else None,'metric':'latent_mse' if truth is None else 'R2','estimate':point,'conditional95_low':ci['low'],'conditional95_high':ci['high'],'valid_bootstrap':ci['valid_replicates'],'source_table_shas':';'.join(sorted({cell[c]['source'] for c in ids}))})
            group=(model,hist,target,component);source_map.setdefault(group,set()).update(cell[c]['source'] for c in ids)
        for model,hist,target,component in sorted(source_map):
            seeds=[0,1,2] if model=='MLP' else [0];items=[arrays[(model,hist,target,component,s)] for s in seeds];err=np.stack([v[0] for v in items]).mean(0);truth=items[0][1]
            for item in items[1:]:
                if truth is not None:np.testing.assert_array_equal(truth,item[1])
            name='MLP_3SEED_MEAN_NOT_ENSEMBLE' if model=='MLP' else 'RIDGE';key=(name,hist,target,component);primary[key]=(err,truth);point,bs=metrics(err,truth,w);ci=st.interval(bs);fc=st.interval(values(fw,err,truth)) if fw is not None else {};state_group=next((k for k,v in protocol['state_targets'][task].items() if v is not None and component in v['labels']),None)
            summary.append({'task':task,'model':name,'history':hist,'target':target,'state_group':state_group,'component':component,'metric':'latent_mse' if truth is None else 'R2','estimate':point,'conditional95_low':ci['low'],'conditional95_high':ci['high'],'valid_bootstrap':ci['valid_replicates'],'family95_low':fc.get('low'),'family95_high':fc.get('high'),'family_valid_bootstrap':fc.get('valid_replicates'),'cases':100,'seeds':len(seeds),'contrast':None,'source_table_shas':';'.join(sorted(source_map[(model,hist,target,component)])),'evidence_label':protocol['evidence_label'],'multiplicity':'UNADJUSTED_POINTWISE_CONDITIONAL','reason':'EVAL target SST is zero; R2 undefined' if point is None else None})
        for name,hist,target,component in sorted(primary):
            if hist!='single':continue
            er,truth=primary[(name,'single',target,component)];er3,tr3=primary[(name,'three',target,component)]
            if truth is not None:np.testing.assert_array_equal(truth,tr3)
            p1,b1=metrics(er,truth,w);p3,b3=metrics(er3,truth,w);ci=st.interval(b1-b3);fc=st.interval(values(fw,er,truth)-values(fw,er3,truth)) if fw is not None else {}
            model='MLP' if name.startswith('MLP') else 'RIDGE';refs=source_map[(model,'single',target,component)]|source_map[(model,'three',target,component)]
            summary.append({'task':task,'model':name,'history':'PAIRED_SINGLE_MINUS_THREE','target':target,'component':component,'metric':'latent_mse_difference' if truth is None else 'R2_difference','estimate':None if p1 is None or p3 is None else p1-p3,'conditional95_low':ci['low'],'conditional95_high':ci['high'],'valid_bootstrap':ci['valid_replicates'],'family95_low':fc.get('low'),'family95_high':fc.get('high'),'family_valid_bootstrap':fc.get('valid_replicates'),'cases':100,'seeds':3 if model=='MLP' else 1,'contrast':'SINGLE_MINUS_THREE','source_table_shas':';'.join(sorted(refs)),'evidence_label':protocol['evidence_label'],'multiplicity':'UNADJUSTED_POINTWISE_CONDITIONAL'})
        # Existing LeWM observations are joined only on exact frozen case/anchor/history/horizon.
        lby={}
        for r in lewm:
            if r['task']!=task:continue
            cid=r['case_id'];assert cid in case_meta and int(r['anchor_raw'])==case_meta[cid]['anchor'];key=(r['model'],r['history_kind'],int(r['horizon_raw']),cid);assert key not in lby;lby[key]=float(r['latent_mse'])
        assert len(lby)==2400
        for h in [1,2,5]:
            for hist,histkind in [('single','H_POLICY'),('three','H_REAL3')]:
                lewm_models={arm:np.array([lby[arm,histkind,5*h,c] for c in ids]) for arm in ['H0','REFIT_103201','REFIT_103202','REFIT_103203']};lewm_models['FIXED3_REFIT_MEAN_NOT_ENSEMBLE']=np.stack([lewm_models[x] for x in ['REFIT_103201','REFIT_103202','REFIT_103203']]).mean(0)
                for probe in ['RIDGE','MLP_3SEED_MEAN_NOT_ENSEMBLE']:
                    e,_=primary[(probe,hist,'latent_'+str(h),'LATENT_ALL_192')]
                    for model,lerr in lewm_models.items():
                        delta=e-lerr;ci=st.interval(values(w,delta));fc=st.interval(values(fw,delta)) if fw is not None else {}
                        comp.append({'task':task,'horizon_macro':h,'history_kind':histkind,'probe':probe,'lewm_model':model,'probe_mse':float(e.mean()),'lewm_mse':float(lerr.mean()),'probe_minus_lewm':float(delta.mean()),'ratio_probe_over_lewm':float(e.mean()/lerr.mean()) if lerr.mean()!=0 else None,'conditional95_low':ci['low'],'conditional95_high':ci['high'],'family95_low':fc.get('low'),'family95_high':fc.get('high'),'case_count':100,'lewm_merged_source_sha256':lewmsha,'probe_source_table_shas':';'.join(sorted(source_map[('MLP' if probe.startswith('MLP') else 'RIDGE',hist,'latent_'+str(h),'LATENT_ALL_192')]))})
            if task=='reacher':
                rv=primary[('RIDGE','single','latent_'+str(h),'LATENT_ALL_192')][0].mean();mv=primary[('MLP_3SEED_MEAN_NOT_ENSEMBLE','single','latent_'+str(h),'LATENT_ALL_192')][0].mean();lv=np.array([[lby[arm,'H_POLICY',5*h,c] for arm in ['REFIT_103201','REFIT_103202','REFIT_103203']] for c in ids]).mean(1).mean();best=min(rv,mv)
                trigger.append({'horizon_macro':h,'ridge_mean_mse':float(rv),'mlp_three_seed_mean_mse':float(mv),'minimum_of_complete_model_class_means':float(best),'fixed3_refit_H_POLICY_mse':float(lv),'threshold_half_refit':float(.5*lv),'condition_met':bool(best<=.5*lv),'scope':'MEASUREMENT_ONLY; no H3 launch authorization','ridge_source_table_shas':';'.join(sorted(source_map[('RIDGE','single','latent_'+str(h),'LATENT_ALL_192')])),'mlp_source_table_shas':';'.join(sorted(source_map[('MLP','single','latent_'+str(h),'LATENT_ALL_192')])),'lewm_merged_source_sha256':lewmsha,'fit_protocol_sha256':sha(Path(__file__).with_name('FIT_PROTOCOL.json')),'report_code_sha256':sha(__file__)})
        csv_write(out/f'{task}_FIT_LEDGER.csv',fit_steps);all_fit_steps.extend(fit_steps)
        if task=='tworoom':summary.append({'task':task,'target':'state','state_group':'velocity','metric':'R2','estimate':None,'reason':'TECHNICALLY_UNAVAILABLE: no recorded velocity labels; none fabricated','source_table_shas':sha(root/'INPUT_AUDIT.json'),'cases':None})
    csv_write(out/'SINGLE_FRAME_INFORMATION_TABLE.csv',summary);csv_write(out/'H1B_PER_SEED_SUMMARY.csv',seed_summary);csv_write(out/'H1B_ALL_RAW_VALUES.csv',raw);csv_write(out/'H1B_LEWM_COMPARISON.csv',comp)
    if trigger:atomic(out/'H3_MEASUREMENT_ONLY.json',{'status':'REACHER_ALL_RIDGE_AND_3SEED_MLP_COMPLETE','rows':trigger,'at_least_two_horizons_meet_measurement_condition':sum(r['condition_met'] for r in trigger)>=2,'H3_started':False,'reason':'Current user scope is C0/H1a/H1b only. This reports the specified measurement; it does not authorize or trigger training.'})
    tech=[dict(read(p),source_receipt_sha256=sha(p)) for p in (root/'technical').glob('*/COMPLETE.json')]
    atomic(out/'MEASUREMENT_UPDATE_LEDGER.json',{'formal_MLP_updates':sum(r['updates'] for r in all_fit_steps if r['model']=='MLP'),'formal_MLP_fits':sum(r['model']=='MLP' for r in all_fit_steps),'formal_MLP_max_updates_per_fit':2000,'CPU_TECH_MLP_updates':sum(r['optimizer_updates'] for r in tech if r['model']=='MLP'),'total_measurement_updates':sum(r['updates'] for r in all_fit_steps if r['model']=='MLP')+sum(r['optimizer_updates'] for r in tech if r['model']=='MLP'),'world_model_updates':0,'technical_receipts':tech,'formal_fit_receipts':all_fit_steps})
    atomic(out/'COMPLETE.json',{'status':'COMPLETE_WITH_TECHNICAL_LIMITATIONS' if 'tworoom' in a.tasks else 'COMPLETE','tasks':a.tasks,'rows':len(summary),'raw_rows':len(raw),'source_raw_files':sources,'source_lewm':file(a.lewm),'statistics_sha256':STATS_SHA,'protocol_sha256':sha(Path(__file__).with_name('FIT_PROTOCOL.json')),'report_code_sha256':sha(__file__),'world_model_updates':0,'limitations':['TwoRoom velocity state unavailable; no fabricated velocity','Probe-class performance is not an information-theoretic limit','Conditional unadjusted intervals on known EVAL cases'],'files':[file(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--lewm',required=True);p.add_argument('--output',required=True);p.add_argument('--tasks',nargs='+',default=['pusht','reacher','tworoom','cube']);main(p.parse_args())
