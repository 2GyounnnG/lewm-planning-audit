"""Independently rebuild formal report statistics from immutable raw case rows."""
import argparse,csv,datetime,hashlib,json,os
from pathlib import Path
import numpy as np

VERSION='R4_V23_CASE_PAIRED_20261003_V1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def csvread(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def verify(r):
    p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    return p
def rec(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def key(r):return (r['task'],r['model'],r['history'],r['target'],r.get('component') or 'LATENT_ALL_192',str(r.get('seed')) if r.get('seed') not in (None,'') else '',r['case_id'])
def rng(task,kind):return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(f'{VERSION}/{task}/{kind}'.encode()).digest()[:8],'big')))
def main(a):
    directory=Path(a.report);receipt_path=directory/'COMPLETE.json';receipt=read(receipt_path)
    code=Path(a.code);assert receipt['report_code_sha256']==sha(code/'report.py')
    assert receipt['protocol_sha256']==sha(code/'FIT_PROTOCOL.json')
    assert receipt['statistics_sha256']==sha(code/'inherited_statistics.py')
    for f in receipt['files']:verify(f)
    references={};source_hashes=set()
    for f in receipt['source_raw_files']:
        p=verify(f);source_hashes.add(f['sha256'])
        for row in read(p):assert key(row) not in references;references[key(row)]=(row,f['sha256'])
    raw=csvread(directory/'H1B_ALL_RAW_VALUES.csv');assert len(raw)==receipt['raw_rows']==len(references)
    for row in raw:
        ref,h=references[key(row)];assert row['source_table_sha256']==h
        for field,value in ref.items():assert row[field]==('' if value is None else str(value)),(field,value,row[field])
    lewm=csvread(verify(receipt['source_lewm']));summary=csvread(directory/'SINGLE_FRAME_INFORMATION_TABLE.csv');perseed=csvread(directory/'H1B_PER_SEED_SUMMARY.csv');comparison=csvread(directory/'H1B_LEWM_COMPARISON.csv')
    maxdiff=0.;numeric_checks=0;counts={};all_primary={};all_lewm={}
    def compare(actual,expected):
        nonlocal maxdiff,numeric_checks
        if expected is None or not np.isfinite(expected):assert actual in ('',None);return
        value=float(actual);delta=abs(value-expected);maxdiff=max(maxdiff,delta);numeric_checks+=1
        assert np.isclose(value,expected,atol=1e-9,rtol=1e-10),(value,expected,delta)
    def ci_check(row,values,prefix='conditional95',count='valid_bootstrap'):
        finite=values[np.isfinite(values)];lo,hi=(np.quantile(finite,[.025,.975]) if len(finite) else (None,None))
        compare(row.get(prefix+'_low'),lo);compare(row.get(prefix+'_high'),hi)
        if count in row and row[count] not in ('',None):assert int(row[count])==len(finite)
    for task in receipt['tasks']:
        rows=[r for r in raw if r['task']==task];ids=sorted({r['case_id'] for r in rows});assert len(ids)==100
        ix=rng(task,'case').integers(0,100,(5000,100));metadata={};group={}
        for r in rows:
            meta=(r['anchor_raw'],r['family_id'])
            if r['case_id'] in metadata:assert metadata[r['case_id']]==meta
            metadata[r['case_id']]=meta
            target=r['target'] if r['target'].startswith('latent_') else 'state';k=(r['model'],r['history'],target,r.get('component') or 'LATENT_ALL_192',int(r['seed']) if r['seed'] else 0)
            group.setdefault(k,{})[r['case_id']]=r
        family_ix=None
        if task=='pusht':
            families=[metadata[c][1] for c in ids];assert all(families);unique=sorted(set(families));members=[np.flatnonzero(np.asarray(families)==f) for f in unique]
            draws=rng(task,'family').integers(0,len(unique),(5000,len(unique)))
            family_ix=[np.concatenate([members[j] for j in draw]) for draw in draws]
        def metrics(err,truth):
            if truth is None:
                return float(err.mean()),err[ix].mean(1),None if family_ix is None else np.array([err[v].mean() for v in family_ix])
            sst=((truth-truth.mean())**2).sum();point=None if sst==0 else 1-err.sum()/sst
            y=truth[ix];sstb=((y-y.mean(1)[:,None])**2).sum(1);bs=1-np.divide(err[ix].sum(1),sstb,out=np.full(5000,np.nan),where=sstb>0)
            family=None
            if family_ix is not None:
                values=[]
                for q in family_ix:
                    t=truth[q];den=((t-t.mean())**2).sum();values.append(np.nan if den==0 else 1-err[q].sum()/den)
                family=np.asarray(values)
            return point,bs,family
        arrays={};stats={}
        for k,cell in group.items():
            assert set(cell)==set(ids);islatent=k[2].startswith('latent_');err=np.array([float(cell[c]['mse'] if islatent else cell[c]['squared_error']) for c in ids]);truth=None if islatent else np.array([float(cell[c]['truth']) for c in ids]);arrays[k]=(err,truth);stats[k]=metrics(err,truth)
        primary={};primary_stats={}
        for model,hist,target,component in sorted({k[:4] for k in arrays}):
            items=[arrays[model,hist,target,component,s] for s in ([0,1,2] if model=='MLP' else [0])]
            truth=items[0][1]
            if truth is not None:
                for item in items[1:]:np.testing.assert_array_equal(truth,item[1])
            err=np.stack([x[0] for x in items]).mean(0);name='MLP_3SEED_MEAN_NOT_ENSEMBLE' if model=='MLP' else 'RIDGE';k=(name,hist,target,component);primary[k]=(err,truth);primary_stats[k]=metrics(err,truth)
        for row in [r for r in perseed if r['task']==task]:
            k=(row['model'],row['history'],row['target'],row['component'],int(row['seed']) if row['seed'] else 0);point,bs,_=stats[k];compare(row['estimate'],point);ci_check(row,bs)
            assert set(row['source_table_shas'].split(';'))<=source_hashes
        for row in [r for r in summary if r['task']==task]:
            if row.get('reason','').startswith('TECHNICALLY_UNAVAILABLE'):
                assert task=='tworoom' and row['state_group']=='velocity' and row['estimate']=='';continue
            name,hist,target,component=row['model'],row['history'],row['target'],row['component']
            if hist=='PAIRED_SINGLE_MINUS_THREE':
                p1,b1,f1=primary_stats[name,'single',target,component];p3,b3,f3=primary_stats[name,'three',target,component]
                point=None if p1 is None or p3 is None else p1-p3;bs=b1-b3;fam=None if f1 is None else f1-f3
            else:point,bs,fam=primary_stats[name,hist,target,component]
            compare(row['estimate'],point);ci_check(row,bs)
            if fam is not None:ci_check(row,fam,prefix='family95',count='family_valid_bootstrap')
            assert int(row['cases'])==100 and set(row['source_table_shas'].split(';'))<=source_hashes
        lby={}
        for row in lewm:
            if row['task']!=task:continue
            assert str(row['anchor_raw'])==metadata[row['case_id']][0];k=(row['model'],row['history_kind'],int(row['horizon_raw']),row['case_id']);assert k not in lby;lby[k]=float(row['latent_mse'])
        assert len(lby)==2400
        for row in [r for r in comparison if r['task']==task]:
            h=int(row['horizon_macro']);hist='single' if row['history_kind']=='H_POLICY' else 'three';e=primary[row['probe'],hist,f'latent_{h}','LATENT_ALL_192'][0]
            arms=['REFIT_103201','REFIT_103202','REFIT_103203'] if row['lewm_model']=='FIXED3_REFIT_MEAN_NOT_ENSEMBLE' else [row['lewm_model']]
            lerr=np.mean([[lby[arm,row['history_kind'],5*h,c] for c in ids] for arm in arms],axis=0);delta=e-lerr
            compare(row['probe_mse'],e.mean());compare(row['lewm_mse'],lerr.mean());compare(row['probe_minus_lewm'],delta.mean());compare(row['ratio_probe_over_lewm'],None if lerr.mean()==0 else e.mean()/lerr.mean())
            _,bs,fam=metrics(delta,None);ci_check(row,bs)
            if fam is not None:ci_check(row,fam,prefix='family95')
            assert row['lewm_merged_source_sha256']==receipt['source_lewm']['sha256'] and set(row['probe_source_table_shas'].split(';'))<=source_hashes
        all_primary[task]=(primary,ids);all_lewm[task]=lby
        counts[task]={'cases':100,'raw_rows':len(rows),'summary_rows':sum(r['task']==task for r in summary),'per_seed_rows':sum(r['task']==task for r in perseed),'comparisons':sum(r['task']==task for r in comparison),'family_sizes':None if family_ix is None else [len(m) for m in members]}
    trigger_path=directory/'H3_MEASUREMENT_ONLY.json';trigger_checked=False
    if trigger_path.exists():
        d=read(trigger_path);primary,ids=all_primary['reacher'];lby=all_lewm['reacher'];flags=[]
        for row in d['rows']:
            h=row['horizon_macro'];r=primary['RIDGE','single',f'latent_{h}','LATENT_ALL_192'][0].mean();m=primary['MLP_3SEED_MEAN_NOT_ENSEMBLE','single',f'latent_{h}','LATENT_ALL_192'][0].mean()
            refit=np.mean([[lby[arm,'H_POLICY',5*h,c] for c in ids] for arm in ['REFIT_103201','REFIT_103202','REFIT_103203']],axis=0).mean();flag=bool(min(r,m)<=.5*refit);flags.append(flag)
            for name,v in [('ridge_mean_mse',r),('mlp_three_seed_mean_mse',m),('minimum_of_complete_model_class_means',min(r,m)),('fixed3_refit_H_POLICY_mse',refit),('threshold_half_refit',.5*refit)]:compare(row[name],v)
            assert row['condition_met']==flag and row['report_code_sha256']==receipt['report_code_sha256'] and row['fit_protocol_sha256']==receipt['protocol_sha256'] and row['lewm_merged_source_sha256']==receipt['source_lewm']['sha256']
            assert set(row['ridge_source_table_shas'].split(';'))<=source_hashes and set(row['mlp_source_table_shas'].split(';'))<=source_hashes
        assert d['at_least_two_horizons_meet_measurement_condition']==(sum(flags)>=2) and not d['H3_started'];trigger_checked=True
    # Fail if the report was replaced while being independently checked.
    assert read(receipt_path)==receipt
    result={'status':'PASS_ACTUAL_FORMAL_REPORT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':counts,
       'source_raw_files_verified':len(source_hashes),'source_raw_rows_exact':len(raw),'numeric_checks':numeric_checks,'maximum_abs_roundoff_difference':maxdiff,
       'comparison_tolerance':{'atol':1e-9,'rtol':1e-10},'case_bootstrap_replicates':5000,'three_seed_error_mean_not_ensemble':True,
       'H3_measurement_checked':trigger_checked,'new_optimizer_updates':0,'new_simulator_calls':0,'source_report':rec(receipt_path),
       'report_source':rec(code/'report.py'),'fit_protocol':rec(code/'FIT_PROTOCOL.json'),'inherited_statistics':rec(code/'inherited_statistics.py'),'review_source':rec(__file__),
       'scope':'All immutable per-case source JSON rows/SHA checked against report raw CSV; summary/per-seed/LeWM contrasts recomputed using direct indexed resampling and independent SSE/SST.'}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--code',required=True);p.add_argument('--output',required=True);main(p.parse_args())
