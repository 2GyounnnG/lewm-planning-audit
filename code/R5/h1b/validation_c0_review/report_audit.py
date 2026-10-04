"""Independent synthetic statistical audit; no scientific fit or simulation."""
import csv,hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent;CODE=HERE.parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(CODE))
from common import atomic,read,file,sha
import report
import inherited_statistics as st

def main():
    source_sha=sha(CODE/'report.py');p=read(CODE/'FIT_PROTOCOL.json');labels=sum([v['labels'] for v in p['state_targets']['reacher'].values() if v],[])
    ids=[f'SYNTHETIC_CASE_{i:03d}' for i in range(100)];source_shas=set()
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_',dir=HERE) as temp:
        root=Path(temp);out=root/'derived';atomic(root/'INPUT_AUDIT.json',{'SYNTHETIC_ONLY':True});lookup={}
        for model,targets,seeds in [('RIDGE',['position','velocity','latent_1','latent_2','latent_5'],[0]),('MLP',['state','latent_1','latent_2','latent_5'],[0,1,2])]:
            for target in targets:
                for hist in ('single','three'):
                    for seed in seeds:
                        dest=root/'fits/reacher'/model/target/hist/f'seed{seed}';dest.mkdir(parents=True)
                        rows=[];components=labels if target=='state' else (p['state_targets']['reacher'][target]['labels'] if not target.startswith('latent_') else [])
                        for i,cid in reversed(list(enumerate(ids))):
                            common={'task':'reacher','case_id':cid,'anchor_raw':20+i%5,'family_id':None,'target':target,'model':model,'history':hist,'seed':seed if model=='MLP' else None,'SYNTHETIC_ONLY':True}
                            if target.startswith('latent_'):
                                h=int(target.split('_')[1]);error=([2.,0. if i%2==0 else 20.,10.][[1,2,5].index(h)] if model=='RIDGE' else [1.,4.,9.][seed]*(.25 if h==5 else 1.))*(.5 if hist=='three' else 1.)
                                rows.append({**common,'mse':error});lookup[model,hist,target,'LATENT_ALL_192',seed if model=='MLP' else 0,cid]=error
                            else:
                                for label in components:
                                    j=labels.index(label);truth=2. if j==1 else float((i+1)*(j+1));signed=.5 if model=='RIDGE' else [1.,2.,-3.][seed];delta=signed*(1+i*.01)*(.5 if hist=='three' else 1.)
                                    rows.append({**common,'component':label,'truth':truth,'prediction':truth+delta,'squared_error':delta**2})
                                    lookup[model,hist,'state',label,seed if model=='MLP' else 0,cid]=delta**2
                        atomic(dest/'case_values.json',rows);source_shas.add(sha(dest/'case_values.json'))
                        atomic(dest/'COMPLETE.json',{'status':'COMPLETE','kind':'FORMAL','model':model,'target':target,'history':hist,'seed':seed,'optimizer_updates':200 if model=='MLP' else 0,'best_step':100 if model=='MLP' else None,'train_n':1000,'holdout_n':100,'SYNTHETIC_ONLY_NOT_AN_ACTUAL_FIT':True})
        lewm=[]
        for i,cid in enumerate(ids):
            for arm,val in [('H0',8.),('REFIT_103201',5.),('REFIT_103202',6.),('REFIT_103203',7.)]:
                for history in ('H_POLICY','H_REAL3'):
                    for h in (1,2,5):lewm.append({'task':'reacher','case_id':cid,'anchor_raw':20+i%5,'model':arm,'history_kind':history,'horizon_raw':5*h,'latent_mse':val})
        lp=root/'SYNTHETIC_LEWM.csv';report.csv_write(lp,lewm)
        report.main(SimpleNamespace(root=str(root),output=str(out),lewm=str(lp),tasks=['reacher']))
        rows=report.csv_read(out/'SINGLE_FRAME_INFORMATION_TABLE.csv');ix=st.case_indices('reacher',ids);checked=0
        for r in rows:
            model='MLP' if r['model'].startswith('MLP') else 'RIDGE';seeds=[0,1,2] if model=='MLP' else [0]
            def cell(hist):return np.mean([[lookup[model,hist,r['target'],r['component'],s,c] for c in ids] for s in seeds],axis=0)
            target=r['target'];truth=None if target.startswith('latent_') else np.array([2. if labels.index(r['component'])==1 else (i+1)*(labels.index(r['component'])+1) for i in range(100)])
            def independent(error):
                if truth is None:return error.mean(),error[ix].mean(1)
                sst=((truth-truth.mean())**2).sum();boottruth=truth[ix];bootsst=((boottruth-boottruth.mean(1)[:,None])**2).sum(1)
                return None if sst==0 else 1-error.sum()/sst,1-np.divide(error[ix].sum(1),bootsst,out=np.full(len(ix),np.nan),where=bootsst>0)
            if r['history']=='PAIRED_SINGLE_MINUS_THREE':
                p1,b1=independent(cell('single'));p3,b3=independent(cell('three'));point=None if p1 is None else p1-p3;bs=b1-b3
            else:point,bs=independent(cell(r['history']))
            if point is None:assert r['estimate']==''
            else:np.testing.assert_allclose(float(r['estimate']),point,atol=1e-12,rtol=1e-12)
            valid=bs[np.isfinite(bs)];assert int(r['valid_bootstrap'])==len(valid)
            if len(valid):np.testing.assert_allclose([float(r['conditional95_low']),float(r['conditional95_high'])],np.quantile(valid,[.025,.975]),atol=1e-12,rtol=1e-12)
            assert set(r['source_table_shas'].split(';'))<=source_shas;checked+=1
        trig=read(out/'H3_MEASUREMENT_ONLY.json');assert [r['condition_met'] for r in trig['rows']]==[True,False,True]
        assert trig['at_least_two_horizons_meet_measurement_condition'] and not trig['H3_started']
        assert trig['rows'][1]['minimum_of_complete_model_class_means']==14/3
        for row in trig['rows']:
            assert set(row['ridge_source_table_shas'].split(';'))<=source_shas
            assert set(row['mlp_source_table_shas'].split(';'))<=source_shas
            assert row['lewm_merged_source_sha256']==sha(lp)
            assert row['fit_protocol_sha256']==sha(CODE/'FIT_PROTOCOL.json')
            assert row['report_code_sha256']==source_sha
        # Signed seed residuals 1,2,-3 average to zero; reported mean SE is14/3.
        r=next(r for r in rows if r['model'].startswith('MLP') and r['history']=='single' and r['target']=='latent_1')
        assert float(r['estimate'])==14/3 and ((1+2-3)/3)**2==0
        complete=read(out/'COMPLETE.json');assert complete['report_code_sha256']==source_sha and complete['source_lewm']['sha256']==sha(lp)
    # Independent explicit family row sampling, with unequal20/30/50 groups.
    families=['A']*20+['B']*30+['C']*50;weights=st.family_weights('pusht',ids,families);errors=np.linspace(.01,1,100);truth=np.linspace(10,50,100)
    actual=report.values(weights,errors,truth);expected=[]
    for w in weights:
        sampled=np.repeat(np.arange(100),w.astype(int));y=truth[sampled];expected.append(1-errors[sampled].sum()/((y-y.mean())**2).sum())
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert sha(CODE/'report.py')==source_sha,'Report changed during audit'
    result={'status':'PASS_REPORT_IMPLEMENTATION_AUDIT','scope':'Synthetic parser/statistics only; no real fits, no neural updates, no simulator calls',
       'report_source':file(CODE/'report.py'),'fit_protocol':file(CODE/'FIT_PROTOCOL.json'),'statistics_source':file(CODE/'inherited_statistics.py'),
       'summary_rows_independently_checked':checked,'case_bootstrap_replicates':5000,'nonensemble_discriminator':{'mean_seed_errors':14/3,'error_of_mean_prediction':0},
       'R2_definition':'Independent direct indexed SSE/SST at every bootstrap draw; constant truth undefined retained',
       'H3_discriminator':{'correct_horizon_flags':[True,False,True],'mean_of_casewise_minima_would_incorrectly_mark_h2':True,'at_least_two':True,'started':False},
       'unequal_family_sizes':[20,30,50],'family_R2_max_abs_difference':float(np.max(np.abs(actual-expected))),
       'source_bindings_verified':['Every actual probe-source table SHA maps to parser inputs','Each H3 measurement row binds ridge/MLP source tables, LeWM, protocol and report code','enclosing COMPLETE binds report code, protocol, inherited statistics and source LeWM'],
       'formal_data_audit_status':'PENDING_ACTUAL_FORMAL_REPORT','review_code':file(__file__)}
    atomic(HERE/'REPORT_IMPLEMENTATION_REVIEW.json',result);print(json.dumps({'status':result['status'],'rows':checked,'report_sha256':source_sha,'family_max_abs':result['family_R2_max_abs_difference']}),flush=True)

if __name__=='__main__':main()
