import csv, json, hashlib, pathlib, numpy as np, datetime
ROOT=pathlib.Path('/workspace')
R6=ROOT/'r6/reacher/raw/FORMAL'
FIX=ROOT/'r8_fix/f2'
FORM=FIX/'FORMAL'
OLD=ROOT/'r8/raw/M3/FORMAL'
def load(root, hist=None):
    out=[]
    for p in sorted(pathlib.Path(root).rglob('result.json')):
        try:d=json.load(open(p))
        except:continue
        if d.get('status')!='COMPLETE':continue
        if hist is None or d.get('history')==hist: out.append(d)
    return out
def grouped(rs):
    d={}
    for r in rs:d.setdefault(r['case_id'],[]).append(float(r['success']))
    return d
def boot(vals,label):
    x=np.asarray(vals,float); n=len(x)
    seed=int.from_bytes(hashlib.sha256(('R8_CASE_BOOTSTRAP_20261004/'+label).encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(5000,n)); m=x[idx].mean(1)
    return {'point':float(x.mean()),'ci95_low':float(np.quantile(m,.025)),'ci95_high':float(np.quantile(m,.975)),
            'bootstrap':5000,'seed_uint64':seed,'indices_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),'n_cases':n}
r6=load(R6); hreal=grouped([r for r in r6 if r.get('history')=='H_REAL3_REPLAN']); hpol=grouped([r for r in r6 if r.get('history')=='H_POLICY'])
pred=load(FORM,'PRED_PAST'); pgrp=grouped(pred)
cases=sorted(set(hreal)&set(pgrp)); main=boot([np.mean(hreal[c])-np.mean(pgrp[c]) for c in cases],'M3_H_REAL3_MINUS_PRED_PAST')
sec=boot([np.mean(pgrp[c])-np.mean(hpol[c]) for c in sorted(set(pgrp)&set(hpol))],'M3_PRED_PAST_MINUS_H_POLICY')
paired=[]
for r in pred:
    q=next(x for x in r6 if x.get('history')=='H_POLICY' and x['case_id']==r['case_id'] and x['arm']==r['arm'] and x['stream']==r['stream'])
    paired.append({'case_id':r['case_id'],'arm':r['arm'],'stream':r['stream'],'pred_success':int(r['success']),'hpolicy_success':int(q['success'])})
flips={'failure_to_success':sum(x['pred_success']==1 and x['hpolicy_success']==0 for x in paired),'success_to_failure':sum(x['pred_success']==0 and x['hpolicy_success']==1 for x in paired),'n':len(paired)}
second=[r['replans'][1] for r in pred if len(r.get('replans',[]))>1]
diff=[not bool(x.get('vs_hreal3_second_plan',{}).get('bitwise_equal')) for x in second]
real2=load(OLD,'REAL2'); repeat=load(OLD,'REPEAT3')
def sample_check(rs,mode):
    rs=sorted(rs,key=lambda r:(r['case_id'],r['arm'],r['stream']))[:20]
    expected={'REAL2':[20,25],'REPEAT3':[25,25,25]}[mode]
    vals=[]
    for r in rs:
        ss=[x for x in r.get('replans',[]) if x.get('replan_index')==1]
        vals.append({'case_id':r['case_id'],'arm':r['arm'],'stream':r['stream'],'history_raw_indices':ss[0].get('history_raw_indices') if ss else None,'expected':expected,'matches':bool(ss and ss[0].get('history_raw_indices')==expected)})
    return {'n':len(vals),'all_match':all(x['matches'] for x in vals),'records':vals}
out={'version':'R8_FIX_F2_INDEPENDENT_STATS_V1','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_rows':len(pred),'main_H_REAL3_minus_PRED_PAST':main,'secondary_PRED_PAST_minus_H_POLICY':sec,'flips':flips,'second_plan_difference':{'n_replanned':len(diff),'n_different':sum(diff),'ratio':float(np.mean(diff)) if diff else None},'design_checks':{'REAL2':sample_check(real2,'REAL2'),'REPEAT3':sample_check(repeat,'REPEAT3')}}
(FIX/'reports').mkdir(exist_ok=True); (FIX/'reports'/'R8_FIX_F2_INDEPENDENT_STATS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))
