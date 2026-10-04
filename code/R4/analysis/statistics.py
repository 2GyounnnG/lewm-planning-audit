"""Frozen R4 case-level paired inference and signed finite-menu diagnostics.

Candidate/source/time repetitions are never independent bootstrap units.
Incomplete cells stay missing. No outcome-driven model or case selection.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np

VERSION='R4_V23_CASE_PAIRED_20261003_V1'
REPLICATES=5000

def seed(task,kind):
    return int.from_bytes(hashlib.sha256(f'{VERSION}/{task}/{kind}'.encode()).digest()[:8],'big')

def case_indices(task,cases):
    if len(cases)!=len(set(cases)) or not cases:raise ValueError('Unique nonempty case universe required')
    return np.random.Generator(np.random.PCG64(seed(task,'case'))).integers(0,len(cases),(REPLICATES,len(cases)))

def family_weights(task,cases,families):
    if task!='pusht':raise ValueError('Family identity is not invented for other tasks')
    if len(cases)!=len(families) or any(f is None or f=='' for f in families):raise ValueError('Verified family per case required')
    unique=sorted(set(families));lookup={f:i for i,f in enumerate(unique)}
    draws=np.random.Generator(np.random.PCG64(seed(task,'family'))).integers(0,len(unique),(REPLICATES,len(unique)))
    count=np.stack([np.bincount(row,minlength=len(unique)) for row in draws])
    return count[:,[lookup[f] for f in families]]

def interval(values):
    x=np.asarray(values,dtype=float);valid=np.isfinite(x)
    if not valid.any():return {'low':None,'high':None,'valid_replicates':0}
    q=np.quantile(x[valid],[.025,.975]);return {'low':float(q[0]),'high':float(q[1]),'valid_replicates':int(valid.sum())}

def paired_summary(task,cases,matrix,*,families=None,reference=0):
    """One row per case; all column pairs use identical resampling indices.

    Columns may be case-level means over predeclared sources/streams, with each
    cell's coverage separately reported by the caller. Missing pairs are never 0.
    """
    x=np.asarray(matrix,dtype=float)
    if x.ndim!=2 or x.shape[0]!=len(cases):raise ValueError('Expected case x arm matrix')
    ix=case_indices(task,cases);out=[]
    for arm in range(x.shape[1]):
        mask=np.isfinite(x[:,arm])&np.isfinite(x[:,reference])
        d=np.where(mask,x[:,arm]-x[:,reference],np.nan)
        sampled=d[ix];n=np.isfinite(sampled).sum(1)
        values=np.divide(np.nansum(sampled,1),n,out=np.full(REPLICATES,np.nan),where=n>0)
        item={'column':arm,'reference':reference,'paired_cases':int(mask.sum()),'universe_cases':len(cases),
              'difference':float(np.mean(d[mask])) if mask.any() else None,'conditional95':interval(values),
              'missing_cases':[c for c,m in zip(cases,mask) if not m],'multiplicity':'UNADJUSTED_POINTWISE_CONDITIONAL'}
        if families is not None:
            w=family_weights(task,cases,families);den=(w*mask).sum(1)
            ratio=np.divide((w*np.nan_to_num(d,nan=0.)).sum(1),den,out=np.full(REPLICATES,np.nan),where=den>0)
            item['family_cluster_episode_weighted95']=interval(ratio)
        out.append(item)
    return out

def ranks(values):
    x=np.asarray(values);order=np.argsort(x,kind='stable');out=np.empty(len(x),dtype=float);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out

def chosen(cost,ids):
    x=np.asarray(cost,dtype=float)
    if len(x)!=len(ids) or len(set(ids))!=len(ids) or not np.isfinite(x).all():raise ValueError('Finite costs and unique logical IDs required')
    return min(range(len(ids)),key=lambda i:(float(x[i]),ids[i]))

def menu_metrics(model,latent,task_cost,ids,*,atol=1e-5,rtol=1e-5):
    m,z,y=[np.asarray(v,dtype=float) for v in (model,latent,task_cost)]
    if not (m.shape==z.shape==y.shape==(len(ids),)):raise ValueError('Same logical menu required')
    im,iz,iy=[chosen(v,ids) for v in (m,z,y)]
    a,b=np.triu_indices(len(ids),1);dm=m[a]-m[b];dz=z[a]-z[b]
    exact_ties=(dm==0)|(dz==0);comparable=~exact_ties
    near_m=np.abs(dm)<=atol+rtol*np.maximum(abs(m[a]),abs(m[b]))
    near_z=np.abs(dz)<=atol+rtol*np.maximum(abs(z[a]),abs(z[b]))
    rm,rz=ranks(m),ranks(z)
    corr=float(np.corrcoef(rm,rz)[0,1]) if np.std(rm)>0 and np.std(rz)>0 else None
    total=float(y[im]-y[iy]);goal=float(y[iz]-y[iy]);signed=float(y[im]-y[iz])
    if total<0 or goal<0 or not math.isclose(total,goal+signed,rel_tol=1e-12,abs_tol=1e-12):raise AssertionError('Signed accounting identity failed')
    return {'model_choice':ids[im],'latent_choice':ids[iz],'task_choice':ids[iy],
        'latent_finite_menu_loss':float(z[im]-z[iz]),'D_total':total,'D_goal':goal,'D_model_signed':signed,
        'spearman_model_sim_lat':corr,'pairwise_pairs':len(a),'pairwise_exact_ties':int(exact_ties.sum()),
        'pairwise_comparable':int(comparable.sum()),'pairwise_misordered':int(((dm*dz<0)&comparable).sum()),
        'pairwise_error_rate':float(((dm*dz<0)&comparable).sum()/comparable.sum()) if comparable.any() else None,
        'near_tie_pairs_model':int(near_m.sum()),'near_tie_pairs_sim_lat':int(near_z.sum()),
        'near_tie_atol':atol,'near_tie_rtol':rtol,'top1_changed':ids[im]!=ids[iz],
        'optimism_bias_mean':float(np.mean(m-z)),'goal_cost_abs_error_mean':float(np.mean(abs(m-z))),
        'logical_candidates':len(ids),'independent_sample_unit':'CASE'}

def freeze(output,source_root):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(source_root).glob('*.py'))}
    contract={'version':VERSION,'replicates':REPLICATES,'files':files,'case_sampling':'shared deterministic indices for all arms within task and case universe',
        'S1':'pair model comparisons on same source/anchor/horizon; average within case before cross-case means; report denominators',
        'S2':'initial forks only for primary; logical aliases retained; candidate count does not increase n; secondary fork separate',
        'S3':'planner streams and refit models repeated within case; no seed pooling as independent cases',
        'fixed_refit_mean':'equal mean of all three fixed refits within case; cell requires all three; not an ensemble policy',
        'history_contrast':'REAL3 minus POLICY on identical source, anchor and horizon with all four models valid in both histories',
        'source_relation':'own/other descriptive conditioning only; not randomized source effect',
        'task_margin':'locked original success geometry; anytime over executed post-action states, initial margin reported separately; terminal always separate',
        'secondary_forks':'fixed20 H0 raw10 visitation/survival conditioned, never pooled with initial100; fixed-horizon invalidity explicitly retained',
        'PushT_family':'sample families uniformly with replacement, divide weighted episode sum by sampled episode count; all arms share draws',
        'Reacher_family':'unknown; no fabricated clusters','interval':'conditional95, pointwise unadjusted; no FWER claim',
        'near_ties':{'atol':1e-5,'rtol':1e-5,'basis':'R3 FP32 replay/cache numerical resolution, frozen before new branch results'},
        'signed_decomposition':'D_total = D_model_signed + D_goal; negative signed term retained',
        'missing':'NOT_RUN/MISSING_LOG/NOT_RESTORABLE/INSUFFICIENT_EXECUTED_SUFFIX/TRUE_TERMINATED remain distinct; never zero imputed',
        'evidence_label':'POST_R3_ATTRIBUTION_SUPPLEMENT_ON_KNOWN_EVALUATION_CASES'}
    target=output/'STATISTICS_FREEZE.json'
    if target.exists() and json.loads(target.read_text())!=contract:raise RuntimeError('Frozen source changed; explicit versioned amendment required')
    target.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n');return contract

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);a=p.parse_args();print(json.dumps(freeze(a.freeze,Path(__file__).parent)))
