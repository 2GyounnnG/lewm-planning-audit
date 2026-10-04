"""Frozen R10 B/C/D case-level analysis with explicit technical missingness."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

ARMS=['H0','REFIT_103201','REFIT_103202','REFIT_103203']
STREAMS=['R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def estimate(module,endpoint,values):
    seed=int.from_bytes(hashlib.sha256(f'R10_CASE_BOOTSTRAP/{module}/{endpoint}'.encode()).digest()[:8],'big')
    if not values:return dict(endpoint=endpoint,point=None,ci95_low=None,ci95_high=None,label='不可估计',complete_cases=0,bootstrap_seed_uint64=seed)
    x=np.asarray(values,float);idx=np.random.default_rng(seed).integers(0,len(x),size=(5000,len(x)));lo,hi=np.quantile(x[idx].mean(1),[.025,.975],method='linear')
    return dict(endpoint=endpoint,point=float(x.mean()),ci95_low=float(lo),ci95_high=float(hi),label='提高' if lo>0 else '降低' if hi<0 else '未检出',complete_cases=len(x),bootstrap_seed_uint64=seed)
def main():
    p=argparse.ArgumentParser();p.add_argument('--module',choices=['B','C','D'],required=True);p.add_argument('--raw',required=True);p.add_argument('--baseline',required=True);p.add_argument('--prereg',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    pr=json.loads(Path(a.prereg).read_text());assert sha(__file__)==pr['analysis_code_sha256'];assert sha(a.baseline)==pr['baseline_sha256']
    base=json.loads(Path(a.baseline).read_text());baseline={(r['case_id'],r['arm'],r['stream']):r for r in base};oldbaseline=dict(baseline);cases=pr['case_order']
    fresh_controls=0
    if pr.get('fresh_paired_controls'):
        baseline={}
        for key,old in oldbaseline.items():
            c,arm,st=key;folder=Path(a.raw)/'FORMAL'/st/c/arm/'H_POLICY'
            if not (folder/'COMPLETE.json').exists():continue
            seal=json.loads((folder/'COMPLETE.json').read_text())
            for n,r in seal['files'].items():assert sha(folder/n)==r['sha256']
            d=json.loads((folder/'result.json').read_text());assert json.loads((folder/'STARTED.json').read_text())['identity']['prereg_sha256']==sha(a.prereg)
            with np.load(folder/'trajectory.npz',allow_pickle=False) as f:
                if not len(f['returned_plans_normalized']):continue
                plan=f['returned_plans_normalized'][0]
            baseline[key]={**d,'source_result_sha256':sha(folder/'result.json'),'first_plan_sha256':hashlib.sha256(plan.tobytes()).hexdigest()};fresh_controls+=1
    modes=['REAL3_NULLACT','REPEAT3_REALACT'] if a.module=='D' else ['H_REAL3_REPLAN']
    endpoints=['E_D1','E_D2'] if a.module=='D' else [f'E_{a.module}']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[];by={};complete=equal=0
    for c in cases:
        for arm in ARMS:
            for st in STREAMS:
                b=baseline.get((c,arm,st),{'success':None,'entered_replanning':None,'source_result_sha256':'','first_plan_sha256':''})
                rows.append(dict(case_id=c,arm=arm,stream=st,mode='H_POLICY',status=('FRESH_PAIRED_CONTROL' if pr.get('fresh_paired_controls') else 'SEALED_BASELINE') if b['success'] is not None else 'TECHNICAL_MISSING',success=b['success'],entered_replanning=b['entered_replanning'],first_plan_equal='',source_result_sha256=b['source_result_sha256'],trajectory_sha256=''))
                for mode in modes:
                    folder=Path(a.raw)/'FORMAL'/st/c/arm/mode
                    r=dict(case_id=c,arm=arm,stream=st,mode=mode,status='TECHNICAL_MISSING',success=None,entered_replanning=None,first_plan_equal=None,source_result_sha256='',trajectory_sha256='')
                    if (folder/'COMPLETE.json').exists():
                        seal=json.loads((folder/'COMPLETE.json').read_text())
                        for n,f in seal['files'].items():assert sha(folder/n)==f['sha256'] and (folder/n).stat().st_size==f['bytes']
                        result=json.loads((folder/'result.json').read_text());assert result['identity_sha256']==seal['identity_sha256']
                        start=json.loads((folder/'STARTED.json').read_text())['identity'];assert start['prereg_sha256']==sha(a.prereg)
                        with np.load(folder/'trajectory.npz',allow_pickle=False) as z:plan=z['returned_plans_normalized'][0]
                        plan_equal=hashlib.sha256(plan.tobytes()).hexdigest()==b['first_plan_sha256']
                        assert plan_equal==result['first_plan_equal']
                        r.update(status='COMPLETE' if plan_equal else 'INVALID_FIRST_PLAN',success=result['success'] if plan_equal else None,entered_replanning=result['entered_replanning'],first_plan_equal=plan_equal,source_result_sha256=sha(folder/'result.json'),trajectory_sha256=sha(folder/'trajectory.npz'))
                        complete+=int(plan_equal);equal+=int(plan_equal)
                    elif (folder/'FAILURE.json').exists():r['status']=json.loads((folder/'FAILURE.json').read_text())['status']
                    rows.append(r);by[c,arm,st,mode]=r
    main=[];secondary=[]
    for endpoint,mode in zip(endpoints,modes):
        vals=[]
        for c in cases:
            if all(by[c,arm,s,mode]['status']=='COMPLETE' and (c,arm,s) in baseline for arm in ARMS for s in STREAMS):
                vals.append(np.mean([by[c,arm,s,mode]['success']-baseline[c,arm,s]['success'] for arm in ARMS for s in STREAMS]))
        main.append(estimate(a.module,endpoint,vals))
        for arm in ARMS:
            selected=[c for c in cases if all(by[c,arm,s,mode]['status']=='COMPLETE' and (c,arm,s) in baseline for s in STREAMS)]
            values=[np.mean([by[c,arm,s,mode]['success']-baseline[c,arm,s]['success'] for s in STREAMS]) for c in selected]
            records=[by[c,arm,s,mode] for c in selected for s in STREAMS]
            secondary.append(dict(endpoint=endpoint,arm=arm,complete_cases=len(selected),difference=float(np.mean(values)) if values else None,entered_replanning_rate=float(np.mean([r['entered_replanning'] for r in records])) if records else None))
    csvout(out/'RAW_VALUES.csv',rows);csvout(out/'MAIN_TABLE.csv',main);csvout(out/'SECONDARY_TABLE.csv',secondary)
    expected=1200*len(modes);pred='未检出' if a.module in ('B','C') else '不预设方向'
    stats=dict(module=a.module,planned=expected,delivered=complete,delivery_rate=complete/expected,first_plan_equal=equal,endpoints=main,prediction=pred,prediction_consistent=(main[0]['label']=='未检出') if a.module in ('B','C') else None,technical_missing=sum(r['status']=='TECHNICAL_MISSING' for r in rows),invalid_first_plan=sum(r['status']=='INVALID_FIRST_PLAN' for r in rows),training_updates=0)
    if pr.get('fresh_paired_controls'):
        stats.update(fresh_controls_delivered=fresh_controls,total_new_planned=expected+1200,total_new_delivered=complete+fresh_controls,
                     descriptive_control_drift=float(np.mean([b['success']-oldbaseline[k]['success'] for k,b in baseline.items()])) if baseline else None)
    if a.module=='D':stats['descriptive_difference_from_real3_13_7pp']={r['endpoint']:r['point']-.137 if r['point'] is not None else None for r in main}
    write(out/'STATS.json',stats)
    pct=lambda v:'不可估计' if v is None else f'{v*100:.2f}'
    lines=[f"模块 {a.module}：{r['endpoint']} = {pct(r['point'])} 个百分点，95% CI [{pct(r['ci95_low'])}, {pct(r['ci95_high'])}]，{r['label']}；完整 case {r['complete_cases']}/100。" for r in main]
    lines.append(f'交付 {complete}/{expected}，首计划门 {equal}/{expected} 逐位相等；技术缺失不补零。预注册预测：{pred}。')
    if pr.get('fresh_paired_controls'):lines.append(f'另交付同机新单帧对照 {fresh_controls}/1200；旧对照未全量通过跨机器逐位门，故主终点使用全量同机配对。')
    lines.append({'B':'影响论文关于“离线单帧/三帧诊断能否预测 TwoRoom 在线历史增益”的表述。','C':'影响论文关于“Cube 单帧并不缺信息、真实历史没有增益”的表述；随机动作48.0%仅作参照。','D':'影响论文关于“Reacher 历史增益由过去帧还是配对动作携带”的表述。两臂均为帧与动作不配对的分布外干预，不能直接外推为部署建议。'}[a.module])
    (out/'CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n');write(out/'SEAL.json',dict(module=a.module,prereg_sha256=sha(a.prereg),files={x.name:dict(bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(out.iterdir()) if x.is_file() and x.name!='SEAL.json'}));print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
