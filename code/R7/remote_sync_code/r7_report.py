from __future__ import annotations
import csv, hashlib, json, math, os, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

RNG_NAMESPACE = "R7_LONG_BOOTSTRAP_20261004"

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def boot(values, task, label, n=5000):
    x=np.asarray(values,dtype=float); seed=int.from_bytes(hashlib.sha256(f"{RNG_NAMESPACE}/{task}/{label}".encode()).digest()[:8],'big')%(2**63-1)
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(n,len(x))); means=x[idx].mean(axis=1)
    return float(x.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)),seed

def read_jsons(root, task):
    out=[]
    for p in sorted((root/'raw'/task/'FORMAL').glob('*/ */ */ */result.json')): pass
    for p in sorted((root/'raw'/task/'FORMAL').glob('**/result.json')):
        d=json.loads(p.read_text());
        if d.get('status')!='COMPLETE': raise RuntimeError(f'noncomplete {p}')
        out.append(d)
    return out

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text(''); return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    root=Path(os.environ.get('R7_ROOT','/workspace/r7')); reports=root/'reports'; reports.mkdir(exist_ok=True)
    all_rows=[]; main_rows=[]; secondary=[]; stats={'identity':'R7_PREREGISTERED_STRESS_TEST_FRESH_CASES_V1','bootstrap_reps':5000,'bootstrap_namespace':RNG_NAMESPACE}
    for task in ('reacher','pusht'):
        sel=json.loads((root/'ops'/f'R7_{task.upper()}_CASE_SELECTION.json').read_text()); cases=sel['formal_cases']; case_ids=[c['case_id'] for c in cases]
        results=read_jsons(root,task); expected=100*4*3*2
        if len(results)!=expected: raise RuntimeError(f'{task}: got {len(results)} expected {expected}')
        keys={(d['case_id'],d['arm'],d['stream'],d['history']) for d in results}
        if len(keys)!=expected or any(d['case_id'] not in case_ids for d in results): raise RuntimeError(f'{task}: duplicate or unknown key')
        pair=[]
        for d in results:
            pair.append(d)
            for rp in d.get('replans',[]):
                if d['history']=='H_REAL3_REPLAN' and rp['replan_index']==0:
                    chk=rp.get('paired_first_plan_check',{}); 
                    if chk.get('status')!='BITWISE_EQUAL' or float(chk.get('max_abs_difference',1))!=0.0: raise RuntimeError(f'{task}: first plan mismatch {d["case_id"]}')
        # raw trajectory table
        for d in results:
            all_rows.append({k:d.get(k) for k in ('task','case_id','arm','stream','history','success','executed_raw_steps','replan_calls','entered_replanning','optimizer_updates','status','evidence_scope','paired_first_plan_checked')})
        by=defaultdict(dict)
        for d in results: by[(d['case_id'],d['arm'],d['stream'],d['history'])]=d
        p1=[]; p2=[]; sec=[]; flip=[]
        for cid in case_ids:
            pol=[by[(cid,a,s,'H_POLICY')]['success'] for a in ('H0','REFIT_103201','REFIT_103202','REFIT_103203') for s in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')]
            real=[by[(cid,a,s,'H_REAL3_REPLAN')]['success'] for a in ('H0','REFIT_103201','REFIT_103202','REFIT_103203') for s in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')]
            p1.append(float(np.mean(real)-np.mean(pol)))
            ref=[by[(cid,a,s,'H_POLICY')]['success'] for a in ('REFIT_103201','REFIT_103202','REFIT_103203') for s in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')]
            h0=[by[(cid,'H0',s,'H_POLICY')]['success'] for s in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')]
            p2.append(float(np.mean(ref)-np.mean(h0)))
            for a in ('H0','REFIT_103201','REFIT_103202','REFIT_103203'):
                for s in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2'):
                    p=by[(cid,a,s,'H_POLICY')]['success']; r=by[(cid,a,s,'H_REAL3_REPLAN')]['success']; flip.append('FAILURE_TO_SUCCESS' if (p==0 and r==1) else 'SUCCESS_TO_FAILURE' if (p==1 and r==0) else 'NO_FLIP')
        p1_stat=boot(p1,task,'P1'); p2_stat=boot(p2,task,'P2')
        if task=='reacher':
            stats['P1_reacher']={'point':p1_stat[0],'ci95':[p1_stat[1],p1_stat[2]],'lower':p1_stat[1],'classification':'成立' if p1_stat[1]>0 else '未成立','seed':p1_stat[3]}
            main_rows.append({'endpoint':'P1','task':task,'contrast':'H_REAL3_REPLAN-H_POLICY','point':p1_stat[0],'ci95_low':p1_stat[1],'ci95_high':p1_stat[2],'classification':stats['P1_reacher']['classification']})
        else:
            stats['secondary_pusht_real3_minus_policy']={'point':p1_stat[0],'ci95':[p1_stat[1],p1_stat[2]],'seed':p1_stat[3]}
        cls='未检出' if p2_stat[1]<=0<=p2_stat[2] else '长时域下 refit 提高成功率' if p2_stat[1]>0 else '降低'
        stats[f'P2_{task}']={'point':p2_stat[0],'ci95':[p2_stat[1],p2_stat[2]],'classification':cls,'seed':p2_stat[3]}
        main_rows.append({'endpoint':'P2','task':task,'contrast':'H_POLICY(refit mean)-H0','point':p2_stat[0],'ci95_low':p2_stat[1],'ci95_high':p2_stat[2],'classification':cls})
        for arm in ('H0','REFIT_103201','REFIT_103202','REFIT_103203'):
            for hist in ('H_POLICY','H_REAL3_REPLAN'):
                vals=[by[(cid,arm,s,hist)] for cid in case_ids for s in ('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2')]
                entered=[int(x['entered_replanning']) for x in vals]; ent_succ=[int(x['success']) for x in vals if x['entered_replanning']]
                secondary.append({'task':task,'arm':arm,'history':hist,'rows':len(vals),'success_rate':float(np.mean([x['success'] for x in vals])),'entered_replanning_rate':float(np.mean(entered)),'replanning_episode_success_rate':float(np.mean(ent_succ)) if ent_succ else '' ,'replan_calls_distribution':json.dumps(dict(Counter(int(x['replan_calls']) for x in vals)),sort_keys=True)})
        stats[f'{task}_counts']={'trajectories':len(results),'cases':len(cases),'pair_first_plan_equal':sum(1 for d in results if d['history']=='H_REAL3_REPLAN' and d['paired_first_plan_checked']), 'flip_counts':dict(Counter(flip)),'max_replan_calls':max(int(d['replan_calls']) for d in results)}
    write_csv(reports/'A_RAW_VALUES.csv',all_rows);write_csv(reports/'A_MAIN_TABLE.csv',main_rows);write_csv(reports/'A_SECONDARY_TABLE.csv',secondary)
    # B descriptive random baseline, case-bootstrap over the three fixed seeds.
    braw=[]; brows=[]; bstats={}
    for task in ('reacher','pusht','tworoom','cube'):
        p=root/'random'/task/f'{task}_RANDOM_RAW_VALUES.csv'; rows=list(csv.DictReader(p.open()))
        if len(rows)!=300 or any(r['error'] for r in rows): raise RuntimeError(f'B invalid {task}')
        braw.extend(rows); bycase=defaultdict(list)
        for r in rows: bycase[r['case_id']].append(int(r['success']))
        vals=[float(np.mean(v)) for _,v in sorted(bycase.items())]; st=boot(vals,task,'B_RANDOM')
        bstats[task]={'rows':len(rows),'cases':len(vals),'successes':sum(int(r['success']) for r in rows),'success_rate':float(np.mean([int(r['success']) for r in rows])),'case_bootstrap_point':st[0],'ci95':[st[1],st[2]],'seed':st[3]}
        brows.append({'task':task,'rows':len(rows),'successes':sum(int(r['success']) for r in rows),'success_rate':bstats[task]['success_rate'],'ci95_low':st[1],'ci95_high':st[2],'description_only':'DESCRIPTIVE_REFERENCE'})
    write_csv(reports/'B_RAW_VALUES.csv',braw);write_csv(reports/'B_MAIN_TABLE.csv',brows)
    stats['B_random']=bstats
    # Preserve the sealed R6 flip table by direction as a separate addendum.
    r6=Path('/workspace/r6/reacher/reports/R6_FLIPS.csv')
    if r6.exists():
        rr=list(csv.DictReader(r6.open())); counts=Counter(r['flip'] for r in rr); write_csv(reports/'R6_FLIPS_BY_DIRECTION.csv',[{'direction':k,'count':v} for k,v in sorted(counts.items())]);stats['R6_flip_counts']=dict(counts)
    (reports/'R7_STATS.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    conclusions=[
      f"Reacher P1 H_REAL3_REPLAN−H_POLICY: {stats['P1_reacher']['point']:.6f}, 95% CI [{stats['P1_reacher']['ci95'][0]:.6f}, {stats['P1_reacher']['ci95'][1]:.6f}]，{stats['P1_reacher']['classification']}。",
      f"Reacher P2 refit−H0: {stats['P2_reacher']['point']:.6f}, 95% CI [{stats['P2_reacher']['ci95'][0]:.6f}, {stats['P2_reacher']['ci95'][1]:.6f}]，{stats['P2_reacher']['classification']}。",
      f"PushT P2 refit−H0: {stats['P2_pusht']['point']:.6f}, 95% CI [{stats['P2_pusht']['ci95'][0]:.6f}, {stats['P2_pusht']['ci95'][1]:.6f}]，{stats['P2_pusht']['classification']}；H_REAL3−H_POLICY 作为描述性次要终点。",
      "B 为四任务固定随机策略描述性参照，按 3 个固定 seed、每任务 300 条有效行统计；不参与判定。",
      "R6 成败翻转另按失败→成功与成功→失败分方向保留；R7 无新增训练、调参或结果后选择。"
    ]
    (reports/'A_CONCLUSION.txt').write_text('\n'.join(conclusions)+'\n')
    (reports/'B_CONCLUSION.txt').write_text(conclusions[3]+'\n')
    print(json.dumps({'status':'PASS','reports':sorted(p.name for p in reports.iterdir()),'stats':stats},ensure_ascii=False))

if __name__=='__main__': main()
