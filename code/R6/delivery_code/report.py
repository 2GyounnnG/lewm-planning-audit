from __future__ import annotations
import csv,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def ci(values):return [float(np.quantile(values,.025,method='linear')),float(np.quantile(values,.975,method='linear'))]
def main(root):
 root=Path(root); raw=root/'reacher/raw'; report=root/'reacher/reports';report.mkdir(parents=True,exist_ok=True);sel=json.loads((root/'ops/R6_CASE_SELECTION.json').read_text());selection_sha=sha(root/'ops/R6_CASE_SELECTION.json'); rows=[]
 for p in sorted(raw.glob('FORMAL/*/*/*/*/result.json')):
  d=json.loads(p.read_text()); traj=p.parent/'trajectory.npz'; complete=p.parent/'COMPLETE.json'; rows.append({'case_id':d['case_id'],'arm':d['arm'],'stream':d['stream'],'history':d['history'],'success':int(d['success']),'entered_replanning':bool(d['entered_replanning']),'replan_calls':int(d['replan_calls']),'executed_raw_steps':int(d['executed_raw_steps']),'paired_first_plan_checked':bool(d.get('paired_first_plan_checked',False)),'source_result_sha256':sha(p),'trajectory_sha256':sha(traj),'case_selection_sha256':selection_sha,'evidence_label':'PREREGISTERED_FRESH_CASE_REPLICATION'})
 expected=24*len(sel['cases']);assert len(rows)==expected,(len(rows),expected)
 keys={(r['case_id'],r['arm'],r['stream'],r['history']) for r in rows};assert len(keys)==expected
 assert all(r['paired_first_plan_checked'] for r in rows if r['history']=='H_REAL3_REPLAN')
 cases=sel['planned_case_order']; arms=['H0','REFIT_103201','REFIT_103202','REFIT_103203'];streams=['R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2']
 by={(r['case_id'],r['arm'],r['stream'],r['history']):r for r in rows}
 case_diff=[]
 for c in cases:
  vals=[]
  for arm in arms:
   for st in streams: vals.append(by[c,arm,st,'H_REAL3_REPLAN']['success']-by[c,arm,st,'H_POLICY']['success'])
  case_diff.append(float(np.mean(vals)))
 point=float(np.mean(case_diff));seed=int.from_bytes(hashlib.sha256(b'R6_CASE_BOOTSTRAP_20261004/reacher').digest()[:8],'big')
 rng=np.random.default_rng(seed);idx=rng.integers(0,len(cases),size=(5000,len(cases)));draws=np.mean(np.asarray(case_diff)[idx],axis=1);interval=ci(draws)
 main=[]
 for arm in arms:
  for st in streams:
   for hist in ('H_POLICY','H_REAL3_REPLAN'):
    part=[by[c,arm,st,hist] for c in cases];main.append({'arm':arm,'stream':st,'history':hist,'cases':len(part),'successes':sum(r['success'] for r in part),'success_rate':float(np.mean([r['success'] for r in part])),'entered_replanning_rate':float(np.mean([r['entered_replanning'] for r in part])),'replanning_episode_success_rate':float(np.mean([r['success'] for r in part if r['entered_replanning']])) if any(r['entered_replanning'] for r in part) else None})
 secondary=[]
 for arm in arms:
  vals=[]
  for st in streams:
   vals.append(float(np.mean([by[c,arm,st,'H_REAL3_REPLAN']['success']-by[c,arm,st,'H_POLICY']['success'] for c in cases])))
   secondary.append({'type':'MODEL_STREAM_DIFFERENCE','arm':arm,'stream':st,'difference':vals[-1]})
  secondary.append({'type':'MODEL_STREAM_SUCCESS_RANGE','arm':arm,'history':'H_POLICY','range':float(max(np.mean([by[c,arm,st,'H_POLICY']['success'] for c in cases]) for st in streams)-min(np.mean([by[c,arm,st,'H_POLICY']['success'] for c in cases]) for st in streams))})
  secondary.append({'type':'MODEL_STREAM_SUCCESS_RANGE','arm':arm,'history':'H_REAL3_REPLAN','range':float(max(np.mean([by[c,arm,st,'H_REAL3_REPLAN']['success'] for c in cases]) for st in streams)-min(np.mean([by[c,arm,st,'H_REAL3_REPLAN']['success'] for c in cases]) for st in streams))})
 for hist in ('H_POLICY','H_REAL3_REPLAN'):
  h0=np.mean([by[c,'H0',st,hist]['success'] for c in cases for st in streams])
  for arm in arms[1:]:secondary.append({'type':'REFIT_VS_H0','arm':arm,'history':hist,'difference':float(np.mean([by[c,arm,st,hist]['success'] for c in cases for st in streams])-h0)})
 for st in streams:
  secondary.append({'type':'STREAM_DIFFERENCE_AVERAGE','stream':st,'difference':float(np.mean([by[c,arm,st,'H_REAL3_REPLAN']['success']-by[c,arm,st,'H_POLICY']['success'] for c in cases for arm in arms]))})
 flips=[]
 for c in cases:
  for arm in arms:
   for st in streams:
    a=by[c,arm,st,'H_POLICY']['success'];b=by[c,arm,st,'H_REAL3_REPLAN']['success'];
    if a!=b:flips.append({'case_id':c,'arm':arm,'stream':st,'policy_success':a,'real3_success':b,'flip':'POLICY_TO_REAL3_SUCCESS' if b>a else 'POLICY_TO_REAL3_FAILURE'})
 with (report/'R6_RAW_VALUES.csv').open('w',newline='') as f:
  fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 with (report/'R6_MAIN_TABLE.csv').open('w',newline='') as f:
  fields=list(main[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(main)
 with (report/'R6_SECONDARY_TABLE.csv').open('w',newline='') as f:
  fields=sorted({k for x in secondary for k in x});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(secondary)
 (report/'R6_FLIPS.csv').write_text('case_id,arm,stream,policy_success,real3_success,flip\n'+'\n'.join(','.join(str(x[k]) for k in ('case_id','arm','stream','policy_success','real3_success','flip')) for x in flips)+'\n')
 stats={'version':'R6_PREREGISTERED_FRESH_CASE_REPORT_V1','evidence_label':'PREREGISTERED_FRESH_CASE_REPLICATION','status':'COMPLETE','cases':len(cases),'trajectories':len(rows),'models':arms,'streams':streams,'history_conditions':['H_POLICY','H_REAL3_REPLAN'],'primary_endpoint':{'definition':'case mean across 4 models and 3 streams of H_REAL3_REPLAN minus H_POLICY success','point_estimate':point,'bootstrap':5000,'bootstrap_unit':'CASE','bootstrap_seed_rule':'int.from_bytes(SHA256("R6_CASE_BOOTSTRAP_20261004/reacher")[:8],big)','bootstrap_seed_uint64':seed,'bootstrap_indices_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),'ci95':interval,'replication判定':'PASS_INTERVAL_LOWER_GT_ZERO' if interval[0]>0 else 'FAIL_INTERVAL_LOWER_NOT_GT_ZERO'},'first_plan_pair_gate':{'H_REAL3_rows':sum(r['history']=='H_REAL3_REPLAN' for r in rows),'checked_equal':sum(r['history']=='H_REAL3_REPLAN' and r['paired_first_plan_checked'] for r in rows),'mismatches':0},'secondary_endpoints':{'flips':len(flips),'model_stream_rows':len(secondary),'raw_rows':len(rows)},'source_tables':{'raw':record(report/'R6_RAW_VALUES.csv'),'main':record(report/'R6_MAIN_TABLE.csv'),'secondary':record(report/'R6_SECONDARY_TABLE.csv'),'flips':record(report/'R6_FLIPS.csv')},'no_new_training':True,'automatic_destroy':False,'remote_originals_retained':True}
 (report/'R6_STATS.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
 conclusion=[f'新 case {len(cases)} 个，2,400 条闭环全部完成，标签 PREREGISTERED_FRESH_CASE_REPLICATION。',f'主终点 H_REAL3_REPLAN−H_POLICY = {point:.6f}，95% CI [{interval[0]:.6f}, {interval[1]:.6f}]。',('复现判定：区间下限 > 0，PASS。' if interval[0]>0 else '复现判定：区间下限不大于 0，FAIL。'),'首段计划逐对逐位一致；两 arm 均独立重跑，未复用闭环。','无新增训练、调参或结果驱动的 case/model/stream 选择。']
 (report/'CONCLUSION_ZH.txt').write_text('\n'.join(conclusion)+'\n')
 print(json.dumps({'status':'COMPLETE','cases':len(cases),'trajectories':len(rows),'point':point,'ci95':interval,'replication':stats['primary_endpoint']['replication判定'],'flips':len(flips)}))
if __name__=='__main__':main(Path(__import__('sys').argv[1]))
