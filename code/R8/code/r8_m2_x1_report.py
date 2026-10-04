from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/r8');TASKS=('cube','tworoom');STREAMS=('R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2');ARMS=('H0','REFIT_103201','REFIT_103202','REFIT_103203')
def seed(text):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
def bootstrap(x,task):
 rng=np.random.default_rng(seed('R8_M2_X1_BOOTSTRAP/'+task));idx=rng.integers(0,len(x),size=(5000,len(x)));return np.quantile(np.asarray(x)[idx].mean(1),[.025,.975])
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];main=[];status={}
for task in TASKS:
 result=[]
 for p in (ROOT/'raw'/'M2'/task/'FORMAL').rglob('result.json'):
  result.append(json.loads(p.read_text()))
 result.sort(key=lambda r:(r['case_id'],r['arm'],r['stream']))
 for r in result: rows.append({k:r.get(k) for k in ('task','case_id','episode_id','arm','stream','success','executed_raw_steps','replan_calls','entered_replanning','first_plan_sha256','status')})
 by={}
 for r in result: by.setdefault(r['case_id'],{})[(r['arm'],r['stream'])]=int(r['success'])
 ds=[]
 for cid in sorted(by):
  d=by[cid];h0=np.mean([d[('H0',s)] for s in STREAMS]);rf=np.mean([d[(f'REFIT_{k}',s)] for k in ('103201','103202','103203') for s in STREAMS]);ds.append(float(rf-h0))
 ci=bootstrap(ds,task);point=float(np.mean(ds));label='提高' if ci[0]>0 else '降低' if ci[1]<0 else '未检出';main.append({'module':'M2','task':task,'endpoint':'fixed_three_refit_mean_minus_H0','case_count':len(by),'trajectory_count':len(result),'point':point,'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'label':label,'bootstrap_reps':5000,'bootstrap_seed_sha256':hashlib.sha256(('R8_M2_X1_BOOTSTRAP/'+task).encode()).hexdigest()});status[task]={'rows':len(result),'cases':len(by),'complete':sum(r['status']=='COMPLETE' for r in result),'bad':sum(r['status']!='COMPLETE' for r in result)}
out=ROOT/'reports'/'M2_X1';write(out/'ALL_RAW_VALUES.csv',rows);write(out/'MAIN_TABLE.csv',main);(out/'CONCLUSION_ZH.txt').write_text('\n'.join([f"{r['task']}：固定三 refit − H0 = {r['point']:+.6f}，95% CI [{r['ci95_low']:+.6f},{r['ci95_high']:+.6f}]，{r['label']}。" for r in main])+'\n');(out/'MODULE_STATUS.json').write_text(json.dumps({'status':'COMPLETE','evidence_label':'R8_FRESH_CASE_REPLICATION','tasks':status,'main_table':'MAIN_TABLE.csv'},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'COMPLETE','main':main},ensure_ascii=False))
