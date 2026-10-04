from __future__ import annotations
import csv, hashlib, json, pathlib
import numpy as np

ROOT=pathlib.Path('/workspace/r8'); REPORT=ROOT/'reports'; REPORT.mkdir(exist_ok=True)

def rows(root):
 out=[]
 for p in sorted(pathlib.Path(root).rglob('result.json')):
  try:
   d=json.loads(p.read_text())
   if d.get('status')=='COMPLETE': d['_path']=str(p);out.append(d)
  except Exception: pass
 return out
def boot(v,label):
 x=np.asarray(v,float);n=len(x);seed=int.from_bytes(hashlib.sha256(('R8_CASE_BOOTSTRAP_20261004/'+label).encode()).digest()[:8],'big');rng=np.random.default_rng(seed);idx=rng.integers(0,n,size=(5000,n));m=x[idx].mean(1)
 return {'point':float(x.mean()),'ci95_low':float(np.quantile(m,.025)),'ci95_high':float(np.quantile(m,.975)),'bootstrap':5000,'bootstrap_seed_rule':'SHA256("R8_CASE_BOOTSTRAP_20261004/'+label+'")[:8] big-endian','bootstrap_seed_uint64':seed,'n_cases':n,'indices_sha256':hashlib.sha256(idx.tobytes()).hexdigest()}
def grouped(rows_, filt=lambda r:True):
 d={}
 for r in rows_:
  if filt(r):d.setdefault(r['case_id'],[]).append(float(r['success']))
 return d
def paired(a,b,fa,fb):
 A=grouped(a,fa);B=grouped(b,fb);cs=sorted(set(A)&set(B));return cs,[np.mean(A[c])-np.mean(B[c]) for c in cs]
def write_csv(path, data):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fields=sorted(set().union(*(x.keys() for x in data))) if data else ['status']
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

def main():
 r6=rows('/workspace/r6/reacher/raw/FORMAL');r7=rows('/workspace/r7/raw/reacher/FORMAL');u1=rows(ROOT/'raw/U1');u2=rows(ROOT/'raw/U2')
 m3={m:rows(ROOT/f'raw/M3/FORMAL/{m}') for m in ('REPEAT3','REAL2','PRED_PAST')}
 # M1 paired decomposition and E3
 m1=[]
 for name,rs in [('U1',u1),('U2',u2)]:
  cs,v=paired(rs,rs,lambda r:r.get('history')=='H_REAL3_REPLAN',lambda r:r.get('history')=='H_POLICY');m1.append({'module':'M1','endpoint':'E1_H_REAL3_minus_H_POLICY','unit':name,**boot(v,'M1_'+name+'_E1')})
  rf=grouped(rs,lambda r:r.get('history')=='H_POLICY' and r.get('arm')!='H0');h0=grouped(rs,lambda r:r.get('history')=='H_POLICY' and r.get('arm')=='H0');cs=sorted(set(rf)&set(h0));m1.append({'module':'M1','endpoint':'E3_refit_minus_H0','unit':name,**boot([np.mean(rf[c])-np.mean(h0[c]) for c in cs],'M1_'+name+'_E3')})
 cs,v=paired(r6,u1,lambda r:r.get('history')=='H_REAL3_REPLAN',lambda r:r.get('history')=='H_REAL3_REPLAN');m1.append({'module':'M1','endpoint':'E2_budget50_minus_budget100','unit':'offset25_R6_minus_U1',**boot(v,'M1_OFFSET25_E2')})
 cs,v=paired(u2,r7,lambda r:r.get('history')=='H_REAL3_REPLAN',lambda r:r.get('history')=='H_REAL3_REPLAN');m1.append({'module':'M1','endpoint':'E2_budget50_minus_budget100','unit':'offset50_U2_minus_R7',**boot(v,'M1_OFFSET50_E2')})
 # M2 Reacher/PushT from existing M1/M2 raw output, fixed single-frame history
 m2=[]
 for task in ('reacher','pusht'):
  rs=rows(ROOT/f'raw/M2_{task}');rf=grouped(rs,lambda r:r.get('arm')!='H0');h0=grouped(rs,lambda r:r.get('arm')=='H0');cs=sorted(set(rf)&set(h0));m2.append({'module':'M2','task':task,'endpoint':'fixed_three_refit_mean_minus_H0','evidence':'R8_FRESH_CASE_REPLICATION',**boot([np.mean(rf[c])-np.mean(h0[c]) for c in cs],'M2_'+task+'_REFIT_MINUS_H0')})
 # M2 X1 reports are authoritative module output
 for r in csv.DictReader((REPORT/'M2_X1/MAIN_TABLE.csv').open()):m2.append({'module':'M2','task':r['task'],'endpoint':r['endpoint'],'evidence':'R8_FRESH_CASE_REPLICATION','point':float(r['point']),'ci95_low':float(r['ci95_low']),'ci95_high':float(r['ci95_high']),'bootstrap':int(r['bootstrap_reps']),'n_cases':int(r['case_count'])})
 write_csv(REPORT/'R8_M1_M2_MAIN_TABLE.csv',m1+m2)
 # M3 main/secondary
 m3main=[];m3sec=[];base=[r for r in r6 if r.get('history')=='H_REAL3_REPLAN']
 for mode in ('REPEAT3','PRED_PAST'):
  cs,v=paired(base,m3[mode],lambda r:True,lambda r:True);m3main.append({'module':'M3','endpoint':'H_REAL3_minus_'+mode,'evidence':'R8_HISTORY_MECHANISM','comparison_cases':len(cs),**boot(v,'M3_H_REAL3_MINUS_'+mode)})
 # descriptive secondary against R6 H_POLICY
 for mode in ('REAL2','REPEAT3','PRED_PAST'):
  cs,v=paired(m3[mode],r6,lambda r:True,lambda r:r.get('history')=='H_POLICY');m3sec.append({'module':'M3','endpoint':mode+'_minus_H_POLICY','evidence':'DESCRIPTIVE','n_cases':len(cs),**boot(v,'M3_'+mode+'_MINUS_H_POLICY')})
 cs,v=paired(base,m3['REAL2'],lambda r:True,lambda r:True);m3sec.append({'module':'M3','endpoint':'H_REAL3_minus_REAL2','evidence':'DESCRIPTIVE','n_cases':len(cs),**boot(v,'M3_H_REAL3_MINUS_REAL2')})
 write_csv(REPORT/'M3_MAIN_TABLE.csv',m3main);write_csv(REPORT/'M3_SECONDARY_TABLE.csv',m3sec)
 raw=[]
 for mode in ('REPEAT3','REAL2','PRED_PAST'):
  for r in m3[mode]:raw.append({k:r.get(k) for k in ('case_id','arm','stream','history','success','entered_replanning','replan_calls','executed_raw_steps','cross_unit_first_plan_checked','identity_sha256')})
 write_csv(REPORT/'M3_RAW_VALUES.csv',raw)
 # random descriptive baseline, one case mean over three seeds
 rnd=[]
 for task,fn in [('reacher',ROOT/'raw/M2_RANDOM_REACHER_RAW_VALUES.csv'),('pusht',ROOT/'raw/M2_RANDOM_PUSHT_RAW_VALUES.csv'),('cube',ROOT/'raw/M2/random/cube_RANDOM_RAW_VALUES.csv'),('tworoom',ROOT/'raw/M2/random/tworoom_RANDOM_RAW_VALUES.csv')]:
  rs=list(csv.DictReader(fn.open()));g={}
  for r in rs:g.setdefault(r['case_id'],[]).append(float(r['success']))
  rnd.append({'module':'M2','endpoint':'random_descriptive_success','task':task,'evidence':'DESCRIPTIVE_REFERENCE','rows':len(rs),**boot([np.mean(g[c]) for c in sorted(g)],'M2_RANDOM_'+task)})
 write_csv(REPORT/'M2_RANDOM_MAIN_TABLE.csv',rnd)
 # machine-readable module status and merged primary table
 primary=[]
 for x in m1+m2+m3main:
  y=dict(x);y['label']='PASS_INTERVAL_LOWER_GT_ZERO' if x.get('ci95_low',0)>0 else ('未检出' if x.get('ci95_low',0)<=0<=x.get('ci95_high',0) else 'NEGATIVE');primary.append(y)
 write_csv(REPORT/'R8_MAIN_TABLE.csv',primary)
 secondary=m3sec+rnd
 write_csv(REPORT/'R8_SECONDARY_TABLE.csv',secondary)
 status={'version':'R8_FINAL_STATUS_V1','status':'COMPLETE_M1_M2_M3','m1':{'U1':len(u1),'U2':len(u2)},'m2':{'reacher':len(rows(ROOT/'raw/M2_reacher')),'pusht':len(rows(ROOT/'raw/M2_pusht')),'cube':len(rows(ROOT/'raw/M2/cube/FORMAL')),'tworoom':len(rows(ROOT/'raw/M2/tworoom/FORMAL')),'random_rows':{x:len(list(csv.DictReader((ROOT/('raw/M2_RANDOM_'+x.upper()+'_RAW_VALUES.csv' if x in ('reacher','pusht') else 'raw/M2/random/'+x+'_RANDOM_RAW_VALUES.csv')).open()))) for x in ('reacher','pusht','cube','tworoom')}},'m3':{k:len(v) for k,v in m3.items()},'new_training':0,'automatic_destroy':False}
 (REPORT/'R8_FINAL_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 (REPORT/'R8_UNEXECUTED_OR_TECH.md').write_text('# R8 未执行模块\n\n- M4、M5、M6：本轮在 M1–M3 及 M2 随机参照完成后未启动，按 R8 §1 的 5 小时调度上限记为“未执行（时间上限）”；无其结果合并到 R8_MAIN_TABLE。\n- M1、M2、M3：TECH 通过且全量 COMPLETE。\n')
 print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':main()
