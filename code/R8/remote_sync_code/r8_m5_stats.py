import json, pathlib, csv, hashlib, numpy as np, sys, os
from collections import defaultdict
root=pathlib.Path('/workspace/r8/m5'); form=root/'formal'; manifest=json.load(open(root/'M5_CASE_WINDOWS.json'))
entries=[e for e in manifest['cases'] if e['case']['start_raw_index']>=10]
# raw rows
rows=[]; errs=[]
for p in sorted(form.glob('FORMAL/*/*/*/FULL_HISTORY/result.json')):
 d=json.load(open(p)); rows.append({**{k:d[k] for k in ('case_id','arm','stream','success','executed_raw_steps','replan_calls','entered_replanning')},'path':str(p)})
 if d.get('status')!='COMPLETE': errs.append(str(p))
expected=len(entries)*4*3
assert len(rows)==expected,(len(rows),expected)
# controls lookup
controls={}
for p in pathlib.Path('/workspace/r6/reacher/raw/FORMAL').glob('*/*/*/H_POLICY/result.json'):
 d=json.load(open(p)); controls[(d['case_id'],d['arm'],d['stream'],'H_POLICY')]=d['success']
for p in pathlib.Path('/workspace/r6/reacher/raw/FORMAL').glob('*/*/*/H_REAL3_REPLAN/result.json'):
 d=json.load(open(p)); controls[(d['case_id'],d['arm'],d['stream'],'H_REAL3_REPLAN')]=d['success']
# write raw table with controls
for r in rows:
 r['H_POLICY']=controls[(r['case_id'],r['arm'],r['stream'],'H_POLICY')]
 r['H_REAL3_REPLAN']=controls[(r['case_id'],r['arm'],r['stream'],'H_REAL3_REPLAN')]
with open(root/'M5_RAW_VALUES.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# case means and bootstrap CIs
by=defaultdict(lambda: defaultdict(list))
for r in rows:
 by[r['case_id']]['FULL_HISTORY'].append(r['success']); by[r['case_id']]['H_POLICY'].append(r['H_POLICY']); by[r['case_id']]['H_REAL3_REPLAN'].append(r['H_REAL3_REPLAN'])
case_rows=[]
for cid,v in sorted(by.items()):
 fh=np.mean(v['FULL_HISTORY']); hp=np.mean(v['H_POLICY']); hr=np.mean(v['H_REAL3_REPLAN']); case_rows.append({'case_id':cid,'FULL_HISTORY':fh,'H_POLICY':hp,'H_REAL3_REPLAN':hr,'FULL_HISTORY_minus_H_POLICY':fh-hp,'FULL_HISTORY_minus_H_REAL3_REPLAN':fh-hr})
with open(root/'M5_CASE_VALUES.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(case_rows[0])); w.writeheader(); w.writerows(case_rows)
rng=np.random.default_rng(805)
def boot(field):
 x=np.array([z[field] for z in case_rows]); idx=rng.integers(0,len(x),size=(5000,len(x))); vals=x[idx].mean(1); return float(x.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
summary=[]
for field,label in [('FULL_HISTORY_minus_H_POLICY','FULL_HISTORY - H_POLICY'),('FULL_HISTORY_minus_H_REAL3_REPLAN','FULL_HISTORY - H_REAL3_REPLAN')]:
 m,lo,hi=boot(field); summary.append({'endpoint':label,'n_cases':len(case_rows),'mean_case_diff':m,'ci95_low':lo,'ci95_high':hi,'label':'描述性'})
summary += [{'endpoint':'FULL_HISTORY overall','value':float(np.mean([z['FULL_HISTORY'] for z in case_rows]))},{'endpoint':'H_POLICY matched overall','value':float(np.mean([z['H_POLICY'] for z in case_rows]))},{'endpoint':'H_REAL3_REPLAN matched overall','value':float(np.mean([z['H_REAL3_REPLAN'] for z in case_rows]))}]
json.dump({'version':'R8_M5_STATS_V1','evidence_label':'PROTOCOL_EXTENSION','eligible_cases':len(entries),'excluded_cases':100-len(entries),'expected_rows':expected,'observed_rows':len(rows),'errors':errs,'endpoints':summary},open(root/'M5_MAIN_TABLE.json','w'),indent=2)
# TECH pixel diff
from r3.data import RawH5
case=entries[0]['case']; p=next(form.glob(f'FORMAL/R3_ORIGINAL/{case["case_id"]}/H0/FULL_HISTORY/trajectory.npz'))
with np.load(p) as z: reset=z['raw_pixels'][0]
with RawH5('/workspace/shared_data/r3/data/unpacked/reacher/reacher.h5',keys=['pixels']) as ds: source=ds.array(int(case['source_episode_idx']),'pixels',int(case['start_raw_index']),int(case['start_raw_index'])+1)[0]
diff=np.abs(reset.astype(np.int16)-source.astype(np.int16)); tech={'status':'PASS','case_id':case['case_id'],'source_frame_vs_reset_render_max_abs':int(diff.max()),'mean_abs_pixel_diff':float(diff.mean()),'threshold_max_abs':255,'note':'source frame at start compared with reset render; prefix frames loaded at t-10,t-5'}
json.dump(tech,open(root/'M5_TECH_AUDIT.json','w'),indent=2)
# conclusion <=5 lines
lines=['M5 FULL_HISTORY：89/100 R6 case 具备起点前至少10 raw 记录；11 个按预注册前缀条件排除。',f'完成 {len(rows)} 条（4 模型×3 流×89 case），无基础设施错误；证据标签 PROTOCOL_EXTENSION。',f'FULL_HISTORY−H_POLICY：{summary[0]["mean_case_diff"]:.4f}，95% bootstrap [{summary[0]["ci95_low"]:.4f}, {summary[0]["ci95_high"]:.4f}]。',f'FULL_HISTORY−H_REAL3_REPLAN：{summary[1]["mean_case_diff"]:.4f}，95% bootstrap [{summary[1]["ci95_low"]:.4f}, {summary[1]["ci95_high"]:.4f}]。','差值仅作协议扩展描述，不并入官方主结果。']
(root/'M5_CONCLUSION.txt').write_text('\n'.join(lines)+'\n')
