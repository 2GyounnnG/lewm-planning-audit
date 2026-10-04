from pathlib import Path
import sys,json,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.recover_r3 import remote_python
code=r'''
from pathlib import Path
import os,json,hashlib,datetime
r=Path('/workspace/r3_official_lewm_predictor_refit');assert os.uname().nodename=='6494ba5e1b1a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent='manifests/RECOVERY_FINAL.json';old='reports/FINAL_SCIENTIFIC_REPORT_ZH.md';source='source/swm_compat/stable_worldmodel/envs/pusht/env.py'
assert sha(r/parent)=='f4e0f4c87ae97f3aa57a2ededf97c92f1fe88c1d9bfefb401b508bae0d7d7b4f'
assert sha(r/old)=='c803c585b10a8f84bcb9afa2fec285963a7fdd23447006bba063425c45e0d158'
parentdoc=json.loads((r/parent).read_text());assert sha(r/source)==parentdoc['files'][source]['sha256']
s=(r/source).read_text();assert 'pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])' in s and 'success = pos_diff < 20 and angle_diff < np.pi / 9' in s
text=(r/old).read_text();wrong='与覆盖率success并非同一个量';correct='与位置—角度阈值定义的success并非同一个量';assert text.count(wrong)==1
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
note=('# R3 最终科学报告勘误001：PushT success术语\n\n'+f'记录时间：{now}。原报告SHA256 {sha(r/old)}，原最终清单SHA256 {sha(r/parent)}。\n\n'
 '原报告“闭环物理误差与实际规划成本”段将PushT写成“覆盖率success”，这是报告术语错误。实际冻结并执行的SWM版本定义为：前四维位置差的L2范数小于20，且环绕角差min(|目标角−当前角|,2π−|目标角−当前角|)小于π/9。step直接以此作为terminated；另返完整7维state L2作为distance并取负reward。应写“与位置—角度阈值定义的success并非同一个量”。\n\n'
 f'一手实现证据：{source}:340–355，SHA256 {sha(r/source)}。只修正报告归纳，不修改执行代码、成功定义、800条原始轨迹、任一原值/区间、统计脚本或科学标签。无需重算或追加实验。\n\n'
 '原报告、原RECOVERY_FINAL清单和所有正在传输的源字节原位保持不变。另提供FINAL_SCIENTIFIC_REPORT_ZH_CORRECTED_001.md：除开头勘误说明及上述一个短语替换外，正文与封存原报告完全相同。该补充独立manifest逐文件SHA校验；完整交付需要主清单VERIFIED、补充清单VERIFIED，以及主科学证据CPU RELEASE_GATE/SEAL通过。\n')
fixed=('# R3 科学报告：术语勘误整合版001\n\n本版只更正PushT success名称；详见FINAL_SCIENTIFIC_REPORT_ZH_ERRATUM_001.md。原报告及封存SHA保持不变，所有数值、区间和结论原样保留。\n\n---\n\n'+text.replace(wrong,correct))
names=['reports/FINAL_SCIENTIFIC_REPORT_ZH_ERRATUM_001.md','reports/FINAL_SCIENTIFIC_REPORT_ZH_CORRECTED_001.md']
for n,b in zip(names,[note.encode(),fixed.encode()]):
 p=r/n;assert not p.exists()
 with p.open('xb') as f:f.write(b)
files={n:{'sha256':sha(r/n),'bytes':(r/n).stat().st_size} for n in names+[parent,old,source]}
d={'version':'R3_REPORT_TERMINOLOGY_SUPPLEMENT_V1','status':'IMMUTABLE_REPORT_SUPPLEMENT_READY','published_at':now,'parent_final_manifest_sha256':sha(r/parent),'corrected_report':names[1],'files':files,'scientific_or_statistical_results_changed':False,'original_published_files_changed':False,'new_updates':0,'new_environment_steps':0,'source':source,'CPU_gate_for_parent_scientific_evidence_still_required':True}
p=r/'manifests/RECOVERY_REPORT_ERRATUM_001.json'
with p.open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'path':str(p.relative_to(r)),'sha256':sha(p),'files':files,'status':d['status']}))
'''
compile(code,'remote_erratum_code','exec')
d=remote_python(code)
p=Path(__file__).with_name('EXECUTION_CONTINUATION.json');c=json.loads(p.read_text());c['report_erratum']=d
c['report_erratum']['reason']='Report terminology only: actual fixed position-angle PushT success was incorrectly called coverage. Original source, results and intervals unchanged.'
c['next']=['Current final main transfer normal; do not duplicate or change published bytes. Wait main VERIFIED and transfer exit.',
 'Then serially recover scripts/recover_r3.py --manifest manifests/RECOVERY_REPORT_ERRATUM_001.json with SHA in report_erratum; do not collide with main canonical writes.',
 'After BOTH main and supplement VERIFIED, run external frozen V2 CPU acceptance using recordedvenv. No training/re-evaluation or relaxed tolerance.',
 'Only if main RELEASE_GATE+SEAL PASS and supplement fullyVERIFIED create separate delivery completion receipt binding both manifest SHAs, gate/seal SHAs and corrected report SHA. Link reports/FINAL_SCIENTIFIC_REPORT_ZH_CORRECTED_001.md to user.',
 'Report four statuses and one next suggestion; pause r3 heartbeat only aftercomplete; instance retained, no destroy.']
c['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(d,indent=2))
