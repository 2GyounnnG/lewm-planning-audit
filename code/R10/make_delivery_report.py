"""Build final Chinese delivery summary from sealed module statistics."""
import csv, datetime, hashlib, json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(v):return '不可估计' if v is None else f'{100*v:+.2f}'
stats={m:json.loads((root/'modules'/m/'STATS.json').read_text()) for m in 'ABDC'}
audits={m:json.loads((root/'ops'/f'{m}_DELIVERY_AUDIT.json').read_text()) for m in 'ABDC'}
assert all(a['status']=='PASS' for a in audits.values())
names={'E_A1':'Reacher：原生 history_len=3 相对1（仅H0）','E_A2':'新旧版本单帧差异（描述性）','E_B':'TwoRoom：真实历史','E_D1':'Reacher：真实帧＋零动作','E_D2':'Reacher：重复当前帧＋真实动作','E_C':'Cube：真实历史'}
table=[]
for m,s in stats.items():
    for e in s['endpoints']:
        table.append(dict(module=m,endpoint=e['endpoint'],comparison=names[e['endpoint']],
         point_pp=None if e['point'] is None else e['point']*100,
         ci95_low_pp=None if e['ci95_low'] is None else e['ci95_low']*100,
         ci95_high_pp=None if e['ci95_high'] is None else e['ci95_high']*100,
         label=e['label'],complete_cases=e.get('complete_cases',e.get('cases')),
         descriptive_only=e.get('descriptive_only',False)))
with (root/'R10_MAIN_TABLE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
lines=['R10 历史补齐评测交付','统计单位：case；每个case先平均模型/规划随机流，5000次case bootstrap，百分位95%区间。数值为成功率差的百分点。','']
for m,s in stats.items():
    for e in s['endpoints']:
        lines.append(f"{e['endpoint']} {names[e['endpoint']]}：{pct(e['point'])}，95% CI [{pct(e['ci95_low'])}, {pct(e['ci95_high'])}]，{e['label']}，完整case {e.get('complete_cases',e.get('cases'))}/100。")
    if m=='A':
        lines.append(f"交付{s['delivered']}/{s['planned']}；首计划门{s['first_plan_pairs_equal']}/{s['planned_first_plan_pairs']}逐位相等。与预注册预测{'一致' if s['prediction_consistent'] else '不一致'}。")
    else:
        consistency='不预设方向，无需判定' if s['prediction_consistent'] is None else ('一致' if s['prediction_consistent'] else '不一致')
        lines.append(f"新增交付{s['total_new_delivered']}/{s['total_new_planned']}（含{s['fresh_controls_delivered']}条同机单帧对照）；历史/干预臂交付率{s['delivery_rate']:.1%}，首计划门{s['first_plan_equal']}/{s['planned']}逐位相等。预测：{s['prediction']}；一致性：{consistency}。")
        lines.append(f"同机新单帧相对旧封存单帧的成功率漂移（描述性）：{pct(s['descriptive_control_drift'])}个百分点。")
    lines.append('')
lines.extend([
 '对论文表述的影响：',
 'A：报告原生history_len=3在官方H0上的实测增益；不据此声称新旧规划器完全等价。',
 'B：TwoRoom未检出增益，与离线单帧信息较充分的预测一致；未检出不等于证明效果为零。',
 'D：分别报告真实帧与动作前缀的干预结果；两臂帧与动作不配对，属于分布外输入，不据此提出部署建议或确定单一因果成分。',
 f"C：Cube真实历史的结果为{stats['C']['endpoints'][0]['label']}，应据实限定“单帧不缺信息、历史没有增益”的表述；随机动作48.0%仅作参照。",
 '',
 '事先记录的协议调整与限制：',
 '1. B/D技术验证发现旧封存结果的跨机器首计划有约1e-6的差异；原始R8单帧实现也复现该差异。因此B/D/C在各自正式运行前统一登记全量同机新单帧对照。Cube的4个TECH首计划与旧封存结果全部相等，采用新对照是正式运行前的统一设计。没有放宽逐位相等门，也没有按结果筛选对照。',
 '2. A固定版本的原生首调用实际使用1帧且无action_history，随后增长至真实历史；没有按提示文件的“复制首帧＋零动作”描述改写库。完整探针保留在TECH结果中。',
 '3. Cube保留CUBE_OFFICIAL_RESET_SYMMETRIC_NONEXACT限制；全局交付率90%停止门在运行前登记，正式轨迹不补跑。',
 '4. A完成必做H0；未执行优先级较低的可选refit扩展。B/D/C均使用H0和三个固定30000步refit，未训练或微调。',
 '5. 技术缺失不补零；B/D/C主分析只使用该终点12个模型/随机流配对齐全的case，A要求H0三条随机流配对齐全。',
 '',
 '目录：modules/<A|B|D|C>含RAW_VALUES.csv、MAIN_TABLE.csv、SEAL.json、CONCLUSION_ZH.txt；raw保留轨迹和逐条收据；prereg为运行前登记；tech保留所有技术尝试；ops含环境冻结、完成收据及独立校验。',
 '主目录：/Volumes/MyProj/lewm_planning_audit/r10_history_completion/',
 '镜像：/Users/richwang/Documents/ChatGPT/热/r10_execution/',
 'GPU实例由作者手动管理；本任务不停止或销毁租用实例。'
])
(root/'R10_CONCLUSION_ZH.txt').write_text('\n'.join(lines)+'\n')
summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),modules=stats,
 total_new_planned=sum(s.get('total_new_planned',s['planned']) for s in stats.values()),
 total_new_delivered=sum(s.get('total_new_delivered',s['delivered']) for s in stats.values()),
 audit_receipts={m:sha(root/'ops'/f'{m}_DELIVERY_AUDIT.json') for m in stats},
 module_seals={m:sha(root/'modules'/m/'SEAL.json') for m in stats},
 preregistrations={m:sha(root/'prereg'/f'MODULE_{m}_FORMAL_PREREG.json') for m in stats},
 training_updates=0,instance_auto_stopped=False)
(root/'R10_DELIVERY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['total_new_planned','total_new_delivered','training_updates']},ensure_ascii=False))
