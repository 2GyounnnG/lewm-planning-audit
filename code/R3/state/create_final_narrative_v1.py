"""Operational report assembly from completed, sealed R3 outputs; no new scoring."""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_r3 import remote_python, guard

guard()
code = r'''
from pathlib import Path
import os,json,hashlib,csv,io,subprocess,datetime,shutil,statistics
r=Path('/workspace/r3_official_lewm_predictor_refit')
assert os.uname().nodename=='6494ba5e1b1a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((r/n).read_text())
assert sha(r/'manifests/MODELS_AND_SELECTION_LOCK.json')=='cbf032b4471f1615cc48181b382fe0b99d399a648146d815d3941791a7ec4f88'
a=read('manifests/ANALYSIS_OUTPUTS.json');assert a['status']=='COMPLETE' and a['formal_training_updates']==180000
for name,item in a['files'].items():
 p=r/name;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name
train=read('state/FORMAL_TRAINING_COMPLETE.json');plan=read('state/PLANNING_FORMAL_COMPLETE.json')
assert train['actual_updates']==180000 and plan['complete_trajectories']==800
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
assert uuids==['GPU-75067296-2e84-cc3f-b93d-19a984e0d7eb','GPU-4eaf00b2-ac9c-6e9e-9718-51f1ca58d628']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True).strip();assert not gpu,gpu
ps=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
active=[line for line in ps if str(r) in line or any(k in line for k in ['scripts/run_planning.py','scripts/summarize_results.py','scripts/run_formal_training.py','r3.train_worker','r3.open_loop','r3.planning_worker'])]
assert not active,active
assert shutil.disk_usage(r).free>=30*(1<<30)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
exitdoc={'status':'ALL_R3_GPU_JOBS_EXITED','hostname':os.uname().nodename,'gpu_uuids':uuids,'active_compute_pids':[],'R3_processes':[],'checked_at':now,'remote_free_GiB':shutil.disk_usage(r).free/(1<<30),'instance_destroyed':False}
exitpath=r/'state/FINAL_GPU_EXIT.json'
if exitpath.exists():assert read('state/FINAL_GPU_EXIT.json')['status']==exitdoc['status']
else:
 with exitpath.open('x') as f:json.dump(exitdoc,f,indent=2);f.write('\n')
stats={t:read(f'artifacts/statistics/{t}/NUMERIC_RESULTS.json') for t in ['pusht','reacher']}
rows=list(csv.DictReader(io.StringIO((r/'tables/open_loop_summary.csv').read_text())))
training=list(csv.DictReader(io.StringIO((r/'tables/training_summary.csv').read_text())))
compute=read('tables/compute_accounting.json')
def table(head,body):
 return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in body)+'\n'
out=['# R3 官方 LeWM 预测器后训练：最终科学报告\n\n',f'生成时间：{now}。此报告仅使用已封存分析，ANALYSIS_OUTPUTS SHA256 `{sha(r/"manifests/ANALYSIS_OUTPUTS.json")}`。科学计算已完成；交付验收状态单独由最终 RELEASE_GATE 与 SEAL 表示。\n\n',
'本轮主要结论为 **PREDICTION_GAIN_WITHOUT_ESTABLISHED_PLANNING_GAIN**。Reacher 的开环误差大幅下降，但闭环成功率没有净提升；PushT 的闭环点估计提高3个百分点，预定区间仍包含0。因此不能确立两个任务的共同控制增益，也不能把区间包含0解释成等效或证明毫无作用。任务点估计方向不同是描述性观察，本轮没有另做任务交互检验。\n\n',
'**执行与模型身份。** 两任务确实使用作者原始检查点，各3个后训练seed（103201/103202/103203）、每个30,000更新，6/6共180,000正式更新完成。只更新predictor与pred_proj参数；encoder、观测projector、action encoder及所有BN运行buffer保持冻结。每任务H0与三个固定30k模型共享100个起点—目标case和同一CEM协议，共800/800正式闭环；H0只运行一次并在配对比较中复用。三seed是同一官方起点的后训练重复，不是独立预训练重复或ensemble policy。\n\n',
table(['任务','作者checkpoint revision','原权重SHA256'],[
['PushT','22b330c28c27ead4bfd1888615af1340e3fe9052','48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'],
['Reacher','62adae4b71dc474ddf8f794c476ebfe737a743ca','eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469']]),
'\nLeWM源码固定8edfeb336732b5f3ce7b8b210d0ba370a09e2cac；兼容官方SWM固定abdced49809d5eae38e24b27dc7b635c502c4812。后者是按源码时间及真实接口核验的兼容版本，未证明就是原论文精确依赖。本轮保持官方任务/规划语义，不声称复现原论文总体百分比。\n\n',
'**闭环原值与配对推断。** 下表success分母均为100；救回是H0失败/refit成功，损失是H0成功/refit失败。没有删除困难case，没有METHOD或未完成基础设施失败混入正式评分。\n\n']
control=[];intervals=[]
for t,s in stats.items():
 for p in s['paired_counts']:
  control.append([t,p['arm'],p['H0_successes'],p['REFIT_successes'],p['delta_success_pp'],p['s01_H0_failure_REFIT_success'],p['s10_H0_success_REFIT_failure'],p['s11_both_success'],p['s00_both_failure']])
 for scheme,d in s['schemes'].items():
  if d['status']=='COMPLETE':
   m=d['metrics']['FIXED_THREE_REFIT_MEAN_minus_H0_pp'];intervals.append([t,scheme,m['estimate'],m['ci95'],m['ci97_5_two_task_bonferroni_approximation']])
  else:intervals.append([t,scheme,d['status'],'不可用','不可用'])
out += [table(['任务','refit seed','H0成功','refit成功','差值pp','救回','损失','共同成功','共同失败'],control),'\n',table(['任务','抽样单位','固定三seed平均差pp','95%条件区间','两任务97.5%近似区间'],intervals),
'\nPushT固定三seed均值94%，H0为91%；Reacher均值84.33333333333333%，H0为86%。统计保留5,000次共同配对重采样，未重采样训练seed。PushT有37个保守初始条件族，整族敏感性并列报告；Reacher来源族未知，不能把episode ID假作可证独立family。95%和97.5%区间条件于当前官方起点、数据和固定三个模型；97.5%为预定Bonferroni近似展示，不是有限样本严格FWER保证。PushT H0已91%，可改善空间为9个百分点；Reacher86%，空间14个百分点。高基线限制可见增益，但本轮不能把ceiling作为没有确立控制提升的唯一解释。\n\n',
'**开环预测。** h=1/2/5个latent转移对应5/10/25个raw环境步，Reacher内部action_repeat=2另计。三帧初始历史与记录动作属于离线预测合法输入；自由递归不回填真实未来观测。以下同坐标原始latent MSE均值直接取封存表；固定TRAIN方差归一值、全部3k/10k/30k和逐case原值见OPEN_LOOP_RESULTS及tables，不跨任务平均latent误差。\n\n']
ol=[];improvements=[]
for t in ['pusht','reacher']:
 for arm in ['H0','REFIT_103201_30000','REFIT_103202_30000','REFIT_103203_30000']:
  vals=[next(float(x['mean_latent_raw_MSE']) for x in rows if x['task']==t and x['arm']==arm and int(x['horizon_macro'])==h) for h in [1,2,5]]
  ol.append([t,arm]+vals)
 for h in [1,2,5]:
  base=next(float(x['mean_latent_raw_MSE']) for x in rows if x['task']==t and x['arm']=='H0' and int(x['horizon_macro'])==h)
  ref=statistics.mean(float(x['mean_latent_raw_MSE']) for x in rows if x['task']==t and x['arm'] in ['REFIT_103201_30000','REFIT_103202_30000','REFIT_103203_30000'] and int(x['horizon_macro'])==h)
  improvements.append([t,h,ref,100*(1-ref/base)])
out += [table(['任务','模型','h1 MSE','h2 MSE','h5 MSE'],ol),'\n',table(['任务','h','三固定refit均值MSE','均值相对H0下降%'],improvements),
'\nPushT h1三个seed均改善；h2的103201、h5的103201与103203略变差。Reacher三个seed在三个报告horizon上均显著降低误差量级，但这里的“量级下降”是原值描述，不是额外显著性检验。它在固定800条控制结果中未带来净成功率提升，说明本范围内更低的记录动作分布预测误差不足以保证CEM控制收益。开环合法历史长度3，而官方在线policy历史长度1，动作分布亦不同；本设计未分离这些原因。均值/方差/范数漂移保留为描述性诊断，不能据此认定唯一因果机制。\n\n',
'**闭环物理误差与实际规划成本。** 下表均为每任务每arm的100例算术均值，完整分布、终点/任意步success、动作和调用数见逐case表及NUMERIC_RESULTS。PushT goal error为作者eval的7维完整state L2（含agent velocity），与覆盖率success并非同一个量；Reacher qpos_match success要求两个关节绝对角差均小于0.05，不是普通point-reaching奖励。\n\n']
aux=[]
for t,s in stats.items():
 for arm,d in s['auxiliary_descriptions'].items():
  aux.append([t,arm]+[d[k]['all_case_mean'] for k in ['final_goal_error','executed_raw_steps','replan_calls','planning_synchronized_wall_seconds','trajectory_wall_seconds']])
out += [table(['任务','arm','终点goal error','raw steps','replan次数','规划同步秒/例','完整轨迹秒/例'],aux),
'\n规划预算保持horizon5、receding5、action_block5、300候选×30迭代、top30、var_scale1，每次最多展开25条raw action，总预算50 raw steps。没有换规划器、增加预算或按结果选checkpoint。各arm耗时受终止步数及重规划次数影响，不代表后训练获得规划算法加速。\n\n',
'**离线算量与耗时。** 正式180,000更新消耗23,040,000个采样窗口、69,120,000个预测target token、345,600,000次raw action暴露，均含重复采样；不等于独立样本数。技术optimizer恰为320/1024，隔离且未继承；完整TECH CEM为8/16，与正式800条分开。正式FP32、AMP/TF32关闭，batch128，4训练worker（每GPU2）；规划每GPU1worker。\n\n',
table(['任务','seed','正式更新','worker秒','compute秒','保存秒'],[[x['task'],x['refit_seed'],x['actual_updates'],x['worker_seconds'],x['compute_seconds'],x['checkpoint_seconds']] for x in training]),
f'\n正式CEM controller记录墙钟{plan["controller_wall_seconds"]}秒。两任务缓存编码1645.52245789906与1517.8251990997232秒；开环forecast/scoring分别152.91366303991526与153.51798638375476秒。源码/数据/模型全SHA核验、初始化与保存成本另列在计算账；worker并行且分项嵌套，不将上述总和冒充端到端墙钟。正式pre-loop setup和全研究总墙钟的统一计量仍为unknown，未填0；数据下载、旧预训练与开发成本没有被抹去。本轮不声称总计算更省。\n\n',
'**暴露、复现与理论边界。** 原官方模型card指向原训练来源，精确episode预训练暴露未知；R3 REFIT_TRAIN/MONITOR/TECH/EVAL隔离仅意味着本次refit留出，不是整个模型从未见过TEST。动作归一统计来自完整原训练来源的有限动作，包含后来留出的输入，但未用EVAL物理目标调参。PushT100例中6条曾在旧G1所选896内，旧XS/lambda/结构开发史不撤销；Reacher源seed和family未知。两任务用有限TECH验证后的固定seed0复位，并非恢复原采集seed。PushT复位会推进一个dt，原像素/状态并非精确原轨迹复刻。全部限制及失败技术尝试原值在技术报告与manifest保留。\n\n',
'这次干预是冻结官方表示后的预测映射继续优化，不产生新表示正则方法；SIGReg在冻结观测表示上对允许参数没有梯度。更好的预测未普遍转化为已确立的控制增益是合法结果。不据此否定JEPA，不归因或推翻未经本轮比较的LEAP/LeFlow/SAGE/BWM。没有金融、EAAI、LIBERO、新PDE或旧scope矩阵；KBS/ESWA只是可能的后续AI写作方向，不构成录用保证。\n\n',
'**完整性与交付。** 全部原始训练journal、monitor逐窗口值、100例×10arm×两任务预测、800条真实闭环动作/物理state/goal原值、共享bootstrap索引/分布、八组PNG/SVG图与紧凑delta均保留。固定3k/10k里程碑仅开环诊断，没有追加CEM选择。科学结果由ANALYSIS_OUTPUTS封存；最终RECOVERY_FINAL、RELEASE_GATE、RELEASE_GATE_SEAL分别负责传输清单、实际CPU恢复和封印。最终报告不预先宣称验收PASS。\n\n',
'交付路径技术修订单独登记：V1以全树basename识别报告，与不可变旧R2报告及BASE接收快照重名。V2仅严格识别reports/下九个本轮报告，保留旧V1、所有历史字节、原模型锁和全部科学/统计/数值门；V2两个辅助源码及测试/独立审查证据纳入最终清单，不改原科学脚本或容差。一次Reacher暂存文件传输损坏已按原SHA重新回传并整单校验，失败字节/日志保留；不能据此断言文件系统原因。两次模型锁操作性失败（reports相对路径、缺原tests部署）及原证据亦保留，不是重训或结果筛选。\n\n',
'最终只保留一个下一步建议：以现有封存结果整理“记录分布预测改善与固定CEM控制收益不等价”的有限机制研究稿，明确暴露和统计边界。本轮停止新科学计算，不自动扩训；完成外置验收后由用户决定实例去留，不自动destroy。\n']
text=''.join(out);p=r/'reports/FINAL_SCIENTIFIC_REPORT_ZH.md';assert not p.exists()
with p.open('x') as f:f.write(text)
receipt={'status':'FINAL_SCIENTIFIC_NARRATIVE_CREATED_FROM_SEALED_ANALYSIS','checked_at':now,'analysis_sha256':sha(r/'manifests/ANALYSIS_OUTPUTS.json'),'report_sha256':sha(p),'report_bytes':p.stat().st_size,'science_label':'PREDICTION_GAIN_WITHOUT_ESTABLISHED_PLANNING_GAIN','new_optimizer_updates':0,'new_environment_steps':0,'acceptance_not_yet_claimed':True}
with (r/'state/FINAL_NARRATIVE_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
'''
print(json.dumps(remote_python(code), indent=2, ensure_ascii=False))
