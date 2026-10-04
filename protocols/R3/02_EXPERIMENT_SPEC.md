# R3-v1技术规格：官方检查点、预测器后训练、固定CEM

本文件是本轮新增设计，不是官方LeWM原实验的逐项复现声明。与主prompt合读。若两者冲突，停止受影响部分并报告，不按结果选择解释。

## 1. 样本与模型的层次

| 对象 | 默认计划 | 解释 |
|---|---:|---|
| 官方任务 | 2 | PushT、DMControl Reacher qpos_match |
| 原官方起点 | 每任务1个 | 不通过评估选版本 |
| 后训练重复 | 每任务3个 | 同一官方表示的优化/采样重复 |
| 正式训练 | 6×30,000 | 180,000 optimizer updates |
| TECH optimizer | ≤1,024 | 完全隔离，不进入主初始化 |
| TECH闭环轨迹 | 总≤16 | 不能进入正式成功率 |
| 每任务正式case | 目标100 | 尽量唯一episode，不足时按第3节降级 |
| 正式闭环trajectory | 2×4×100=800上限 | 每任务H0一次、3个refit各一次 |
| 官方encoder再训练 | 0 | projectors/normalizers的冻结见第2节 |
| 新state decoder、少标签head | 0 | 这轮不重复标签效率矩阵 |
| bilinear、新scope、其他规划器 | 0 | 不自动扩大 |

不要将6个后训练模型称为6个独立pretrained encoder，也不要将100×3当成300个独立起点。

## 2. “预测器”的精确定义与梯度路径

本轮预测映射写作：

`z = observation_projector(visual_encoder(image))`

`a_emb = action_encoder(normalized_macro_action)`

`z_hat = pred_proj(predictor(z_history, a_emb_history))`

全部可见表征坐标与action embedding固定。训练白名单为`predictor`及`pred_proj`参数，含预测侧BN affine；所有BN的running buffers固定。观测projector完全冻结，不能把它与预测侧pred_proj混淆。

实现顺序建议：
1. model.eval(); 全部requires_grad_(False)。
2. 只给白名单参数requires_grad_(True)。
3. predictor/pred_proj按训练模式管理dropout；再次将所有BatchNorm模块设eval以锁running统计。
4. optimizer严格接白名单去重后的参数列表。
5. 每100步验证冻结module hash或周期末完整hash；最终完整核验。

缓存来自`visual_encoder+observation_projector`在eval模式的实际官方输出；不额外白化、移除均值、换CLS/patch坐标、归一化到球面，也不复用自有G1 encoder latent。

SIGReg若只作用于冻结z，对白名单无梯度，训练不用计算该常数项。不能把predicted z强制高斯化，这会引入另一实验因素。

H0和全部refit推理时均eval；同一图像、action、goal输入转换完全相同。

## 3. 本轮数据角色与确定性划分

### 3.1 来源与暴露先记录

模型card链接的原train数据、官方独立validation/test若存在，均记录来源revision与内容身份。三个状态独立：
- `OFFICIAL_PRETRAIN_EXPOSURE`：明确见过 / 明确未见 / 无法核实。
- `R3_REFIT_EXPOSURE`：是否参加本轮权重更新。
- `PRIOR_USER_STUDY_EXPOSURE`：是否曾在G1/R2等被观察。

官方config从训练数据评价时，不能因本轮排除其再拟合就改称原模型独立TEST。报告“原分布、对本轮后训练留出”的含义。数据合格性依据schema、可重置状态、长度和身份，不依据任务成功或误差。

### 3.2 有官方独立评价数据

若有可核实的独立评价集合，优先从该集合建EVAL与TECH，训练只用作者train集合。TECH先固定，EVAL不包含TECH。官方原预训练是否真的排除该集合仍须来源证据。

### 3.3 无独立评价数据的默认规则

从官方train来源的全部合法episode中，按可核实的family分组后，用固定盐`R3_GROUP_SPLIT_20261002`的SHA256排序，最后约20%的groups划为EVAL_POOL，其余为FIT_POOL。只有episode信息时以episode为group，显式记`FAMILY_UNKNOWN`。

FIT_POOL中再按固定盐选择约10%的groups作MONITOR，其余REFIT_TRAIN。所有seed共享同一划分。

EVAL_POOL按group-round-robin分配：先取4个不同episode作TECH；剩余最多100个不同episode作正式EVAL。正式case只从选定episode里按固定hash选一个合法start；不从一个episode取多对来凑100。

若源数据实际不足，优先保护组隔离：记录实际数量并完成现有合法case，使用`LIMITED_EVALUATION_SAMPLE`；不能根据风险重拆family。若少于20个合法EVAL episode或无法建立非空REFIT_TRAIN，则先报告具体数据阻塞，不宣称完整两任务确认。

记录每个case的source ID、family ID/unknown、episode哈希、start raw index、goal raw index、goal offset、环境初状态、任务goal信息和随机seed。配置与选择名单先冻结，之后才读模型风险。

### 3.4 动作与像素归一化

优先检查checkpoint或原资产保存的normalization；没有单独保存时，从明确的原training source按作者对应实现重建。记录样本/总体std约定、NaN边界、原生动作单位。

注意公开train/eval脚本的归一化实现不完全同名就代表完全相同。目标是本轮H0与refit的输入域一致并接近可核实官方管线，而不是悄悄“修正”H0使其变弱或变强。

不得用R3 EVAL物理目标拟合任何统计。若重建stats使用了全部官方原train观测，明确包含的source暴露，不能称为严格从未见过的评价总体。

## 4. 官方值锚点与必须实例化的单位

以下值来自2026-10-02本次读取的官方公开配置；执行前固定实际commit与权重合同。它们不是本包已完成复现的证明。

| 字段 | PushT | Reacher |
|---|---|---|
| env | swm/PushT-v1 | swm/ReacherDMControl-v0 |
| task | 依官方PushT | qpos_match |
| 图像 | 224 | 224 |
| goal_offset_steps | 25原始步 | 25原始步 |
| eval_budget | 50原始步 | 50原始步 |
| plan horizon | 5 | 5 |
| receding_horizon字段 | 5 | 5 |
| action_block | 5 | 5 |
| CEM num_samples | 300 | 300 |
| CEM n_steps | 30 | 30 |
| CEM topk | 30 | 30 |
| CEM var_scale | 1.0 | 1.0 |

`receding_horizon`实际由policy怎样转换成原始动作条数必须读源码，不能凭表自行解释。原脚本num_eval=50、seed=42；本轮将num_eval改成固定最多100个case并使用显式case seeds，这是已登记的评价样本扩展，不是原论文数值逐点复现。

macro-action内各个原始动作按先后完整拼接，禁止平均、取首动作或误把block=5变成每步重复同一个动作。环境action bounds/内部控制器语义完全使用锁定版本。

训练history_size、latent维度、预测侧输出形状按官方checkpoint config核实。通常三帧历史、一帧预测，但不能不看config硬填。

## 5. 起点、历史和goal合同

官方重置/历史padding/goal生成的真实方式先复现；若policy在第一轮重复当前图像补历史，H0与refit都保留，不能某一arm获得专家过去轨迹而另一arm只得重复帧。

若为离线开环使用source真实L帧历史，需明确它与闭环起点历史的区别。禁止把goal图像伪装为历史中的未来帧。

环境真状态可用于`set_state`和`set_target_qpos`等重置API，必须由evaluation接口隔离；planner只得pixels、goal pixels、合法已执行动作历史。不要给planner正确的未来专家动作序列。

同case各arm起点和goal完全相同，策略状态清空；执行后每个arm分别在自身动作驱动的环境里运行，不每一步重置到录制的下一状态。

## 6. 预测器后训练数据和损失

每个REFIT_TRAIN合法窗口包含原history_size加一个未来macro观测；保留该窗口相应动作块。按锁定官方预测目标对齐产生teacher-forced输出，所有目标都来自冻结encoder，目标无梯度。

若原predict()返回多个历史token的下一时刻预测，沿用正确的全token一步对齐；不可因截取最后一个token改变预测token预算而不登记。技术检查用确定性人工状态/动作索引验证所有shift。

窗口均匀抽样，单独记录每个episode与每个raw/macro transition的使用次数。训练状态标签、reward、goal任务标签一律不用。开启官方predictor原dropout；不存在自适应window/目标难度筛选。

缓存至少覆盖所有REFIT_TRAIN/MONITOR合法时刻，可按原始episode分块建立。磁盘估算按`sum(encoded frames)×latent_dim×4 bytes`加索引，不按压缩视频原大小盲猜。中断可按episode恢复，不复制第二份像素数据。

## 7. 严格固定训练预算

2任务×3个seed，种子103201/103202/103203；同task所有seed起始权重相同，数据顺序和dropout随机数不同。

30k额外update；新AdamW状态；500step warmup/cosine；默认5e-5→5e-6，wd1e-3，clip1；batch128；FP32。公开原trainer的bf16设置与本轮区别必须写进protocol delta。

只在正式前允许固定microbatch及数值无关的缓存/设备调度优化。禁止换decoder、rank或训练目标；关闭TEST early stopping。

保存last+3k/10k/30k的predictor/pred_proj delta，last若与30k同hash用alias；resume保存optimizer及所有RNG。MONITOR每1000step记录相同固定窗口的teacher-forced与自由5macro步loss；它不是后验延训依据。

## 8. 正式开环评价

以同一EVAL名单的episode构造共同合法历史锚点。主horizon 5个macro转移，通常25原始步；辅助1、2macro。若一个规划case起点没有足够的离线历史，在episode内按事前hash另选共同有效离线锚点，记录`OPEN_LOOP_ANCHOR_NOT_PLANNING_START`，不能删除失败case。

target是真未来图像经同一冻结encoder得到的z，只在评分时读取；forecast只从起点历史和该段记录动作递归生成。

每case每horizon保存：原坐标latent MSE、TRAIN标量方差归一化MSE、预测和target范数、有限值标记。各arm使用相同坐标，因此可比；不把各自再归一化得到的更小值当预测改善。

H0和每个seed的3k/10k/30k可做这些离线比较；正式规划只H0和fixed30k。两任务报告分开，不用不同latent风险直接比较哪任务提升更大。

## 9. CEM执行、随机流与计时

每任务case100、模型H0/3seed，共400条。一次完整trajectory定义为从reset到官方success/time budget终止，不是一次CEM求解。

固定随机seed：从`R3_CEM_CASE_20261002/task/case_id` SHA256派生64bit整数；环境随机流、CEM随机流、显示视频随机流分离。每个case每次重规划的随机流最好由(case_id, replan_index)确定，确保worker调度或某模型提前成功不改变后续case样本。

不同模型观察/候选分布可分化，公共基础随机数不保证最后动作相同；这是正常的。但完全相同模型的clone必须通过精度/决策一致检查。

科学预算固定：300候选×30迭代、top30等。GPU候选microbatch或同时环境数可作纯执行优化，但不能减少候选、缩短horizon、提前根据success rate退出整个矩阵。

同一case的各模型尽量在同一GPU上串行执行，跨case并行；按case hash平衡两GPU。主正式计时每GPU至多一个planning进程，避免与训练混跑造成时间不可比。训练和评价阶段可按任务独立，但用于方法时间比较的数据须标明是否同机竞争；只用无竞争正式段作主时间结果。

每trajectory保存总墙钟、GPU推理/编码/solver计时、环境耗时、replan次数、执行raw步数、各步原始actions、goal error和success。首个JIT/kernel预热独立计入setup，不从总研究账删除。

对耗时更少的success early stop，用每episode时间和每次规划时间并列，不说模型单步加速了。需要固定长度比较时仅使用已有同预算计时，不新增后验轨迹。

## 10. success统计与计数

对任务k、case i：`d_i=(s_i_seed1+s_i_seed2+s_i_seed3)/3 - s_i_H0`。

`delta_success_pp=100*mean_i(d_i)`。

逐seed另保存(s01成功/s10失去成功/两者成功/两者失败)配对计数。三模型的平均不是ensemble policy，每个模型独立控制。

普通bootstrap同case所有arm共用采样索引，5000次；family敏感性按family整簇采样后按episode加权。若一个family内episode大小不等，用重采样总差/重采样总case数保持同一episode等权estimand。

主任务两项只报告task-wise结果。若写“两任务均有证据”，给各任务97.5%区间(1.25%,98.75%)作为Bonferroni近似口径并保留95%区间；不能因family结果跨0只展示episode显著性。簇太少应注明不稳定，不把显著性做技术gate。

可选描述理论最大可提升=100−H0成功率，但不据此降低正式阈值，因为本轮无必须达到5/10pp阈值。

基础设施失败重跑同case同seed并保存失败日志，不能替换case。算法失败按0 success纳入；timeout是原50step预算定义，不使用随负载变化的墙钟早停决定失败。

## 11. 结论范围和不允许的推论

- 官方模型也改善：支持指定官方起点与本轮后训练配方中的现象，不证明所有官方检查点训练不足。
- 预测改善、控制不改善：说明当前离线预测指标不足以保证规划增益，不说明所有预测器训练无用。
- 控制改善：额外训练成本清楚保留；不宣称新算法/无额外成本。
- 数据原来参与官方预训练：标为source-distribution post-refit holdout，不声称整个模型从未见过。
- 不进行其他planner交叉：不能重写LEAP/LeFlow/SAGE论文的增益归因。
- 没有这轮少标签decode：不宣称新增标签效率结果。
- 未达5/10pp：可决定不继续投入当前候选，不等于否定整个JEPA方向。

## 12. 执行结束后的唯一扩展讨论

先完成规定两任务与归档。若要补TwoRoom/Cube或增强planner，必须另行说明具体缺口、真实计时和新授权。不能为满足“至少两个任务赢”而不停加任务，再只选有利者进入正文。
