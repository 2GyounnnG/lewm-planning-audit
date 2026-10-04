# Codex执行协议｜R3：官方LeWM预测器后训练与CEM规划验证

版本：2026-10-02 / R3-v1  
建议分支：`r3_official_lewm_predictor_refit`  
必须同时读取：`02_EXPERIMENT_SPEC.md`、`03_PRIMARY_SOURCES.md`。  
性质：实际执行，不仅准备；复用经核验的原G1/R2.1同一台2×RTX 5090。  
研究方向：AI/世界模型；不考虑EAAI，不加入金融。

## 0. 本轮授权及科学问题

用户明确同意续用尚未销毁的原双5090实例。本文件仅覆盖这一台经核验的机器，替代旧文件中“只能连接新SSH”的限制；不授权租卡、访问其他历史主机、杀死无关任务或销毁实例。

唯一主问题：

> 在官方发布的LeWM表示和现有CEM规划管线保持固定时，离线继续拟合预测器，能否改善开环预测，并转化为闭环控制收益？

允许实际执行：
1. 必要的实例接收与R2.1证据保护；读取旧产物，但不修改旧结论。
2. 下载/核实作者官方PushT与Reacher检查点及对应数据，复用相同内容的现有缓存。
3. 实现严格参数冻结、同坐标latent缓存和原预测管线的等价性检查。
4. 每任务一个官方起点，3个后训练随机种子，每个30,000更新；共6个作业、180,000正式更新。
5. 每任务固定最多100个起点—目标对；原官方模型与3个后训练模型用相同CEM协议评价，共最多800条正式闭环轨迹。
6. 固定开环评价、统计分析、实际算量/耗时报告、增量回收和独立恢复验收。

不授权：
- 新视觉编码器训练、改变表示、改变目标度量或学习奖励。
- TC/XS scope矩阵、双线性新方法、BWM联合复现、LEAP/LeFlow/SAGE复现、GN/L-BFGS交叉。
- TwoRoom/Cube自动补入、LIBERO、热学PDE、金融、测试时在线适配。
- 新lambda/LR搜索、按TEST选择checkpoint、增加seed或延长正式更新。
- 自动投稿、arXiv发布、发邮件、删除唯一旧产物、自动释放实例。

这是新的实验协议，不是对旧R2草案“已经预注册”的补录。不要求成功率必须提高5%或10%；技术错误与科学无效分开。结果不理想也完成已授权比较，不扩训追正结果。

## 1. 续用原双5090，保护历史证据

通过G1/R2.1实际部署收据和SSH配置确认：主机身份、SSH host-key指纹、GPU UUID、工作区和任务标记。只使用可唯一匹配的原主机；无法匹配才向用户索要当前SSH，不猜端口。

记录两卡、驱动、CUDA/PyTorch、运行进程及归属、CPU/RAM、磁盘和缓存路径。无关进程不得中止。

R2.1最新报告表示科学完成、回收待验收。先确认原文件处于不可变快照并已有可靠manifest：允许旧回收与R3并行，不要求重复下载全历史；若旧唯一产物面临空间或覆盖风险，先保护该部分。旧RELEASE_GATE通过不等于新R3完成后可退卡。

在旧目录旁创建独立R3目录，旧代码/结果只读。可复用工作环境；需要兼容依赖时创建隔离venv，不升级破坏G1/R2.1。保留≥30 GiB磁盘余量，数据、缓存与官方权重只存一份共享副本。缺空间只停止新分配并报告，不自动删除旧证据。

本轮不设美元或墙钟截止；范围有限。不得承诺1–2天完成。CEM评估吞吐必须实测。

## 2. 源记录接收：官方模型不能被我们自己的XS模型替代

必要历史只读：
- R2.1 `FINAL_SCIENTIFIC_REPORT_ZH.md`、实际参数冻结/精度/恢复合同及硬件收据；
- G1真实数据版本、族映射、源码和动作宏步接口；
- 本包来源表和执行规格。

本轮模型起点：
- `quentinll/lewm-pusht`
- `quentinll/lewm-reacher`

这两个名称在作者LeWM README中有明确链接。执行时从该链条取具体revision和实际文件，锁定仓库SHA、权重SHA256、配置SHA；不要凭同名第三方仓库代替作者模型。

官方README公开了state_dict/config与object checkpoint两种接口。优先安全的state_dict严格加载；需要object格式时，用核实过的源码在本地构造，禁止下载未知pickle后直接运行。

每任务只用一个正式作者检查点，不在多个epoch或版本里按表现选最佳。优先作者默认/明确指定最终模型；若两个不同最终候选无法由来源确定，列明歧义并停该任务选择，不任意挑。3个seed是同一官方起点的后训练重复，不是3次独立官方预训练。

最终`OFFICIAL_ASSET_MANIFEST.json`必须列出原检查点、转换结果、严格load结果、数据名称/revision、预训练暴露状态和环境版本。

## 3. 官方协议核对，不用默认值猜“复现成功”

阅读并锁定作者`jepa.py`、`module.py`、`train.py`、`utils.py`、eval.py、两个eval config、CEM config及实际数据schema。记录stable-worldmodel及其依赖commit。

本包提供2026-10-02公开配置的数值锚点；完整语义以执行端核实后的固定源码为准。若有实质版本差异，先登记具体差异；不能一边沿用旧名称一边使用另一任务/另一控制预算。

必须核实：
- 图像resize/normalize顺序、范围、插值；
- encoder.projector的真实latent坐标和历史长度；
- raw动作到macro-action的拼接、单位、归一化/反归一化和环境裁剪；
- predictor输出及pred_proj的角色；
- 目标图像/goal embedding构造；
- 起点重置、历史补齐、目标设置、success判定和停止时刻；
- CEM候选数/迭代数/elite/初始方差/动作块/重规划策略。

不得为了复现某个论文成功率而调参数。基线百分比不同可以是来源/协议差异，先排查可证明的接口问题；不是必须达到一个预期成功率才允许训练。

特别防止：Reacher `qpos_match`不是普通distance-to-point reach奖励；PushT主success不得改成agent到点。环境内部真状态只用于复位、目标生成与评价，不输入预测器或规划代价。

## 4. 只改变预测映射：精确冻结白名单

根据本轮定义：

- 冻结`encoder.*`、观测`projector.*`、`action_encoder.*`的全部参数与buffer；
- 所有BatchNorm的running_mean/running_var/计数固定，训练中不更新；
- 可训练参数只有`predictor.*`和预测输出投影`pred_proj.*`的参数；后者属于预测映射，不是观测编码器；
- 若pred_proj内有BN，其running统计冻结，affine参数属于白名单；
- 正式训练前导出每个可训练parameter的完整名称、shape、参数数与hash。

全局model.train()后必须重新把冻结子模块和BN切回eval。冻结路径无梯度，训练前后字节hash相同；不因缓存生成改变BN统计。forward/backward smoke必须证明白名单外无参数更新。

原参考H0完整保留。H0_CLONE是同权重的纯包装/序列化等价性测试，不是新方法。

官方SIGReg施加于观测表示，表示冻结后它对允许训练参数无梯度。因此本轮只优化原一步预测MSE，不增加输出Gaussian约束，不通过删改权重重新定义一种scope方法。若实际source显示不同梯度路径，先查清，不照抄这个结论。

## 5. 数据与评估身份：不是原G1的896条就自动正确

复用数据必须满足source版本、原始episode内容、动作含义和像素转换一致；旧G1 latent由我们自己的编码器产生，不能喂给官方检查点。本轮每任务重新以官方冻结encoder/projector编码一次，可由三个后训练seed共享。

优先使用作者分离的train/evaluation数据。若官方eval仍从作者训练数据抽起点，保留这个事实：可以研究“官方分布上的控制”，不能称整套模型未见TEST。

本轮在实际source中额外分开REFIT_TRAIN、MONITOR、TECH、EVAL，优先按可核实family隔离，详细确定性规则见规格。EVAL不参与任何本轮再拟合/选模；不代表排除了原官方预训练的暴露。

目标最大规模：每任务100个唯一episode上的起点—目标对；可用身份不足时按规格完整保留实际数量并明确降级，不重复一个episode冒充更多独立案例，不新增更容易的数据。正式起点/目标根据ID与长度等元数据确定，不看模型成功率筛选。

所有图像变换、action归一化来自官方固定流程/原train来源，一次锁定供H0和R3所有模型共用。不用EVAL状态拟合scaler/decoder；本轮没有新状态decoder。

## 6. 训练：六作业，180,000更新

任务PushT/Reacher，各3个后训练seed：`103201,103202,103203`。全部从对应任务同一个官方checkpoint出发。

主配方：
- AdamW，lr=5e-5、weight_decay=1e-3、betas=(0.9,0.999)、eps=1e-8；
- 新optimizer，不冒称从官方joint optimizer完整恢复；
- 500更新线性warmup，之后cosine到5e-6；
- global grad clip=1；
- effective batch=128，FP32，AMP/TF32关闭；
- 每job30,000更新，fixed30k为唯一主checkpoint；
- 存3k/10k/30k，里程碑只做固定离线预测诊断，不额外跑正式CEM矩阵；
- 使用原一步teacher-forced预测目标/对齐，所有预测token和数据采样次数独立记账。

这些是本轮的固定后训练配方，不声称等同官方全部端到端训练。当前公开官方trainer采用bf16，本轮FP32是明确的技术选择；H0/REFIT正式推理精度完全一致。

microbatch不足时，只在正式首步前以128=64×2或32×4确定积累；所有匹配作业一致。BN统计固定，不跨batch更新。不能改latent维度、图像尺寸、历史长度或CEM候选数来挤显存。

缓存latent使正式predictor训练不重复跑ViT；无随机增强时缓存逐值核验。若实际官方变换含随机增强，必须先固定本轮deterministic预处理并登记，不能悄悄把有增强训练称为完全等价缓存。

每任务MONITOR仅诊断曲线，不选学习率、checkpoint或延训。训练不收敛要报告，不因30k自动宣称原模型“充分优化”。

## 7. 真正GPU与规划技术检查，随后连续运行

默认每GPU一个训练worker，至多4个总worker；1/2 workers per GPU先短profile，实际增益不足10%选较低并发。与R2.1同硬件不等于新工作自动同吞吐。不要恢复其额外6worker扫描。

规划默认每GPU一个policy进程，按小批/逐case运行；可在技术case上验证更大batch，但所有模型的科学CEM预算不变、每case RNG独立。同一case的H0和REFIT尽量在同一GPU上比较。

最少实际检查：
1. 两任务官方模型能严格加载、真实环境能headless reset/step；
2. source预处理与包装后的zero-update预测、目标代价一致；
3. H0与纯克隆同case同CEM随机流闭环一致，允许精度合同内微差但需解释决策影响；
4. 真正prediction backward，冻结hash不变；
5. train→save→exit→resume连续、RNG和游标正确；
6. 已知动作脉冲与末端索引对齐；自由rollout不读真实未来观测；
7. 一个完整CEM episode的实测耗时，不只测模型一步；
8. TEST/final-goal标签不能进入训练Loader。

技术optimizer合计≤1,024更新，使用隔离scratch权重，不能继承为正式初始化；完整技术闭环轨迹总上限16，必须与正式100对不重叠。纯无optimizer检查另计。技术规模不足时报告具体需求，不能暗中扩成科学试验。

通过技术检查就完成正式6个训练及配对评价；不等CPU胜负、不要求某方法先提高成功率。若Reacher确有缺依赖/权重阻塞，PushT可独立继续，不能偷偷替换成TwoRoom。

## 8. CEM正式评价：只比较官方原模型与后训练

每任务H0一次，REFIT三个seed各一次；每arm使用完全相同的最多100对。

`2任务 × (1原模型+3后训练模型) ×100对 = 最多800条正式闭环轨迹`。

H0结果可以在与三个refit的比较中复用，但不能记成三次独立H0运行。每case重置policy历史、warmstart、CEM RNG和环境；共用其预先冻结的随机seed。不用全局RNG随worker顺序漂移。

CEM参数、goal cost、action limits、history、eval预算全保持一致。只执行原官方CEM，不加GN/L-BFGS、proposal网络、真状态代价或新的goal窗口损失。

采用同case真实环境闭环：执行动作后重新观察当前环境并重规划；不是teacher forcing专家轨迹。未来goal图像是任务允许输入，但录制的专家未来动作不是控制时的已知正确答案。

主指标：作者任务success率的配对变化，两个任务分别报告；同时保留final goal error、任意/终点成功的原定义、执行动作数、规划总时长/调用次数及异常率。不得只报告达到的最佳时刻而改变官方success定义。

没有固定5–10个百分点或100倍提速准入标准。可描述≥5pp这一实际量级，但主结果无论是否达到都完整交付。相同算力不等于相同总研究成本：额外离线训练必须计入。

## 9. 开环预测与闭环规划严格分开

每个固定EVAL case，同时取得其记录轨迹上的合法历史和动作，用冻结目标encoder计算future latent。开放环不读未来真观测修正预测。

主/辅助horizon按原始模拟步明确：通常frameskip=5，对应1/2/5个latent转移=5/10/25原始步。不得把“25原始步”写成“25个macro步”。详细合法索引见规格；如果checkpoint使用其他stride，必须报告真实对应而不改数据凑数。

报告同坐标latent MSE、按固定TRAIN方差归一的误差、均值/方差/范数漂移。各任务单独给数值，不直接用不同任务latent单位平均出一个赢家。

本轮不训练state probe或读取G1旧decoder给官方encoder解码。需要物理误差时，只用环境原有的真实闭环goal误差；开环预测与物理状态预测的区别保留。

记录prediction与planning是否同向，但不能把相关关系直接宣称为误差对规划的唯一因果路径。

## 10. 统计与无效案例

单位是起点—目标case，若每episode只有一对，case与episode相同；推断族存在则同时报告family整簇敏感性。三个refit种子并不是三个独立官方预训练世界，也不是把100变成300个独立TEST。

主效应每任务：先对同case三个固定refit的success均值求值，再减该case H0，最后平均case。逐seed差值、赢/输/同计数完整展示。

同一批5,000次配对重采样保留所有arm/seed；普通95%条件区间与family敏感性并列。需要描述两个任务同时支持时，按预定Bonferroni口径给每任务97.5%区间，注明仍是有限样本bootstrap近似，不宣称有限样本严格FWER保证。其他horizon/里程碑/时间比较探索性，未校正清楚注明。

统计不能按结果选更窄的episode或family区间。推断族独立性不保证，少簇导致不稳定要说明。不要复用R2.1尚未通过覆盖率核验的max-stat方法。

基础设施失败可同seed同配置重跑并单记；模型输出非有限、计划失败应保留为方法失败，规划success记0并保留原因。不能删困难case后重新计算成功率。

## 11. 实際ETA：由完整闭环计时计算

输出两项而非只报训练ETA：
- 六个predictor作业剩余更新/实测训练吞吐；
- 每任务剩余完整CEM episode × 实測耗时，按真实并行分配估算。

给出数据准备、训练、规划、统计/回收分项，使用中位数和较慢分位范围；并行部分按关键路径，不把所有worker小时简单相加成墙钟。

首次实际profile后写`ETA_PROFILE.json`，每阶段更新。若CEM明显慢，报告，不默认降候选数、迭代数、case数或改为离线评分。当前没有时限；用户明确修改预算时才登记新修订并保护已开始的配对完整性。

## 12. 紧凑保存与回收，避免重新塞满本机

每任务共享一份官方base encoder/权重、source快照与变换；每job仅保存可训练参数delta、optimizer、LR/RNG/数据游标、完整构造manifest。保存3k/10k/30k所需delta，不为每个seed复制整个ViT/原数据。

紧凑结果必须足够重评分：case manifest、环境版本、action序列/原始尺度、step success/goal error、latents/离线误差、时间与CEM调用账、逐case指标与bootstrap身份。

正式不保存全量视频；最多每任务4条预先选ID的展示视频，总≤8条，不能按漂亮程度挑。视频是辅助，不能替代轨迹数据。

每job及每批完整case原子落盘、生成SHA并增量回传。归档验证必须逐成员核验关键科学文件及所有原始必要文件清单，不只抽样检查。旧原始数据不重复回传；新官方权重和source可单独保留一次。

最终CPU只读重建：两任务H0和各3个final predictor delta均可加载；重算所有保存预测的指标与配对表；运行同环境容差合同，数据hash必须精确一致。跨GPU/BLAS差异用预先冻结的数值容差，不为方法赢了放宽。

## 13. 最少交付集及最终答复

报告：
- `INTAKE_AND_INSTANCE_REUSE.md`
- `OFFICIAL_SOURCE_AND_PROTOCOL_AUDIT.md`
- `FREEZE_AND_CACHE_VALIDATION.md`
- `BASELINE_TECHNICAL_REPRODUCTION.md`
- `TRAINING_COMPLETION.md`
- `OPEN_LOOP_RESULTS.md`
- `CEM_PAIRED_RESULTS.md`
- `COMPUTE_STORAGE_AND_ETA.md`
- `FINAL_SCIENTIFIC_REPORT_ZH.md`

机器文件：
- 官方权重/源码/数据/统计及原预训练暴露manifest；
- 冻结parameter白名单、6job表、每任务cases/family/seed表；
- 训练曲线、逐case规划与预测值、时间/失败账、共享bootstrap索引；
- source/config/weights/predictor delta SHA及恢复收据。

最后分别报告：
1. 两任务是否确实使用官方checkpoint；哪些暴露未知；
2. 6/6训练与实际更新、最多800规划的真实完成数；
3. 开环是否改善、规划是否改善，两个任务与每seed原值；
4. 效果是否受ceiling、置信区间/样本量、优化预算限制；
5. 原参数冻结是否通过、是否有接口或数值缺项；
6. 总离线成本与每次规划成本；
7. 科学结论和单独的交付/退卡状态。

允许科学结果：
- `REFIT_IMPROVES_PREDICTION_AND_PLANNING_IN_TESTED_TASKS`
- `PREDICTION_GAIN_WITHOUT_ESTABLISHED_PLANNING_GAIN`
- `TASK_DEPENDENT_REFIT_EFFECT`
- `NO_MATERIAL_GAIN_ESTABLISHED_IN_TESTED_RANGE`
- `TECHNICAL_OR_OPTIMIZATION_INCONCLUSIVE`

这些是本轮标签，不裁定所有JEPA方法和期刊录用。没有规划器交叉就不能说LEAP/LeFlow/SAGE的增益“来自预测器”，也不能声称它们被推翻。

完成后停止新科学计算。`RELEASE_GATE`和seal应覆盖G1/R2.1尚未闭合的必要证据及本轮产物；GPU任务全部退出且数据验证通过才通知用户可手动退卡。不要自动destroy。

## 14. 执行顺序

1. 复用实例、保护R2.1快照与必要回传。
2. 核实两个官方权重、代码、原数据与action/goal合同。
3. 建立无暴露混淆的本轮分割、技术case、主case身份，冻结分析规则。
4. 两任务真实官方基线、无更新clone、恢复/冻结检查；获取完整CEM计时。
5. latent缓存可按任务独立完成；任务技术通过即开始3个refit，不必等另一任务下载结束。
6. 全部权重锁定后进行正式H0/REFIT配对规划和开环评分；不看中间成功率选model。
7. 统计、紧凑回收、CPU恢复验收、最终报告。
8. 停止，不自动扩任务/模型/规划器，不自动释放实例。
