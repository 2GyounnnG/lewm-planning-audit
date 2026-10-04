# R3官方来源与协议差异审计

本轮使用[LeWM官方README](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/README.md)链接的[PushT](https://huggingface.co/quentinll/lewm-pusht)和[Reacher](https://huggingface.co/quentinll/lewm-reacher)原始weights.pt与config.json。两份权重均完成LFS SHA核验与303个state_dict键的strict加载，未删键、改名或改用自有G1权重。

|任务|model revision|weights SHA256|
|---|---|---|
|PushT|22b330c28c27ead4bfd1888615af1340e3fe9052|48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607|
|Reacher|62adae4b71dc474ddf8f794c476ebfe737a743ca|eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469|

两份config的SHA均为2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09。结构是224像素ViT tiny/patch14、192维投影latent、3帧predictor历史、10维宏动作。可训练参数11,584,128，仅predictor与pred_proj；冻结参数6,450,350。所有BN运行buffer固定，包括预测侧BN；预测侧BN affine属于可训练白名单。冻结observed z上的SIGReg对该白名单无梯度，未另加predicted-z高斯化目标。

LeWM源码固定8edfeb336732b5f3ce7b8b210d0ba370a09e2cac；SPT构造函数固定ab836bf699a2be712dfcf980a5eb70d36391e876。配置类映射采用作者README给出的原JEPA/ARPredictor加载方式，使用weights_only=True读取纯tensor字典，不执行下载的pickle对象。

SWM规划/环境实现固定abdced49809d5eae38e24b27dc7b635c502c4812：按日期选取固定LeWM源码之前最后的兼容官方版本，保留其CEMSolver(model=)接口和默认std correction=1。这是有记录的兼容版本选择，无法证明就是原论文运行时的精确依赖。当前SWM main的规划接口、std约定及PushT成功语义已变化，当前main仅作为差异审计来源。

官方[CEM参数](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval)保持horizon5、receding_horizon5、action_block5、300候选、30迭代、top30、var_scale1。实际每次重规划执行25条依次展开的raw action，最多50 raw steps。官方policy默认当前图像history_len=1；没有给某个模型额外专家过去/未来帧。Reacher为qpos_match，成功是每个关节绝对角差小于0.05；dm_control action_repeat=2，raw step指SWM环境step。

原模型card明确链接相应原训练数据。因此本轮留出的EVAL只表示对本次refit留出；不能声称整个模型从未接触这些来源。精确episode预训练暴露仍未证实。有限核查作者collection、固定eval脚本和公开文件树未找到可核验的独立TEST资产；Reacher eval配置名dmc/reacher_random与原训练档案reacher.h5的名字差异继续保留，不将两者称为同一独立评价文件。

实际数据与暴露核验截点：2026-10-02T17:43:31.727115+00:00（UTC）。两份原始压缩档案完成固定revision/LFS SHA下载核验，解压成员单独SHA核验；未替换成旧G1权重或旧Lance输入来运行R3。

|任务|官方数据revision|解压H5 SHA256|源episode / raw帧|
|---|---|---|---:|
|PushT|655cd446b9929369d7d406001da85c15d1457850|b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb|18,685 / 2,336,736|
|Reacher|e70a080d0d04c6072123c9ebd343acf7fff28dbf|85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06|10,000 / 2,010,000|

PushT实际H5与先前G1固定Lance来源完成全体episode ID/长度/offset/step值、全部action及初始proprio的逐值核对；原选896条也逐动作完全对应。PUSHT_PRIOR_SOURCE_ALIGNMENT_V2为PASS，V1因索引列整数dtype不同而未建立匹配的记录保留。V2披露dtype不同但整数值完全相同；未验证全部pixels字节完全一致，也未证明原demo lineage。以实际初始proprio精确分组得到185个保守initial-condition family，每族101条；该归组是防泄漏的保守身份，不冒充已证明185条独立原demo。

|任务|REFIT_TRAIN|MONITOR|TECH|EVAL|EVAL_POOL_UNUSED|家族边界|
|---|---:|---:|---:|---:|---:|---|
|PushT|13,433|1,515|4|100|3,633|TRAIN133族、MONITOR15族、EVAL_POOL37族互斥|
|Reacher|7,200|800|4|100|1,896|family_id=null，按episode作为分割单位|

按协议TECH先取4条，再取100条EVAL，二者episode不重叠；PushT允许TECH与EVAL共享4个已知族，两者所属整个EVAL_POOL与TRAIN/MONITOR仍整族隔离。PushT本轮EVAL100条中6条曾在G1所选896内；其余角色中的旧896重合为TRAIN587、MONITOR96、TECH0、unused207。旧G1/R2的已知来源/开发暴露按独立账本披露，R2逐episode标签访问历史未在此重新构造；不能把R3 EVAL包装为从未使用过的新测试数据。

Reacher源为10,000条各201帧，qpos/qvel为float64二维joint，末行action NaN每条一行。源seed列和可证明family列均缺失；初始qpos/qvel恰好各自唯一并不能证明demo独立。固定family_id=null/FAMILY_UNKNOWN，不把source ID猜成seed，不据此输出有家族身份依据的cluster敏感性CI。两任务物理状态只用于预授权metadata admissibility、固定reset/goal和技术诊断；没有按误差/成功率选case。

原始seed均缺失，两个独立真实reset助手各通过固定2TECH×seed0/1×2重复×两条路径的16reset/16step门。独立fallback manifest绑定成功收据，保持原roles中的null不变，仅为后续共同case路由提供固定seed0；这不是原seed恢复或全EVAL环境等价证明。失败的类型接口尝试与Reacher有效接触检查器修订均保留，详见BASELINE_TECHNICAL_REPRODUCTION。

数据/暴露证据SHA：PushT alignmentV2 `63d497936aec0cdf7a75530ad1a52036f301c724bfd1b800e9f59b6b1ddf01ed`；Reacher metadata audit `cd6215bf91d86fad84dfb43d2d43fb792f0a8d8aa5b0336972e981110603d467`；roles分别 `b8a2bc273996b01f59bc67f25532898c3c8cc3ffba2048857e897fd5944ca0f6` 与 `e3ddd8d7dff80f229a2375a29ea0957981232ee672a43ec06e910ffbabd40c4a`。

R3是新增后训练实验：新AdamW、FP32且关闭AMP/TF32、冻结表示与BN、180,000额外更新；原公开trainer的bf16不同。动作归一化优先采用官方eval的StandardScaler总体std，公开train工具使用样本std；实际共同动作统计已由完整原作者训练来源的所有有限raw action拟合并冻结（包括R3后来留出的episode输入；Reacher排除10,000个末行NaN），H0和全部refit使用同一输入域；未使用EVAL物理目标拟合统计。

截至上述截点，strict加载、两任务缓存、真实TECH数据模型/目标cost包装、两reset fallback均已PASS；真退出恢复smoke也已PASS，320次技术更新及合并训练技术门已于2026-10-02T17:52:22 UTC实际核获PASS，固定4 workers（每GPU2）与microbatch128，未决技术更新为0。另于2026-10-02T18:00:18 UTC实际核获完整CEM技术门PASS（8条、4对clone exact；收据SHA `9ad86eacea01bb9dd927afa9fc0b80c2541a01a7e3f89dc68e52263afc763b58`），正式科学结果仍待实际完成。CPU/合成检查、真实技术前向或短smoke均不等于规划成功率复现或方法收益。
