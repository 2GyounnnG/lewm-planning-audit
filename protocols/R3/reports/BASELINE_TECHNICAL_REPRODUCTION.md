# 官方基线技术复现进度

证据截点：2026-10-02T18:00:18.357677+00:00（UTC）。本报告记录实际原模型/数据/环境接口核验，**尚不构成原论文成功率复现，也没有形成R3正式科学结论**。两任务strict加载、真实TECH输入包装、真实目标cost包装与有限reset fallback均已PASS；完整CEM技术门8条也已PASS；正式训练及评价仍待实际完成。

| 技术项 | PushT | Reacher | 可支持的范围 |
|---|---|---|---|
| 官方原权重strict加载 | PASS | PASS | 303个原state_dict键，无删键/改名 |
| 原数据→编码缓存/预测/自由rollout包装 | PASS | PASS | 固定两个TECH窗口；详见FREEZE_AND_CACHE_VALIDATION |
| 原get_cost与计时wrapper/goal编码 | PASS | PASS | 两固定TECH case、每case两组固定合成候选动作 |
| 缺seed固定0的有限reset核验 | attempt2 PASS | attempt2 PASS | 每任务两个固定TECH case、seed0/1、各两重复、两条路径 |
| 真进程退出+恢复smoke | PASS | PASS | 两任务合计32技术更新，不继承到正式初始化 |
| 合并训练技术门/320更新profile | PASS | PASS | 320/1024；4 workers、每GPU2、microbatch128；无未决技术更新 |
| H0/clone完整CEM | PASS | PASS | 每任务固定2 TECH case×2 arms，共8/16条技术轨迹，4对逐步exact，计时不纳相等比较 |
| 正式三seed×30k refit及正式CEM | PENDING | PENDING | 不能把技术前向/反传当成正式训练或规划结果 |

COST_WRAPPER两任务每case的原始get_cost与包装输出逐值exact，官方goal编码exact，手算最后预测latent与goal latent的平方差之和在已冻结1e-6容差内一致，所有数组finite。候选动作固定为全0以及第一宏动作首坐标0.25；不使用源专家未来action。每任务该门0 optimizer、0环境step、0完整CEM轨迹。原值保存在 `COST_ARRAYS.npz` 与 `COST_WRAPPER_CHECK.json`。

reset诊断单独计账如下；raw step指SWM环境step。Reacher内部action_repeat=2，不能把内部积分次数冒充外层轨迹数。

| 固定尝试 | 实际状态 | reset尝试 | 成功reset | raw step | 比较数 | optimizer / 完整CEM |
|---|---|---:|---:|---:|---:|---:|
| pusht attempt1 | BLOCKED_RESET_FALLBACK | 16 | 0 | 0 | 0 | 0 / 0 |
| pusht attempt2 | PASS_TECH_RESET_FALLBACK | 16 | 16 | 16 | 14 | 0 / 0 |
| reacher attempt1 | BLOCKED_RESET_FALLBACK | 16 | 16 | 0 | 0 | 0 / 0 |
| reacher attempt2 | PASS_TECH_RESET_FALLBACK | 16 | 16 | 16 | 14 | 0 / 0 |

PushT attempt1是NumPy整数seed被Gym拒绝的接口类型错误；修正只将uint32范围整数无损转Python int。Reacher attempt1误把局部geom碰撞位1/1当成有效接触：实际原XML全局contact=disable。修正检查实际编译模型disableflags与已核实MuJoCo3.11.0的mjDSBL_CONTACT=16，保留局部位原值、旧助手快照和IMPLEMENTATION_CORRECTION；没有修改环境、XML、case、seed或数值容差。两次失败记录均保留，未覆盖为成功。四份reset收据合计48次成功reset、32个raw step、0完整CEM。

PushT attempt2的14比较对物理state7、隐藏刚体动力学、render、非覆盖variation与一步结果均exact。原 `_set_state` 会推进一个dt，因此源state/pixel与实际reset可以不同；首个固定TECH试次记录最大state差0.7856159973144372、pixel最大差150、RMS2.9271896239204858。全部试次偏差原值保留，不能称精确恢复原始轨迹状态。

Reacher attempt2有12条同seed全字段exact比较、2条跨seed任务字段exact比较。qpos/qvel、目标joint、隐藏积分/控制/外力状态、物理参数、render、一步结果与qpos_match终止均为硬门。随机point target位置/native observation/point reward/累积分数只有跨seed时作为finite诊断；同seed重复/两路径仍必须exact。目标透明且全局禁用接触由真实模型配置核实；qpos_match成功只由两关节相对目标的绝对差均小于0.05决定。源图像/关节重建偏差逐试次保留，未根据实际结果调整门槛。

两任务源文件均缺seed列；原roles中TECH4+EVAL100的104个case保持source_seed/reset_seed=null。独立RESET_FALLBACK_MANIFEST把有效运行seed固定为0，身份是 `SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED`。该有限验证证明这些固定TECH条件的可重复性，**不恢复原seed，不证明任意seed或全部EVAL与原采集环境完全等价**。所有正式模型必须使用相同case映射，不能按方法挑seed。

实际完整CEM技术门已通过：四个预选TECH case的H0/clone原始动作、success/truncation、物理state、goal误差及goal state、reward逐值一致。总执行246 raw steps、12次replan、0 optimizer、无额外未计warmup轨迹。controller墙钟474.09506067913026秒；8条轨迹计时之和31.2217058618553秒、planning同步时间之和26.49104908388108秒、环境step时间之和0.8172938316129148秒。各设备计时可能嵌套，不能相加冒称墙钟；数据身份SHA和模型/环境setup单列。TECH case的成功/失败仅用于此固定技术核验，不用来推断正式EVAL成功率或调整设计。正式规划仍固定H5、receding H5、五raw action拼块、300候选/30迭代/top30/var_scale1；每次重规划执行25条raw action，最多50 raw steps。后续门槛的科学好坏不等同于实现错误。

证据身份：

| 收据 | SHA256 |
|---|---|
| `artifacts/technical/pusht_cost_wrapper_attempt1/COST_WRAPPER_CHECK.json` | `58b4b6ab156cf413598e0080356a7b052101db0ff746ca32cb98bfc27a83444e` |
| `artifacts/technical/reacher_cost_wrapper_attempt1/COST_WRAPPER_CHECK.json` | `8db4f51d894b80ddc5ee53d3a48df72072704cb7ea49d9648ec8b122c1e7e6f7` |
| `artifacts/technical/pusht_reset_fallback_attempt1/RESET_VALIDATION_RECEIPT.json` | `b19c5b9a454c26afea4eff8d82bd708bacf01b568eb05bb5479e5f11837171d3` |
| `artifacts/technical/pusht_reset_fallback_attempt2/RESET_VALIDATION_RECEIPT.json` | `2c36989bd9b48b8ab505c2d04cf2a3c5660005ad92e3809afcc5af9a19e96c8a` |
| `artifacts/technical/reacher_reset_fallback_attempt1/RESET_VALIDATION_RECEIPT.json` | `3b994c6db092b3d9bdc458eb2071963b0dc25a05fde4199657db747ee4a9c3a5` |
| `artifacts/technical/reacher_reset_fallback_attempt2/RESET_VALIDATION_RECEIPT.json` | `8d7814bbe4df48052cf3c6262394d1fa67e11c262e962c263e52c78281baea35` |
| `manifests/RESET_FALLBACK_MANIFEST.json` | `7269e60f770cdcb0f14f5febab7e8c6510d5cb5eb8500763caa3067500634573` |
| `state/GPU_SMOKE.json` | `1034e9777835954f61418e6daf545dc5ca6d6d0388d4339f7698cc0827cc22d6` |
| `state/TRAINING_TECHNICAL_GATE.json` | `57d5cf66ee9d38294ddfd5cc925841b129229e0dd46b98c13833e54b6f6a87f0` |
| `state/GPU_PROFILE.json` | `2fe0da83181c39954bc114de8d4e70cbeaee6283fcd0af2bff006a941ef3d090` |
| `state/PLANNING_TECH_GATE.json` | `9ad86eacea01bb9dd927afa9fc0b80c2541a01a7e3f89dc68e52263afc763b58` |
