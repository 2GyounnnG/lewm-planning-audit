# 冻结与缓存实际核验

本报告是可更新的技术记录。证据截点：2026-10-02T18:00:18.357677+00:00（UTC）。两任务观测缓存与真实TECH模型门均已完成；这些结果没有提供正式EVAL的预测质量、规划成功率或方法收益。此处汇总现有JSON收据，未新增编码、训练、环境步或科学评价。

| 任务 | 已缓存episode | 已缓存raw帧 | NPZ逻辑字节 | 编码墙钟秒 | REFIT_TRAIN方差帧数 | 固定总体标量方差 |
|---|---:|---:|---:|---:|---:|---:|
| pusht | 15,052 | 1,862,147 | 1,491,558,248 | 1645.52245789906 | 1,665,187 | 1.0067291625340873 |
| reacher | 8,104 | 1,628,904 | 1,299,103,616 | 1517.8251990997232 | 1,447,200 | 0.989531499622593 |

两个manifest的状态均为 `FROZEN_OBSERVED_CACHE_COMPLETE`。缓存仅覆盖REFIT_TRAIN/MONITOR/TECH/EVAL；EVAL_POOL_UNUSED未编码，故缓存episode数小于完整原始源。缓存EVAL观测不等于把它们交给refit。缓存文件保留所有raw相位：`z[N,192]`、标准化`actions[N,2]`、raw_indices、stride5与legal_starts；按窗口选三帧历史，每个10维宏动作按时间顺序保留五条二维raw action，未取均值或只取首动作。

编码使用官方weights、encoder/projector的eval模式、FP32、AMP/TF32关闭，batch128。每episode检查所有参数/buffer版本、eval/requires_grad；开始、每100个新增episode、结束核对完整冻结SHA。两任务缓存均记录0 optimizer更新、0 state标签读取。用于开放环误差归一化的标量是REFIT_TRAIN全部raw帧的逐坐标总体方差再平均，用float64累积；没有使用MONITOR/EVAL潜变量拟合该方差。

动作变换采用固定官方eval `StandardScaler` 的ddof0。它使用完整原作者训练来源的有限raw action，**包括后来被R3划为EVAL的来源episode**；这是预先声明的输入统计暴露，不是EVAL物理目标拟合。PushT有限行2,336,736、排除0；Reacher有限行2,000,000、排除每episode末行NaN共10,000。H0及全部refit共用同一变换。具体mean/scale全部原值位于两份normalization manifest；不得称为仅用R3 REFIT_TRAIN拟合的动作统计。

真实模型门固定使用每任务TECH预选列表前两个窗口，共16个原始图像帧、70条raw action。只读取pixels/action；EVAL数组、state/reward/goal标签读取均为0。70个标量脉冲映射全部exact。预登记容差：encoder/cache atol=rtol=1e-5，predict与相同初始history的rollout包装 atol=rtol=1e-6，未根据结果放宽。

| 比较 | PushT最大绝对差 | Reacher最大绝对差 | 实际状态 |
|---|---:|---:|---|
| 原始图像encode vs cached z | 1.6689300537109375e-06 | 1.9073486328125e-06 | PASS / PASS |
| 相同cached history的原predict vs wrapper | 0.0 | 0.0 | PASS / PASS |
| 相同原始编码history的原H5 rollout vs wrapper | 0.0 | 0.0 | PASS / PASS |
| 原rollout初始编码 vs cache | 2.7418136596679688e-06 | 2.3245811462402344e-06 | PASS / PASS |
| 原pixels-start vs production cache-start rollout（诊断） | 2.816319465637207e-06 | 1.430511474609375e-06 | DIAGNOSTIC_ONLY / DIAGNOSTIC_ONLY |

最后一行是诊断，不能冒充另一个通过容差的独立硬门。两任务全部保存数组finite。零更新反传以 `z[:,1:L+1]` 为三位置teacher-forced target：PushT技术loss=0.007895032875239849，Reacher=0.01063538808375597。这两个小样本loss仅说明反传实际执行，不是神经训练结果。

两任务 `predictor.` 81个参数张量、`pred_proj.` 6个参数张量均有finite非None梯度，此次各87个均非零。所有非白名单参数grad=None；冻结参数及所有BN运行buffer逐字节不变；全部参数值在无optimizer的反传前后不变。预测侧BN affine可训练，运行统计仍固定。训练白名单共11,584,128个标量参数，冻结参数6,450,350个；这属于结构/实现计数。

截至本截点，合并 `TRAINING_TECHNICAL_GATE.json` 已实际核获 `TRAINING_TECHNICAL_GATES_PASS`：真进程退出恢复smoke32次、profile A96次、profile B192次，共320/1024次技术更新，账本无未决更新。4 workers（每GPU2个）与microbatch128被选中；两任务负载比例相同，未使用科学结果选择并发。技术optimizer状态不会继承为正式初始化。该训练门本身不覆盖完整CEM；另已实际核获 `PLANNING_TECH_GATE.json` PASS：两任务各固定2 TECH case×H0/clone，共8条完整轨迹、4对逐动作/物理state/goal/reward/终止exact比较（计时除外）。技术轨迹8/16，不改变正式800轨迹预算。正式180,000更新与正式800轨迹尚无完成证据。

| 并发profile | 完成更新 | 含启动/监控/保存墙钟秒 | 端到端更新/秒 | 排除初始观测前缀后稳态总更新/秒（观测界） |
|---|---:|---:|---:|---:|
| A：2 workers | 96 | 16.017633724957705 | 5.993394632967454 | 28.13747740204902–29.994458674904532 |
| B：4 workers | 192 | 16.759730016812682 | 11.45603179808943 | 54.99654217596583–58.96670221060912 |

B/A端到端吞吐比1.9114429300340117，稳态保守比1.8335567503333472，均超过预设10%增益门槛。稳态范围是短技术profile的轮询观测界，不是长期正式训练吞吐保证，也不是统计置信区间。

证据身份（路径均相对R3根目录；本报告不代替原始收据/数组）：

| 收据 | SHA256 |
|---|---|
| `manifests/pusht_cache.json` | `b4eb5cc382c602ee31c2e8887e7033a481d0ff23744a2885c26d2c01349cdf15` |
| `manifests/reacher_cache.json` | `6cbf5a4e76df91f13735152d6d0d93a8d31f95fffaf2e449bb055d732c5555db` |
| `manifests/pusht_normalization.json` | `5e7fb8811c3fa8e6665180bfa169ac3860ab9e6a1d2484504414b0256a305b2f` |
| `manifests/reacher_normalization.json` | `b84bda3aca392acd6660ab5dd415dc6c15fbeb5726ce139e70dd26bca2ae092c` |
| `artifacts/technical/pusht_real_model_attempt1/REAL_DATA_MODEL_CHECK.json` | `a9924562e56e68c9e799f4c989e6d7e3fe5ec18a36c6188c3eade0df9b9e3cc5` |
| `artifacts/technical/reacher_real_model_attempt1/REAL_DATA_MODEL_CHECK.json` | `506a52455e8e4e305a5e3c6c3ac3eede1f44bfa365051d701be66e00650aa067` |
| `state/GPU_SMOKE.json` | `1034e9777835954f61418e6daf545dc5ca6d6d0388d4339f7698cc0827cc22d6` |
| `state/TECHNICAL_LEDGER.json` | `7e58db8f92f635c56ca162c5e0fcbea92b76a3574a613dc00fba199c34ec2738` |
| `state/TRAINING_TECHNICAL_GATE.json` | `57d5cf66ee9d38294ddfd5cc925841b129229e0dd46b98c13833e54b6f6a87f0` |
| `state/GPU_PROFILE.json` | `2fe0da83181c39954bc114de8d4e70cbeaee6283fcd0af2bff006a941ef3d090` |
| `state/PLANNING_TECH_GATE.json` | `9ad86eacea01bb9dd927afa9fc0b80c2541a01a7e3f89dc68e52263afc763b58` |
