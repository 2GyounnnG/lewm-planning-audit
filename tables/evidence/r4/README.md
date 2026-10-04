# R4-v2.3 执行入口

状态（2026-10-04 00:32 北京时间）：四线主批及符合条件的第二批已合并、交付并完成独立CPU统计复算，全部科学计算结束。R4和X2必要文件已完成恢复；X1的Cube主包及TwoRoom第二批4.24MB轨迹增量继续回收。Cube三个固定30k后训练及开环均完成，闭环因冻结exact复位门槛失败留空。TwoRoom严格CPU恢复未通过，释放门槛保持HOLD。

执行依据为 `../r4_v2_3_package/R4_V2_3_SINGLE_HOST_ZH.md`。旧 R3 与 E0 台账只读。

- `manifests/EXECUTION_LEDGER.json`：主机、8张GPU UUID、CPU/GPU分区、训练预算及存储差异。
- `manifests/HOST_AUDIT.json`：新机 PyTorch 2.8.0+cu128、8卡实测原值。
- `tables/CPU_ACCEPTANCE_OVERVIEW.csv`：PushT/Reacher各600条CPU锚点核验摘要；逐条值在 `r4/cpu_boot_*/boot_anchor_raw.csv`。
- `x2/reports/CPU_ACCEPTANCE_RAW_VALUES.csv`：3编码器CPU小样例，神经臂只有3步TECH。
- `x2/reports/COMPACT_ACCEPTANCE_RAW_VALUES.csv`：原/精简输入等价核验；原SHA到派生SHA全链在 `x2/compact_transfer/COMPACT_TRANSFER_MANIFEST.json`。
- `x1/reports/CPU_SEMANTICS_RAW.csv`：X1 CPU语义测试原值。
- `r4/R4_CONTRACT.json`：时间轴、成功、历史、复位重放与随机流科学合同。
- `r4/EXECUTION_COMMANDS.json`：R4各模块最小依赖和入口。
- `analysis/aggregate.py`：从逐case/候选表重算S1/S2/S3；最终预结果冻结在 `manifests/statistics_full_v3/STATISTICS_FREEZE.json`。

远端固定主机 `72531573d994`。执行目录：`/workspace/r4_pusht`、`/workspace/r4_reacher`、`/workspace/x1_tworoom`、`/workspace/x1_cube`、`/workspace/x2`；共享环境 `/workspace/env`；共享只读输入 `/workspace/shared_data`。

真实磁盘只有300GiB（协议预期约2TB），可重新获取的原始HDF5允许用RAM临时解压；训练权重、正式输出、哈希证据写持久磁盘。没有自动租卡或destroy。

每条线只按技术门槛阻塞自身依赖模块，负面科学结果不阻塞。四任务主表、规划随机性、归因和X2总表只在必要原值到齐后合并；未跑项目不填零。

模块原值与不超过5行的结论索引：`manifests/DELIVERY_INDEX.json`。本地独立数值/覆盖核验在 `manifests/main_validation/`。合并入口为 `ops/deliver.py --spec manifests/MAIN_MERGE_SPEC.json --out tables/main`，会等全部主批门槛。

明确技术限制：Reacher固定25raw模拟归因/SIM重排受真实DMC LAST限制；Cube当前冻结exact复位协议未通过，正式闭环缺失而非零成功。两者完整证据、原值和未执行原因分别保留，离线可评价模块继续。

恢复范围与科学交付分开：X2全部324编码器对应的PRED、MLP及2592个闭式头已回收，固定两DEV episode上的CPU推理探针通过，全部256 episode已存风险可重聚合；这不是完整256 episode推理重跑。R4全主批统计已从已回收原result与显式派生state子集独立复算，两任务完整菜单证据已通过本地SHA核验，最小轨迹包继续回收。X1全量必要模型/固定100例开环输入/闭环轨迹回收在后台。保留远端原始日志，不自动销毁主机。

`reports/CONTROL_ATTRIBUTION_FINAL_ZH.md`集中列出主张、证据、替代解释和未识别范围，已纳入第二批；`reports/STATISTICAL_INTERPRETATION_NOTES.md`说明缺失、分母、时间归约、浮点舍入与区间的解释限制。

主批合并在 `tables/main/`：四任务主表80行、原值表8000行、R4规划随机性24行、归因1149行、初始候选12800行、R4正式闭环2640行、X2四臂主表1296行和全部头原值11664行。8000行包括显式case内均值与500条Cube闭环缺失，不是8000个独立观测。`MERGE_STATUS.json`逐项保留来源SHA与技术限制。

第二批合并在 `tables/secondary/`：四任务规划随机性48行、X1三流原单元2400行（TwoRoom1200观测、Cube1200缺失）、PushT中途1280候选、两任务固定20例可用性共40行，以及SIM_CEM未运行理由/估算4行。两批分开保存；`manifests/SECONDARY_BATCH_DELIVERED.json`核对主批原SHA未变。

在本机执行一次CPU复算：

```sh
cd "/Users/richwang/Documents/ChatGPT/热/r4_v23_execution"
"/Users/richwang/Documents/ChatGPT/热/jepa_low_label_bridge_gpu_v2/g1/.venv/bin/python" -B ops/recompute_delivery.py --r3-root "/Volumes/MyProj/r3_official_lewm_predictor_refit/recovery"
```

入口默认新建 `recomputed/<UTC>-<pid>/`，拒绝覆盖交付目录。R4 S1/S2/S3从原值调用SHA固定的v3统计；R3/X1/X2保留原始区间并验证覆盖与算术。它不训练、不调用模型或模拟器，也不重新执行闭环。实际完整验收为 `manifests/MAIN_CPU_RECOMPUTATION.json`：46份R4表逐字段相同，8007个输入SHA运行前后不变。第二批中途fork与新增X1随机流由独立第二批验收文件覆盖。

严格CPU恢复限制（与上述统计复算分开）：TwoRoom的文件SHA全部通过，但Mac预测有4/384000坐标、Linux预测有11/384000坐标超出原冻结CPU容差。失败原值、outlier坐标与平台指纹位于 `x1/recovery/tworoom/`。未放宽门槛或修改正式GPU原值；最终文件可以如实封存，`RELEASE_GATE`须保持HOLD，不能声称CPU恢复全部验收通过。
