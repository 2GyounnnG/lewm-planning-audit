# R3 官方LeWM预测器后训练执行包

## 交给Codex的文件

主执行指示：`01_CODEX_PROMPT.md`  
必需技术规格：`02_EXPERIMENT_SPEC.md`  
一手来源与核实边界：`03_PRIMARY_SOURCES.md`  
启动消息：`04_START_MESSAGE.txt`  
训练与评价逻辑表：`05_TRAIN_JOBS.csv`、`06_EVAL_ARMS.csv`

整包交付可以避免只上传主prompt而遗漏必需技术规格。

## 需要执行端实际取得的内容

- 原G1/R2.1实例SSH记录、GPU UUID与工作区；
- 旧结果的可靠快照与必要回收状态；
- 作者官方PushT和Reacher检查点/数据/兼容环境；
- source与权重的具体revision、哈希、预处理和规划接口。

本包不是模型/数据上传包，没有SSH密钥、真实官方权重或已跑出的R3结果。
`07_PROTOCOL_CHECKS.py`只检查本包计数及若干隔离约定，不验证远端数据、模型、显卡或实验效果。

## 范围

- 只用两个任务、同一个官方起点的三次后训练重复；
- 6个后训练作业，共180,000更新；
- 每任务最多100对，最多800条正式CEM轨迹；
- 没有新encoder、state head、bilinear、其他planner或领域矩阵；
- 使用原两张5090，按任务profile，不默认DDP；
- 数据与checkpoint复用但源码分支隔离；完成后不自动销毁实例。

历史报告在`inputs_readonly/FINAL_SCIENTIFIC_REPORT_ZH.md`，只是R2.1背景，不是官方LeWM的本轮结果。
