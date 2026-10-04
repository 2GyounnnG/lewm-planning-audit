# R8-FIX 更正记录 V3

## F1 M1 E2

审计 `r8_finalize.py` 时发现原 E2 计算的是 H_REAL3_REPLAN 成功率在预算 50 与预算 100 之间的差，不是协议定义的每 case 12 条 arm 的历史效应差。已按 100 个配对 case、每 case 4 模型×3 流先算 `(H_REAL3_REPLAN−H_POLICY)`，再计算预算 50−预算 100；bootstrap seed 与 `indices_sha256` 与原封存一致。偏移25：+0.128333，95% CI [+0.099167,+0.159167]；偏移50：+0.100000，95% CI [+0.074167,+0.127500]。原两行保留为描述性成功率差。

## F2 M3 PRED_PAST

旧实现把预测历史写进 `info['emb']`，而 `AuditedCost.get_cost` 按 pixels 重新编码，预测历史被覆盖/忽略；旧 1,200 条保留并标记 `INVALID_IMPLEMENTATION`。修正版使用显式 latent rollout 和 `[ẑ15, ẑ20, z25]`。

修正版正式运行产生 1,200 条，但事后严格 TECH 验证中入口上下文断言通过、z25 保持不变；只对前两项预测 latent 加固定 1e−2 扰动时计划未改变。因此 F2 按预注册门槛记为 `POSTHOC_AFTER_TECH_FAIL / INVALID_TECH_GATE`，数值不并入有效 R8 主结论，不再重跑。REAL2/REPEAT3 各抽20条索引审计通过。

原 R4–R8 封存文件、原 R8 PRED_PAST、F2 原 seal 均未修改；V3 独立保存。
