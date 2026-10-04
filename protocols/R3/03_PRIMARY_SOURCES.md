# 一手来源与本轮新增设计的区别

访问日期：2026-10-02。以下公开URL已在生成本包时查看。远端main是可变化引用；执行端须下载实际revision并保存SHA，本文不冒充已经固定了远端commit。

## 已核实的官方来源

1. LeWorldModel v3原文
   https://arxiv.org/html/2603.19312v3
   用途：动作条件latent预测、像素规划、任务与训练/评价定义。本轮只检验官方起点后训练，不改变论文既有实证的归因。

2. 作者README与模型链接
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/README.md
   用途：确认代码、数据与权重来源链，state_dict/config与object加载接口。
   https://huggingface.co/quentinll/lewm-pusht
   https://huggingface.co/quentinll/lewm-reacher
   两张模型card分别明确为PushT与DMControl Reacher的官方预训练模型。具体下载权重/revision仍由执行端核验。

3. 模型接口
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/jepa.py
   用途：区分观测projector、action_encoder、predictor、预测投影pred_proj；核对真正rollout、目标编码与末端代价。不要从名称猜冻结边界。

4. 训练与优化配置
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/train.py
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/train/lewm.yaml
   用途：一步预测目标与官方优化器参数锚点。当前公开配置含AdamW、5e-5学习率、1e-3权重衰减；原trainer含bf16。本轮更新数、schedule、FP32与冻结方式均是新增设计，不能称全部原配方复现。

5. 组件实现与模型结构配置
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/module.py
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/train/model/lewm.yaml
   用途：核对dropout、BatchNorm、预测投影及SIGReg对哪些张量/参数有梯度。

6. 官方eval和CEM配置
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/eval.py
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/pusht.yaml
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/reacher.yaml
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/solver/cem.yaml
   用途：固定任务、复位、success、目标偏移与CEM实际预算。当前eval配置会引用训练来源；必须如实披露原pretraining暴露，不包装为整模型全新holdout。

7. 图像及动作归一化
   https://raw.githubusercontent.com/lucas-maes/le-wm/main/utils.py
   用途：检查图片管线、边界NaN与std convention。不同文件里的实现不能仅凭名字认定一致。

8. 作者依赖的平台
   https://github.com/galilai-group/stable-worldmodel
   用途：执行端读取实际policy、environment、CEM、数据实现；本包没有逐行审计其所有版本。

## 本次不能确认的具体细节

- 两个HF repo内部config.json直接URL本轮抓取失败；模型card存在不等于完整权重已在本地验收。
- 猜测的`config/train/data/reacher.yaml`路径本轮返回404。执行端应从真实目录找到Reacher数据配置，不按这个不存在的路径造文件并称官方。
- 本包未连接用户2×5090、未读取私有R3缓存、未运行官方闭环复现。
- 本包未认证期刊分区、审稿速度或论文录用成熟度。

## 新设计，不是来源里的事实

- 仅PushT+Reacher，6次后训练×30k。
- 100对/任务、800条正式trajectory上限。
- param白名单、BN统计冻结、FP32后训练、显式case随机数。
- refit/monitor/eval隔离及原官方预训练暴露审计。
- 95%与两任务97.5%条件bootstrap呈现规则。
- 两卡续用、紧凑delta存储、增量回收，不自动destroy。

这些是用户本轮授权后执行的研究选择；不是声称官方论文采用同一设置。
