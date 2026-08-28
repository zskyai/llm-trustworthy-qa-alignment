# Trustworthy QA Alignment

面向专业知识问答可信生成的后训练对齐项目。仓库当前聚焦数据规则、损失层实现与可复核评测，不将未完成的大规模训练描述为已落地结果。

## 项目目标

专业知识问答中，基础大模型容易出现答案流畅但证据不足、引用不一致、信息缺失时仍强行作答等问题。本项目围绕“可信回答”构建 SFT + DPO/SimPO/ORPO 可对比后训练流程，让模型在专业问答场景中更倾向于：

- 基于证据回答
- 引用一致、可追溯
- 证据不足时保守回答或拒答
- 减少无依据扩展和幻觉

## 技术路线

1. 构建 instruction tuning 数据，学习专业问答格式、引用格式和保守回答规范。
2. 构造偏好对，包括：有证据支撑 vs 无证据臆断、引用一致 vs 引用错误、保守回答 vs 过度生成。
3. 在统一 completion log-prob 口径上实现 DPO、SimPO 和 ORPO，对比有无参考模型、长度归一化和 SFT 锚定的影响。
4. 使用 citation precision、unsupported claim rate、refusal accuracy 等指标评估可信生成行为。

## 算法实现

`src/losses.py` 不依赖 TRL/OpenRLHF Trainer 封装，直接用 PyTorch 暴露以下计算环节：

- `completion_logps_from_logits`：完成 causal shift，通过 completion mask 排除 prompt/padding token，并支持序列 log-prob 求和或按 token 平均。DPO 可使用 sum 口径，SimPO/ORPO 使用 mean 口径减少长度偏置。
- `dpo_loss`：计算 policy 相对于冻结 reference policy 的偏好 margin，支持 label smoothing 和可选 SFT anchor。
- `simpo_loss`：使用无 reference 的长度归一化 reward，显式加入 `gamma / beta` 目标 margin，并支持可选 SFT anchor。
- `orpo_loss`：将 chosen 回答的 SFT NLL 与 odds-ratio 偏好损失合并；使用分段 `log1mexp` 处理 log-prob 接近 0 的数值边界。

`tests/test_losses.py` 使用 toy tensor 检查 completion mask/长度归一化、DPO 的 `policy == reference` 边界、SimPO 目标 margin、ORPO 极端 log-prob 的有限值，以及三种损失对 chosen/rejected 的梯度方向。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 仓库结构

```text
data/       数据样例与数据格式说明
configs/    SFT/DPO 训练配置
scripts/    数据构造、训练、评测脚本
src/        completion log-prob 与偏好优化损失层
eval/       评测指标与样例结果
docs/       设计文档与实验记录
tests/      损失公式、数值边界与梯度方向测试
```

## 真实完成边界

- [x] 项目目标与后训练任务定义
- [x] 偏好对构造规则设计
- [x] 仓库结构初始化
- [x] 纯 PyTorch 实现 completion log-prob、DPO、SimPO 与 ORPO 损失层
- [x] 建立 toy unit tests，覆盖 mask、长度归一化、数值边界与梯度方向
- [ ] 补充可公开样例数据
- [ ] 将损失层接入仓库内的批量 LoRA 训练脚本
- [ ] 在同一数据切分上完成 SFT/DPO/SimPO/ORPO 对比训练
- [ ] 输出评测报告

当前新增内容是损失层和单元测试，不代表已用新目标完成模型训练，也不代表已得到 SimPO/ORPO 的指标提升。

## 后续实验矩阵

下列是待执行的对照设计，不是已完成结果：

| 维度 | 对照项 | 要回答的问题 |
| --- | --- | --- |
| 优化目标 | SFT / DPO / SimPO / ORPO | 拒答、引用和答案质量分别发生了什么变化？ |
| log-prob 口径 | sum / completion-token mean | 得分提升是真偏好改善还是长度偏置？ |
| reference | 冻结 SFT reference / reference-free | 参考模型显存与稳定性之间的取舍是什么？ |
| 偏好数据 | 表面负例 / 同长同格式 hard negative | 模型是学到证据判断，还是学到引用编号和拒答前缀捷径？ |
| 稳定化 | SFT anchor 权重、beta、SimPO margin | 偏好 margin 增长时是否伴随 F1/引用率退化？ |
| 评测 | pair accuracy、margin、token F1、引用率、拒答 balanced accuracy、生成长度 | 如何防止用单一偏好分数掩盖任务能力退化？ |
