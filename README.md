# Trustworthy QA Alignment

面向专业知识问答可信生成的后训练对齐项目。当前版本直接消费官方 Qasper parquet 与 PubMedQA `ori_pqal.json`，按 paper/document ID 固定切分，执行 LoRA SFT、DPO，并在独立 held-out split 上报告偏好、答案、引用和拒答指标。

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

## 真实数据与正式运行

- [x] 项目目标与后训练任务定义
- [x] 偏好对构造规则设计
- [x] 仓库结构初始化
- [x] 纯 PyTorch 实现 completion log-prob、DPO、SimPO 与 ORPO 损失层
- [x] 建立 toy unit tests，覆盖 mask、长度归一化、数值边界与梯度方向
- `data/qasper_train.parquet` 是 AllenAI Qasper 官方 train split（888 篇论文、2,593 个问题）；`data/pubmedqa.json` 是 PubMedQA 官方 `ori_pqal.json`。
- `scripts/real_pipeline.py` 是完整训练/评测入口，默认使用全部真实记录，不会生成 synthetic/sample 数据。验证集按 `SHA1(source:document_id)` 做 80/20 paper-level split，避免同一论文泄漏。
- 训练产物包含 `train.jsonl`、`validation.jsonl`、LoRA adapter、逐阶段 predictions/metrics、运行配置和耗时。

安装依赖后运行（本地模型路径也可以替换为 Hugging Face 模型）：

```bash
pip install -r requirements.txt
python scripts/real_pipeline.py \
  --qasper data/qasper_train.parquet \
  --pubmedqa data/pubmedqa.json \
  --model ../models/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/qasper_real
```

默认 `--max-train 0` 和 `--max-generation-eval 0` 表示完整真实训练集与完整 held-out 生成评测；只有在资源受限时才显式设置上限，并在 `run_config.json` 中保留分母。单独构造偏好数据：

```bash
python scripts/build_preference_pairs.py --qasper data/qasper_train.parquet \
  --pubmedqa data/pubmedqa.json --output-dir outputs/pairs
```

对已有预测重算正式指标：

```bash
python scripts/evaluate_trustworthy_qa.py \
  --predictions outputs/qasper_real/predictions_sft_dpo.json \
  --output outputs/qasper_real/metrics_recheck.json
```

`results/pilot_cpu_96_48.json` records a small local CPU pilot with explicit denominators and split
metadata. It is included for reproducibility and failure analysis only; it is not evidence of a
large-scale training gain or medical accuracy.

SimPO/ORPO 的损失层仍保留在 `src/losses.py`，但本次正式数据入口默认执行 SFT+DPO；没有实际跑出的目标不会写入结果。

数据审计与无泄漏上下文基线：

```bash
python scripts/audit_datasets.py --qasper data/qasper_train.parquet \
  --pubmedqa data/pubmedqa.json --output results/dataset_audit.json
python scripts/run_qasper_protocol.py --qasper data/qasper_train.parquet \
  --output results/protocol_eval.json --top-k 8
```

`run_qasper_protocol.py` 的 full-context、lexical retrieval 和 extractive
模式只从论文原文构造输入，gold evidence 只用于计算 Evidence
Recall/Precision/F1；`--dense-model` 可传入本地 sentence-transformers
模型启用真正的 dense cosine retrieval。由于当前公开文件只有 Qasper
train parquet 和 PubMedQA `ori_pqal.json`，报告会明确标注这一事实，不能
把该 train-only/pool 数据包装成官方 dev/test 结果。

真实模型负例与校准工具：

- `scripts/mine_model_rejections.py` 使用本地 Base 模型实际生成 rejected，
  不再只依赖规则模板。
- `scripts/audit_preference_pairs.py` 对生成负例检查引用和答案一致性，并
  导出人工审核样本；首批 32 条模型负例经过真实 DeBERTa NLI 与规则联合
  过滤后保留 12 条，联合通过率 37.50%，NLI 子检查通过率 53.13%。
- `scripts/coverage_risk.py` 输出 coverage-risk/ECE 诊断；当前 pilot 使用
  引用/拒答启发式 confidence，因此明确标记为 diagnostic-only，不能当成
  模型概率校准结果。
- `scripts/run_filtered_dpo.py` 只接受 `rejected_source=base_model_generation`
  且通过 NLI 的偏好对，按论文 ID 隔离 train/held-out。12-pair pilot 使用 8
  条训练、4 条 held-out；sum log-prob 首次运行发生 DPO loss 发散，保留为负
  结果。改为 completion-token mean 且将 LR 降到 `1e-5` 后，DPO loss 从
  0.6930 降至 0.6670，held-out mean margin 从 SFT 的 3.17 增至 4.94。样本太小，
  该结果仅证明训练稳定性，不能声称回答质量提升。
- `scripts/run_model_coverage_risk.py` 使用模型实际生成 token 的 mean log-prob，
  在互斥论文集合上拟合 Platt scaling。40 条 Qasper validation pilot 的独立
  evaluation 为 16 条，严格正确率为 0、ECE 为 0.0418；低 ECE 只说明模型
  正确地保持低置信度，不能掩盖 Qwen2.5-0.5B 在该任务上能力不足。

### 全量真实训练进度（2026-09）

在上述 pilot 之后，已对本地可用的官方训练池执行全量、论文级隔离流程：

| 阶段 | 真实分母/结果 | 状态 |
| --- | ---: | --- |
| Qasper + PubMedQA 固定数据池 | 3,593 records；2,862 train / 731 validation | complete |
| Base-model rejected generation | 2,862 / 2,862 | complete |
| DeBERTa NLI + citation/consistency filter | 767 / 2,862 kept（26.80%） | complete |
| NLI pair scoring | 10,840 pairs；条件通过率 32.53% | complete |
| Filtered DPO paper split | 712 train pairs / 55 held-out pairs；462 / 34 documents | complete |
| Full Qwen LoRA SFT | 2,862 training records；adapter written | complete |
| Full SFT/DPO answer generation metrics | 731 validation records | pending on CPU run |

全量 rejected、过滤报告和 DPO 指标分别保存在
`results/model_rejections_full_train_v2_manifest.json`、
`results/model_rejection_audit_full_nli/filter_report.json` 和
`results/filtered_dpo_full/metrics.json`。全量 SFT adapter 位于
`results/qasper_full_real/sft_adapter/`，默认被 `.gitignore` 排除；仓库只提交
脚本和小型 metrics，不提交原始 parquet、生成 JSONL、向量库或权重。
由于本地公开文件不包含官方完整 Qasper test 标注，所有上述训练、偏好构造和
阈值选择均未读取 test，不能将 validation 结果包装为官方 test 成绩。人工复核
样本已导出但仍需人工完成，不能称为已完成人工评审。

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
