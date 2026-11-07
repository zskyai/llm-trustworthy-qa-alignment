# Trustworthy QA Alignment

面向 Qasper 与 PubMedQA 的证据约束问答后训练系统。项目把回答质量拆成可审计的结构化输出：最终答案、引用证据、证据位置、置信度和拒答原因；再用真实模型生成 rejected answers，经过引用校验、答案一致性检查和 NLI 过滤后训练 DPO。

## 项目主线

```text
Qasper / PubMedQA
        |
        +--> paper-level split + dataset audit
        |
        +--> full-context / lexical / dense retrieval baseline
        |
        +--> structured SFT
        |       final_answer + citations + confidence + abstention
        |
        +--> base-model rejected generation
        |       invalid citation / unsupported extension / over-refusal
        |
        +--> citation + consistency + NLI filter
        |
        +--> filtered DPO
        |
        +--> coverage-risk calibration + answer / refusal evaluation
```

核心目标是让模型在证据充分时给出可追溯答案，在证据不足或冲突时降低无依据扩展，并把引用正确性与拒答行为纳入同一套评测。

## 全量真实数据进度

当前本地官方数据池包含 3,593 条记录，按论文级 ID 固定划分为 2,862 条训练记录和 731 条 validation 记录：

| 阶段 | 结果 |
|---|---:|
| Qasper + PubMedQA 数据池 | 3,593 records |
| Paper-level split | 2,862 train / 731 validation |
| Base-model rejected generation | 2,862 / 2,862 |
| DeBERTa NLI + citation/consistency filter | 767 / 2,862 kept（26.80%） |
| NLI pair scoring | 10,840 pairs，条件通过率 32.53% |
| Filtered DPO paper split | 712 train pairs / 55 held-out pairs |
| Full LoRA SFT | 2,862 training records，adapter 已写出 |

过滤后的 DPO 数据进一步隔离为 462 个训练文档和 34 个 held-out 文档。原始 rejected、过滤报告、DPO 指标和 SFT adapter 分别写入 `results/`，大文件默认不提交 Git。

## 真实基线

在 validation 上，Base + full context 的当前基线为：

| 指标 | 结果 |
|---|---:|
| Token F1 | 17.45% |
| Citation validity / support | 25.14% |
| Answerable recall | 62.26% |
| Refusal recall | 66.00% |

协议入口同时支持 full-context、lexical retrieval、dense retrieval 和 extractive evidence 模式，统一输出答案、引用和拒答指标。SFT、filtered DPO 与 refusal calibration 使用同一 paper-level validation 评测脚本，便于比较“回答变好”是否伴随引用质量、过度拒答或长文档遗漏变化。

## 核心模块

| 模块 | 作用 |
|---|---|
| `src/losses.py` | PyTorch 原生 DPO、SimPO、ORPO；支持 completion mask、长度归一化和 SFT anchor |
| `src/structured_answer.py` | 严格答案协议：`final_answer`、`citations`、`confidence`、`abstained`、`refusal_reason` |
| `src/preference_data.py` | invalid citation、citation removed、unsupported extension、over-refusal、polarity flip hard negatives |
| `src/selective_calibration.py` | validation-only coverage-risk threshold、selective risk、abstention rate、ECE |
| `src/evaluation.py` | Token F1、citation precision/recall/support、拒答 precision/recall、过度拒答率、PubMedQA macro-F1、长文档分桶 |
| `scripts/real_pipeline.py` | 全量数据加载、paper split、LoRA SFT、预测和阶段产物 |
| `scripts/mine_model_rejections.py` | 使用 Base 模型实际生成 rejected answers |
| `scripts/audit_preference_pairs.py` | 引用、答案一致性、NLI 过滤和人工审核样本导出 |
| `scripts/run_filtered_dpo.py` | 仅消费通过过滤器的真实偏好对并执行 paper-level DPO |
| `scripts/run_model_coverage_risk.py` | 用模型 token mean log-prob 拟合 coverage-risk calibration |

## 结构化输出

模型完成不是自由文本，而是下面的可验证协议：

```json
{
  "final_answer": "The intervention reduced readmission risk.",
  "citations": [
    {"id": "E1", "evidence": "The study reports a lower 30-day readmission rate.", "location": "results.p3"}
  ],
  "confidence": 0.82,
  "abstained": false,
  "refusal_reason": null
}
```

引用 ID 必须来自输入证据集合；`abstained=true` 时必须给出拒答原因。评测阶段会检查引用位置、证据支持关系和答案与 gold answer 的一致性。

## 偏好数据

chosen 需要同时满足：答案被证据支持、引用位置有效、结论不过度扩展，以及需要拒答时确实拒答。rejected 来自真实模型生成，并覆盖：

- 看似流畅但没有证据支持；
- 引用 ID 或位置错误；
- 证据与结论冲突；
- 应拒答却强行回答；
- 过度拒答。

过滤器组合 evidence span 匹配、NLI/蕴含判断、答案一致性和结构校验，再按论文 ID 隔离 train/held-out。首批全量流程从 2,862 条模型 rejected 中保留 767 条，最终 DPO 训练使用 712 对，55 对用于 held-out 检查。

## 评测维度

### Qasper

- 官方答案 Token F1；
- Evidence Recall / Precision / F1；
- Citation Validity；
- Citation Support Rate；
- 长文档长度分桶；
- Answerable recall、拒答 precision/recall 和过度拒答率。

### PubMedQA

- Accuracy；
- Macro-F1；
- yes/no/maybe 分类分布；
- 拒答 Precision / Recall；
- 过度拒答率和校准误差。

评测脚本还支持至少 100 条人工抽样的一致性记录格式，以及自动评测与人工标签的 agreement 统计。

## 快速开始

环境：Python 3.10+，模型可以使用本地 Hugging Face 路径。

```bash
pip install -r requirements.txt
```

运行全量真实数据的 SFT、预测和阶段指标：

```bash
python scripts/real_pipeline.py \
  --qasper data/qasper_train.parquet \
  --pubmedqa data/pubmedqa.json \
  --model ../models/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/qasper_real
```

`--max-train 0` 和 `--max-generation-eval 0` 表示不截断数据。资源受限时再显式设置上限，运行配置会记录真实分母。

审计数据并运行检索协议：

```bash
python scripts/audit_datasets.py \
  --qasper data/qasper_train.parquet \
  --pubmedqa data/pubmedqa.json \
  --output results/dataset_audit.json
python scripts/run_qasper_protocol.py \
  --qasper data/qasper_train.parquet \
  --output results/protocol_eval.json \
  --top-k 8 \
  --dense-model sentence-transformers/all-MiniLM-L6-v2
```

构造偏好对、审计过滤并执行 DPO：

```bash
python scripts/build_preference_pairs.py \
  --qasper data/qasper_train.parquet \
  --pubmedqa data/pubmedqa.json \
  --output-dir outputs/pairs
python scripts/mine_model_rejections.py \
  --input outputs/pairs/train.jsonl \
  --model ../models/Qwen2.5-0.5B-Instruct \
  --output outputs/model_rejections/train_rejected.jsonl
python scripts/audit_preference_pairs.py \
  --input outputs/model_rejections/train_rejected.jsonl \
  --output-dir outputs/pair_audit
python scripts/run_filtered_dpo.py \
  --pairs outputs/pair_audit/filtered.jsonl \
  --model ../models/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/filtered_dpo
```

对已有预测重算指标并运行校准：

```bash
python scripts/evaluate_trustworthy_qa.py \
  --predictions outputs/qasper_real/predictions_sft_dpo.json \
  --output outputs/qasper_real/metrics_recheck.json
python scripts/run_model_coverage_risk.py \
  --qasper data/qasper_train.parquet \
  --model ../models/Qwen2.5-0.5B-Instruct \
  --output outputs/qasper_real/coverage_risk.json
```

## 实验协议

- 使用官方 Qasper 与 PubMedQA 文件；数据审计记录问题类型、answerability、evidence span、yes/no/maybe、文档长度和重复情况。
- 论文级划分后，train 用于 SFT/DPO，validation 用于阈值和模型选择；test 不参与偏好构造、校准或训练。
- 结果同时报告答案、证据、引用、拒答和长上下文分桶，避免只用单一准确率评价可信问答。
- `results/pilot_cpu_96_48.json` 等小规模记录用于复现和失败分析；全量运行的分母、配置和产物以对应 `run_config.json` 与 `results/` 清单为准。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 completion mask、长度归一化、DPO/SimPO/ORPO 数值边界、结构化答案、citation ID、偏好过滤、coverage-risk 校准和 ECE。

## 数据与许可证

Qasper 由 AllenAI 发布，PubMedQA 使用官方 `ori_pqal.json`。原始数据、生成 JSONL、向量库和模型权重不提交仓库；代码使用 MIT License。
