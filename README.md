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

## 应用：Paper Evidence Desk

研究者可以先按标题/摘要搜索 Qasper 论文，再针对选定论文提问。系统按章节召回、在段落级用 BM25 + MMR 选择证据，返回逐句原文、section/paragraph 位置和可复制 citation ID；没有词项证据时明确拒答。每次问题、证据和 Useful / Needs review 反馈会持久化到 SQLite，形成可回放的审阅历史。

工作台提供两种模式：原文摘录，以及本地模型生成后进行 citation / sentence / NLI 检查。生成模式通过原文位置解析引用，失败时扩展证据、修复一次，仍不通过则拒答；trace 保存模型原始输出、验证结果、输入输出 token 和耗时。当前接入 Base checkpoint；已有 SFT/DPO adapter 的生成评测单独报告。

支持导入真实论文 PDF，按 SHA-256 去重，保存原文件和页码；重启后恢复论文库。界面可打开完整段落、查看原 PDF、提交复核备注并下载 JSON 证据报告。PubMedQA 可加载官方 context 参与文献检索与问答，final_decision、LONG_ANSWER 和预测标签不进入索引；yes/no/maybe 指标仍走独立分类评测。

```bash
pip install -r requirements.txt
python scripts/serve_research_workbench.py \
  --qasper data/qasper_train.parquet \
  --pubmedqa data/pubmedqa.json \
  --history-db results/paper_qa_history.sqlite3 \
  --port 8100
```

打开 `http://127.0.0.1:8100/`。要启用本地生成，追加 `--model ../../models/Qwen2.5-0.5B-Instruct --nli-model ../../models/cross-encoder__nli-deberta-v3-base`。PDF 默认保存在 `data/paper_library/`，历史和反馈保存在 SQLite。扫描件需要先做 OCR，双栏论文需要对照原文件检查文本顺序。服务默认仅监听本机。

应用评测在固定论文划分的全部 **189 篇 / 525 个留出问题**上完成。推理输入仅含正文，未输入标注证据。结果见 `results/research_application_validation_2026-11-07/summary.json`：

| 原文摘录模式指标 | 结果 |
|---|---:|
| 完成问题 / 失败 | 525 / 0 |
| Answer Token F1 proxy | 6.69% |
| Evidence Recall / Precision proxy | 24.95% / 6.96% |
| 原文句子位置有效率 | 100.00% |
| 拒答 Recall / 过度拒答率 | 9.80% / 1.27% |
| 本地函数耗时 P50 / P95 | 1.83 / 4.62 ms |

这些是 token overlap / span 匹配代理评分，不是官方 Qasper scorer；耗时不含 HTTP、PDF 导入或模型生成。525 问题划分与混合训练流程的 543 条 Qasper validation 不同。模型生成完成真实接入检查，仍出现漏引和错误句号并触发拒答，尚无全量生成模式收益结论。

```bash
python scripts/evaluate_research_application.py \
  --qasper data/qasper_train.parquet \
  --output results/research_application_validation_new
python scripts/export_research_feedback.py \
  --db results/paper_qa_history.sqlite3 \
  --output results/review_candidates/feedback.jsonl
```

反馈导出用于逐条审计，包含原始回答和复核意见。转成训练样本前需要检查论文划分、证据位置、答案一致性及 NLI，人工反馈不会自动进入 SFT/DPO。

使用 `python scripts/import_evaluation_reviews.py --predictions results/research_application_validation_2026-11-07/predictions.jsonl` 将全部 525 条预测导入审阅历史。导入保留 split、run hash 和代理评分，重复执行跳过已有记录；不会代填人工意见。

## 全量段落检索优化（2026-11-08）

在相同的 189 篇论文、525 个 official-train 论文留出问题上，比较章节优先检索、全局 BM25，以及真实预训练 Cross-Encoder 重排。推理只接收问题和正文；标注证据仅用于评分。

| 方法 | Evidence Precision | Evidence Recall | Evidence F1 |
|---|---:|---:|---:|
| 章节优先 Top-6 | 8.62% | 37.03% | 13.30% |
| 全局 BM25 Top-6 | 9.40% | 39.52% | 14.24% |
| 全局 BM25 Top-24 + Cross-Encoder Top-6 | 12.41% | 51.15% | 18.69% |
| 全局 BM25 Top-24 + Cross-Encoder Top-12 | 7.97% | 58.94% | 13.00% |

按官方 evaluator 的完整段落精确匹配、多个标注取最佳 Evidence F1 评分。全部 525 问、3,150 次评分与官方代码逐项一致，见 `results/paragraph_retrieval_holdout525_v2/official_scorer_parity.json`。旧表中的 highlighted-span proxy 使用不同口径，不能与此表直接比较。

长文档（至少 4,096 words，192 问）的 Top-6 Recall 从 37.72% 提升至 49.22%；短文档 Evidence F1 则从 17.40% 降到 16.67%。这一轮验证的是检索改善，生成答案质量仍需独立评测。MiniLM Cross-Encoder 为现有预训练模型，未在这 525 问上训练。

工作台增加 `--paragraph-reranker LOCAL_CHECKPOINT`，使用相同的全局召回和神经重排，保留原始 section / paragraph / sentence 位置。完整运行见 `scripts/evaluate_paragraph_retrieval.py`；v2 修正了 Top-6 章节检索的候选构造，替代 v1 的结果汇总。

## PubMedQA 划分审计

`src/clean_pubmedqa.py` 固定 PQA-L 的 450/50/500 manifest，并在 SFT、DPO、校准和评测入口校验 source hash、base checkpoint fingerprint 与 ID 集合。`scripts/prepare_clean_pubmedqa.py` 生成 450 条训练和 50 条 dev 的完整 evidence-position 记录；`scripts/train_clean_pubmedqa.py` 使用可恢复 LoRA SFT，`scripts/evaluate_clean_pubmedqa_stages.py` 对同一 base checkpoint 配对评测 Base/SFT/SFT+DPO，test 不被物化到训练数据。

```bash
python scripts/prepare_clean_pubmedqa.py --source data/pubmedqa.json --manifest results/pubmedqa_official_protocol_audit_v1/split_manifest.json --nli-model ../../models/cross-encoder__nli-deberta-v3-base --output results/pubmedqa_clean450_data_v1
python scripts/train_clean_pubmedqa.py --source data/pubmedqa.json --manifest results/pubmedqa_official_protocol_audit_v1/split_manifest.json --prepared-data results/pubmedqa_clean450_data_v1 --base-model ../../models/Qwen2.5-0.5B-Instruct --output results/pubmedqa_clean450_sft_v1 --resume
```

`src/led_evidence_scaffold.py` 提供 LED-16K 的证据 scaffold，保留段落、句子和 citation provenance，并在输入超过 encoder budget 时拒绝截断。`src/sentence_evidence_selector.py` 将 Qasper 段落候选送入真实 Cross-Encoder 做句子选择，官方兼容 scorer 按来源段落计算结果。

`src/evidence_set_selector.py` 在句子 rerank 分数之外加入 question-term coverage gain、section prior、跨句冗余和冲突词惩罚，采用有界 listwise greedy 选择；`SentenceEvidenceSelector.select_coverage_aware` 直接返回保留原始 paragraph/sentence ID 的证据集合。它解决 Top-k 只扩上下文导致 precision/F1 下降的问题，选择参数应在 paper-level validation 上拟合，不能从 test 反推。

`src/preference_data.py::audit_model_preference_candidate` 将真实模型 rejected 按 chosen NLI entailment、citation 合法性、答案一致性和 failure type 过滤，保存每个 pair 的检查结果和 `test_used=false` provenance；未通过的流畅回答不会进入 DPO。

### 当前 PubMedQA 训练产物

按固定的 450 train / 50 dev / 500 test manifest 重建证据监督。450 条训练记录全部包含 evidence span、位置和 NLI silver confidence；`pubmedqa_clean450_sft_v2` 完成 1 epoch CPU LoRA SFT，最终平均 loss 为 0.6189。基于真实模型 rejected 的 2,420 条 train-only 候选，经过 citation、答案一致性和 DeBERTa NLI 过滤保留 736 对（通过率 30.41%），并导出 100 条人工审核样本。产物分别见 `results/pubmedqa_clean450_sft_v2/sft_adapter/provenance.json` 和 `results/alignment_pair_audit_v2/filter_report.json`。

这 736 对已经通过数据过滤，`filtered_dpo_train_only_v2` 记录了 CPU 训练状态和输入清单；该运行尚未产生 checkpoint 与正式指标。当前可复现结论是：证据监督与偏好数据链路已完成，DPO 仍需在同一 manifest 上完成训练后再进入 Base/SFT/DPO 配对评测。

官方 PQA-L 的 1,000 条有标签记录先隔离 500 条 test，当前固定使用 fold-0 的 450 train / 50 dev manifest。`src/pubmedqa_protocol.py` 按官方代码重建划分，`results/pubmedqa_official_protocol_audit_v1/split_manifest.json` 保存三组 ID、source hash 和 fold 配置；训练、偏好构造、校准均只读取 train/dev。

历史 `qasper_full_real` 与 `pairs_from_repo` 的 812 个 PubMedQA 训练 ID 中，有 404 个落入重建 test；`filtered_dpo_full` 的 53 个 PubMedQA 训练 ID 中有 29 个落入。相关历史权重不能用于声称该 test 上的独立泛化效果。新混合训练与 filtered DPO 必须提供 ID manifest，并在训练前拒绝 test 或未知 ID。详见 `results/pubmedqa_official_protocol_audit_v1/`。

## 历史全量训练记录

历史混合数据池包含 3,593 条记录，曾按论文级 ID 划分为 2,862 条训练记录和 731 条 validation 记录。下表保留作模型行为对照，当前 PubMedQA 结论以固定 450/50/500 manifest 为准：

| 阶段 | 结果 |
|---|---:|
| Qasper + PubMedQA 数据池 | 3,593 records |
| Paper-level split | 2,862 train / 731 validation |
| Base-model rejected generation | 2,862 / 2,862 |
| DeBERTa NLI + citation/consistency filter | 767 / 2,862 kept（26.80%） |
| NLI pair scoring | 10,840 pairs，条件通过率 32.53% |
| Filtered DPO paper split | 712 train pairs / 55 held-out pairs |
| Full LoRA SFT | 2,862 training records，adapter 已写出 |

该历史 run 的过滤数据进一步隔离为 462 个训练文档和 34 个 held-out 文档。原始 rejected、过滤报告、DPO 指标和 SFT adapter 分别写入 `results/`，大文件默认不提交 Git。

## 生成基线（Oracle Evidence）

`results/qasper_full_real/metrics_base.json` 在 731 条混合 paper-level validation（543 条 Qasper、188 条 PubMedQA）上评估 Base 模型。Qasper prompt 使用官方标注的 evidence span 作为输入，因此这是 **oracle evidence 条件下的生成基线**，不代表 full-context 或检索端到端结果：

| 指标 | 结果 |
|---|---:|
| Token F1 | 17.45% |
| 输出 citation ID 合法率 | 25.14% |
| 可回答问题中至少给出一个合法 citation | 3.52% |
| 混合数据中可回答问题回答率 | 62.26% |
| 不可回答问题拒答召回率 | 66.00% |

该历史基线没有运行语义 NLI 引用支持评测，因此合法 citation ID 不等同于 citation support。用同一 731 条 validation、同一 Base checkpoint 和固定 greedy 解码完成的 Base、全量 SFT 与 filtered-pair SFT+DPO 对照保存在 `results/qasper_stage_eval_2026-11-07/comparison.json`；它用于回归分析，严格 PubMedQA manifest 的训练状态见上方。

| 历史混合划分阶段 | Token F1 | PubMedQA Accuracy / Macro-F1 | Citation ID 合法率 | Answerable recall | 拒答 recall | 过度拒答率 | 长文档 Token F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | **17.74%** | **56.91% / 33.77%** | 34.51% | 50.10% | **66.00%** | 49.90% | **16.05%** |
| 全量 SFT（2,862 条） | 4.42% | 0.00% / 0.00% | 56.64% | 100.00% | 0.00% | 0.00% | 4.05% |
| Filtered-pair SFT+DPO（712 对，独立训练） | 13.44% | 56.38% / **35.89%** | **85.04%** | 45.84% | 62.00% | 54.16% | 15.33% |

结果没有显示训练带来整体答案质量提升：全量 SFT 发生明显退化，预测集中为格式化数字序列；DPO 恢复了大部分答案和分类表现，并显著提高 citation ID 合法率，但 Token F1 仍比 Base 低 4.30 个百分点。按 Qasper 子集重算后，DPO 的 answerable recall 比 Base 低 4.26 个百分点，过度拒答率高 4.26 个百分点，说明引用合法率的提升伴随更多拒答。这里的 Citation ID 合法率只检查引用编号是否属于输入证据，`citation_support_rate` 未测，不能据此声称引用语义正确。

此比较使用 Qasper 标注 evidence span 作为模型输入，属于 oracle-evidence 生成评测，不是检索端到端结果；数据是训练池内按论文划分的 validation，不是官方 test。PubMedQA 来自 `ori_pqal` 训练池，也没有官方 test 指标。模型 checkpoint 为本地 `Qwen2.5-0.5B-Instruct`，解码为 greedy、最多生成 64 tokens、CPU float32、batch size 2；SFT adapter 和 filtered-pair SFT+DPO adapter 是各自独立训练产物。该 run 未记录逐阶段生成延迟，也没有使用 test 数据、NLI citation support 标签或人工评审。

`run_qasper_protocol.py` 独立评测 full-context、lexical、dense、extractive 和 hierarchical context 的证据覆盖；hierarchical 模式先聚合章节相关性，再用 MMR 选择互补段落。

在实际 Qasper paper-level validation（525 个问题，418 个长文档问题）上，章节优先的 BM25 + MMR 已优于段落词项重叠基线：

| 方法 | Evidence Recall | Evidence Precision | 长文档 Recall |
|---|---:|---:|---:|
| Paragraph lexical Top-8 | 26.36% | 4.33% | 24.95% |
| Section-first BM25 + MMR Top-8 | **29.75%** | **4.91%** | **28.03%** |

结果保存在 `results/qasper_hierarchical_validation_2026-11-07.json`；这是训练池内的 paper-level validation，不是官方 test。

## 核心模块

| 模块 | 作用 |
|---|---|
| `src/losses.py` | PyTorch 原生 DPO、SimPO、ORPO；支持 completion mask、长度归一化和 SFT anchor |
| `src/structured_answer.py` | 严格答案协议：`final_answer`、`citations`、`confidence`、`abstained`、`refusal_reason` |
| `src/agentic_qa.py` | 有界 QA agent：检索、生成、引用/NLI 核验、扩大候选后重试或拒答 |
| `src/hierarchical_retrieval.py` | 章节级父节点召回、段落级 BM25 与 MMR 多样性选择 |
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

## Agent 工作流

`EvidenceQAAgent` 将检索器、生成模型和支持性 verifier 作为工具注入，按固定轮次运行：初次检索和结构化生成、引用 ID 校验、NLI 支持度检查；若校验失败则扩大检索 top-k 并要求模型基于新增证据修正，重试预算耗尽后输出带原因的拒答。每次工具调用和校验结果都会写入 `trace`，便于审计和离线分析。

```python
agent = EvidenceQAAgent(
    retrieve=hybrid_retriever.search,
    generate=local_model.generate_structured,
    support_score=nli_verifier.score_answer,
    max_retries=1,
    top_k=8,
)
result = agent.run(question)
```

检索和生成通过适配器接入，因而可以替换为 BM25 + Milvus、不同本地模型或线上推理端点；支持性核验应使用经过目标语料验证的 NLI 模型。

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
- 新实验先冻结数据集的 train/dev/test ID，再在 train 内按论文划分模型选择数据；历史混合训练的 PubMedQA 划分问题以上方审计为准。
- 结果同时报告答案、证据、引用、拒答和长上下文分桶，避免只用单一准确率评价可信问答。
- `results/pilot_cpu_96_48.json` 等小规模记录用于复现和失败分析；全量运行的分母、配置和产物以对应 `run_config.json` 与 `results/` 清单为准。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 completion mask、长度归一化、DPO/SimPO/ORPO 数值边界、结构化答案、citation ID、偏好过滤、coverage-risk 校准和 ECE。

拒答阈值在满足 validation 风险上限的候选中选择最低阈值，以最大化覆盖率；可行性使用 95% Wilson 风险上界，降低小样本零错误造成的乐观偏差。test 不参与阈值搜索。

## 数据与许可证

Qasper 由 AllenAI 发布，PubMedQA 使用官方 `ori_pqal.json`。原始数据、生成 JSONL、向量库和模型权重不提交仓库；代码使用 MIT License。
