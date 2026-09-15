# Advanced Development Roadmap

1. Long context: compare retrieval-augmented Qwen with LongLoRA/YaRN-capable
   checkpoints. Report answer F1 and evidence recall by document-length bucket.
2. Retrieval: compare BM25, dense bi-encoder and ColBERT late interaction on
   paragraph evidence; keep gold evidence out of model inputs during evaluation.
3. Preference quality: mine rejected answers from actual model generations,
   verify them with span matching plus an NLI checkpoint, then manually audit at
   least 100 pairs before DPO.
4. Alignment: compare evidence-constrained SFT, DPO with SFT anchor, SimPO and
   ORPO. Track answer F1, citation support, preference margin and output length.
5. Refusal: calibrate on validation using temperature scaling/selective
   prediction and publish coverage-risk and expected calibration error curves.
6. Agentic retrieval: implement CRAG/Self-RAG-style retry only for low evidence
   coverage. Query rewrite, decomposition or Wikipedia lookup must be logged as
   separate tool calls and separately cited.

No model is promoted from an accuracy-only result. It must improve citation
support or evidence recall without hiding degradation through over-refusal.
