# Reproducibility records

`pilot_cpu_96_48.json` is a local CPU pilot record for the public Qasper/PubMedQA data protocol.
It contains configuration, fixed document-level split metadata, metric denominators and aggregate
results only. Raw source documents, model weights and generated answers are intentionally excluded.

The pilot is a small protocol check, not a production or clinical result. Generation metrics use 24
held-out examples; pair metrics use 48 preference examples. In particular, `valid_citation_rate` only
checks whether an emitted citation ID is syntactically valid, not whether the cited evidence entails
the answer. `qasper_answerable_non_refusal_rate` measures non-refusal on answerable examples and is not
an answer-correctness metric.

