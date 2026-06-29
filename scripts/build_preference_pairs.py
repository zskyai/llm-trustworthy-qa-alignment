"""Build preference pairs for trustworthy QA alignment.

This is a placeholder script. Replace sample records with public or self-built data.
"""

import json
from pathlib import Path

out = Path("data/dpo_train_sample.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

sample = {
    "prompt": "根据给定证据回答问题，并在证据不足时说明无法确定。\n问题：该制度是否适用于外包人员？\n证据：制度原文仅说明适用于正式员工。",
    "chosen": "根据当前证据，无法确定该制度是否适用于外包人员。现有证据只明确提到正式员工，未覆盖外包人员，因此不应扩大解释。",
    "rejected": "该制度适用于外包人员，因为外包人员也属于公司管理范围。",
}

with out.open("w", encoding="utf-8") as f:
    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"wrote {out}")
