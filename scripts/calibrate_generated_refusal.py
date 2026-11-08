"""Select a refusal confidence threshold on validation predictions only."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selective_calibration import CoverageRiskCalibrator
from src.clean_pubmedqa import sha256, read_jsonl

def calibration_inputs(rows, manifest):
    expected = set(map(str, manifest['validation_ids']))
    ids = [str(r['document_id']) for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Calibration requires exactly the frozen dev IDs')
    confidences, correct = [], []
    for row in rows:
        if row.get('split') != 'validation':
            raise ValueError('Calibration predictions must explicitly be validation')
        confidence = row.get('classification_confidence')
        if confidence is None:
            pred = row.get('prediction')
            try: pred = json.loads(pred) if isinstance(pred, str) else pred
            except json.JSONDecodeError as exc: raise ValueError('Invalid structured confidence JSON') from exc
            confidence = pred.get('confidence') if isinstance(pred, dict) else None
        if isinstance(confidence, bool) or not isinstance(confidence, (int,float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Every row needs a finite model confidence in [0,1]')
        label = row.get('classification_label', row.get('predicted_label'))
        if row.get('gold_label') not in ('yes','no','maybe'):
            raise ValueError('This calibrator uses PQA-L exact label correctness')
        confidences.append(confidence)
        correct.append(label == row['gold_label'])
    return confidences, correct

def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--target-risk',type=float,default=.10); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    data=a.predictions.read_text(encoding='utf8')
    rows=json.loads(data) if data.lstrip().startswith('[') else read_jsonl(a.predictions)
    manifest=json.loads(a.manifest.read_text(encoding='utf8'))
    confidences, correct=calibration_inputs(rows,manifest)
    calibrator=CoverageRiskCalibrator(target_risk=a.target_risk); threshold=calibrator.fit(confidences,correct); summary=calibrator.summarize(confidences,correct)
    report={'threshold':threshold,'reject_all':calibrator.reject_all,'coverage':summary.coverage,'selective_risk':summary.selective_risk,'abstention_rate':summary.abstention_rate,'ece':summary.ece,'risk_upper_bound':summary.risk_upper_bound,'target_risk':a.target_risk,'validation_only':True,'test_used':False,'manifest_sha256':sha256(a.manifest),'predictions_sha256':sha256(a.predictions),'correctness_definition':'Exact PQA-L label, not semantic rationale support','risk_bound_definition':'Pointwise Wilson upper bound selected on dev; not a simultaneous or test-risk guarantee'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report))
if __name__=='__main__': main()
