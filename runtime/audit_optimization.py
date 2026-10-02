"""Offline equivalence of every optimized class/QP map to the old mapper."""
from pathlib import Path
import csv
import json
import numpy as np
from core import classify, block_classes
from live_model import normalize_saliency_map
from map_projection import reference_scores


def audit_run(run):
    run=Path(run)
    with (run/'inference_classes.csv').open(newline='',encoding='utf-8') as f:
        labels=list(csv.DictReader(f))
    raw_path=run/'map_audit/raw_saliency.f32'
    expected=len(labels)*144*192*4
    if not labels or raw_path.stat().st_size!=expected:
        raise RuntimeError('Raw saliency audit is missing or incomplete.')
    scores=np.load(run/'map_audit/compact_scores.npy',allow_pickle=False)
    if scores.shape!=(len(labels),5,9):
        raise RuntimeError('Compact tile score audit is incomplete.')
    raw=np.memmap(raw_path,mode='r',dtype=np.float32,shape=(len(labels),144,192))
    ema=None; max_error=0.; mismatches=[]
    for i,row in enumerate(labels):
        normalized,_,_=normalize_saliency_map(raw[i])
        reference=reference_scores(normalized)
        max_error=max(max_error,float(np.max(np.abs(reference-scores[i]))))
        ema=reference if ema is None else .7*ema+.3*reference
        expected_classes=classify(ema)
        actual=np.frombuffer(bytes.fromhex(row['classes_5x9_hex']),np.uint8).reshape(5,9)
        if not np.array_equal(expected_classes,actual):
            mismatches.append(int(row['frame_id']))
        # The unchanged block expansion maps the same class grid to the same
        # H.264 per-block offsets. Verify expansion as well as tile decisions.
        if not np.array_equal(block_classes(expected_classes,4096,2048),block_classes(actual,4096,2048)):
            if int(row['frame_id']) not in mismatches: mismatches.append(int(row['frame_id']))
    report={'checked_inferences':len(labels),'class_or_qp_mismatches':len(mismatches),
            'first_mismatched_source_ids':mismatches[:20],'max_abs_tile_score_error':max_error,
            'tile_score_abs_tolerance':1e-6,
            'status':'PASS' if not mismatches and max_error<=1e-6 else 'FAIL',
            'oracle':'previous OpenCV full-resolution resize, blur, ERP weighting and integral sums, followed by original EMA and class rules',
            'scope':'all completed inferences in this session, including warm-up; identical inputs to both postprocessors; performed after streaming'}
    (run/'optimization_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
