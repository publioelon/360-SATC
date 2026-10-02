#!/usr/bin/env bash
set -euo pipefail

# Full frozen zero-shot FlowSal evaluation on Sports-360.
# Exact released SST-Sal temporal schedule:
#   T=20, advance=16, discard output positions 0..3, score 4..19.
# Evaluates all 104 numbered videos and all 75,312 frozen target frames.

GPU_ACTIVATE="${GPU_ACTIVATE:-/data/Sports-360/activate_satc_gpu.sh}"
VENV="${VENV:-/home/mininet-ovs/venvs/satc}"
DATASET_ROOT="${DATASET_ROOT:-/data/Sports-360/360_Saliency_dataset_2018ECCV}"
PROTOCOL_DIR="${PROTOCOL_DIR:-/data/Sports-360/protocols/sstsal_official_all104_t20_stride16_output4to19}"
WINDOW_MANIFEST="${WINDOW_MANIFEST:-$PROTOCOL_DIR/window_manifest.csv}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$PROTOCOL_DIR/evaluation_manifest.csv}"
PROTOCOL_HASHES="${PROTOCOL_HASHES:-$PROTOCOL_DIR/SHA256SUMS.txt}"

FLOWSAL_ONNX="${FLOWSAL_ONNX:-/data/D-SAV360/unified_rgb_motionstem_3seed_full_validation/models/Unified_RGB_Motion48_C8_3Seed_Selected_static_opset16.onnx}"
EXPECTED_FLOWSAL_SHA256="${EXPECTED_FLOWSAL_SHA256:-ef437ef88f39e52a7c231529103cd7e18cf80f34607379ba7cb23f4d260e42aa}"

EXP_ROOT="${EXP_ROOT:-/data/Sports-360/flowsal_full104_official_sstsal_protocol}"
RESULT_ROOT="${RESULT_ROOT:-$EXP_ROOT/selected_model_${EXPECTED_FLOWSAL_SHA256:0:12}}"
TRT_CACHE="${TRT_CACHE:-$EXP_ROOT/tensorrt/selected_model}"
GT_CACHE="${GT_CACHE:-$EXP_ROOT/shared_gt_u8_320}"

REQUIRE_TENSORRT="${REQUIRE_TENSORRT:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
RESUME="${RESUME:-1}"
ONLY_SEQUENCES="${ONLY_SEQUENCES:-}"

STAMP="$(date +%F_%H-%M-%S)"
RUN_LOG_DIR="$RESULT_ROOT/logs/$STAMP"
PY_SCRIPT="$RUN_LOG_DIR/run_flowsal_full104.py"
CONSOLE_LOG="$RUN_LOG_DIR/console.log"

mkdir -p "$RUN_LOG_DIR" "$RESULT_ROOT" "$TRT_CACHE" "$GT_CACHE"

finish() {
    status=$?
    echo
    echo "============================================================"
    echo "Sports-360 full FlowSal exit status: $status"
    echo "Experiment directory: $RESULT_ROOT"
    echo "Console log: $CONSOLE_LOG"
    echo "Your terminal remains open."
    echo "============================================================"
}
trap finish EXIT

{
    echo "===== SPORTS-360 FLOWSAL FULL 104-VIDEO ZERO-SHOT EVALUATION ====="
    echo "Dataset: $DATASET_ROOT"
    echo "Protocol: $PROTOCOL_DIR"
    echo "Frozen FlowSal: $FLOWSAL_ONNX"
    echo "Experiment: $RESULT_ROOT"
    echo "Resume: $RESUME"
    echo "Save predictions: $SAVE_PREDICTIONS"
    echo "Only sequences: ${ONLY_SEQUENCES:-ALL}"
    echo

    if [[ -f "$GPU_ACTIVATE" ]]; then
        # shellcheck disable=SC1090
        source "$GPU_ACTIVATE"
    else
        # shellcheck disable=SC1090
        source "$VENV/bin/activate"
    fi

    echo "===== REQUIRED PATHS ====="
    for path in \
        "$VENV/bin/python" \
        "$DATASET_ROOT" \
        "$WINDOW_MANIFEST" \
        "$EVAL_MANIFEST" \
        "$PROTOCOL_HASHES" \
        "$FLOWSAL_ONNX"
    do
        [[ -e "$path" ]] || { echo "ERROR: missing required path: $path"; exit 1; }
        ls -ld "$path"
    done

    echo
    echo "===== VERIFYING FROZEN ARTIFACTS ====="
    echo "$EXPECTED_FLOWSAL_SHA256  $FLOWSAL_ONNX" | sha256sum -c -
    (
        cd "$PROTOCOL_DIR"
        sha256sum -c SHA256SUMS.txt
    )

    export SPORTS360_DATASET_ROOT="$DATASET_ROOT"
    export SPORTS360_WINDOW_MANIFEST="$WINDOW_MANIFEST"
    export SPORTS360_EVAL_MANIFEST="$EVAL_MANIFEST"
    export SPORTS360_PROTOCOL_HASHES="$PROTOCOL_HASHES"
    export SPORTS360_FLOWSAL_ONNX="$FLOWSAL_ONNX"
    export SPORTS360_EXPECTED_FLOWSAL_SHA256="$EXPECTED_FLOWSAL_SHA256"
    export SPORTS360_RESULT_ROOT="$RESULT_ROOT"
    export SPORTS360_TRT_CACHE="$TRT_CACHE"
    export SPORTS360_GT_CACHE="$GT_CACHE"
    export SPORTS360_REQUIRE_TENSORRT="$REQUIRE_TENSORRT"
    export SPORTS360_SAVE_PREDICTIONS="$SAVE_PREDICTIONS"
    export SPORTS360_RESUME="$RESUME"
    export SPORTS360_ONLY_SEQUENCES="$ONLY_SEQUENCES"
    export PYTHONDONTWRITEBYTECODE=1

    cat > "$PY_SCRIPT" <<'PY'
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import onnxruntime as ort

DATASET_ROOT = Path(os.environ["SPORTS360_DATASET_ROOT"])
WINDOW_MANIFEST = Path(os.environ["SPORTS360_WINDOW_MANIFEST"])
EVAL_MANIFEST = Path(os.environ["SPORTS360_EVAL_MANIFEST"])
PROTOCOL_HASHES = Path(os.environ["SPORTS360_PROTOCOL_HASHES"])
FLOWSAL_ONNX = Path(os.environ["SPORTS360_FLOWSAL_ONNX"])
EXPECTED_FLOWSAL_SHA256 = os.environ["SPORTS360_EXPECTED_FLOWSAL_SHA256"]
RESULT_ROOT = Path(os.environ["SPORTS360_RESULT_ROOT"])
TRT_CACHE = Path(os.environ["SPORTS360_TRT_CACHE"])
GT_CACHE = Path(os.environ["SPORTS360_GT_CACHE"])
REQUIRE_TENSORRT = os.environ["SPORTS360_REQUIRE_TENSORRT"] == "1"
SAVE_PREDICTIONS = os.environ["SPORTS360_SAVE_PREDICTIONS"] == "1"
RESUME = os.environ["SPORTS360_RESUME"] == "1"
ONLY_SEQUENCES_RAW = os.environ.get("SPORTS360_ONLY_SEQUENCES", "").strip()

T = 20
ADVANCE = 16
DISCARD_PREFIX = 4
FLOW_W = 192
FLOW_H = 144
EVAL_W = 320
EVAL_H = 240
EPSILON = np.finfo(float).eps
METRIC_PROTOCOL = "author_style_latitude_CC_SIM_bidirectional_symmetric_KLD_u8_v1"

VIDEOS_DIR = RESULT_ROOT / "videos"
GLOBAL_DIR = RESULT_ROOT / "global"
PRED_ROOT = RESULT_ROOT / "predictions"
PREVIEW_ROOT = RESULT_ROOT / "previews"
PROGRESS_JSON = RESULT_ROOT / "progress.json"
SOURCE_HASHES = RESULT_ROOT / "source_hashes.txt"
PER_VIDEO_CSV = GLOBAL_DIR / "per_video_metrics.csv"
PER_FRAME_CSV = GLOBAL_DIR / "per_frame_metrics.csv"
REPORT_JSON = GLOBAL_DIR / "report.json"
REPORT_TXT = GLOBAL_DIR / "report.txt"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def normalize_to_u8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        fail(f"Invalid map range: min={minimum}, max={maximum}")
    return ((array - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)


def normalize(array: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if method == "standard":
        std = float(np.std(values))
        if std <= 0:
            fail("Cannot standard-normalize a constant map")
        return (values - np.mean(values)) / std
    if method == "range":
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum <= minimum:
            fail("Cannot range-normalize a constant map")
        return (values - minimum) / (maximum - minimum)
    if method == "sum":
        total = float(np.sum(values))
        if total <= 0:
            fail("Cannot sum-normalize a non-positive map")
        return values / total
    raise ValueError(method)


def metric_cc(map1: np.ndarray, map2: np.ndarray) -> float:
    a = normalize(map1, "standard")
    b = normalize(map2, "standard")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def metric_sim(map1: np.ndarray, map2: np.ndarray) -> float:
    a = normalize(normalize(map1, "range"), "sum")
    b = normalize(normalize(map2, "range"), "sum")
    return float(np.minimum(a, b).sum())


def metric_kld(p: np.ndarray, q: np.ndarray) -> float:
    p_norm = normalize(p, "sum")
    q_norm = normalize(q, "sum")
    return float(
        np.sum(
            np.where(
                p_norm != 0,
                p_norm * np.log((p_norm + EPSILON) / (q_norm + EPSILON)),
                0,
            )
        )
    )


def author_scores(gt_u8: np.ndarray, pred_u8: np.ndarray) -> tuple[float, float, float]:
    gt = normalize(gt_u8.astype(np.float32), "range")
    pred = normalize(pred_u8.astype(np.float32), "range")
    vertical = np.sin(np.linspace(0, np.pi, gt.shape[0], dtype=np.float64))
    gt = gt * vertical[:, None] + EPSILON
    pred = pred * vertical[:, None] + EPSILON
    gt = normalize(gt, "sum")
    pred = normalize(pred, "sum")
    cc = 0.5 * (metric_cc(gt, pred) + metric_cc(pred, gt))
    sim = 0.5 * (metric_sim(gt, pred) + metric_sim(pred, gt))
    kld = 0.5 * (metric_kld(gt, pred) + metric_kld(pred, gt))
    return float(cc), float(sim), float(kld)


def read_protocol_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    with PROTOCOL_HASHES.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                hashes[Path(parts[1]).name] = parts[0]
    return hashes


def parse_only_sequences() -> set[str] | None:
    if not ONLY_SEQUENCES_RAW:
        return None
    selected: set[str] = set()
    for token in ONLY_SEQUENCES_RAW.replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            selected.update(str(value) for value in range(int(left), int(right) + 1))
        else:
            selected.add(str(int(token)))
    return selected


def load_manifests() -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    windows: dict[str, list[dict[str, str]]] = defaultdict(list)
    evaluations: dict[str, list[dict[str, str]]] = defaultdict(list)

    with WINDOW_MANIFEST.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            windows[row["sequence"]].append(
                {
                    "video_window_index": row["video_window_index"],
                    "input_frame_ids": row["input_frame_ids"],
                    "window_start_frame": row["window_start_frame"],
                    "window_end_frame": row["window_end_frame"],
                }
            )

    with EVAL_MANIFEST.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            evaluations[row["sequence"]].append(
                {
                    "video_sample_index": row["video_sample_index"],
                    "target_frame": row["target_frame"],
                    "official_prediction_name": row["official_prediction_name"],
                    "gt_path": row["gt_path"],
                    "window_output_position_zero_based": row["window_output_position_zero_based"],
                }
            )

    for rows in windows.values():
        rows.sort(key=lambda row: int(row["video_window_index"]))
    for rows in evaluations.values():
        rows.sort(key=lambda row: int(row["video_sample_index"]))

    return dict(windows), dict(evaluations)


def create_session() -> ort.InferenceSession:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")

    available = ort.get_available_providers()
    providers: list[Any] = []
    if "TensorrtExecutionProvider" in available:
        TRT_CACHE.mkdir(parents=True, exist_ok=True)
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(TRT_CACHE),
                    "trt_engine_cache_prefix": "FLOWSAL_SELECTED_SPORTS360_FULL104",
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(TRT_CACHE),
                    "trt_min_subgraph_size": 1,
                    "trt_max_workspace_size": 2147483648,
                },
            )
        )
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(FLOWSAL_ONNX), sess_options=options, providers=providers)

    selected = session.get_providers()
    if REQUIRE_TENSORRT and (not selected or selected[0] != "TensorrtExecutionProvider"):
        fail(f"TensorRT was required but not selected: {selected}")

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.shape != [1, T, 3, FLOW_H, FLOW_W]:
        fail(f"Unexpected FlowSal input metadata: {input_meta.shape}")
    if output_meta.shape != [1, T, 1, FLOW_H, FLOW_W]:
        fail(f"Unexpected FlowSal output metadata: {output_meta.shape}")

    dummy = np.zeros((1, T, 3, FLOW_H, FLOW_W), dtype=np.float32)
    for _ in range(10):
        output = session.run(None, {input_meta.name: dummy})[0]
    if output.shape != (1, T, 1, FLOW_H, FLOW_W) or not np.isfinite(output).all():
        fail("FlowSal warm-up output was invalid")

    return session


def load_video_rgb(sequence: str) -> tuple[np.ndarray, list[str]]:
    sequence_dir = DATASET_ROOT / sequence
    frame_paths = sorted(sequence_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not frame_paths:
        fail(f"No JPEG frames found for sequence {sequence}")

    expected_ids = [f"{index:04d}" for index in range(1, len(frame_paths) + 1)]
    actual_ids = [path.stem for path in frame_paths]
    if actual_ids != expected_ids:
        fail(f"Sequence {sequence} frame IDs are not contiguous from 0001")

    frames = np.empty((len(frame_paths), 3, FLOW_H, FLOW_W), dtype=np.uint8)
    for index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            fail(f"Could not read {frame_path}")
        resized = cv2.resize(frame, (FLOW_W, FLOW_H), interpolation=cv2.INTER_LINEAR)
        frames[index] = resized.transpose(2, 0, 1)
        if (index + 1) % 250 == 0 or index + 1 == len(frame_paths):
            print(f"  [RGB] sequence={sequence} frames={index + 1}/{len(frame_paths)}", flush=True)

    return frames, actual_ids


def load_gt_u8(sequence: str, frame_id: str, gt_path: Path) -> np.ndarray:
    cached = GT_CACHE / sequence / f"{frame_id}.png"
    if cached.is_file():
        image = cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE)
        if image is not None and image.shape == (EVAL_H, EVAL_W):
            return image

    gt = np.squeeze(np.load(gt_path, allow_pickle=False)).astype(np.float32)
    if gt.ndim != 2:
        fail(f"Unexpected GT shape for {gt_path}: {gt.shape}")
    gt_resized = cv2.resize(gt, (EVAL_W, EVAL_H), interpolation=cv2.INTER_AREA)
    gt_u8 = normalize_to_u8(gt_resized)
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(cached), gt_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        fail(f"Could not cache GT {cached}")
    return gt_u8


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def marker_valid(marker_path: Path, expected: dict[str, Any], per_frame_path: Path) -> bool:
    if not marker_path.is_file() or not per_frame_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not all(marker.get(key) == value for key, value in expected.items()):
        return False
    return count_csv_rows(per_frame_path) == expected["evaluated_frames"]


def save_preview(sequence: str, name: str, gt_u8: np.ndarray, pred_u8: np.ndarray) -> None:
    output = PREVIEW_ROOT / sequence
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / f"{name}_gt.png"), gt_u8)
    cv2.imwrite(str(output / f"{name}_flowsal.png"), pred_u8)


def evaluate_video(
    sequence: str,
    window_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    session: ort.InferenceSession,
    model_sha: str,
    protocol_hashes: dict[str, str],
) -> dict[str, Any]:
    video_dir = VIDEOS_DIR / sequence
    video_dir.mkdir(parents=True, exist_ok=True)
    per_frame_path = video_dir / "per_frame_metrics.csv"
    marker_path = video_dir / "complete.json"
    prediction_dir = PRED_ROOT / sequence

    expected = {
        "sequence": sequence,
        "model_sha256": model_sha,
        "window_manifest_sha256": protocol_hashes["window_manifest.csv"],
        "evaluation_manifest_sha256": protocol_hashes["evaluation_manifest.csv"],
        "window_count": len(window_rows),
        "evaluated_frames": len(eval_rows),
        "metric_protocol": METRIC_PROTOCOL,
        "save_predictions": SAVE_PREDICTIONS,
    }

    if RESUME and marker_valid(marker_path, expected, per_frame_path):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        print(f"[SKIP] sequence={sequence} already complete", flush=True)
        return marker

    if SAVE_PREDICTIONS:
        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[VIDEO] sequence={sequence} windows={len(window_rows)} "
        f"evaluated={len(eval_rows)}",
        flush=True,
    )
    video_started = time.perf_counter()
    frames_u8, frame_ids = load_video_rgb(sequence)
    input_name = session.get_inputs()[0].name

    output_rows: list[dict[str, Any]] = []
    inference_times: list[float] = []
    eval_index = 0
    preview_indices = {1, max(1, len(eval_rows) // 2), len(eval_rows)}

    for window_index, window_row in enumerate(window_rows, start=1):
        ids = window_row["input_frame_ids"].split("|")
        if len(ids) != T:
            fail(f"Sequence {sequence} window {window_index} has {len(ids)} inputs")
        indices = [int(frame_id) - 1 for frame_id in ids]
        if any(index < 0 or index >= len(frames_u8) for index in indices):
            fail(f"Sequence {sequence} window {window_index} references an invalid frame")

        clip = frames_u8[indices].astype(np.float32)[None] / 255.0
        started = time.perf_counter()
        prediction = session.run(None, {input_name: clip})[0]
        inference_times.append((time.perf_counter() - started) * 1000.0)
        if prediction.shape != (1, T, 1, FLOW_H, FLOW_W):
            fail(f"Unexpected prediction shape for sequence {sequence}: {prediction.shape}")

        for output_position in range(DISCARD_PREFIX, T):
            if eval_index >= len(eval_rows):
                fail(f"Too many outputs for sequence {sequence}")
            eval_row = eval_rows[eval_index]
            target_frame = ids[output_position]
            expected_name = f"{sequence}_{target_frame}"
            if eval_row["target_frame"] != target_frame:
                fail(
                    f"Target mismatch sequence {sequence}: window gives {target_frame}, "
                    f"manifest gives {eval_row['target_frame']}"
                )
            if eval_row["official_prediction_name"] != expected_name:
                fail(f"Prediction-name mismatch for {expected_name}")
            if int(eval_row["window_output_position_zero_based"]) != output_position:
                fail(f"Output-position mismatch for {expected_name}")

            raw_map = np.asarray(prediction[0, output_position, 0], dtype=np.float32)
            resized = cv2.resize(raw_map, (EVAL_W, EVAL_H), interpolation=cv2.INTER_LINEAR)
            pred_u8 = normalize_to_u8(resized)
            gt_u8 = load_gt_u8(sequence, target_frame, Path(eval_row["gt_path"]))
            cc, sim, kld = author_scores(gt_u8, pred_u8)
            if not all(math.isfinite(value) for value in (cc, sim, kld)):
                fail(f"Non-finite metric for {expected_name}")

            if SAVE_PREDICTIONS:
                destination = prediction_dir / f"{expected_name}.png"
                if not cv2.imwrite(str(destination), pred_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    fail(f"Could not save prediction {destination}")

            sample_number = eval_index + 1
            if sample_number in preview_indices:
                save_preview(sequence, expected_name, gt_u8, pred_u8)

            output_rows.append(
                {
                    "sequence": sequence,
                    "video_sample_index": sample_number,
                    "target_frame": target_frame,
                    "official_prediction_name": expected_name,
                    "CC": cc,
                    "SIM": sim,
                    "KLD_symmetric": kld,
                    "gt_path": eval_row["gt_path"],
                    "prediction_path": str(prediction_dir / f"{expected_name}.png") if SAVE_PREDICTIONS else "",
                }
            )
            eval_index += 1

        if window_index % 10 == 0 or window_index == len(window_rows):
            print(
                f"  [PRED] sequence={sequence} windows={window_index}/{len(window_rows)} "
                f"scored={eval_index}/{len(eval_rows)}",
                flush=True,
            )

    if eval_index != len(eval_rows):
        fail(f"Sequence {sequence}: produced {eval_index}, expected {len(eval_rows)}")
    if len({row["official_prediction_name"] for row in output_rows}) != len(output_rows):
        fail(f"Sequence {sequence}: duplicate target names")

    temporary_csv = per_frame_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_csv.replace(per_frame_path)

    marker = dict(expected)
    marker.update(
        {
            "rgb_frames": len(frame_ids),
            "first_frame": frame_ids[0],
            "last_frame": frame_ids[-1],
            "first_evaluated_frame": output_rows[0]["target_frame"],
            "last_evaluated_frame": output_rows[-1]["target_frame"],
            "CC": float(np.mean([row["CC"] for row in output_rows])),
            "SIM": float(np.mean([row["SIM"] for row in output_rows])),
            "KLD_symmetric": float(np.mean([row["KLD_symmetric"] for row in output_rows])),
            "inference_mean_ms_per_window": float(np.mean(inference_times)),
            "inference_p50_ms_per_window": percentile(inference_times, 50),
            "inference_p95_ms_per_window": percentile(inference_times, 95),
            "duration_seconds": time.perf_counter() - video_started,
            "providers": session.get_providers(),
            "per_frame_metrics": str(per_frame_path),
            "prediction_directory": str(prediction_dir) if SAVE_PREDICTIONS else None,
            "status": "success",
        }
    )
    atomic_write_json(marker_path, marker)

    del frames_u8
    del output_rows
    gc.collect()
    return marker


def write_progress(
    sequences: list[str],
    completed: list[str],
    current: str | None,
    started_at: float,
) -> None:
    atomic_write_json(
        PROGRESS_JSON,
        {
            "status": "running" if len(completed) < len(sequences) else "complete",
            "selected_videos": len(sequences),
            "completed_videos": len(completed),
            "remaining_videos": len(sequences) - len(completed),
            "current_sequence": current,
            "completed_sequences": completed,
            "elapsed_seconds": time.perf_counter() - started_at,
        },
    )


def aggregate(
    sequences: list[str],
    video_markers: dict[str, dict[str, Any]],
    model_sha: str,
    protocol_hashes: dict[str, str],
    total_duration: float,
    providers: list[str],
) -> dict[str, Any]:
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

    per_video_rows: list[dict[str, Any]] = []
    global_sums = {"CC": 0.0, "SIM": 0.0, "KLD_symmetric": 0.0}
    global_count = 0

    temporary_global = PER_FRAME_CSV.with_suffix(".csv.tmp")
    global_writer = None
    with temporary_global.open("w", newline="", encoding="utf-8") as global_handle:
        for sequence in sequences:
            marker = video_markers[sequence]
            video_csv = VIDEOS_DIR / sequence / "per_frame_metrics.csv"
            with video_csv.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if global_writer is None:
                    global_writer = csv.DictWriter(global_handle, fieldnames=reader.fieldnames)
                    global_writer.writeheader()
                for row in reader:
                    global_writer.writerow(row)
                    global_sums["CC"] += float(row["CC"])
                    global_sums["SIM"] += float(row["SIM"])
                    global_sums["KLD_symmetric"] += float(row["KLD_symmetric"])
                    global_count += 1

            per_video_rows.append(
                {
                    "sequence": sequence,
                    "rgb_frames": marker["rgb_frames"],
                    "official_windows": marker["window_count"],
                    "evaluated_frames": marker["evaluated_frames"],
                    "CC": marker["CC"],
                    "SIM": marker["SIM"],
                    "KLD_symmetric": marker["KLD_symmetric"],
                    "inference_mean_ms_per_window": marker["inference_mean_ms_per_window"],
                    "duration_seconds": marker["duration_seconds"],
                }
            )
    temporary_global.replace(PER_FRAME_CSV)

    with PER_VIDEO_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_video_rows)

    frame_weighted = {key: value / global_count for key, value in global_sums.items()}
    macro = {
        key: float(np.mean([float(row[key]) for row in per_video_rows]))
        for key in ("CC", "SIM", "KLD_symmetric")
    }
    video_std = {
        key: float(np.std([float(row[key]) for row in per_video_rows], ddof=1))
        for key in ("CC", "SIM", "KLD_symmetric")
    }
    all_window_latencies = [float(row["inference_mean_ms_per_window"]) for row in per_video_rows]

    report = {
        "status": "success",
        "purpose": "Frozen zero-shot FlowSal evaluation on Sports-360 using the released SST-Sal temporal schedule",
        "method": "FlowSal RGB-only learned motion representation",
        "model": {
            "path": str(FLOWSAL_ONNX),
            "sha256": model_sha,
            "input": [1, T, 3, FLOW_H, FLOW_W],
            "output": [1, T, 1, FLOW_H, FLOW_W],
            "providers": providers,
        },
        "protocol": {
            "videos": len(sequences),
            "official_windows": int(sum(int(row["official_windows"]) for row in per_video_rows)),
            "evaluated_frames": global_count,
            "sequence_length": T,
            "window_advance": ADVANCE,
            "discarded_output_positions": [0, 1, 2, 3],
            "scored_output_positions": list(range(4, 20)),
            "prediction_resize": "raw 144x192 output to 240x320 with OpenCV INTER_LINEAR",
            "prediction_serialization": "per-map minmax, multiply by 255, uint8 truncation",
            "gt_serialization": "official _gt.npy to 240x320 with INTER_AREA, per-map minmax, multiply by 255, uint8 truncation",
            "metric_protocol": METRIC_PROTOCOL,
            "window_manifest_sha256": protocol_hashes["window_manifest.csv"],
            "evaluation_manifest_sha256": protocol_hashes["evaluation_manifest.csv"],
        },
        "zero_shot_controls": {
            "sports360_training": False,
            "sports360_finetuning": False,
            "sports360_checkpoint_selection": False,
            "sports360_threshold_tuning": False,
            "frozen_model": True,
        },
        "metrics": {
            "frame_weighted_global_mean": frame_weighted,
            "macro_video_mean": macro,
            "video_level_standard_deviation": video_std,
        },
        "performance": {
            "mean_of_video_mean_inference_ms_per_window": float(np.mean(all_window_latencies)),
            "total_duration_seconds": total_duration,
        },
        "published_original_sstsal_reference": {
            "CC": 0.439,
            "SIM": 0.284,
            "KLD": 8.610,
            "note": "Literature reference only; this report measures FlowSal.",
        },
        "files": {
            "per_video_metrics": str(PER_VIDEO_CSV),
            "per_frame_metrics": str(PER_FRAME_CSV),
            "predictions": str(PRED_ROOT) if SAVE_PREDICTIONS else None,
            "previews": str(PREVIEW_ROOT),
            "source_hashes": str(SOURCE_HASHES),
        },
    }
    atomic_write_json(REPORT_JSON, report)

    text = f"""SPORTS-360 FLOWSAL FULL ZERO-SHOT EVALUATION
{'=' * 100}
Status: SUCCESS
Videos: {len(sequences)}
Official windows: {report['protocol']['official_windows']}
Evaluated frames: {global_count}
Model SHA-256: {model_sha}
Providers: {providers}

FRAME-WEIGHTED GLOBAL MEAN
{'-' * 100}
CC:             {frame_weighted['CC']:.7f}
SIM:            {frame_weighted['SIM']:.7f}
Symmetric KLD:  {frame_weighted['KLD_symmetric']:.7f}

MACRO VIDEO MEAN ± VIDEO-LEVEL STANDARD DEVIATION
{'-' * 100}
CC:             {macro['CC']:.7f} ± {video_std['CC']:.7f}
SIM:            {macro['SIM']:.7f} ± {video_std['SIM']:.7f}
Symmetric KLD:  {macro['KLD_symmetric']:.7f} ± {video_std['KLD_symmetric']:.7f}

PUBLISHED ORIGINAL SST-SAL SPORTS-360 REFERENCE — LITERATURE COMPARATOR
{'-' * 100}
CC:  0.439
SIM: 0.284
KLD: 8.610

PERFORMANCE
{'-' * 100}
Mean of per-video FlowSal window means: {report['performance']['mean_of_video_mean_inference_ms_per_window']:.3f} ms
Total wall duration: {total_duration / 60.0:.2f} minutes

OUTPUTS
{'-' * 100}
Per-video metrics: {PER_VIDEO_CSV}
Per-frame metrics: {PER_FRAME_CSV}
Report JSON: {REPORT_JSON}
Predictions: {PRED_ROOT if SAVE_PREDICTIONS else 'disabled'}
Previews: {PREVIEW_ROOT}
Experiment directory: {RESULT_ROOT}
"""
    atomic_write_text(REPORT_TXT, text)
    return report


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_PREDICTIONS:
        PRED_ROOT.mkdir(parents=True, exist_ok=True)

    model_sha = sha256(FLOWSAL_ONNX)
    if model_sha != EXPECTED_FLOWSAL_SHA256:
        fail(f"FlowSal SHA mismatch: {model_sha}")
    protocol_hashes = read_protocol_hashes()
    for required in ("window_manifest.csv", "evaluation_manifest.csv"):
        if required not in protocol_hashes:
            fail(f"Missing {required} in protocol hash file")

    SOURCE_HASHES.write_text(
        "\n".join(
            [
                f"{model_sha}  {FLOWSAL_ONNX}",
                f"{sha256(WINDOW_MANIFEST)}  {WINDOW_MANIFEST}",
                f"{sha256(EVAL_MANIFEST)}  {EVAL_MANIFEST}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("[MANIFEST] Loading frozen windows and evaluation targets...", flush=True)
    windows, evaluations = load_manifests()
    sequences = sorted(set(windows) & set(evaluations), key=int)
    selected = parse_only_sequences()
    if selected is not None:
        unknown = selected - set(sequences)
        if unknown:
            fail(f"Requested unknown sequences: {sorted(unknown, key=int)}")
        sequences = [sequence for sequence in sequences if sequence in selected]

    total_windows = sum(len(windows[sequence]) for sequence in sequences)
    total_evaluations = sum(len(evaluations[sequence]) for sequence in sequences)
    if selected is None:
        if len(sequences) != 104 or total_windows != 4707 or total_evaluations != 75312:
            fail(
                f"Frozen protocol mismatch: videos={len(sequences)}, "
                f"windows={total_windows}, frames={total_evaluations}"
            )

    for sequence in sequences:
        expected_count = len(windows[sequence]) * (T - DISCARD_PREFIX)
        if len(evaluations[sequence]) != expected_count:
            fail(
                f"Sequence {sequence}: {len(evaluations[sequence])} targets, "
                f"expected {expected_count}"
            )

    print("SPORTS-360 FLOWSAL FULL ZERO-SHOT EVALUATION")
    print("=" * 100)
    print("Videos:", len(sequences))
    print("Official windows:", total_windows)
    print("Evaluated frames:", total_evaluations)
    print("Frozen model:", FLOWSAL_ONNX)
    print("Model SHA-256:", model_sha)
    print("Save predictions:", SAVE_PREDICTIONS)
    print("Resume:", RESUME)

    print("[ORT] Creating FlowSal session and TensorRT engine...", flush=True)
    session = create_session()
    providers = session.get_providers()
    print("[ORT] Selected providers:", providers, flush=True)

    started = time.perf_counter()
    completed: list[str] = []
    markers: dict[str, dict[str, Any]] = {}
    write_progress(sequences, completed, sequences[0] if sequences else None, started)

    for video_number, sequence in enumerate(sequences, start=1):
        print()
        print(f"===== VIDEO {video_number}/{len(sequences)} — {sequence} =====", flush=True)
        marker = evaluate_video(
            sequence,
            windows[sequence],
            evaluations[sequence],
            session,
            model_sha,
            protocol_hashes,
        )
        markers[sequence] = marker
        completed.append(sequence)
        next_sequence = sequences[video_number] if video_number < len(sequences) else None
        write_progress(sequences, completed, next_sequence, started)
        print(
            f"[DONE] sequence={sequence} "
            f"CC={marker['CC']:.6f} SIM={marker['SIM']:.6f} "
            f"KLD={marker['KLD_symmetric']:.6f} "
            f"completed={video_number}/{len(sequences)}",
            flush=True,
        )

    total_duration = time.perf_counter() - started
    report = aggregate(
        sequences,
        markers,
        model_sha,
        protocol_hashes,
        total_duration,
        providers,
    )
    write_progress(sequences, completed, None, started)

    print()
    print(REPORT_TXT.read_text(encoding="utf-8"))
    print("Frame-weighted metrics:", report["metrics"]["frame_weighted_global_mean"])


if __name__ == "__main__":
    main()
PY

    echo
    echo "===== ACTIVE GPU ====="
    nvidia-smi || true

    echo
    echo "===== PYTHON ENVIRONMENT ====="
    "$VENV/bin/python" - <<'PYENV'
import cv2
import numpy
import onnxruntime
import sys
if hasattr(onnxruntime, "preload_dlls"):
    onnxruntime.preload_dlls(directory="")
print("Python:", sys.version)
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("Available providers:", onnxruntime.get_available_providers())
PYENV

    echo
    echo "===== RUNNING FULL FLOWSAL EVALUATION ====="
    PYTHONUNBUFFERED=1 "$VENV/bin/python" "$PY_SCRIPT"

    echo
    echo "===== FINAL REPORT ====="
    cat "$RESULT_ROOT/global/report.txt"
} 2>&1 | tee "$CONSOLE_LOG"
