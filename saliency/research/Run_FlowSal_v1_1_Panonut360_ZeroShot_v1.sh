#!/usr/bin/env bash
set -euo pipefail

VENV="/home/mininet-ovs/venvs/satc"
PANONUT_ROOT="/data/Panonut360"
VIDEO_DIR="$PANONUT_ROOT/extracted/videos/Video"
SALIENCY_DIR="$PANONUT_ROOT/extracted/tracking/SaliencyMap"
SOURCE_MANIFEST="$PANONUT_ROOT/audit/panonut360_source_manifest.csv"

FLOWSAL_ONNX="/data/D-SAV360/flowsal/models/FlowSal_R192_T20_C8_v1_1_Robust1000_static_opset16.onnx"
EXPECTED_FLOWSAL_SHA256="51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa"

BASELINE_RUN="$PANONUT_ROOT/evaluation_sstsal_teacher_vs_kd_r192_w24_t20/results/2026-07-26_22-24-25"
BASELINE_CSV="$BASELINE_RUN/per_second_metrics.csv"
BASELINE_REPORT="$BASELINE_RUN/report.json"
BASELINE_PROTOCOL="$BASELINE_RUN/frozen_protocol.json"
BASELINE_SALIENCY_HASHES="$BASELINE_RUN/saliency_array_hashes.txt"
BASELINE_MANIFEST_HASH="$BASELINE_RUN/dataset_manifest_hash.txt"

EXPECTED_BASELINE_CSV_SHA256="043a2b8bb1242f59dfba3504c983c6f724df346ecdc5e8732aaedaf482e3d7a6"
EXPECTED_BASELINE_REPORT_SHA256="18f8f83b08491d0595792777dc7f8e3916220c2c2f5858372312675144ff5032"
EXPECTED_BASELINE_PROTOCOL_SHA256="9de5336b2f599eb14277362db24293d7934541747b23993e04d65f503eb7b8f1"

EXP_ROOT="$PANONUT_ROOT/evaluation_flowsal_v1_1_robust1000_zero_shot"
STAMP="$(date +%F_%H-%M-%S)"
RUN_DIR="$EXP_ROOT/results/$STAMP"
PY_SCRIPT="$RUN_DIR/run_flowsal_panonut360_zero_shot.py"
CONSOLE_LOG="$RUN_DIR/console.log"
COMPLETION_MARKER="$EXP_ROOT/FLOWSAL_V1_1_PANONUT360_ALREADY_EVALUATED.json"

BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-100000}"
KEEP_TEMP="${KEEP_TEMP:-0}"
SAVE_PREVIEWS="${SAVE_PREVIEWS:-1}"
ALLOW_RERUN="${ALLOW_RERUN:-0}"

mkdir -p "$RUN_DIR"

NVIDIA_SITE_LIBS="$("$VENV/bin/python" - <<'PY'
from pathlib import Path
import site

patterns = (
    "libcudnn.so*", "libcublas.so*", "libcudart.so*", "libcufft.so*",
    "libcurand.so*", "libcusolver.so*", "libcusparse.so*", "libnvrtc.so*",
    "libnvJitLink.so*", "libnvinfer.so*", "libnvinfer_plugin.so*",
    "libnvonnxparser.so*",
)

directories = set()
for root_name in site.getsitepackages():
    root = Path(root_name)
    if not root.is_dir():
        continue
    for pattern in patterns:
        for library in root.rglob(pattern):
            if library.is_file() or library.is_symlink():
                directories.add(str(library.parent.resolve()))
print(":".join(sorted(directories)))
PY
)"

GPU_LIBRARY_PATH=""
for directory in \
    "/usr/local/cuda/lib64" \
    "/usr/local/cuda-12.6/lib64" \
    "/usr/local/cuda-12.8/lib64" \
    "/usr/local/TensorRT/lib" \
    "/usr/lib/x86_64-linux-gnu" \
    "/lib/x86_64-linux-gnu"
do
    if [[ -d "$directory" ]]; then
        GPU_LIBRARY_PATH="${GPU_LIBRARY_PATH:+$GPU_LIBRARY_PATH:}$directory"
    fi
done
if [[ -n "$NVIDIA_SITE_LIBS" ]]; then
    GPU_LIBRARY_PATH="${GPU_LIBRARY_PATH:+$GPU_LIBRARY_PATH:}$NVIDIA_SITE_LIBS"
fi
export LD_LIBRARY_PATH="${GPU_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

finish_shell() {
    status=$?
    echo
    echo "============================================================"
    echo "FlowSal Panonut360 zero-shot exit status: $status"
    echo "Run directory: $RUN_DIR"
    echo "Console log: $CONSOLE_LOG"
    echo "============================================================"
    echo "The terminal remains open. Type exit when finished."
    exec bash -i
}
trap finish_shell EXIT

{
    echo "===== FLOWSAL v1.1 PANONUT360 ZERO-SHOT EVALUATION ====="
    echo "FlowSal frozen before evaluation: YES"
    echo "Training/fine-tuning on Panonut360: NO"
    echo "Checkpoint selection on Panonut360: NO"
    echo "SEA-RAFT executed in this run: NO"
    echo "Teacher/KD inference repeated: NO"
    echo "Frozen baseline rows expected: 3901"
    echo "Bootstrap repetitions: $BOOTSTRAP_REPEATS"
    echo

    if [[ -f "$COMPLETION_MARKER" && "$ALLOW_RERUN" != "1" ]]; then
        echo "ERROR: A completed FlowSal Panonut360 marker already exists:"
        echo "$COMPLETION_MARKER"
        echo
        echo "This safeguard prevents repeated external-test evaluation."
        exit 1
    fi

    echo "===== REQUIRED PATHS ====="
    for path in \
        "$VENV/bin/python" \
        "$VIDEO_DIR" \
        "$SALIENCY_DIR" \
        "$FLOWSAL_ONNX" \
        "$BASELINE_CSV" \
        "$BASELINE_REPORT" \
        "$BASELINE_PROTOCOL" \
        "$BASELINE_SALIENCY_HASHES"
    do
        if [[ ! -e "$path" ]]; then
            echo "ERROR: missing required path: $path"
            exit 1
        fi
        ls -ld "$path"
    done

    command -v ffmpeg >/dev/null
    command -v ffprobe >/dev/null

    echo
    echo "===== VERIFYING FROZEN MODEL AND BASELINE ====="
    echo "$EXPECTED_FLOWSAL_SHA256  $FLOWSAL_ONNX" | sha256sum -c -
    echo "$EXPECTED_BASELINE_CSV_SHA256  $BASELINE_CSV" | sha256sum -c -
    echo "$EXPECTED_BASELINE_REPORT_SHA256  $BASELINE_REPORT" | sha256sum -c -
    echo "$EXPECTED_BASELINE_PROTOCOL_SHA256  $BASELINE_PROTOCOL" | sha256sum -c -

    echo
    echo "===== VERIFYING PANONUT360 GROUND TRUTH ====="
    sha256sum -c "$BASELINE_SALIENCY_HASHES"
    if [[ -f "$SOURCE_MANIFEST" && -f "$BASELINE_MANIFEST_HASH" ]]; then
        sha256sum -c "$BASELINE_MANIFEST_HASH"
    fi

    VIDEO_COUNT="$(find "$VIDEO_DIR" -maxdepth 1 -type f -iname '*.mp4' | wc -l)"
    NPY_COUNT="$(find "$SALIENCY_DIR" -maxdepth 1 -type f -iname '*.npy' | wc -l)"
    echo "Videos: $VIDEO_COUNT"
    echo "Saliency arrays: $NPY_COUNT"
    if [[ "$VIDEO_COUNT" -ne 15 || "$NPY_COUNT" -ne 15 ]]; then
        echo "ERROR: expected 15 videos and 15 saliency arrays."
        exit 1
    fi

    echo
    echo "===== ENVIRONMENT ====="
    nvidia-smi
    "$VENV/bin/python" - <<'PY'
import sys
import cv2
import numpy
import onnxruntime
if hasattr(onnxruntime, "preload_dlls"):
    onnxruntime.preload_dlls(directory="")
print("Python:", sys.executable)
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("Providers:", onnxruntime.get_available_providers())
PY

    export PANONUT_FLOWSAL_RUN_DIR="$RUN_DIR"
    export PANONUT_FLOWSAL_EXP_ROOT="$EXP_ROOT"
    export PANONUT_FLOWSAL_ONNX="$FLOWSAL_ONNX"
    export PANONUT_BASELINE_CSV="$BASELINE_CSV"
    export PANONUT_BASELINE_REPORT="$BASELINE_REPORT"
    export PANONUT_COMPLETION_MARKER="$COMPLETION_MARKER"
    export PANONUT_BOOTSTRAP_REPEATS="$BOOTSTRAP_REPEATS"
    export PANONUT_KEEP_TEMP="$KEEP_TEMP"
    export PANONUT_SAVE_PREVIEWS="$SAVE_PREVIEWS"

    cat > "$PY_SCRIPT" <<'PY'
from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")

ROOT = Path("/data/Panonut360")
VIDEO_DIR = ROOT / "extracted" / "videos" / "Video"
SALIENCY_DIR = ROOT / "extracted" / "tracking" / "SaliencyMap"

RUN_DIR = Path(os.environ["PANONUT_FLOWSAL_RUN_DIR"])
EXP_ROOT = Path(os.environ["PANONUT_FLOWSAL_EXP_ROOT"])
FLOWSAL_ONNX = Path(os.environ["PANONUT_FLOWSAL_ONNX"])
BASELINE_CSV = Path(os.environ["PANONUT_BASELINE_CSV"])
BASELINE_REPORT = Path(os.environ["PANONUT_BASELINE_REPORT"])
COMPLETION_MARKER = Path(os.environ["PANONUT_COMPLETION_MARKER"])
BOOTSTRAP_REPEATS = int(os.environ["PANONUT_BOOTSTRAP_REPEATS"])
KEEP_TEMP = os.environ["PANONUT_KEEP_TEMP"] == "1"
SAVE_PREVIEWS = os.environ["PANONUT_SAVE_PREVIEWS"] == "1"

TEMP_ROOT = RUN_DIR / "temporary_rgb"
TRT_CACHE = EXP_ROOT / "flowsal_trt_cache"
PREVIEW_DIR = RUN_DIR / "previews"
PER_SAMPLE_CSV = RUN_DIR / "per_second_metrics.csv"
PER_VIDEO_CSV = RUN_DIR / "per_video_metrics.csv"
REPORT_JSON = RUN_DIR / "report.json"
REPORT_TXT = RUN_DIR / "report.txt"
PROTOCOL_JSON = RUN_DIR / "frozen_protocol.json"
INVENTORY_JSON = RUN_DIR / "dataset_inventory.json"

SEED = 20260727
T = 20
FULL_H = 240
FULL_W = 320
MODEL_H = 144
MODEL_W = 192
SAMPLE_RATE_HZ = 60.0 / 8.0
SAMPLE_PERIOD_SECONDS = 1.0 / SAMPLE_RATE_HZ
WINDOW_SPAN_SECONDS = (T - 1) * SAMPLE_PERIOD_SECONDS
FLOWSAL_INPUT_SHAPE = (1, T, 3, MODEL_H, MODEL_W)
FLOWSAL_PARAMS = 27731
EXPECTED_ROWS = 3901
EXPECTED_VIDEOS = 15

for directory in (RUN_DIR, TEMP_ROOT, TRT_CACHE, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

cv2.setNumThreads(1)
np.random.seed(SEED)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"No rows found in {path}")
    return rows


def ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size",
            "-of", "json", str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def parse_fraction(value: str) -> float:
    numerator, denominator = value.split("/")
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def provider_stack(cache_dir: Path, prefix: str) -> list[Any]:
    available = set(ort.get_available_providers())
    providers: list[Any] = []
    if "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_engine_cache_prefix": prefix,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache_dir),
                    "trt_context_memory_sharing_enable": True,
                    "trt_min_subgraph_size": 1,
                    "trt_max_workspace_size": 2147483648,
                },
            )
        )
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    return providers


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=provider_stack(TRT_CACHE, "FLOWSAL_V1_1_PANONUT360"),
    )
    if session.get_providers()[0] != "TensorrtExecutionProvider":
        fail(f"TensorRT was not selected first: {session.get_providers()}")
    return session


def extract_resampled_raw_video(video_path: Path, raw_path: Path) -> int:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-an", "-sn",
            "-vf", f"fps={SAMPLE_RATE_HZ:.12g},scale={FULL_W}:{FULL_H}:flags=bilinear,format=bgr24",
            "-f", "rawvideo", str(raw_path),
        ],
        check=True,
    )
    bytes_per_frame = FULL_H * FULL_W * 3
    size = raw_path.stat().st_size
    if size % bytes_per_frame != 0:
        fail(f"Raw frame file has incomplete final frame: {raw_path}")
    count = size // bytes_per_frame
    if count < T:
        fail(f"Not enough sampled frames in {video_path}: {count}")
    return int(count)


def build_rgb_memmap(video_name: str, video_path: Path) -> tuple[np.memmap, np.memmap, int]:
    video_temp = TEMP_ROOT / video_name
    video_temp.mkdir(parents=True, exist_ok=True)
    raw_path = video_temp / "sampled_bgr_320x240.raw"
    low_path = video_temp / "sampled_bgr_192x144_uint8.dat"
    frame_count = extract_resampled_raw_video(video_path, raw_path)
    rgb_full = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(frame_count, FULL_H, FULL_W, 3),
    )
    rgb_low = np.memmap(
        low_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frame_count, 3, MODEL_H, MODEL_W),
    )
    for frame_index in range(frame_count):
        chw = np.asarray(rgb_full[frame_index]).transpose(2, 0, 1)
        for channel in range(3):
            rgb_low[frame_index, channel] = cv2.resize(
                chw[channel],
                (MODEL_W, MODEL_H),
                interpolation=cv2.INTER_LINEAR,
            )
        if (frame_index + 1) % 250 == 0 or frame_index + 1 == frame_count:
            print(f"[{video_name}] RGB frames {frame_index + 1}/{frame_count}")
    rgb_low.flush()
    return rgb_full, rgb_low, frame_count


def extract_last_map(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 5:
        return np.asarray(array[0, -1, 0], dtype=np.float32)
    if array.ndim == 4:
        if array.shape[1] == 1:
            return np.asarray(array[0, 0], dtype=np.float32)
        return np.asarray(array[0, -1], dtype=np.float32)
    if array.ndim == 3:
        return np.asarray(array[-1], dtype=np.float32)
    fail(f"Unsupported model output shape: {array.shape}")


def normalize_01(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        fail("Map contains non-finite values")
    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def probability(array: np.ndarray) -> np.ndarray:
    values = np.clip(array.astype(np.float64), 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def metric_cc(prediction: np.ndarray, target: np.ndarray) -> float:
    p = prediction.ravel().astype(np.float64)
    t = target.ravel().astype(np.float64)
    if float(p.std()) <= 0 or float(t.std()) <= 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def metric_sim(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.minimum(probability(prediction), probability(target)).sum())


def metric_kld(prediction: np.ndarray, target: np.ndarray) -> float:
    epsilon = 1e-12
    p = probability(prediction)
    t = probability(target)
    return float(np.sum(t * np.log((t + epsilon) / (p + epsilon))))


def saliency_metrics(prediction_raw: np.ndarray, target_raw: np.ndarray) -> dict[str, float]:
    prediction = normalize_01(prediction_raw)
    target = normalize_01(target_raw)
    return {
        "CC": metric_cc(prediction, target),
        "SIM": metric_sim(prediction, target),
        "KLD": metric_kld(prediction, target),
        "MSE": float(np.mean((prediction - target) ** 2)),
    }


def finite_mean(values: list[float]) -> float:
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def finite_percentile(values: list[float], percentile: float) -> float:
    return float(np.nanpercentile(np.asarray(values, dtype=np.float64), percentile))


def summarize_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in ("CC", "SIM", "KLD", "MSE"):
        values = [float(row[f"{prefix}_{metric}"]) for row in rows]
        output[metric] = finite_mean(values)
        output[f"{metric}_median"] = finite_percentile(values, 50)
        output[f"{metric}_p05"] = finite_percentile(values, 5)
        output[f"{metric}_p95"] = finite_percentile(values, 95)
    return output


def exact_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability_value = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability_value)


def exact_sign_flip_p(deltas: np.ndarray) -> float:
    n = len(deltas)
    observed = abs(float(np.mean(deltas)))
    exceed = 0
    total = 2 ** n
    for bits in itertools.product((-1.0, 1.0), repeat=n):
        value = abs(float(np.mean(deltas * np.asarray(bits, dtype=np.float64))))
        if value >= observed - 1e-15:
            exceed += 1
    return exceed / total


def paired_video_inference(
    per_video_rows: list[dict[str, Any]],
    candidate: str,
    reference: str,
    repeats: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + (17 if reference == "student" else 29))
    output: dict[str, Any] = {
        "candidate": candidate,
        "reference": reference,
        "videos": len(per_video_rows),
        "bootstrap_repeats": repeats,
        "sign_flip_method": "exact 2^N enumeration",
    }
    for metric in ("CC", "SIM", "KLD", "MSE"):
        deltas = np.asarray(
            [
                float(row[f"{candidate}_{metric}"])
                - float(row[f"{reference}_{metric}"])
                for row in per_video_rows
            ],
            dtype=np.float64,
        )
        n = len(deltas)
        observed = float(np.mean(deltas))
        indices = rng.integers(0, n, size=(repeats, n))
        bootstrap_means = deltas[indices].mean(axis=1)
        raw_positive = int(np.count_nonzero(deltas > 0))
        raw_negative = int(np.count_nonzero(deltas < 0))
        wins = raw_positive
        losses = raw_negative
        if metric in {"KLD", "MSE"}:
            wins, losses = raw_negative, raw_positive
        loo = np.asarray(
            [float(np.mean(np.delete(deltas, index))) for index in range(n)],
            dtype=np.float64,
        )
        output[metric] = {
            "mean_delta": observed,
            "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
            "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
            "wins": wins,
            "losses": losses,
            "ties": int(n - raw_positive - raw_negative),
            "sign_p": exact_sign_test(wins, losses),
            "sign_flip_p_exact": exact_sign_flip_p(deltas),
            "loo_min": float(np.min(loo)),
            "loo_max": float(np.max(loo)),
        }
    return output


def latency_summary(values_seconds: list[float]) -> dict[str, float]:
    values_ms = np.asarray(values_seconds, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(np.mean(values_ms)),
        "median_ms": float(np.median(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "p99_ms": float(np.percentile(values_ms, 99)),
        "samples": int(len(values_ms)),
        "note": "session.run wall time; includes host-device transfers",
    }


def heatmap(values: np.ndarray) -> np.ndarray:
    image = np.clip(normalize_01(values) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 29), (0, 0, 0), -1)
    cv2.putText(
        result, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def save_preview(
    path: Path,
    frame_bgr: np.ndarray,
    target: np.ndarray,
    flowsal: np.ndarray,
    title: str,
) -> None:
    frame = label(cv2.resize(frame_bgr, (FULL_W, FULL_H)), "RGB endpoint")
    target_hm = label(heatmap(target), "Panonut360 ground truth")
    flowsal_hm = label(heatmap(flowsal), "FlowSal v1.1")
    canvas = cv2.hconcat([frame, target_hm, flowsal_hm])
    cv2.putText(
        canvas, title, (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        fail(f"Could not write preview: {path}")


def main() -> None:
    baseline_rows = read_csv_rows(BASELINE_CSV)
    required_fields = {
        "video",
        "saliency_second_index",
        "model_endpoint_sample_index",
        "model_endpoint_seconds",
    }
    for prefix in ("teacher", "student"):
        for metric in ("CC", "SIM", "KLD", "MSE"):
            required_fields.add(f"{prefix}_{metric}")
    missing_fields = sorted(required_fields - set(baseline_rows[0]))
    if missing_fields:
        fail(f"Baseline CSV is missing fields: {missing_fields}")
    if len(baseline_rows) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS} baseline rows, found {len(baseline_rows)}")

    baseline_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    baseline_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in baseline_rows:
        video = str(row["video"]).strip()
        second_index = int(row["saliency_second_index"])
        key = (video, second_index)
        if key in baseline_by_key:
            fail(f"Duplicate baseline key: {key}")
        baseline_by_key[key] = row
        baseline_by_video[video].append(row)
    for rows in baseline_by_video.values():
        rows.sort(key=lambda row: int(row["saliency_second_index"]))

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if len(videos) != EXPECTED_VIDEOS:
        fail(f"Expected {EXPECTED_VIDEOS} videos, found {len(videos)}")
    names = {video.stem for video in videos}
    if set(baseline_by_video) != names:
        fail(
            "Baseline CSV videos do not match current Panonut360 videos: "
            f"baseline_only={sorted(set(baseline_by_video) - names)}, "
            f"current_only={sorted(names - set(baseline_by_video))}"
        )

    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    teacher_latency = baseline_report["teacher"]["evaluation_wall_latency"]
    student_latency = baseline_report["student"]["evaluation_wall_latency"]

    protocol = {
        "status": "frozen",
        "external_dataset": "Panonut360",
        "models_frozen": True,
        "training_or_finetuning": False,
        "checkpoint_selection_on_panonut360": False,
        "flowsal_sha256": sha256(FLOWSAL_ONNX),
        "baseline_csv": str(BASELINE_CSV),
        "baseline_csv_sha256": sha256(BASELINE_CSV),
        "baseline_rows": len(baseline_rows),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "timesteps": T,
        "window_span_seconds": WINDOW_SPAN_SECONDS,
        "evaluation_seconds_and_endpoints": "reused exactly from frozen teacher/KD per-second CSV",
        "ground_truth": "official Panonut360 SaliencyMap NPY resized from 240x120 to 320x240",
        "runtime_input": "RGB only at [1,20,3,144,192]",
        "sea_raft_executed": False,
        "teacher_inference_executed": False,
        "kd_inference_executed": False,
        "metrics": ["CC", "SIM", "KLD", "MSE"],
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "sign_flip": "exact enumeration over 2^15 video sign assignments",
    }
    PROTOCOL_JSON.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("PANONUT360 FLOWSAL v1.1 ZERO-SHOT EVALUATION")
    print("=" * 132)
    print("Videos:", len(videos))
    print("Frozen baseline rows:", len(baseline_rows))
    print("FlowSal:", FLOWSAL_ONNX)
    print("FlowSal SHA256:", sha256(FLOWSAL_ONNX))
    print("SEA-RAFT executed: NO")
    print("Teacher/KD inference repeated: NO")
    print("Training or checkpoint selection: NO")

    session = create_session(FLOWSAL_ONNX)
    input_name = session.get_inputs()[0].name
    warmup = np.zeros(FLOWSAL_INPUT_SHAPE, dtype=np.float32)
    for _ in range(10):
        session.run(None, {input_name: warmup})

    all_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    flowsal_times: list[float] = []
    inventory: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()

    for video_number, video_path in enumerate(videos, start=1):
        name = video_path.stem
        video_started = time.perf_counter()
        saliency_path = SALIENCY_DIR / f"{name}.npy"
        if not saliency_path.is_file():
            fail(f"Missing saliency array: {saliency_path}")
        saliency = np.load(saliency_path, mmap_mode="r", allow_pickle=False)
        if saliency.ndim != 3 or saliency.shape[1:] != (120, 240):
            fail(f"Unexpected saliency shape for {name}: {saliency.shape}")

        metadata = ffprobe(video_path)
        stream = metadata["streams"][0]
        duration = float(metadata["format"]["duration"])
        source_fps = parse_fraction(str(stream["avg_frame_rate"]))

        print()
        print(f"===== VIDEO {video_number:02d}/{len(videos):02d}: {name} =====")
        rgb_full, rgb_low, frame_count = build_rgb_memmap(name, video_path)
        frozen_rows = baseline_by_video[name]
        video_rows: list[dict[str, Any]] = []
        preview_saved = False

        for local_index, baseline in enumerate(frozen_rows, start=1):
            second_index = int(baseline["saliency_second_index"])
            endpoint_index = int(baseline["model_endpoint_sample_index"])
            expected_endpoint = int(round(second_index * SAMPLE_RATE_HZ))
            if endpoint_index != expected_endpoint:
                fail(
                    f"Frozen endpoint mismatch for {name} second={second_index}: "
                    f"baseline={endpoint_index}, expected={expected_endpoint}"
                )
            if second_index < 0 or second_index >= saliency.shape[0]:
                fail(f"Saliency second out of range: {name}/{second_index}")
            if endpoint_index < T - 1 or endpoint_index >= frame_count:
                fail(
                    f"Endpoint outside sampled RGB for {name}: "
                    f"endpoint={endpoint_index}, frames={frame_count}"
                )

            start_index = endpoint_index - (T - 1)
            input_rgb = (
                np.asarray(
                    rgb_low[start_index : endpoint_index + 1],
                    dtype=np.float32,
                )[None]
                / 255.0
            )
            if input_rgb.shape != FLOWSAL_INPUT_SHAPE:
                fail(f"Unexpected FlowSal input shape: {input_rgb.shape}")

            started = time.perf_counter()
            output = session.run(None, {input_name: input_rgb})[0]
            elapsed = time.perf_counter() - started
            flowsal_times.append(elapsed)

            flowsal_low = extract_last_map(output)
            flowsal_map = cv2.resize(
                flowsal_low,
                (FULL_W, FULL_H),
                interpolation=cv2.INTER_LINEAR,
            )
            target_map = cv2.resize(
                np.asarray(saliency[second_index], dtype=np.float32),
                (FULL_W, FULL_H),
                interpolation=cv2.INTER_LINEAR,
            )
            metrics = saliency_metrics(flowsal_map, target_map)

            row: dict[str, Any] = dict(baseline)
            row["video"] = name
            row["flowsal_wall_latency_ms"] = elapsed * 1000.0
            for metric, value in metrics.items():
                row[f"flowsal_{metric}"] = value
                row[f"flowsal_minus_student_{metric}"] = value - float(baseline[f"student_{metric}"])
                row[f"flowsal_minus_teacher_{metric}"] = value - float(baseline[f"teacher_{metric}"])
            all_rows.append(row)
            video_rows.append(row)

            if SAVE_PREVIEWS and not preview_saved:
                save_preview(
                    PREVIEW_DIR / f"{name}_flowsal_preview.png",
                    np.asarray(rgb_full[endpoint_index]).copy(),
                    target_map,
                    flowsal_map,
                    f"{name} second={second_index} endpoint={endpoint_index}",
                )
                preview_saved = True

            if local_index % 50 == 0 or local_index == len(frozen_rows):
                print(f"[{name}] evaluated {local_index}/{len(frozen_rows)}")

        teacher_summary = summarize_rows(video_rows, "teacher")
        student_summary = summarize_rows(video_rows, "student")
        flowsal_summary = summarize_rows(video_rows, "flowsal")
        video_result: dict[str, Any] = {
            "video": name,
            "video_path": str(video_path),
            "saliency_path": str(saliency_path),
            "source_duration_seconds": duration,
            "source_fps": source_fps,
            "sampled_frames": frame_count,
            "saliency_maps_total": int(saliency.shape[0]),
            "evaluated_seconds": len(video_rows),
        }
        for metric in ("CC", "SIM", "KLD", "MSE"):
            for prefix, summary in (
                ("teacher", teacher_summary),
                ("student", student_summary),
                ("flowsal", flowsal_summary),
            ):
                video_result[f"{prefix}_{metric}"] = summary[metric]
            video_result[f"flowsal_minus_student_{metric}"] = flowsal_summary[metric] - student_summary[metric]
            video_result[f"flowsal_minus_teacher_{metric}"] = flowsal_summary[metric] - teacher_summary[metric]
        per_video_rows.append(video_result)
        inventory.append(
            {
                "video": name,
                "video_path": str(video_path),
                "video_size_bytes": video_path.stat().st_size,
                "video_duration_seconds": duration,
                "source_fps": source_fps,
                "saliency_path": str(saliency_path),
                "saliency_sha256": sha256(saliency_path),
                "saliency_shape": list(saliency.shape),
                "sampled_frames": frame_count,
                "evaluated_seconds": len(video_rows),
            }
        )

        del rgb_full, rgb_low, saliency
        gc.collect()
        if not KEEP_TEMP:
            shutil.rmtree(TEMP_ROOT / name, ignore_errors=True)
        print(f"[{name}] complete in {time.perf_counter() - video_started:.1f}s")

    if len(all_rows) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS} rows, found {len(all_rows)}")
    if len(per_video_rows) != EXPECTED_VIDEOS:
        fail(f"Expected {EXPECTED_VIDEOS} videos, found {len(per_video_rows)}")

    with PER_SAMPLE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with PER_VIDEO_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_video_rows)
    INVENTORY_JSON.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    micro = {
        prefix: summarize_rows(all_rows, prefix)
        for prefix in ("teacher", "student", "flowsal")
    }
    macro = {
        prefix: {
            metric: finite_mean([float(row[f"{prefix}_{metric}"]) for row in per_video_rows])
            for metric in ("CC", "SIM", "KLD", "MSE")
        }
        for prefix in ("teacher", "student", "flowsal")
    }
    flowsal_latency = latency_summary(flowsal_times)
    inference_vs_student = paired_video_inference(
        per_video_rows, "flowsal", "student", BOOTSTRAP_REPEATS
    )
    inference_vs_teacher = paired_video_inference(
        per_video_rows, "flowsal", "teacher", BOOTSTRAP_REPEATS
    )

    report = {
        "status": "success",
        "experiment": "FlowSal v1.1 frozen zero-shot evaluation on Panonut360",
        "models_frozen": True,
        "training_or_finetuning": False,
        "checkpoint_selection_on_panonut360": False,
        "sea_raft_executed": False,
        "teacher_or_kd_inference_repeated": False,
        "videos": len(per_video_rows),
        "evaluated_saliency_seconds": len(all_rows),
        "protocol": protocol,
        "models": {
            "teacher": {
                "micro_summary": micro["teacher"],
                "macro_video_summary": macro["teacher"],
                "wall_latency_from_frozen_baseline": teacher_latency,
            },
            "student": {
                "micro_summary": micro["student"],
                "macro_video_summary": macro["student"],
                "wall_latency_from_frozen_baseline": student_latency,
            },
            "flowsal": {
                "onnx": str(FLOWSAL_ONNX),
                "sha256": sha256(FLOWSAL_ONNX),
                "parameters": FLOWSAL_PARAMS,
                "runtime_input": list(FLOWSAL_INPUT_SHAPE),
                "micro_summary": micro["flowsal"],
                "macro_video_summary": macro["flowsal"],
                "wall_latency": flowsal_latency,
            },
        },
        "paired_video_inference": {
            "flowsal_minus_student": inference_vs_student,
            "flowsal_minus_teacher": inference_vs_teacher,
        },
        "duration_seconds": time.perf_counter() - evaluation_started,
        "files": {
            "per_second_csv": str(PER_SAMPLE_CSV),
            "per_video_csv": str(PER_VIDEO_CSV),
            "protocol_json": str(PROTOCOL_JSON),
            "dataset_inventory_json": str(INVENTORY_JSON),
            "preview_directory": str(PREVIEW_DIR),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fs = micro["flowsal"]
    kd = micro["student"]
    teacher = micro["teacher"]
    fsm = macro["flowsal"]
    kdm = macro["student"]
    tm = macro["teacher"]

    def delta_line(candidate: dict[str, float], reference: dict[str, float]) -> str:
        return (
            f"{candidate['CC'] - reference['CC']:+.7f}  "
            f"{candidate['SIM'] - reference['SIM']:+.7f}  "
            f"{candidate['KLD'] - reference['KLD']:+.7f}  "
            f"{candidate['MSE'] - reference['MSE']:+.7f}"
        )

    text = f"""PANONUT360 FLOWSAL v1.1 ZERO-SHOT EVALUATION REPORT
========================================================================================================================
Status: SUCCESS
Training/fine-tuning on Panonut360: NO
Checkpoint selection on Panonut360: NO
FlowSal frozen before evaluation: YES
SEA-RAFT executed in this run: NO
Teacher/KD inference repeated: NO
Videos: {len(per_video_rows)}
Evaluated saliency seconds: {len(all_rows)}
FlowSal SHA256: {sha256(FLOWSAL_ONNX)}

FROZEN TEMPORAL PROTOCOL
------------------------------------------------------------------------------------------------------------------------
Model input rate: {SAMPLE_RATE_HZ:.6f} Hz
Timesteps: {T}
Window span: {WINDOW_SPAN_SECONDS:.6f} seconds
Evaluation seconds and endpoints: reused exactly from frozen teacher/KD CSV
Ground truth: official Panonut360 map resized from 240x120 to 320x240
FlowSal runtime input: RGB only, [1,20,3,144,192]

PRIMARY MICRO AVERAGE ACROSS ALL {len(all_rows)} EVALUATED SECONDS
------------------------------------------------------------------------------------------------------------------------
Method                              CC          SIM         KLD         MSE
Original SST-Sal teacher        {teacher['CC']:.7f}   {teacher['SIM']:.7f}   {teacher['KLD']:.7f}   {teacher['MSE']:.7f}
KD R192-W24-T20 + SEA-RAFT      {kd['CC']:.7f}   {kd['SIM']:.7f}   {kd['KLD']:.7f}   {kd['MSE']:.7f}
FlowSal v1.1 RGB-only           {fs['CC']:.7f}   {fs['SIM']:.7f}   {fs['KLD']:.7f}   {fs['MSE']:.7f}
FlowSal minus KD                {delta_line(fs, kd)}
FlowSal minus teacher           {delta_line(fs, teacher)}

PRIMARY MACRO AVERAGE ACROSS {len(per_video_rows)} VIDEOS
------------------------------------------------------------------------------------------------------------------------
Method                              CC          SIM         KLD         MSE
Original SST-Sal teacher        {tm['CC']:.7f}   {tm['SIM']:.7f}   {tm['KLD']:.7f}   {tm['MSE']:.7f}
KD R192-W24-T20 + SEA-RAFT      {kdm['CC']:.7f}   {kdm['SIM']:.7f}   {kdm['KLD']:.7f}   {kdm['MSE']:.7f}
FlowSal v1.1 RGB-only           {fsm['CC']:.7f}   {fsm['SIM']:.7f}   {fsm['KLD']:.7f}   {fsm['MSE']:.7f}
FlowSal minus KD                {delta_line(fsm, kdm)}
FlowSal minus teacher           {delta_line(fsm, tm)}

PAIRED VIDEO-LEVEL INFERENCE: FLOWSAL MINUS KD+SEA-RAFT
------------------------------------------------------------------------------------------------------------------------
Metric      Mean delta    95% CI low   95% CI high    Wins    Loss      Sign p      Flip p       LOO min       LOO max
"""
    for metric in ("CC", "SIM", "KLD", "MSE"):
        item = inference_vs_student[metric]
        text += (
            f"{metric:<8s}{item['mean_delta']:+13.7f}{item['ci95_low']:+14.7f}{item['ci95_high']:+15.7f}"
            f"{item['wins']:8d}{item['losses']:8d}{item['sign_p']:12.7f}{item['sign_flip_p_exact']:12.7f}"
            f"{item['loo_min']:+14.7f}{item['loo_max']:+14.7f}\n"
        )

    text += f"""
INFERENCE LATENCY, SESSION.RUN WALL TIME
------------------------------------------------------------------------------------------------------------------------
Method                              Mean ms      P95 ms      P99 ms
Original SST-Sal teacher          {float(teacher_latency['mean_ms']):9.3f}   {float(teacher_latency['p95_ms']):9.3f}   {float(teacher_latency['p99_ms']):9.3f}
KD R192-W24-T20 + SEA-RAFT        {float(student_latency['mean_ms']):9.3f}   {float(student_latency['p95_ms']):9.3f}   {float(student_latency['p99_ms']):9.3f}
FlowSal v1.1 RGB-only             {flowsal_latency['mean_ms']:9.3f}   {flowsal_latency['p95_ms']:9.3f}   {flowsal_latency['p99_ms']:9.3f}
Note: model-call latency excludes video decoding, ground-truth loading and SEA-RAFT.
FlowSal requires no SEA-RAFT execution; teacher and KD require external flow in deployment.

FILES
------------------------------------------------------------------------------------------------------------------------
Per-second CSV: {PER_SAMPLE_CSV}
Per-video CSV: {PER_VIDEO_CSV}
Protocol JSON: {PROTOCOL_JSON}
Dataset inventory JSON: {INVENTORY_JSON}
Report JSON: {REPORT_JSON}
Previews: {PREVIEW_DIR}
"""
    REPORT_TXT.write_text(text, encoding="utf-8")
    print()
    print(text)

    COMPLETION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    COMPLETION_MARKER.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": "Panonut360",
                "flowsal_sha256": sha256(FLOWSAL_ONNX),
                "baseline_csv_sha256": sha256(BASELINE_CSV),
                "run_directory": str(RUN_DIR),
                "report_json": str(REPORT_JSON),
                "evaluated_seconds": len(all_rows),
                "videos": len(per_video_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
PY

    echo
    echo "===== RUNNING FLOWSAL PANONUT360 ZERO-SHOT EVALUATION ====="
    "$VENV/bin/python" "$PY_SCRIPT"

    echo
    echo "===== RESULT HASHES ====="
    (
        cd "$RUN_DIR"
        find . \
            -type f \
            ! -path './temporary_rgb/*' \
            ! -name result_hashes.txt \
            -print0 \
            | sort -z \
            | xargs -0 sha256sum \
            | tee result_hashes.txt
    )

    ln -sfn "$RUN_DIR" "$EXP_ROOT/latest"

    echo
    echo "===== FINAL REPORT ====="
    cat "$RUN_DIR/report.txt"
    echo
    echo "Completed successfully."
} 2>&1 | tee "$CONSOLE_LOG"
