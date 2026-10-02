#!/usr/bin/env bash
set -euo pipefail

VENV="/home/mininet-ovs/venvs/satc"
DATASET_ROOT="/data/VR-EyeTracking/VR Eye-Tracking Dataset"
VIDEO_DIR="$DATASET_ROOT/videos/videos"
GAZE_DIR="$DATASET_ROOT/Gaze_txt_files/Gaze_txt_files"
SPLIT_XLSX="$DATASET_ROOT/train_test_set.xlsx"

FLOWSAL_ONNX="/data/D-SAV360/flowsal/models/FlowSal_R192_T20_C8_v1_1_Robust1000_static_opset16.onnx"
EXPECTED_FLOWSAL_SHA256="51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa"

BASELINE_CSV="/data/VR-EyeTracking/evaluation_sstsal_teacher_vs_kd_r192_w24_t20/results/2026-07-27_00-20-37/per_sample_metrics.csv"
EXPECTED_BASELINE_SHA256="07a8bb1d337f9139c17f83b1f1f20c30da3370a2f4a55430e9d0f08c8f7ba0d9"

# Reuse the already-built FlowSal TensorRT engine cache from the successful zero-shot run.
FLOWSAL_CACHE="/data/VR-EyeTracking/evaluation_flowsal_v1_1_robust1000_zero_shot/flowsal_trt_cache"

EXP_ROOT="/data/VR-EyeTracking/evaluation_flowsal_author_protocol_lean"
STAMP="$(date +%F_%H-%M-%S)"
RUN_DIR="$EXP_ROOT/results/$STAMP"
PY_SCRIPT="$RUN_DIR/run_flowsal_author_protocol_lean.py"
CONSOLE_LOG="$RUN_DIR/console.log"

# Optional: explicitly point this to the authors' pre-generated saliency_maps_5deg
# directory. If omitted, the script auto-detects common locations and otherwise
# reconstructs 5-degree spherical maps directly from the frozen gaze frames.
AUTHOR_GT_ROOT="${AUTHOR_GT_ROOT:-}"
GT_FRAME_OFFSET="${GT_FRAME_OFFSET:-0}"
KEEP_TEMP="${KEEP_TEMP:-0}"
ALLOW_CONCURRENT="${ALLOW_CONCURRENT:-0}"

mkdir -p "$RUN_DIR" "$FLOWSAL_CACHE"

finish() {
    status=$?
    echo
    echo "============================================================"
    echo "Lean author-protocol evaluation exit status: $status"
    echo "Run directory: $RUN_DIR"
    echo "Console log: $CONSOLE_LOG"
    echo "============================================================"
}
trap finish EXIT

{
    echo "===== FLOWSAL VR-EYETRACKING — LEAN AUTHOR-PROTOCOL EVALUATION ====="
    echo "FlowSal only: YES"
    echo "Original SST-Sal inference: NO"
    echo "KD-SST-Sal inference: NO"
    echo "SEA-RAFT: NO"
    echo "Scoring variants: ONE"
    echo "Bootstrap: NO"
    echo "Previews: NO"
    echo "Protocol: 5-degree GT, 8-bit PNG-equivalent maps, sin(linspace(0,pi,H)), authors' CC/SIM, symmetric KLD"
    echo

    if [[ "$ALLOW_CONCURRENT" != "1" ]] && pgrep -f '[r]un_protocol_sensitivity.py' >/dev/null 2>&1; then
        echo "ERROR: The previous heavy protocol-sensitivity evaluation is still running."
        echo "Stop it with Ctrl+C in its original terminal before starting this lean run."
        echo "Running both concurrently would make both much slower."
        exit 1
    fi

    echo "===== REQUIRED FILES ====="
    for path in \
        "$VENV/bin/python" \
        "$VIDEO_DIR" \
        "$GAZE_DIR" \
        "$SPLIT_XLSX" \
        "$FLOWSAL_ONNX" \
        "$BASELINE_CSV"
    do
        [[ -e "$path" ]] || { echo "ERROR: missing required path: $path"; exit 1; }
        ls -ld "$path"
    done
    command -v ffmpeg >/dev/null

    echo
    echo "===== VERIFYING FROZEN ARTIFACTS ====="
    echo "$EXPECTED_FLOWSAL_SHA256  $FLOWSAL_ONNX" | sha256sum -c -
    echo "$EXPECTED_BASELINE_SHA256  $BASELINE_CSV" | sha256sum -c -

    echo
    echo "===== SOURCE HASHES ====="
    sha256sum "$FLOWSAL_ONNX" "$BASELINE_CSV" "$SPLIT_XLSX" | tee "$RUN_DIR/source_hashes.txt"

    NVIDIA_SITE_LIBS="$("$VENV/bin/python" - <<'PYLIB'
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
PYLIB
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
        [[ -d "$directory" ]] && GPU_LIBRARY_PATH="${GPU_LIBRARY_PATH:+$GPU_LIBRARY_PATH:}$directory"
    done
    [[ -n "$NVIDIA_SITE_LIBS" ]] && GPU_LIBRARY_PATH="${GPU_LIBRARY_PATH:+$GPU_LIBRARY_PATH:}$NVIDIA_SITE_LIBS"
    export LD_LIBRARY_PATH="${GPU_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

    export LEAN_RUN_DIR="$RUN_DIR"
    export LEAN_FLOWSAL_ONNX="$FLOWSAL_ONNX"
    export LEAN_FLOWSAL_CACHE="$FLOWSAL_CACHE"
    export LEAN_BASELINE_CSV="$BASELINE_CSV"
    export LEAN_AUTHOR_GT_ROOT="$AUTHOR_GT_ROOT"
    export LEAN_GT_FRAME_OFFSET="$GT_FRAME_OFFSET"
    export LEAN_KEEP_TEMP="$KEEP_TEMP"

    cat > "$PY_SCRIPT" <<'PYCODE'
from __future__ import annotations

import csv
import gc
import hashlib
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
from openpyxl import load_workbook

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")

DATASET_ROOT = Path("/data/VR-EyeTracking/VR Eye-Tracking Dataset")
VIDEO_DIR = DATASET_ROOT / "videos" / "videos"
GAZE_DIR = DATASET_ROOT / "Gaze_txt_files" / "Gaze_txt_files"
SPLIT_XLSX = DATASET_ROOT / "train_test_set.xlsx"

RUN_DIR = Path(os.environ["LEAN_RUN_DIR"])
FLOWSAL_ONNX = Path(os.environ["LEAN_FLOWSAL_ONNX"])
FLOWSAL_CACHE = Path(os.environ["LEAN_FLOWSAL_CACHE"])
BASELINE_CSV = Path(os.environ["LEAN_BASELINE_CSV"])
EXPLICIT_AUTHOR_GT_ROOT = os.environ.get("LEAN_AUTHOR_GT_ROOT", "").strip()
GT_FRAME_OFFSET = int(os.environ.get("LEAN_GT_FRAME_OFFSET", "0"))
KEEP_TEMP = os.environ.get("LEAN_KEEP_TEMP", "0") == "1"

TEMP_ROOT = RUN_DIR / "temporary_rgb"
PER_SAMPLE_CSV = RUN_DIR / "per_sample_author_metrics.csv"
PER_VIDEO_CSV = RUN_DIR / "per_video_author_metrics.csv"
REPORT_TXT = RUN_DIR / "report.txt"
REPORT_JSON = RUN_DIR / "report.json"
PROTOCOL_JSON = RUN_DIR / "protocol.json"

T = 20
INPUT_H = 144
INPUT_W = 192
OUTPUT_H = 240
OUTPUT_W = 320
SAMPLE_RATE_HZ = 7.5
SIGMA_DEGREES = 5.0
SIGMA_RADIANS = math.radians(SIGMA_DEGREES)
EXPECTED_ROWS = 17129
EXPECTED_VIDEOS = 74
FLOWSAL_INPUT_SHAPE = (1, T, 3, INPUT_H, INPUT_W)
FLOWSAL_PARAMS = 27731
EPSILON = np.finfo(float).eps

for directory in (RUN_DIR, TEMP_ROOT, FLOWSAL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)

cv2.setNumThreads(2)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_ids() -> list[int]:
    workbook = load_workbook(SPLIT_XLSX, data_only=True, read_only=True)
    if "test_set" not in workbook.sheetnames:
        fail("Missing test_set sheet")
    values = [
        int(row[0])
        for row in workbook["test_set"].iter_rows(values_only=True)
        if row[0] is not None
    ]
    if len(values) != EXPECTED_VIDEOS:
        fail(f"Expected {EXPECTED_VIDEOS} test videos, found {len(values)}")
    return values


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS} baseline rows, found {len(rows)}")
    required = {
        "video", "model_endpoint_sample_index", "selected_gaze_frame_1based", "valid_observers"
    }
    missing = required - set(rows[0])
    if missing:
        fail(f"Baseline CSV missing fields: {sorted(missing)}")
    return rows


def provider_stack() -> list[Any]:
    available = set(ort.get_available_providers())
    providers: list[Any] = []
    if "TensorrtExecutionProvider" in available:
        providers.append((
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(FLOWSAL_CACHE),
                "trt_engine_cache_prefix": "FLOWSAL_V1_1_VREYE",
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(FLOWSAL_CACHE),
                "trt_context_memory_sharing_enable": True,
                "trt_min_subgraph_size": 1,
                "trt_max_workspace_size": 2147483648,
            },
        ))
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    return providers


def create_session() -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(FLOWSAL_ONNX), sess_options=options, providers=provider_stack()
    )
    print("Execution providers:", session.get_providers())
    if session.get_providers()[0] not in {
        "TensorrtExecutionProvider", "CUDAExecutionProvider"
    }:
        fail("FlowSal did not select a GPU execution provider")
    return session


def extract_video_direct_192x144(video_name: str, video_path: Path) -> tuple[np.memmap, int]:
    video_temp = TEMP_ROOT / video_name
    video_temp.mkdir(parents=True, exist_ok=True)
    raw_path = video_temp / "sampled_bgr_192x144.raw"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4",
        "-i", str(video_path), "-an", "-sn",
        "-vf", f"fps={SAMPLE_RATE_HZ:.12g},scale={INPUT_W}:{INPUT_H}:flags=bilinear,format=bgr24",
        "-f", "rawvideo", str(raw_path),
    ]
    subprocess.run(command, check=True)
    bytes_per_frame = INPUT_H * INPUT_W * 3
    size = raw_path.stat().st_size
    if size % bytes_per_frame:
        fail(f"Incomplete raw video: {raw_path}")
    count = size // bytes_per_frame
    if count < T:
        fail(f"Not enough sampled frames in {video_path}: {count}")
    frames = np.memmap(
        raw_path, mode="r", dtype=np.uint8, shape=(count, INPUT_H, INPUT_W, 3)
    )
    return frames, int(count)


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
    fail(f"Unsupported FlowSal output shape: {array.shape}")


def parse_gaze_file(path: Path) -> dict[int, tuple[float, float, float, float]]:
    rows: dict[int, tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = [value.strip() for value in line.strip().split(",")]
            if len(fields) != 8:
                continue
            try:
                if (
                    fields[0].lower() != "frame"
                    or fields[2].lower() != "forward"
                    or fields[5].lower() != "eye"
                ):
                    continue
                frame = int(fields[1])
                values = tuple(float(fields[index]) for index in (3, 4, 6, 7))
                if all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                    rows[frame] = values  # head_x, head_y, eye_x, eye_y_bottom
            except (TypeError, ValueError):
                continue
    return rows


def load_video_gaze(video_id: int) -> dict[str, dict[int, tuple[float, float, float, float]]]:
    participant_data: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    for participant_dir in sorted(GAZE_DIR.glob("p*")):
        if not participant_dir.is_dir():
            continue
        matches = sorted(participant_dir.glob(f"{video_id:03d}.*.txt"))
        if not matches:
            continue
        candidates = [(len(parse_gaze_file(path)), path) for path in matches]
        candidates.sort(reverse=True, key=lambda item: item[0])
        chosen = candidates[0][1]
        participant_data[participant_dir.name] = parse_gaze_file(chosen)
    return participant_data


def gaze_points_at(
    participant_data: dict[str, dict[int, tuple[float, float, float, float]]],
    frame_index: int,
) -> list[tuple[float, float]]:
    return [
        (rows[frame_index][2], rows[frame_index][3])
        for rows in participant_data.values()
        if frame_index in rows
    ]


def make_spherical_grid(width: int, height: int) -> np.ndarray:
    longitude = (np.arange(width, dtype=np.float32) + 0.5) / width * (2 * math.pi) - math.pi
    y_top = (np.arange(height, dtype=np.float32) + 0.5) / height
    latitude = math.pi / 2 - y_top * math.pi
    cos_lat = np.cos(latitude)[:, None]
    sin_lat = np.sin(latitude)[:, None]
    cos_lon = np.cos(longitude)[None, :]
    sin_lon = np.sin(longitude)[None, :]
    return np.stack(
        [
            np.broadcast_to(cos_lat * cos_lon, (height, width)),
            np.broadcast_to(sin_lat, (height, width)),
            np.broadcast_to(cos_lat * sin_lon, (height, width)),
        ],
        axis=2,
    ).reshape(-1, 3).astype(np.float32)


GT_GRID = make_spherical_grid(OUTPUT_W, OUTPUT_H)
VERTICAL_WEIGHT = np.sin(np.linspace(0, np.pi, OUTPUT_H, dtype=np.float64))[:, None]


def make_gt_map_5deg(points: list[tuple[float, float]]) -> np.ndarray:
    eye_x = np.asarray([point[0] for point in points], dtype=np.float32)
    eye_y_bottom = np.asarray([point[1] for point in points], dtype=np.float32)
    longitude = eye_x * (2 * math.pi) - math.pi
    latitude = eye_y_bottom * math.pi - math.pi / 2
    cos_lat = np.cos(latitude)
    vectors = np.stack(
        [cos_lat * np.cos(longitude), np.sin(latitude), cos_lat * np.sin(longitude)],
        axis=1,
    ).astype(np.float32)

    # One matrix multiply for all observers is faster than four scoring variants
    # and exactly preserves the great-circle 5-degree Gaussian construction.
    dots = np.clip(GT_GRID @ vectors.T, -1.0, 1.0)
    angular_distance = np.arccos(dots)
    saliency = np.exp(-0.5 * (angular_distance / SIGMA_RADIANS) ** 2).sum(axis=1)
    return saliency.reshape(OUTPUT_H, OUTPUT_W).astype(np.float32)


def normalize_range(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        fail("Non-finite saliency map")
    if maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def normalize_sum(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0:
        fail("Saliency map has invalid sum")
    return values / total


def png_equivalent_u8(array: np.ndarray) -> np.ndarray:
    return np.rint(normalize_range(array) * 255.0).clip(0, 255).astype(np.uint8)


def author_kld(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_sum(p)
    q = normalize_sum(q)
    return float(np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0)))


def author_cc(map1: np.ndarray, map2: np.ndarray) -> float:
    map1 = (map1 - np.mean(map1)) / np.std(map1)
    map2 = (map2 - np.mean(map2)) / np.std(map2)
    return float(np.corrcoef(map1.ravel(), map2.ravel())[0, 1])


def author_sim(map1: np.ndarray, map2: np.ndarray) -> float:
    map1 = normalize_sum(normalize_range(map1))
    map2 = normalize_sum(normalize_range(map2))
    return float(np.minimum(map1, map2).sum())


def author_scores(gt_u8: np.ndarray, pred_u8: np.ndarray) -> dict[str, float]:
    # This mirrors the supplied evaluate_salmaps.py exactly after image loading:
    # min-max -> multiply by sin(latitude) + machine epsilon -> sum normalize ->
    # CC/SIM and bidirectional-average KLD through getSimVal().
    gt = normalize_range(gt_u8.astype(np.float32))
    pred = normalize_range(pred_u8.astype(np.float32))
    gt = normalize_sum(gt * VERTICAL_WEIGHT + EPSILON)
    pred = normalize_sum(pred * VERTICAL_WEIGHT + EPSILON)
    cc = 0.5 * (author_cc(gt, pred) + author_cc(pred, gt))
    sim = 0.5 * (author_sim(gt, pred) + author_sim(pred, gt))
    kld = 0.5 * (author_kld(gt, pred) + author_kld(pred, gt))
    return {"CC": cc, "SIM": sim, "KLD": kld}


def candidate_gt_roots() -> list[Path]:
    values: list[Path] = []
    if EXPLICIT_AUTHOR_GT_ROOT:
        values.append(Path(EXPLICIT_AUTHOR_GT_ROOT))
    values.extend(
        [
            DATASET_ROOT / "saliency_maps_5deg",
            Path("/data/VR-EyeTracking/saliency_maps_5deg"),
            Path("/data/VR-EyeTracking/VR-EyeTracking/saliency_maps_5deg"),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def index_gt_root(root: Path, required_by_video: dict[str, set[int]]) -> dict[tuple[str, int], Path] | None:
    if not root.is_dir():
        return None
    lookup: dict[tuple[str, int], Path] = {}
    for video, required in required_by_video.items():
        folders = [root / video, root / str(int(video))]
        folder = next((candidate for candidate in folders if candidate.is_dir()), None)
        if folder is None:
            return None
        numeric_files: dict[int, Path] = {}
        for path in folder.glob("*.png"):
            try:
                numeric_files[int(path.stem)] = path
            except ValueError:
                continue
        for frame in required:
            adjusted = frame + GT_FRAME_OFFSET
            path = numeric_files.get(adjusted)
            if path is None:
                return None
            lookup[(video, frame)] = path
    return lookup


def main() -> None:
    test_ids = load_test_ids()
    baseline_rows = read_csv_rows(BASELINE_CSV)
    baseline_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_rows:
        video = str(row["video"]).strip().zfill(3)
        baseline_by_video[video].append(row)
    for rows in baseline_by_video.values():
        rows.sort(key=lambda row: int(row["model_endpoint_sample_index"]))
    expected = {f"{video_id:03d}" for video_id in test_ids}
    if set(baseline_by_video) != expected:
        fail("Baseline CSV does not match official 74-video test split")

    required_gt: dict[str, set[int]] = {
        video: {int(row["selected_gaze_frame_1based"]) for row in rows}
        for video, rows in baseline_by_video.items()
    }
    gt_lookup: dict[tuple[str, int], Path] | None = None
    gt_root: Path | None = None
    for root in candidate_gt_roots():
        indexed = index_gt_root(root, required_gt)
        if indexed is not None:
            gt_lookup = indexed
            gt_root = root
            break
    gt_mode = "authors_pre_generated_5deg_png" if gt_lookup is not None else "reconstructed_5deg_from_frozen_gaze"

    protocol = {
        "status": "frozen_lean_author_metric_reproduction",
        "models_evaluated": ["FlowSal v1.1"],
        "teacher_inference": False,
        "kd_inference": False,
        "sea_raft": False,
        "scoring_variants": 1,
        "bootstrap": False,
        "previews": False,
        "official_test_videos": EXPECTED_VIDEOS,
        "expected_samples": EXPECTED_ROWS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "timesteps": T,
        "frozen_endpoints_and_gaze_frames": str(BASELINE_CSV),
        "ground_truth_mode": gt_mode,
        "ground_truth_root": str(gt_root) if gt_root else None,
        "ground_truth_sigma_degrees": SIGMA_DEGREES,
        "prediction_quantization": "min-max normalized then rounded to uint8 [0,255] in memory",
        "ground_truth_quantization_when_reconstructed": "min-max normalized then rounded to uint8 [0,255] in memory",
        "latitude_weight": "sin(linspace(0, pi, H))",
        "metric_implementation": "supplied SST-Sal authors code: weighted maps, ordinary CC/SIM, symmetric bidirectional KLD",
        "aggregation": {
            "micro": "mean across all evaluated frames/endpoints",
            "macro": "mean of per-video means across 74 videos",
        },
        "sample_set_limitation": "Uses the previously frozen 17129 endpoints because the authors' VREyeTracking.txt frame list was not supplied.",
    }
    PROTOCOL_JSON.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Official test videos:", len(test_ids))
    print("Frozen endpoints:", len(baseline_rows))
    print("Ground-truth mode:", gt_mode)
    if gt_root:
        print("Author GT root:", gt_root)
    else:
        print("Pre-generated author maps not found; reconstructing one 5-degree map per endpoint.")
    print("FlowSal SHA256:", sha256(FLOWSAL_ONNX))

    session = create_session()
    input_name = session.get_inputs()[0].name
    warmup = np.zeros(FLOWSAL_INPUT_SHAPE, dtype=np.float32)
    for _ in range(3):
        session.run(None, {input_name: warmup})

    all_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    inference_times: list[float] = []
    started_all = time.perf_counter()

    for video_number, video_id in enumerate(test_ids, start=1):
        video_started = time.perf_counter()
        video = f"{video_id:03d}"
        video_path = VIDEO_DIR / f"{video}.mp4"
        if not video_path.is_file():
            fail(f"Missing video: {video_path}")
        frozen_rows = baseline_by_video[video]
        frames, sampled_count = extract_video_direct_192x144(video, video_path)
        participant_data = None if gt_lookup is not None else load_video_gaze(video_id)
        if gt_lookup is None and not participant_data:
            fail(f"No gaze data for {video}")

        print()
        print(f"===== VIDEO {video_number:02d}/{EXPECTED_VIDEOS}: {video} | endpoints={len(frozen_rows)} =====")
        video_rows: list[dict[str, Any]] = []

        for local_index, baseline in enumerate(frozen_rows, start=1):
            endpoint = int(baseline["model_endpoint_sample_index"])
            gaze_frame = int(baseline["selected_gaze_frame_1based"])
            observers = int(float(baseline["valid_observers"]))
            if endpoint < T - 1 or endpoint >= sampled_count:
                fail(f"Endpoint {endpoint} outside video {video} sampled_count={sampled_count}")

            start = endpoint - (T - 1)
            window_bgr = np.asarray(frames[start : endpoint + 1], dtype=np.float32)
            model_input = window_bgr.transpose(0, 3, 1, 2)[None] / 255.0
            if model_input.shape != FLOWSAL_INPUT_SHAPE:
                fail(f"Unexpected model input shape: {model_input.shape}")

            inference_started = time.perf_counter()
            output = session.run(None, {input_name: model_input})[0]
            inference_seconds = time.perf_counter() - inference_started
            inference_times.append(inference_seconds)

            pred_low = extract_last_map(output)
            pred = cv2.resize(pred_low, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            pred_u8 = png_equivalent_u8(pred)

            if gt_lookup is not None:
                gt_path = gt_lookup[(video, gaze_frame)]
                gt_u8 = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt_u8 is None:
                    fail(f"Could not read GT map: {gt_path}")
                if gt_u8.shape != (OUTPUT_H, OUTPUT_W):
                    gt_u8 = cv2.resize(gt_u8, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_AREA)
            else:
                assert participant_data is not None
                points = gaze_points_at(participant_data, gaze_frame)
                if len(points) != observers:
                    fail(
                        f"Observer mismatch {video} endpoint={endpoint}: "
                        f"baseline={observers}, current={len(points)}"
                    )
                gt_u8 = png_equivalent_u8(make_gt_map_5deg(points))

            scores = author_scores(gt_u8, pred_u8)
            row = {
                "video": video,
                "model_endpoint_sample_index": endpoint,
                "selected_gaze_frame_1based": gaze_frame,
                "valid_observers": observers,
                "flowsal_CC": scores["CC"],
                "flowsal_SIM": scores["SIM"],
                "flowsal_KLD_symmetric": scores["KLD"],
                "flowsal_wall_latency_ms": inference_seconds * 1000.0,
            }
            all_rows.append(row)
            video_rows.append(row)

            if local_index % 50 == 0 or local_index == len(frozen_rows):
                print(f"[{video}] evaluated {local_index}/{len(frozen_rows)}")

        video_summary = {
            "video": video,
            "evaluated_samples": len(video_rows),
            "flowsal_CC": float(np.mean([row["flowsal_CC"] for row in video_rows])),
            "flowsal_SIM": float(np.mean([row["flowsal_SIM"] for row in video_rows])),
            "flowsal_KLD_symmetric": float(np.mean([row["flowsal_KLD_symmetric"] for row in video_rows])),
        }
        per_video_rows.append(video_summary)

        del frames, participant_data
        gc.collect()
        if not KEEP_TEMP:
            shutil.rmtree(TEMP_ROOT / video, ignore_errors=True)
        print(f"[{video}] complete in {time.perf_counter() - video_started:.1f}s")

    if len(all_rows) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS} evaluated rows, found {len(all_rows)}")

    with PER_SAMPLE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with PER_VIDEO_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_video_rows)

    micro = {
        "CC": float(np.mean([row["flowsal_CC"] for row in all_rows])),
        "SIM": float(np.mean([row["flowsal_SIM"] for row in all_rows])),
        "KLD_symmetric": float(np.mean([row["flowsal_KLD_symmetric"] for row in all_rows])),
    }
    macro = {
        "CC": float(np.mean([row["flowsal_CC"] for row in per_video_rows])),
        "SIM": float(np.mean([row["flowsal_SIM"] for row in per_video_rows])),
        "KLD_symmetric": float(np.mean([row["flowsal_KLD_symmetric"] for row in per_video_rows])),
    }
    duration = time.perf_counter() - started_all
    latency = {
        "mean_ms": float(np.mean(inference_times) * 1000.0),
        "p50_ms": float(np.percentile(inference_times, 50) * 1000.0),
        "p95_ms": float(np.percentile(inference_times, 95) * 1000.0),
    }

    report = {
        "status": "success",
        "experiment": "FlowSal v1.1 VR-EyeTracking lean evaluation using supplied SST-Sal author metric implementation",
        "protocol": protocol,
        "flowsal": {
            "onnx": str(FLOWSAL_ONNX),
            "sha256": sha256(FLOWSAL_ONNX),
            "parameters": FLOWSAL_PARAMS,
            "latency": latency,
        },
        "evaluated_samples": len(all_rows),
        "evaluated_videos": len(per_video_rows),
        "micro_frame_mean": micro,
        "macro_video_mean": macro,
        "published_sstsal_reference": {"CC": 0.500, "SIM": 0.338, "KLD": 7.371},
        "duration_seconds": duration,
        "files": {
            "per_sample_csv": str(PER_SAMPLE_CSV),
            "per_video_csv": str(PER_VIDEO_CSV),
            "protocol_json": str(PROTOCOL_JSON),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    text = f"""FLOWSAL v1.1 VR-EYETRACKING — LEAN SST-SAL AUTHOR-PROTOCOL REPORT
========================================================================================================================
Status: SUCCESS
FlowSal only: YES
Teacher/KD/SEA-RAFT executed: NO
Official test videos: {len(per_video_rows)}
Evaluated frozen endpoints: {len(all_rows)}
Ground-truth mode: {gt_mode}
Ground-truth sigma: 5 degrees
Prediction/GT representation: 8-bit PNG-equivalent
Latitude compensation: sin(linspace(0, pi, H)) multiplied into both maps
Metrics: authors' ordinary CC, SIM, and bidirectional-average symmetric KLD
Authors' exact VREyeTracking.txt frame list available: NO — frozen 17129 endpoints reused

PRIMARY AUTHOR-STYLE FRAME MEAN
------------------------------------------------------------------------------------------------------------------------
FlowSal CC:            {micro['CC']:.7f}
FlowSal SIM:           {micro['SIM']:.7f}
FlowSal symmetric KLD: {micro['KLD_symmetric']:.7f}

SECONDARY MACRO MEAN ACROSS 74 VIDEOS
------------------------------------------------------------------------------------------------------------------------
FlowSal CC:            {macro['CC']:.7f}
FlowSal SIM:           {macro['SIM']:.7f}
FlowSal symmetric KLD: {macro['KLD_symmetric']:.7f}

PUBLISHED SST-SAL REFERENCE
------------------------------------------------------------------------------------------------------------------------
CC: 0.500
SIM: 0.338
KLD: 7.371

FLOWSAL INFERENCE LATENCY ONLY
------------------------------------------------------------------------------------------------------------------------
Mean: {latency['mean_ms']:.3f} ms
P50:  {latency['p50_ms']:.3f} ms
P95:  {latency['p95_ms']:.3f} ms

Total evaluation duration: {duration / 60.0:.2f} minutes
Run directory: {RUN_DIR}
"""
    REPORT_TXT.write_text(text, encoding="utf-8")
    print()
    print(text)


if __name__ == "__main__":
    main()
PYCODE

    echo
    echo "===== ENVIRONMENT ====="
    nvidia-smi
    "$VENV/bin/python" - <<'PYENV'
import cv2, numpy, onnxruntime, sys
if hasattr(onnxruntime, "preload_dlls"):
    onnxruntime.preload_dlls(directory="")
print("Python:", sys.executable)
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("Providers:", onnxruntime.get_available_providers())
PYENV

    echo
    echo "===== RUNNING LEAN EVALUATION ====="
    "$VENV/bin/python" -u "$PY_SCRIPT"

} 2>&1 | tee "$CONSOLE_LOG"
