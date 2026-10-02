#!/usr/bin/env bash
set -euo pipefail

VENV="/home/mininet-ovs/venvs/satc"
DATASET_ROOT="/data/VR-EyeTracking/VR Eye-Tracking Dataset"
VIDEO_DIR="$DATASET_ROOT/videos/videos"
GAZE_DIR="$DATASET_ROOT/Gaze_txt_files/Gaze_txt_files"
SPLIT_XLSX="$DATASET_ROOT/train_test_set.xlsx"
README_PATH="$DATASET_ROOT/README.txt"

FLOWSAL_ONNX="/data/D-SAV360/flowsal/models/FlowSal_R192_T20_C8_v1_1_Robust1000_static_opset16.onnx"
EXPECTED_FLOWSAL_SHA256="51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa"
BASELINE_RUN="/data/VR-EyeTracking/evaluation_sstsal_teacher_vs_kd_r192_w24_t20/results/2026-07-27_00-20-37"
BASELINE_CSV="$BASELINE_RUN/per_sample_metrics.csv"
EXPECTED_BASELINE_SHA256="07a8bb1d337f9139c17f83b1f1f20c30da3370a2f4a55430e9d0f08c8f7ba0d9"

EXP_ROOT="/data/VR-EyeTracking/evaluation_flowsal_v1_1_robust1000_zero_shot"
STAMP="$(date +%F_%H-%M-%S)"
RUN_DIR="$EXP_ROOT/results/$STAMP"
PY_SCRIPT="$RUN_DIR/run_flowsal_vreye_zero_shot.py"
CONSOLE_LOG="$RUN_DIR/console.log"
COMPLETION_MARKER="$EXP_ROOT/FLOWSAL_V1_1_VREYE_ALREADY_EVALUATED.json"

BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-100000}"
MAX_GAZE_FRAME_SHIFT="${MAX_GAZE_FRAME_SHIFT:-2}"
MIN_OBSERVERS="${MIN_OBSERVERS:-10}"
KEEP_TEMP="${KEEP_TEMP:-0}"
SAVE_PREVIEWS="${SAVE_PREVIEWS:-1}"
ALLOW_RERUN="${ALLOW_RERUN:-0}"

mkdir -p "$RUN_DIR"

finish_shell() {
    status=$?
    echo
    echo "============================================================"
    echo "FlowSal VR-EyeTracking zero-shot exit status: $status"
    echo "Run directory: $RUN_DIR"
    echo "Console log: $CONSOLE_LOG"
    echo "============================================================"
    echo "The terminal remains open. Type exit when finished."
    exec bash -i
}
trap finish_shell EXIT

{
    echo "===== FLOWSAL v1.1 VR-EYETRACKING ZERO-SHOT EVALUATION ====="
    echo "FlowSal frozen before evaluation: YES"
    echo "Training/fine-tuning: NO"
    echo "Checkpoint selection: NO"
    echo "SEA-RAFT executed in this run: NO"
    echo "Teacher/KD inference repeated: NO"
    echo "Frozen baseline rows reused: 17129"
    echo "Bootstrap/sign-flip repetitions: $BOOTSTRAP_REPEATS"
    echo

    if [[ -f "$COMPLETION_MARKER" && "$ALLOW_RERUN" != "1" ]]; then
        echo "ERROR: A completed FlowSal VR-EyeTracking marker already exists:"
        echo "$COMPLETION_MARKER"
        echo
        echo "This safeguard prevents repeated external-test evaluation."
        exit 1
    fi

    echo "===== REQUIRED PATHS ====="
    for path in \
        "$VENV/bin/python" \
        "$VIDEO_DIR" \
        "$GAZE_DIR" \
        "$SPLIT_XLSX" \
        "$README_PATH" \
        "$FLOWSAL_ONNX" \
        "$BASELINE_CSV"
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
    echo "===== VERIFYING FROZEN ARTIFACTS ====="
    echo "$EXPECTED_FLOWSAL_SHA256  $FLOWSAL_ONNX" | sha256sum -c -
    echo "$EXPECTED_BASELINE_SHA256  $BASELINE_CSV" | sha256sum -c -

    echo
    echo "===== SOURCE HASHES ====="
    sha256sum \
        "$FLOWSAL_ONNX" \
        "$BASELINE_CSV" \
        "$SPLIT_XLSX" \
        "$README_PATH" \
        | tee "$RUN_DIR/source_hashes.txt"

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
        if [[ -d "$directory" ]]; then
            GPU_LIBRARY_PATH="${GPU_LIBRARY_PATH:+$GPU_LIBRARY_PATH:}$directory"
        fi
    done
    if [[ -n "$NVIDIA_SITE_LIBS" ]]; then
        GPU_LIBRARY_PATH="${GPU_LIBRARY_PATH:+$GPU_LIBRARY_PATH:}$NVIDIA_SITE_LIBS"
    fi
    export LD_LIBRARY_PATH="${GPU_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    echo
    echo "===== ENVIRONMENT ====="
    nvidia-smi
    "$VENV/bin/python" - <<'PYENV'
import sys, cv2, numpy, onnxruntime
if hasattr(onnxruntime, "preload_dlls"):
    onnxruntime.preload_dlls(directory="")
print("Python:", sys.executable)
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("Providers:", onnxruntime.get_available_providers())
PYENV

    export VREYE_RUN_DIR="$RUN_DIR"
    export VREYE_EXP_ROOT="$EXP_ROOT"
    export VREYE_EVAL_STEP="1"
    export VREYE_BOOTSTRAP_REPEATS="$BOOTSTRAP_REPEATS"
    export VREYE_MAX_GAZE_SHIFT="$MAX_GAZE_FRAME_SHIFT"
    export VREYE_MIN_OBSERVERS="$MIN_OBSERVERS"
    export VREYE_KEEP_TEMP="$KEEP_TEMP"
    export VREYE_SAVE_PREVIEWS="$SAVE_PREVIEWS"
    export VREYE_FLOWSAL_ONNX="$FLOWSAL_ONNX"
    export VREYE_BASELINE_CSV="$BASELINE_CSV"
    export VREYE_COMPLETION_MARKER="$COMPLETION_MARKER"

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

PROJECT = Path("/home/mininet-ovs/Documents/SST-Sal")
TEACHER_ONNX = Path(
    "/data/D-SAV360/sst_sal_onnx_parity/models/"
    "SST_Sal_fp32_static_1x20x6x240x320_opset16.onnx"
)
STUDENT_ONNX = Path(
    "/data/D-SAV360/sst_sal_kd_r192_w24_t20/models/"
    "SST_Sal_R192_W24_T20_KD_static_opset16.onnx"
)
STUDENT_CONFIG = Path(
    "/data/D-SAV360/sst_sal_kd_r192_w24_t20/models/"
    "SST_Sal_R192_W24_T20_KD_config.json"
)
RAFT_ONNX = (
    PROJECT
    / "models"
    / "SEA-RAFT"
    / "Tartan-C-T-TSKH-kitti432x960-M.onnx"
)

RUN_DIR = Path(os.environ["VREYE_RUN_DIR"])
EXP_ROOT = Path(os.environ["VREYE_EXP_ROOT"])
EVAL_STEP = int(os.environ["VREYE_EVAL_STEP"])
BOOTSTRAP_REPEATS = int(os.environ["VREYE_BOOTSTRAP_REPEATS"])
MAX_GAZE_SHIFT = int(os.environ["VREYE_MAX_GAZE_SHIFT"])
MIN_OBSERVERS = int(os.environ["VREYE_MIN_OBSERVERS"])
KEEP_TEMP = os.environ["VREYE_KEEP_TEMP"] == "1"
SAVE_PREVIEWS = os.environ["VREYE_SAVE_PREVIEWS"] == "1"

TEMP_ROOT = RUN_DIR / "temporary_video_features"
PREVIEW_DIR = RUN_DIR / "previews"
RAFT_CACHE = EXP_ROOT / "raft_trt_cache"
TEACHER_CACHE = EXP_ROOT / "teacher_trt_cache"
STUDENT_CACHE = EXP_ROOT / "student_trt_cache"

PER_SAMPLE_CSV = RUN_DIR / "per_sample_metrics.csv"
PER_VIDEO_CSV = RUN_DIR / "per_video_metrics.csv"
REPORT_JSON = RUN_DIR / "report.json"
REPORT_TXT = RUN_DIR / "report.txt"
PROTOCOL_JSON = RUN_DIR / "frozen_protocol.json"
INVENTORY_JSON = RUN_DIR / "dataset_inventory.json"
GAZE_INVENTORY_CSV = RUN_DIR / "gaze_file_inventory.csv"

SEED = 20260727
T = 20
CHANNELS = 6
FULL_H = 240
FULL_W = 320
STUDENT_H = 144
STUDENT_W = 192
SAMPLE_RATE_HZ = 60.0 / 8.0
SAMPLE_PERIOD_SECONDS = 1.0 / SAMPLE_RATE_HZ
WINDOW_SPAN_SECONDS = (T - 1) * SAMPLE_PERIOD_SECONDS
SIGMA_DEGREES = 9.35
SIGMA_RADIANS = math.radians(SIGMA_DEGREES)
GT_PARITY_THRESHOLD = 0.9999
GT_BATCH_FIXATIONS = 8

TEACHER_PARAMS = 55912
STUDENT_PARAMS = 26920

TEACHER_INPUT_SHAPE = (1, T, CHANNELS, FULL_H, FULL_W)
STUDENT_INPUT_SHAPE = (1, T, CHANNELS, STUDENT_H, STUDENT_W)

for directory in (
    RUN_DIR,
    TEMP_ROOT,
    PREVIEW_DIR,
    RAFT_CACHE,
    TEACHER_CACHE,
    STUDENT_CACHE,
):
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


def parse_fraction(value: str) -> float:
    numerator, denominator = value.split("/")
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:"
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def load_official_split() -> tuple[list[int], list[int]]:
    workbook = load_workbook(
        SPLIT_XLSX,
        data_only=True,
        read_only=True,
    )

    def sheet_values(name: str) -> list[int]:
        if name not in workbook.sheetnames:
            fail(f"Missing workbook sheet: {name}")
        return [
            int(row[0])
            for row in workbook[name].iter_rows(values_only=True)
            if row[0] is not None
        ]

    train_ids = sheet_values("train_set")
    test_ids = sheet_values("test_set")

    if len(train_ids) != 134:
        fail(f"Expected 134 train IDs, found {len(train_ids)}")
    if len(test_ids) != 74:
        fail(f"Expected 74 test IDs, found {len(test_ids)}")
    if set(train_ids) & set(test_ids):
        fail("Official train and test sets overlap")
    if len(set(train_ids) | set(test_ids)) != 208:
        fail("Official split does not contain 208 unique IDs")

    return train_ids, test_ids


def provider_stack(
    cache_dir: Path,
    prefix: str,
    fp16: bool,
) -> list[Any]:
    available = set(ort.get_available_providers())
    providers: list[Any] = []

    if "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": fp16,
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


def create_session(
    path: Path,
    cache_dir: Path,
    prefix: str,
    fp16: bool,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=provider_stack(
            cache_dir,
            prefix,
            fp16,
        ),
    )

    if session.get_providers()[0] != "TensorrtExecutionProvider":
        fail(
            f"TensorRT was not selected first for {path}: "
            f"{session.get_providers()}"
        )
    return session


def create_raft_session() -> tuple[ort.InferenceSession, int, int]:
    session = create_session(
        RAFT_ONNX,
        RAFT_CACHE,
        "SEA_RAFT_VREYE",
        False,
    )
    shape = session.get_inputs()[0].shape
    input_h = shape[-2] if isinstance(shape[-2], int) else FULL_H
    input_w = shape[-1] if isinstance(shape[-1], int) else FULL_W
    return session, int(input_h), int(input_w)


def extract_flow(outputs: list[np.ndarray]) -> np.ndarray:
    for output in outputs:
        array = np.asarray(output)
        if array.ndim == 4 and array.shape[1] == 2:
            return array[0].transpose(1, 2, 0).astype(np.float32)
        if array.ndim == 4 and array.shape[-1] == 2:
            return array[0].astype(np.float32)

    fail(
        "Cannot identify SEA-RAFT flow output: "
        f"{[np.asarray(item).shape for item in outputs]}"
    )


def resize_flow_vectors(
    flow: np.ndarray,
    output_w: int,
    output_h: int,
) -> np.ndarray:
    source_h, source_w = flow.shape[:2]
    if (source_h, source_w) == (output_h, output_w):
        return flow

    resized = cv2.resize(
        flow,
        (output_w, output_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    resized[..., 0] *= output_w / source_w
    resized[..., 1] *= output_h / source_h
    return resized


def run_raft_pair(
    session: ort.InferenceSession,
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    input_w: int,
    input_h: int,
) -> np.ndarray:
    previous = cv2.resize(
        previous_bgr,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )
    current = cv2.resize(
        current_bgr,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )

    tensor1 = (
        previous.transpose(2, 0, 1)[None].astype(np.float32)
        / 255.0
    )
    tensor2 = (
        current.transpose(2, 0, 1)[None].astype(np.float32)
        / 255.0
    )

    names = [item.name for item in session.get_inputs()]
    outputs = session.run(
        None,
        {
            names[0]: tensor1,
            names[1]: tensor2,
        },
    )

    return resize_flow_vectors(
        extract_flow(outputs),
        FULL_W,
        FULL_H,
    )


def flow_to_rgb(flow_uv: np.ndarray) -> np.ndarray:
    try:
        import sys

        sys.path.insert(0, str(PROJECT))
        from utils.flow_viz import flow_to_image  # type: ignore

        return np.clip(
            flow_to_image(flow_uv, True),
            0,
            255,
        ).astype(np.uint8)
    except Exception:
        u = flow_uv[..., 0].astype(np.float32)
        v = flow_uv[..., 1].astype(np.float32)
        magnitude, angle = cv2.cartToPolar(
            u,
            v,
            angleInDegrees=True,
        )
        maximum = (
            float(np.nanmax(magnitude))
            if magnitude.size
            else 0.0
        )

        hsv = np.zeros(
            (flow_uv.shape[0], flow_uv.shape[1], 3),
            dtype=np.uint8,
        )
        hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
        hsv[..., 1] = 255

        if maximum > 0:
            hsv[..., 2] = np.clip(
                magnitude / maximum * 255.0,
                0,
                255,
            ).astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def downsample_six_channel(
    frame_chw: np.ndarray,
) -> np.ndarray:
    output = np.empty(
        (CHANNELS, STUDENT_H, STUDENT_W),
        dtype=np.uint8,
    )
    for channel in range(CHANNELS):
        output[channel] = cv2.resize(
            frame_chw[channel],
            (STUDENT_W, STUDENT_H),
            interpolation=cv2.INTER_LINEAR,
        )
    return output


def extract_resampled_raw_video(
    video_path: Path,
    raw_path: Path,
) -> int:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-vf",
        (
            f"fps={SAMPLE_RATE_HZ:.12g},"
            f"scale={FULL_W}:{FULL_H}:flags=bilinear,"
            "format=bgr24"
        ),
        "-f",
        "rawvideo",
        str(raw_path),
    ]
    subprocess.run(command, check=True)

    bytes_per_frame = FULL_H * FULL_W * 3
    size = raw_path.stat().st_size

    if size % bytes_per_frame:
        fail(f"Incomplete raw frame file: {raw_path}")

    frame_count = size // bytes_per_frame
    if frame_count < T:
        fail(
            f"Not enough 7.5-Hz frames in {video_path}: "
            f"{frame_count}"
        )
    return int(frame_count)


def build_feature_memmaps(
    video_name: str,
    video_path: Path,
    raft_session: ort.InferenceSession,
    raft_w: int,
    raft_h: int,
) -> tuple[np.memmap, np.memmap, int]:
    video_temp = TEMP_ROOT / video_name
    video_temp.mkdir(parents=True, exist_ok=True)

    raw_path = video_temp / "sampled_bgr_320x240.raw"
    full_path = video_temp / "features_full_uint8.dat"
    low_path = video_temp / "features_low_uint8.dat"

    frame_count = extract_resampled_raw_video(
        video_path,
        raw_path,
    )

    rgb = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(frame_count, FULL_H, FULL_W, 3),
    )
    full = np.memmap(
        full_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frame_count, CHANNELS, FULL_H, FULL_W),
    )
    low = np.memmap(
        low_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frame_count, CHANNELS, STUDENT_H, STUDENT_W),
    )

    previous: np.ndarray | None = None
    for frame_index in range(frame_count):
        current = np.asarray(rgb[frame_index]).copy()

        if previous is None:
            flow_rgb = np.zeros(
                (FULL_H, FULL_W, 3),
                dtype=np.uint8,
            )
        else:
            flow_uv = run_raft_pair(
                raft_session,
                previous,
                current,
                raft_w,
                raft_h,
            )
            flow_rgb = flow_to_rgb(flow_uv)

        combined = np.concatenate(
            [
                current.transpose(2, 0, 1),
                flow_rgb.transpose(2, 0, 1),
            ],
            axis=0,
        ).astype(np.uint8)

        full[frame_index] = combined
        low[frame_index] = downsample_six_channel(combined)
        previous = current

        if (
            (frame_index + 1) % 250 == 0
            or frame_index + 1 == frame_count
        ):
            print(
                f"[{video_name}] feature frames "
                f"{frame_index + 1}/{frame_count}"
            )

    full.flush()
    low.flush()
    del rgb
    gc.collect()
    return full, low, frame_count


def parse_gaze_file(
    path: Path,
) -> tuple[dict[int, tuple[float, float, float, float]], int]:
    rows: dict[int, tuple[float, float, float, float]] = {}
    malformed = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            fields = [
                value.strip()
                for value in line.strip().split(",")
            ]

            if len(fields) != 8:
                malformed += 1
                continue

            try:
                if (
                    fields[0].lower() != "frame"
                    or fields[2].lower() != "forward"
                    or fields[5].lower() != "eye"
                ):
                    malformed += 1
                    continue

                frame = int(fields[1])
                head_x = float(fields[3])
                head_y = float(fields[4])
                eye_x = float(fields[6])
                eye_y = float(fields[7])

                values = (head_x, head_y, eye_x, eye_y)
                if not all(
                    math.isfinite(value)
                    and 0.0 <= value <= 1.0
                    for value in values
                ):
                    malformed += 1
                    continue

                rows[frame] = values
            except (TypeError, ValueError):
                malformed += 1

    return rows, malformed


def load_video_gaze(
    video_id: int,
) -> tuple[
    dict[str, dict[int, tuple[float, float, float, float]]],
    list[dict[str, Any]],
]:
    participant_data: dict[
        str,
        dict[int, tuple[float, float, float, float]],
    ] = {}
    inventory: list[dict[str, Any]] = []

    for participant_dir in sorted(GAZE_DIR.glob("p*")):
        if not participant_dir.is_dir():
            continue

        matches = sorted(
            participant_dir.glob(f"{video_id:03d}.*.txt")
        )
        if not matches:
            continue

        candidates = []
        for path in matches:
            rows, malformed = parse_gaze_file(path)
            candidates.append(
                (
                    len(rows),
                    -malformed,
                    str(path),
                    path,
                    rows,
                    malformed,
                )
            )

        candidates.sort(reverse=True)
        valid_rows, _, _, chosen_path, rows, malformed = (
            candidates[0]
        )

        participant_data[participant_dir.name] = rows
        inventory.append(
            {
                "video_id": f"{video_id:03d}",
                "participant": participant_dir.name,
                "matching_files": len(matches),
                "chosen_file": str(chosen_path),
                "valid_rows": valid_rows,
                "malformed_rows": malformed,
                "first_frame": min(rows) if rows else None,
                "last_frame": max(rows) if rows else None,
            }
        )

    return participant_data, inventory


def gaze_points_at(
    participant_data: dict[
        str,
        dict[int, tuple[float, float, float, float]],
    ],
    frame_index: int,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    return [
        (participant, rows[frame_index])
        for participant, rows in participant_data.items()
        if frame_index in rows
    ]


def select_gaze_frame(
    participant_data: dict[
        str,
        dict[int, tuple[float, float, float, float]],
    ],
    target_frame: int,
    source_frame_count: int,
) -> tuple[int | None, list[Any]]:
    candidates: list[
        tuple[
            tuple[int, int, int],
            int,
            list[Any],
        ]
    ] = []

    for delta in range(
        -MAX_GAZE_SHIFT,
        MAX_GAZE_SHIFT + 1,
    ):
        frame = target_frame + delta
        if frame < 1 or frame > source_frame_count:
            continue
        points = gaze_points_at(
            participant_data,
            frame,
        )
        candidates.append(
            (
                (
                    len(points),
                    -abs(delta),
                    -frame,
                ),
                frame,
                points,
            )
        )

    if not candidates:
        return None, []

    candidates.sort(reverse=True)
    _, frame, points = candidates[0]

    if len(points) < MIN_OBSERVERS:
        return None, []

    return frame, points


def make_spherical_grid(
    width: int,
    height: int,
) -> np.ndarray:
    longitude = (
        (np.arange(width, dtype=np.float32) + 0.5)
        / float(width)
        * (2.0 * math.pi)
        - math.pi
    )
    y_top = (
        np.arange(height, dtype=np.float32) + 0.5
    ) / float(height)
    latitude = (
        math.pi / 2.0
        - y_top * math.pi
    )

    cos_latitude = np.cos(latitude)[:, None]
    sin_latitude = np.sin(latitude)[:, None]
    cos_longitude = np.cos(longitude)[None, :]
    sin_longitude = np.sin(longitude)[None, :]

    return np.stack(
        [
            np.broadcast_to(
                cos_latitude * cos_longitude,
                (height, width),
            ),
            np.broadcast_to(
                sin_latitude,
                (height, width),
            ),
            np.broadcast_to(
                cos_latitude * sin_longitude,
                (height, width),
            ),
        ],
        axis=2,
    ).reshape(-1, 3).astype(np.float32)


GT_GRID_320 = make_spherical_grid(
    FULL_W,
    FULL_H,
)


def fixation_vectors(
    eye_x: np.ndarray,
    eye_y_bottom: np.ndarray,
) -> np.ndarray:
    longitude = (
        eye_x * (2.0 * math.pi)
        - math.pi
    )
    latitude = (
        eye_y_bottom * math.pi
        - math.pi / 2.0
    )
    cos_latitude = np.cos(latitude)

    return np.stack(
        [
            cos_latitude * np.cos(longitude),
            np.sin(latitude),
            cos_latitude * np.sin(longitude),
        ],
        axis=1,
    ).astype(np.float32)


def spherical_gaussian_from_grid(
    grid: np.ndarray,
    vectors: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    output = np.zeros(
        grid.shape[0],
        dtype=np.float32,
    )

    for start in range(
        0,
        vectors.shape[0],
        GT_BATCH_FIXATIONS,
    ):
        current = vectors[
            start : start + GT_BATCH_FIXATIONS
        ]
        dots = np.clip(
            grid @ current.T,
            -1.0,
            1.0,
        )
        angular_distance = np.arccos(dots)
        output += np.exp(
            -0.5
            * (
                angular_distance
                / SIGMA_RADIANS
            )
            ** 2
        ).sum(axis=1)

    output = output.reshape(
        height,
        width,
    )
    total = float(output.sum())

    if not math.isfinite(total) or total <= 0:
        fail("Generated ground-truth map has invalid mass")

    output /= total
    return output


def make_gt_map(
    points: list[
        tuple[
            str,
            tuple[float, float, float, float],
        ]
    ],
) -> np.ndarray:
    eye_x = np.asarray(
        [values[2] for _, values in points],
        dtype=np.float32,
    )
    eye_y_bottom = np.asarray(
        [values[3] for _, values in points],
        dtype=np.float32,
    )
    vectors = fixation_vectors(
        eye_x,
        eye_y_bottom,
    )
    return spherical_gaussian_from_grid(
        GT_GRID_320,
        vectors,
        FULL_W,
        FULL_H,
    )


def gt_resolution_parity(
    points: list[
        tuple[
            str,
            tuple[float, float, float, float],
        ]
    ],
    direct_320: np.ndarray,
) -> dict[str, float]:
    eye_x = np.asarray(
        [values[2] for _, values in points],
        dtype=np.float32,
    )
    eye_y_bottom = np.asarray(
        [values[3] for _, values in points],
        dtype=np.float32,
    )
    vectors = fixation_vectors(
        eye_x,
        eye_y_bottom,
    )

    high_grid = make_spherical_grid(
        2048,
        1024,
    )
    high = spherical_gaussian_from_grid(
        high_grid,
        vectors,
        2048,
        1024,
    )
    resized = cv2.resize(
        high,
        (FULL_W, FULL_H),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    resized /= float(resized.sum())

    cc = metric_cc(
        direct_320,
        resized,
    )
    mae = float(
        np.mean(
            np.abs(
                direct_320.astype(np.float64)
                - resized.astype(np.float64)
            )
        )
    )
    max_abs = float(
        np.max(
            np.abs(
                direct_320.astype(np.float64)
                - resized.astype(np.float64)
            )
        )
    )

    if cc < GT_PARITY_THRESHOLD:
        fail(
            "Direct 320x240 GT failed 2048x1024 parity: "
            f"CC={cc}"
        )

    del high_grid, high, resized
    gc.collect()

    return {
        "pearson": cc,
        "mae": mae,
        "max_absolute_error": max_abs,
        "threshold": GT_PARITY_THRESHOLD,
    }


def extract_last_map(
    output: np.ndarray,
) -> np.ndarray:
    array = np.asarray(output)

    if array.ndim == 5:
        return np.asarray(
            array[0, -1, 0],
            dtype=np.float32,
        )

    if array.ndim == 4:
        if array.shape[1] == 1:
            return np.asarray(
                array[0, 0],
                dtype=np.float32,
            )
        return np.asarray(
            array[0, -1],
            dtype=np.float32,
        )

    if array.ndim == 3:
        return np.asarray(
            array[-1],
            dtype=np.float32,
        )

    fail(
        f"Unsupported model output shape: {array.shape}"
    )


def normalize_01(
    array: np.ndarray,
) -> np.ndarray:
    values = array.astype(np.float64)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))

    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
    ):
        fail("Map contains non-finite values")

    if maximum <= minimum:
        return np.zeros_like(
            values,
            dtype=np.float64,
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def probability(
    array: np.ndarray,
) -> np.ndarray:
    values = np.clip(
        array.astype(np.float64),
        0.0,
        None,
    )
    total = float(values.sum())

    if total <= 0:
        return np.full(
            values.shape,
            1.0 / values.size,
            dtype=np.float64,
        )

    return values / total


def metric_cc(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    p = prediction.ravel().astype(np.float64)
    t = target.ravel().astype(np.float64)

    if (
        float(p.std()) <= 0
        or float(t.std()) <= 0
    ):
        return float("nan")

    return float(
        np.corrcoef(p, t)[0, 1]
    )


def metric_sim(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    return float(
        np.minimum(
            probability(prediction),
            probability(target),
        ).sum()
    )


def metric_kld(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    epsilon = 1e-12
    p = probability(prediction)
    t = probability(target)

    return float(
        np.sum(
            t
            * np.log(
                (t + epsilon)
                / (p + epsilon)
            )
        )
    )


def saliency_metrics(
    prediction_raw: np.ndarray,
    target_raw: np.ndarray,
) -> dict[str, float]:
    prediction = normalize_01(
        prediction_raw,
    )
    target = normalize_01(
        target_raw,
    )

    return {
        "CC": metric_cc(
            prediction,
            target,
        ),
        "SIM": metric_sim(
            prediction,
            target,
        ),
        "KLD": metric_kld(
            prediction,
            target,
        ),
        "MSE": float(
            np.mean(
                (
                    prediction
                    - target
                )
                ** 2
            )
        ),
    }


def probability_preserving_metrics(
    prediction_raw: np.ndarray,
    target_probability: np.ndarray,
) -> dict[str, float]:
    prediction = probability(
        normalize_01(prediction_raw)
    )
    target = probability(
        target_probability
    )
    epsilon = 1e-12

    return {
        "PP_SIM": float(
            np.minimum(
                prediction,
                target,
            ).sum()
        ),
        "PP_KLD": float(
            np.sum(
                target
                * np.log(
                    (target + epsilon)
                    / (prediction + epsilon)
                )
            )
        ),
        "PP_scaled_MSE": float(
            np.mean(
                (
                    prediction
                    - target
                )
                ** 2
            )
            * prediction.size
        ),
    }


def agreement_metrics(
    student_raw: np.ndarray,
    teacher_raw: np.ndarray,
) -> dict[str, float]:
    student = normalize_01(
        student_raw,
    )
    teacher = normalize_01(
        teacher_raw,
    )
    difference = student - teacher
    denominator = float(
        np.linalg.norm(
            teacher.ravel()
        )
    )

    return {
        "teacher_student_CC": metric_cc(
            student,
            teacher,
        ),
        "teacher_student_relative_L2": (
            float(
                np.linalg.norm(
                    difference.ravel()
                )
                / denominator
            )
            if denominator > 0
            else float("inf")
        ),
        "teacher_student_MAE": float(
            np.mean(
                np.abs(difference)
            )
        ),
    }


def finite_mean(
    values: list[float],
) -> float:
    return float(
        np.nanmean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def finite_percentile(
    values: list[float],
    percentile: float,
) -> float:
    return float(
        np.nanpercentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            percentile,
        )
    )


def summarize_rows(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    result: dict[str, float] = {}

    for metric in (
        "CC",
        "SIM",
        "KLD",
        "MSE",
        "PP_SIM",
        "PP_KLD",
        "PP_scaled_MSE",
    ):
        values = [
            float(
                row[
                    f"{prefix}_{metric}"
                ]
            )
            for row in rows
        ]
        result[metric] = finite_mean(
            values
        )
        result[
            f"{metric}_median"
        ] = finite_percentile(
            values,
            50,
        )
        result[
            f"{metric}_p05"
        ] = finite_percentile(
            values,
            5,
        )
        result[
            f"{metric}_p95"
        ] = finite_percentile(
            values,
            95,
        )

    return result


def bootstrap_video_deltas(
    per_video_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    video_count = len(per_video_rows)

    if video_count < 2:
        fail(
            "At least two videos required for bootstrap"
        )

    output: dict[str, Any] = {
        "method": "paired video-level bootstrap",
        "repeats": BOOTSTRAP_REPEATS,
        "videos": video_count,
        "confidence_level": 0.95,
    }

    for metric in (
        "CC",
        "SIM",
        "KLD",
        "MSE",
    ):
        deltas = np.asarray(
            [
                float(
                    row[
                        f"delta_{metric}"
                    ]
                )
                for row in per_video_rows
            ],
            dtype=np.float64,
        )
        samples = np.empty(
            BOOTSTRAP_REPEATS,
            dtype=np.float64,
        )

        for repeat in range(
            BOOTSTRAP_REPEATS
        ):
            indices = rng.integers(
                0,
                video_count,
                size=video_count,
            )
            samples[repeat] = float(
                np.mean(
                    deltas[indices]
                )
            )

        output[metric] = {
            "mean_video_delta": float(
                np.mean(deltas)
            ),
            "ci95_low": float(
                np.percentile(
                    samples,
                    2.5,
                )
            ),
            "ci95_high": float(
                np.percentile(
                    samples,
                    97.5,
                )
            ),
        }

    return output


def latency_summary(
    values_seconds: list[float],
) -> dict[str, float]:
    values_ms = (
        np.asarray(
            values_seconds,
            dtype=np.float64,
        )
        * 1000.0
    )
    return {
        "mean_ms": float(
            np.mean(values_ms)
        ),
        "median_ms": float(
            np.median(values_ms)
        ),
        "p95_ms": float(
            np.percentile(
                values_ms,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                values_ms,
                99,
            )
        ),
        "samples": int(
            len(values_ms)
        ),
        "note": (
            "session.run wall time; includes host-device transfers; "
            "excludes SEA-RAFT, video decoding, feature preparation "
            "and ground-truth construction"
        ),
    }


def heatmap(
    values: np.ndarray,
) -> np.ndarray:
    normalized = normalize_01(
        values
    )
    image = np.clip(
        normalized * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return cv2.applyColorMap(
        image,
        cv2.COLORMAP_JET,
    )


def label(
    image: np.ndarray,
    text: str,
) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 29),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        result,
        text,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def save_preview(
    path: Path,
    frame_bgr: np.ndarray,
    target: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
    title: str,
) -> None:
    tiles = [
        label(
            frame_bgr,
            title,
        ),
        label(
            heatmap(target),
            "eye GT",
        ),
        label(
            heatmap(teacher),
            "teacher",
        ),
        label(
            heatmap(student),
            "KD student",
        ),
    ]
    cv2.imwrite(
        str(path),
        np.concatenate(
            tiles,
            axis=1,
        ),
    )



FLOWSAL_ONNX = Path(os.environ["VREYE_FLOWSAL_ONNX"])
BASELINE_CSV = Path(os.environ["VREYE_BASELINE_CSV"])
COMPLETION_MARKER = Path(os.environ["VREYE_COMPLETION_MARKER"])
FLOW_CACHE = EXP_ROOT / "flowsal_trt_cache"
FLOW_CACHE.mkdir(parents=True, exist_ok=True)
FLOWSAL_INPUT_SHAPE = (1, T, 3, STUDENT_H, STUDENT_W)
FLOWSAL_PARAMS = 27731
EXPECTED_BASELINE_ROWS = 17129
EXPECTED_TEST_VIDEOS = 74


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"No rows found in {path}")
    return rows


def build_rgb_memmap(
    video_name: str,
    video_path: Path,
) -> tuple[np.memmap, np.memmap, int]:
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
        shape=(frame_count, 3, STUDENT_H, STUDENT_W),
    )

    for frame_index in range(frame_count):
        current = np.asarray(rgb_full[frame_index])
        chw = current.transpose(2, 0, 1)
        for channel in range(3):
            rgb_low[frame_index, channel] = cv2.resize(
                chw[channel],
                (STUDENT_W, STUDENT_H),
                interpolation=cv2.INTER_LINEAR,
            )
        if (
            (frame_index + 1) % 250 == 0
            or frame_index + 1 == frame_count
        ):
            print(
                f"[{video_name}] RGB frames "
                f"{frame_index + 1}/{frame_count}"
            )

    rgb_low.flush()
    return rgb_full, rgb_low, frame_count


def save_flowsal_preview(
    path: Path,
    frame_bgr: np.ndarray,
    target: np.ndarray,
    flowsal: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = cv2.resize(frame_bgr, (FULL_W, FULL_H), interpolation=cv2.INTER_LINEAR)
    target_hm = heatmap(target)
    flowsal_hm = heatmap(flowsal)
    frame = label(frame, "RGB endpoint")
    target_hm = label(target_hm, "Gaze ground truth")
    flowsal_hm = label(flowsal_hm, "FlowSal v1.1")
    canvas = cv2.hconcat([frame, target_hm, flowsal_hm])
    cv2.putText(
        canvas,
        title,
        (10, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        fail(f"Could not write preview: {path}")


def exact_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


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
        "sign_flip_repeats": repeats,
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

        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(repeats, n))
        flipped = (signs * deltas[None, :]).mean(axis=1)
        flip_p = float((1 + np.count_nonzero(np.abs(flipped) >= abs(observed))) / (repeats + 1))

        wins = int(np.count_nonzero(deltas > 0))
        losses = int(np.count_nonzero(deltas < 0))
        if metric in {"KLD", "MSE"}:
            wins, losses = losses, wins

        loo = np.asarray(
            [float(np.mean(np.delete(deltas, i))) for i in range(n)],
            dtype=np.float64,
        )

        output[metric] = {
            "mean_delta": observed,
            "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
            "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
            "wins": wins,
            "losses": losses,
            "ties": int(n - wins - losses),
            "sign_p": exact_sign_test(wins, losses),
            "sign_flip_p_monte_carlo": flip_p,
            "loo_min": float(np.min(loo)),
            "loo_max": float(np.max(loo)),
        }

    return output


def main() -> None:
    _, test_ids = load_official_split()
    baseline_rows = read_csv_rows(BASELINE_CSV)

    required_baseline_fields = {
        "video",
        "model_endpoint_sample_index",
        "selected_gaze_frame_1based",
        "valid_observers",
        "teacher_wall_latency_ms",
        "student_wall_latency_ms",
    }
    for prefix in ("teacher", "student"):
        for metric in ("CC", "SIM", "KLD", "MSE", "PP_SIM", "PP_KLD", "PP_scaled_MSE"):
            required_baseline_fields.add(f"{prefix}_{metric}")

    missing_fields = sorted(required_baseline_fields - set(baseline_rows[0]))
    if missing_fields:
        fail(f"Baseline CSV is missing fields: {missing_fields}")
    if len(baseline_rows) != EXPECTED_BASELINE_ROWS:
        fail(
            f"Expected {EXPECTED_BASELINE_ROWS} baseline rows, "
            f"found {len(baseline_rows)}"
        )

    baseline_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    baseline_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in baseline_rows:
        video = str(row["video"]).strip().zfill(3)
        endpoint = int(row["model_endpoint_sample_index"])
        key = (video, endpoint)
        if key in baseline_by_key:
            fail(f"Duplicate baseline key: {key}")
        baseline_by_key[key] = row
        baseline_by_video[video].append(row)

    expected_names = {f"{value:03d}" for value in test_ids}
    if set(baseline_by_video) != expected_names:
        fail("Baseline CSV videos do not match the official 74-video split")

    for rows in baseline_by_video.values():
        rows.sort(key=lambda row: int(row["model_endpoint_sample_index"]))

    protocol = {
        "status": "frozen",
        "external_dataset": "VR-EyeTracking official 74-video test split",
        "training_or_finetuning": False,
        "checkpoint_selection": False,
        "flowsal_frozen_before_evaluation": True,
        "flowsal_sha256": sha256(FLOWSAL_ONNX),
        "baseline_csv": str(BASELINE_CSV),
        "baseline_csv_sha256": sha256(BASELINE_CSV),
        "baseline_samples": len(baseline_rows),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "timesteps": T,
        "window_span_seconds": WINDOW_SPAN_SECONDS,
        "evaluation_endpoints": "exactly reused from frozen teacher/KD per-sample CSV",
        "gaze_frames": "exactly reused from frozen teacher/KD per-sample CSV",
        "ground_truth": "eye-coordinate spherical Gaussian, sigma 9.35 degrees",
        "runtime_input": "RGB only at [1,20,3,144,192]",
        "sea_raft_executed": False,
        "teacher_inference_executed": False,
        "kd_inference_executed": False,
    }
    PROTOCOL_JSON.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("VR-EYETRACKING FLOWSAL v1.1 ZERO-SHOT EVALUATION")
    print("=" * 132)
    print("Official test videos:", len(test_ids))
    print("Frozen baseline samples:", len(baseline_rows))
    print("FlowSal:", FLOWSAL_ONNX)
    print("FlowSal SHA256:", sha256(FLOWSAL_ONNX))
    print("SEA-RAFT executed: NO")
    print("Teacher/KD inference repeated: NO")
    print("Training or checkpoint selection: NO")

    session = create_session(
        FLOWSAL_ONNX,
        FLOW_CACHE,
        "FLOWSAL_V1_1_VREYE",
        True,
    )
    input_name = session.get_inputs()[0].name
    warmup = np.zeros(FLOWSAL_INPUT_SHAPE, dtype=np.float32)
    for _ in range(10):
        session.run(None, {input_name: warmup})

    all_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    flowsal_times: list[float] = []
    dataset_inventory: list[dict[str, Any]] = []
    gaze_inventory: list[dict[str, Any]] = []
    parity_result: dict[str, float] | None = None

    evaluation_started = time.perf_counter()

    for video_number, video_id in enumerate(test_ids, start=1):
        video_started = time.perf_counter()
        name = f"{video_id:03d}"
        video_path = VIDEO_DIR / f"{name}.mp4"
        if not video_path.is_file():
            fail(f"Missing video: {video_path}")

        metadata = ffprobe(video_path)
        stream = metadata["streams"][0]
        source_fps = parse_fraction(str(stream["avg_frame_rate"]))
        duration = float(metadata["format"]["duration"])
        raw_frame_count = stream.get("nb_frames")
        source_frame_count = (
            int(raw_frame_count)
            if raw_frame_count and str(raw_frame_count).isdigit()
            else int(round(duration * source_fps))
        )

        participant_data, current_inventory = load_video_gaze(video_id)
        gaze_inventory.extend(current_inventory)
        if not participant_data:
            fail(f"No participant gaze files for {name}")

        print()
        print(f"===== VIDEO {video_number:02d}/{len(test_ids):02d}: {name} =====")
        rgb_full, rgb_low, sampled_count = build_rgb_memmap(name, video_path)

        video_rows: list[dict[str, Any]] = []
        preview_saved = False
        frozen_rows = baseline_by_video[name]

        for local_index, baseline in enumerate(frozen_rows, start=1):
            endpoint_index = int(baseline["model_endpoint_sample_index"])
            selected_gaze_frame = int(baseline["selected_gaze_frame_1based"])
            expected_observers = int(float(baseline["valid_observers"]))

            if endpoint_index < T - 1 or endpoint_index >= sampled_count:
                fail(
                    f"Endpoint {endpoint_index} is outside sampled RGB for {name}: "
                    f"frames={sampled_count}"
                )

            points = gaze_points_at(participant_data, selected_gaze_frame)
            if len(points) != expected_observers:
                fail(
                    f"Observer mismatch for {name} endpoint={endpoint_index}: "
                    f"baseline={expected_observers}, current={len(points)}"
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
            target_map = make_gt_map(points)

            if parity_result is None:
                parity_result = gt_resolution_parity(points, target_map)
                print(f"[GT] parity Pearson={parity_result['pearson']:.9f}")

            flowsal_metrics = saliency_metrics(flowsal_map, target_map)
            flowsal_pp = probability_preserving_metrics(flowsal_map, target_map)

            row: dict[str, Any] = dict(baseline)
            row["video"] = name
            row["flowsal_wall_latency_ms"] = elapsed * 1000.0
            for metric, value in flowsal_metrics.items():
                row[f"flowsal_{metric}"] = value
                row[f"flowsal_minus_student_{metric}"] = value - float(baseline[f"student_{metric}"])
                row[f"flowsal_minus_teacher_{metric}"] = value - float(baseline[f"teacher_{metric}"])
            for metric, value in flowsal_pp.items():
                row[f"flowsal_{metric}"] = value
                row[f"flowsal_minus_student_{metric}"] = value - float(baseline[f"student_{metric}"])
                row[f"flowsal_minus_teacher_{metric}"] = value - float(baseline[f"teacher_{metric}"])

            all_rows.append(row)
            video_rows.append(row)

            if SAVE_PREVIEWS and not preview_saved:
                endpoint_bgr = np.asarray(rgb_full[endpoint_index]).copy()
                save_flowsal_preview(
                    PREVIEW_DIR / f"{name}_flowsal_preview.png",
                    endpoint_bgr,
                    target_map,
                    flowsal_map,
                    f"{name} endpoint={endpoint_index} observers={len(points)}",
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
            "source_duration_seconds": duration,
            "source_fps": source_fps,
            "source_frame_count": source_frame_count,
            "sampled_7_5hz_frames": sampled_count,
            "participant_recordings": len(participant_data),
            "evaluated_samples": len(video_rows),
        }
        for metric in ("CC", "SIM", "KLD", "MSE", "PP_SIM", "PP_KLD", "PP_scaled_MSE"):
            for prefix, summary in (
                ("teacher", teacher_summary),
                ("student", student_summary),
                ("flowsal", flowsal_summary),
            ):
                video_result[f"{prefix}_{metric}"] = summary[metric]
            video_result[f"flowsal_minus_student_{metric}"] = flowsal_summary[metric] - student_summary[metric]
            video_result[f"flowsal_minus_teacher_{metric}"] = flowsal_summary[metric] - teacher_summary[metric]

        per_video_rows.append(video_result)
        dataset_inventory.append(
            {
                "video": name,
                "video_path": str(video_path),
                "video_duration_seconds": duration,
                "source_fps": source_fps,
                "source_frame_count": source_frame_count,
                "sampled_frames": sampled_count,
                "participant_recordings": len(participant_data),
                "evaluated_samples": len(video_rows),
            }
        )

        del rgb_full, rgb_low, participant_data
        gc.collect()
        if not KEEP_TEMP:
            shutil.rmtree(TEMP_ROOT / name, ignore_errors=True)
        print(f"[{name}] complete in {time.perf_counter() - video_started:.1f}s")

    if len(all_rows) != EXPECTED_BASELINE_ROWS:
        fail(f"Expected {EXPECTED_BASELINE_ROWS} evaluated rows, found {len(all_rows)}")
    if len(per_video_rows) != EXPECTED_TEST_VIDEOS:
        fail(f"Expected {EXPECTED_TEST_VIDEOS} videos, found {len(per_video_rows)}")
    if parity_result is None:
        fail("GT parity check did not run")

    with PER_SAMPLE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with PER_VIDEO_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_video_rows)
    with GAZE_INVENTORY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gaze_inventory[0].keys()))
        writer.writeheader()
        writer.writerows(gaze_inventory)
    INVENTORY_JSON.write_text(json.dumps(dataset_inventory, indent=2), encoding="utf-8")

    micro = {
        prefix: summarize_rows(all_rows, prefix)
        for prefix in ("teacher", "student", "flowsal")
    }
    macro = {
        prefix: {
            metric: finite_mean([float(row[f"{prefix}_{metric}"]) for row in per_video_rows])
            for metric in ("CC", "SIM", "KLD", "MSE", "PP_SIM", "PP_KLD", "PP_scaled_MSE")
        }
        for prefix in ("teacher", "student", "flowsal")
    }

    flowsal_latency = latency_summary(flowsal_times)
    baseline_teacher_latency = latency_summary(
        [float(row["teacher_wall_latency_ms"]) / 1000.0 for row in baseline_rows]
    )
    baseline_student_latency = latency_summary(
        [float(row["student_wall_latency_ms"]) / 1000.0 for row in baseline_rows]
    )

    inference_vs_student = paired_video_inference(
        per_video_rows, "flowsal", "student", BOOTSTRAP_REPEATS
    )
    inference_vs_teacher = paired_video_inference(
        per_video_rows, "flowsal", "teacher", BOOTSTRAP_REPEATS
    )

    report = {
        "status": "success",
        "experiment": "FlowSal v1.1 frozen zero-shot evaluation on VR-EyeTracking official test split",
        "models_frozen": True,
        "training_or_finetuning": False,
        "checkpoint_selection_on_vr_eyetracking": False,
        "sea_raft_executed": False,
        "teacher_or_kd_inference_repeated": False,
        "official_test_videos": len(per_video_rows),
        "evaluated_samples": len(all_rows),
        "protocol": protocol,
        "gt_direct_resolution_parity": parity_result,
        "models": {
            "teacher": {
                "parameters": TEACHER_PARAMS,
                "micro_summary": micro["teacher"],
                "macro_video_summary": macro["teacher"],
                "wall_latency_from_frozen_baseline": baseline_teacher_latency,
            },
            "student": {
                "parameters": STUDENT_PARAMS,
                "micro_summary": micro["student"],
                "macro_video_summary": macro["student"],
                "wall_latency_from_frozen_baseline": baseline_student_latency,
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
            "per_sample_csv": str(PER_SAMPLE_CSV),
            "per_video_csv": str(PER_VIDEO_CSV),
            "gaze_inventory_csv": str(GAZE_INVENTORY_CSV),
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

    text = f"""VR-EYETRACKING FLOWSAL v1.1 ZERO-SHOT EVALUATION REPORT
========================================================================================================================
Status: SUCCESS
Training/fine-tuning on VR-EyeTracking: NO
Checkpoint selection on VR-EyeTracking: NO
FlowSal frozen before evaluation: YES
SEA-RAFT executed in this run: NO
Teacher/KD inference repeated: NO
Official test videos: {len(per_video_rows)}
Evaluated samples: {len(all_rows)}
FlowSal SHA256: {sha256(FLOWSAL_ONNX)}

FROZEN PROTOCOL
------------------------------------------------------------------------------------------------------------------------
Model input rate: {SAMPLE_RATE_HZ:.6f} Hz
Timesteps: {T}
Window span: {WINDOW_SPAN_SECONDS:.6f} seconds
Evaluation endpoints and gaze frames: reused exactly from frozen baseline CSV
Ground truth: eye-coordinate spherical Gaussian, sigma {SIGMA_DEGREES:.2f} degrees
GT resolution: 320x240
Direct 320 vs 2048->320 parity Pearson: {parity_result['pearson']:.9f}

PRIMARY MICRO AVERAGE ACROSS ALL {len(all_rows)} SAMPLES
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
            f"{item['wins']:8d}{item['losses']:8d}{item['sign_p']:12.7f}{item['sign_flip_p_monte_carlo']:12.7f}"
            f"{item['loo_min']:+14.7f}{item['loo_max']:+14.7f}\n"
        )

    text += f"""
INFERENCE LATENCY, SESSION.RUN WALL TIME
------------------------------------------------------------------------------------------------------------------------
Method                              Mean ms      P95 ms      P99 ms
Original SST-Sal teacher          {baseline_teacher_latency['mean_ms']:9.3f}   {baseline_teacher_latency['p95_ms']:9.3f}   {baseline_teacher_latency['p99_ms']:9.3f}
KD R192-W24-T20 + SEA-RAFT        {baseline_student_latency['mean_ms']:9.3f}   {baseline_student_latency['p95_ms']:9.3f}   {baseline_student_latency['p99_ms']:9.3f}
FlowSal v1.1 RGB-only             {flowsal_latency['mean_ms']:9.3f}   {flowsal_latency['p95_ms']:9.3f}   {flowsal_latency['p99_ms']:9.3f}
Note: model-call latency excludes video decoding, ground-truth construction and SEA-RAFT.
FlowSal requires no SEA-RAFT execution; teacher and KD require external flow in deployment.

SECONDARY PROBABILITY-PRESERVING MICRO METRICS
------------------------------------------------------------------------------------------------------------------------
Method                          PP_SIM      PP_KLD      PP_scaled_MSE
Original SST-Sal teacher        {teacher['PP_SIM']:.7f}   {teacher['PP_KLD']:.7f}   {teacher['PP_scaled_MSE']:.7f}
KD R192-W24-T20 + SEA-RAFT      {kd['PP_SIM']:.7f}   {kd['PP_KLD']:.7f}   {kd['PP_scaled_MSE']:.7f}
FlowSal v1.1 RGB-only           {fs['PP_SIM']:.7f}   {fs['PP_KLD']:.7f}   {fs['PP_scaled_MSE']:.7f}

FILES
------------------------------------------------------------------------------------------------------------------------
Per-sample CSV: {PER_SAMPLE_CSV}
Per-video CSV: {PER_VIDEO_CSV}
Protocol JSON: {PROTOCOL_JSON}
Dataset inventory JSON: {INVENTORY_JSON}
Report JSON: {REPORT_JSON}
Previews: {PREVIEW_DIR}
"""

    REPORT_TXT.write_text(text, encoding="utf-8")
    COMPLETION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    COMPLETION_MARKER.write_text(
        json.dumps(
            {
                "status": "success",
                "completed_run": str(RUN_DIR),
                "report_json": str(REPORT_JSON),
                "report_txt": str(REPORT_TXT),
                "flowsal_sha256": sha256(FLOWSAL_ONNX),
                "baseline_csv_sha256": sha256(BASELINE_CSV),
                "official_test_videos": len(per_video_rows),
                "evaluated_samples": len(all_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(text)


if __name__ == "__main__":
    main()

PYCODE

    echo
    echo "===== RUNNING FROZEN FLOWSAL EXTERNAL TEST ====="
    "$VENV/bin/python" "$PY_SCRIPT"

    echo
    echo "===== RESULT HASHES ====="
    (
        cd "$RUN_DIR"
        find . -type f ! -name SHA256SUMS.txt -print0 \
            | sort -z \
            | xargs -0 sha256sum \
            | tee SHA256SUMS.txt
    )

    ln -sfn "$RUN_DIR" "$EXP_ROOT/latest"

    echo
    echo "===== FINAL REPORT ====="
    cat "$RUN_DIR/report.txt"

    echo
    echo "Completed successfully."
} 2>&1 | tee "$CONSOLE_LOG"
