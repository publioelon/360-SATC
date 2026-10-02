#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.amp import autocast


INPUT_SHAPE = (1, 20, 6, 240, 320)


def load_module_from_file(module_name: str, file_path: Path):
    """Load a module under the exact name stored in the old PyTorch pickle."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import specification for {module_name}: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_checkpoint_compatibility(model_path: Path) -> dict[str, str]:
    """
    SST_Sal.pth is a full-object pickle created by the official repository.
    It stores references to:
      models.SST_Sal
      Modules.SpherConvLSTM_EncoderCell
      Modules.SpherConvLSTMCell
      Modules.SpherConvLSTM_DecoderCell
      spherenet.sphere_cnn.*

    The local research tree also contains directories named models/ and Modules/,
    which can shadow the original flat modules. Load the intended files explicitly.
    """
    project_root = model_path.parents[2]
    modules_file = project_root / "Modules.py"

    candidate_model_files = [
        project_root / "models.py",
        project_root / "satc_modules" / "models.py",
    ]
    models_file = next((p for p in candidate_model_files if p.is_file()), None)

    if not modules_file.is_file():
        raise FileNotFoundError(f"Required compatibility module is missing: {modules_file}")
    if models_file is None:
        raise FileNotFoundError(
            "Could not find the SST-Sal model definition. Checked: "
            + ", ".join(str(p) for p in candidate_model_files)
        )
    if not (project_root / "spherenet").is_dir():
        raise FileNotFoundError(f"Required spherenet package is missing: {project_root / 'spherenet'}")

    root_string = str(project_root)
    sys.path = [root_string] + [p for p in sys.path if p != root_string]

    # Remove any previously imported shadow packages.
    for name in ("models", "Modules"):
        sys.modules.pop(name, None)

    modules_module = load_module_from_file("Modules", modules_file)
    models_module = load_module_from_file("models", models_file)

    required_modules_symbols = (
        "SpherConvLSTM_EncoderCell",
        "SpherConvLSTMCell",
        "SpherConvLSTM_DecoderCell",
    )
    missing_modules = [
        symbol for symbol in required_modules_symbols
        if not hasattr(modules_module, symbol)
    ]
    if missing_modules:
        raise RuntimeError(
            f"{modules_file} is missing checkpoint symbols: {missing_modules}"
        )
    if not hasattr(models_module, "SST_Sal"):
        raise RuntimeError(f"{models_file} does not define SST_Sal")

    return {
        "project_root": str(project_root),
        "Modules": str(modules_file),
        "models": str(models_file),
        "spherenet": str(project_root / "spherenet"),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "std_ms": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min_ms": float(min(values)),
        "median_ms": float(statistics.median(values)),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": float(max(values)),
        "throughput_fps_from_mean": float(1000.0 / statistics.fmean(values)),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deadline_stats(values: list[float]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for fps in (30, 60, 90, 120):
        deadline = 1000.0 / fps
        misses = sum(v > deadline for v in values)
        result[str(fps)] = {
            "deadline_ms": deadline,
            "misses": misses,
            "miss_pct": 100.0 * misses / len(values) if values else float("nan"),
        }
    return result


def create_events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def time_cuda_stage(
    stage_name: str,
    repeats: int,
    operation: Callable[[], object],
) -> tuple[list[dict[str, float | int | str]], list[float], list[float]]:
    rows: list[dict[str, float | int | str]] = []
    gpu_values: list[float] = []
    wall_values: list[float] = []

    for iteration in range(repeats):
        start_event, end_event = create_events()
        torch.cuda.synchronize()
        wall_start = time.perf_counter_ns()
        start_event.record()
        result = operation()
        end_event.record()
        torch.cuda.synchronize()
        wall_end = time.perf_counter_ns()

        # Keep result alive until after synchronization.
        if result is None:
            raise RuntimeError(f"{stage_name} returned None")

        gpu_ms = float(start_event.elapsed_time(end_event))
        wall_ms = (wall_end - wall_start) / 1_000_000.0
        gpu_values.append(gpu_ms)
        wall_values.append(wall_ms)
        rows.append(
            {
                "stage": stage_name,
                "iteration": iteration,
                "gpu_ms": gpu_ms,
                "wall_ms": wall_ms,
            }
        )

    return rows, gpu_values, wall_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--mixed-precision", choices=["true", "false"], required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mixed_precision = args.mixed_precision == "true"

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": None,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "input_shape": list(INPUT_SHAPE),
        "mixed_precision": mixed_precision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    try:
        import torchvision
        environment["torchvision"] = torchvision.__version__
    except Exception:
        pass

    # Recreate the exact module names stored in the official full-object pickle.
    compatibility_modules = install_checkpoint_compatibility(model_path)
    environment["checkpoint_compatibility_modules"] = compatibility_modules
    print("Checkpoint compatibility modules:")
    for name, path in compatibility_modules.items():
        print(f"  {name}: {path}")

    load_start = time.perf_counter_ns()
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device).eval()
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter_ns() - load_start) / 1_000_000.0

    gpu_input = torch.rand(INPUT_SHAPE, dtype=torch.float32, device=device)
    cpu_input = torch.rand(INPUT_SHAPE, dtype=torch.float32, device="cpu")
    try:
        cpu_input = cpu_input.pin_memory()
        pinned_memory = True
    except RuntimeError:
        pinned_memory = False

    def infer_gpu():
        with torch.inference_mode(), autocast(
            device_type="cuda", enabled=mixed_precision
        ):
            return model(gpu_input)

    # True cold inference: before explicit warm-up.
    cold_start_event, cold_end_event = create_events()
    torch.cuda.synchronize()
    cold_wall_start = time.perf_counter_ns()
    cold_start_event.record()
    cold_output = infer_gpu()
    cold_end_event.record()
    torch.cuda.synchronize()
    cold_wall_ms = (time.perf_counter_ns() - cold_wall_start) / 1_000_000.0
    cold_gpu_ms = float(cold_start_event.elapsed_time(cold_end_event))
    output_shape = list(cold_output.shape)

    # Explicit warm-up.
    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast(device_type="cuda", enabled=mixed_precision):
                _ = model(gpu_input)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)

    all_rows: list[dict[str, float | int | str]] = []
    summaries: dict[str, object] = {}

    rows, gpu_ms, wall_ms = time_cuda_stage(
        "gpu_resident_inference",
        args.repeats,
        infer_gpu,
    )
    all_rows.extend(rows)
    summaries["gpu_resident_inference"] = {
        "gpu": summarize(gpu_ms),
        "wall": summarize(wall_ms),
        "deadline_compliance_wall": deadline_stats(wall_ms),
    }

    def transfer_and_infer():
        transferred = cpu_input.to(device, non_blocking=pinned_memory)
        with torch.inference_mode(), autocast(
            device_type="cuda", enabled=mixed_precision
        ):
            return model(transferred)

    rows, gpu_ms, wall_ms = time_cuda_stage(
        "host_to_device_plus_inference",
        args.repeats,
        transfer_and_infer,
    )
    all_rows.extend(rows)
    summaries["host_to_device_plus_inference"] = {
        "gpu": summarize(gpu_ms),
        "wall": summarize(wall_ms),
        "deadline_compliance_wall": deadline_stats(wall_ms),
        "pinned_memory": pinned_memory,
    }

    # Reuse one model output to isolate the exact postprocessing pattern used by
    # satc_modules/saliency.py: last timestep, normalization, CPU conversion,
    # uint8 conversion, and OpenCV resize.
    with torch.inference_mode(), autocast(
        device_type="cuda", enabled=mixed_precision
    ):
        reusable_output = model(gpu_input)
    torch.cuda.synchronize()

    postprocess_results: dict[str, object] = {}
    for width, height in ((1920, 960), (2732, 1366), (4096, 2048)):
        values: list[float] = []
        local_repeats = min(args.repeats, 200)
        for _ in range(local_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            tensor = reusable_output[:, -1, ...]
            if tensor.dim() == 4 and tensor.shape[1] == 1:
                tensor = tensor[0, 0]
            elif tensor.dim() == 4 and tensor.shape[1] == 3:
                tensor = tensor[0].mean(0)
            elif tensor.dim() == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.mean(0)
            tensor = tensor.detach().float()
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
            sal = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
            sal = cv2.resize(sal, (width, height), interpolation=cv2.INTER_LINEAR)
            if sal.shape != (height, width):
                raise RuntimeError("Unexpected postprocessed saliency-map shape.")
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            values.append(dt_ms)
        key = f"{width}x{height}"
        postprocess_results[key] = {
            "wall": summarize(values),
            "repeats": local_repeats,
        }

    # Full inference + production of the last saliency map at each output size.
    full_stage_results: dict[str, object] = {}
    full_stage_repeats = min(args.repeats, 200)
    for width, height in ((1920, 960), (2732, 1366), (4096, 2048)):
        wall_values: list[float] = []
        for _ in range(full_stage_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            with torch.inference_mode(), autocast(
                device_type="cuda", enabled=mixed_precision
            ):
                output = model(gpu_input)
            tensor = output[:, -1, ...]
            if tensor.dim() == 4 and tensor.shape[1] == 1:
                tensor = tensor[0, 0]
            elif tensor.dim() == 4 and tensor.shape[1] == 3:
                tensor = tensor[0].mean(0)
            elif tensor.dim() == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.mean(0)
            tensor = tensor.detach().float()
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
            sal = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
            sal = cv2.resize(sal, (width, height), interpolation=cv2.INTER_LINEAR)
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            wall_values.append(dt_ms)
        key = f"{width}x{height}"
        full_stage_results[key] = {
            "wall": summarize(wall_values),
            "deadline_compliance_wall": deadline_stats(wall_values),
            "repeats": full_stage_repeats,
        }

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    csv_path = output_dir / "per_iteration_timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stage", "iteration", "gpu_ms", "wall_ms"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "environment": environment,
        "model_load_ms": model_load_ms,
        "cold_inference": {
            "gpu_ms": cold_gpu_ms,
            "wall_ms": cold_wall_ms,
            "output_shape": output_shape,
        },
        "summaries": summaries,
        "postprocess_only": postprocess_results,
        "full_sst_saliency_stage": full_stage_results,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "interpretation": {
            "one_model_call_processes_time_steps": INPUT_SHAPE[1],
            "one_model_call_produces_output_shape": output_shape,
            "pipeline_uses_last_time_step_only": True,
            "inference_latency_is_per_20_frame_clip_and_per_new_saliency_map": True,
        },
    }

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    txt_path = output_dir / "summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("SST-Sal inference benchmark v3\n")
        f.write("=" * 72 + "\n")
        f.write(json.dumps(report, indent=2))
        f.write("\n")

    print(json.dumps(report, indent=2))
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"TXT:  {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
