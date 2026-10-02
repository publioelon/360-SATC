#!/usr/bin/env python3
"""
RUN_360SATC_FINAL_QUALITY_MATRIX_v1.py
=====================================

Final fixed-rate full-reference quality experiment for the 360-SATC paper.

This runner is intentionally limited to the ONE experiment still needed to
complete the per-video quality heatmaps.

FINAL MATRIX
------------
Videos:
  - basketball
  - rollercoaster
  - ballet
  - london_tower

Codecs:
  - H.264
  - H.265/HEVC
  - AV1

Bandwidth targets:
  - 12 Mbit/s (Low)
  - 35 Mbit/s (High)

Methods:
  - 360-SATC
  - CBR
  - MUC
  - PC
  - AUC
  - 360-ST

Protocol:
  - 4096x2048 ERP, 60 FPS
  - NVIDIA NVENC P7
  - CBR
  - GOP 240
  - no B-frames
  - 240 warm-up frames
  - 896 measured frames
  - PSNR
  - SSIM
  - LPIPS (AlexNet v0.1)
  - WS-PSNR
  - actual measured bitrate
  - measured payload / mean frame size
  - complete encoded-file size

Final 360-SATC spatial allocation:
  - 5x9 tile grid
  - continuous MoST-Sal score after the existing ERP/EMA processing
  - 6 highest-scoring tiles: QP delta -2
  - next 6 tiles: QP delta -1
  - remaining 33 tiles: QP delta 0
  - no positive QP offsets
  - no neighbor ring

IMPORTANT:
  "RG-6-6" is NOT used as a paper-facing method name. The method is 360-SATC.

REUSE POLICY
------------
By default, this runner reuses ONLY exact-compatible existing London Tower
baseline quality rows if they are present under:

  /home/mininet-ovs/Downloads/360SATC_ALL_EXPERIMENTS_20260925_000038/exp2_quality

Eligible reuse methods:
  CBR, MUC, PC, AUC, 360-ST

The old 360-SATC row is NEVER reused because the spatial policy changed.

If the exact old London baseline files/protocol are unavailable or fail audit,
the corresponding cases are encoded again automatically.

With all 30 London baseline rows reusable, the fresh workload is:
  - 24 new 360-SATC encodes
  - 18 CBR encodes (3 non-London videos x 3 codecs x 2 rates)
  - 72 Caruso-family encodes
  = 114 fresh encodes

If nothing can be reused, the runner completes all 144 encodes.

RESUME
------
The output directory is resumable. Completed valid encodes and completed metric
rows are reused. Failed/partial individual jobs are restarted without deleting
completed jobs.

OUTPUT
------
Primary results:
  FINAL_QUALITY_RESULTS.csv
  FINAL_QUALITY_RESULTS.json
  HEATMAP_READY.csv
  FINAL_QUALITY_PROTOCOL.json
  FINAL_QUALITY_SUMMARY.json
  REUSE_REPORT.json

Small diagnostic upload package:
  /home/mininet-ovs/Downloads/files_to_upload/
      SATC_FINAL_QUALITY_EXP09_RESULTS.zip

Raw encoded streams stay in the experiment directory and are not placed in the
upload ZIP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VERSION = "360SATC_FINAL_QUALITY_MATRIX_20260926_v1"

DEFAULT_MASTER = Path(__file__).with_name("all_experiments_v6.py")
DEFAULT_BASE = Path(__file__).with_name("allocation_candidates.py")
DEFAULT_LONDON = Path(os.environ.get("SATC_LONDON", str(Path(__file__).resolve().parents[2] / "data/prepared/london_tower_4096x2048_60fps.mp4")))
DEFAULT_OLD_QUALITY = Path(
    "/home/mininet-ovs/Downloads/360SATC_ALL_EXPERIMENTS_20260925_000038/exp2_quality"
)
DEFAULT_OUTPUT = Path("/home/mininet-ovs/Downloads/SATC_FINAL_QUALITY_EXP09")
UPLOAD_DIR = Path("/home/mininet-ovs/Downloads/files_to_upload")

VIDEOS = ("basketball", "rollercoaster", "ballet", "london_tower")
CODECS = ("h264", "hevc", "av1")
RATES = (12, 35)
METHODS = ("360-SATC", "CBR", "MUC", "PC", "AUC", "360-ST")
CARUSO_METHODS = ("muc", "pc", "auc", "360st")
CARUSO_LABEL = {"muc": "MUC", "pc": "PC", "auc": "AUC", "360st": "360-ST"}
EXT = {"h264": "h264", "hevc": "hevc", "av1": "ivf"}

WARMUP = 240
MEASURED = 896
FPS = 60
WIDTH = 4096
HEIGHT = 2048


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_py(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def result_key(video, codec, rate, method):
    return f"{video}|{codec}|{int(rate)}|{method}"


def expected_keys():
    return {
        result_key(v, c, r, m)
        for v in VIDEOS
        for c in CODECS
        for r in RATES
        for m in METHODS
    }


def measured_byte_stats(events_csv, warmup=WARMUP, frames=MEASURED):
    rows = read_csv(events_csv)
    wanted = [
        r for r in rows
        if warmup <= int(r["frame_id"]) < warmup + frames
    ]
    ids = [int(r["frame_id"]) for r in wanted]
    if ids != list(range(warmup, warmup + frames)):
        raise RuntimeError(
            f"{events_csv}: measured frame IDs are not exactly "
            f"{warmup}..{warmup + frames - 1}"
        )
    total = sum(int(r["encoded_bytes"]) for r in wanted)
    return {
        "measured_frames": frames,
        "measured_payload_bytes": int(total),
        "measured_payload_mb": float(total / 1e6),
        "mean_frame_size_bytes": float(total / frames),
        "actual_bitrate_mbps": float(total * 8.0 * FPS / frames / 1e6),
    }


def stable_rows(results_by_key):
    return sorted(
        results_by_key.values(),
        key=lambda r: (
            VIDEOS.index(r["video"]),
            CODECS.index(r["codec"]),
            RATES.index(int(r["bandwidth_mbps"])),
            METHODS.index(r["method"]),
        ),
    )


FINAL_FIELDS = [
    "video",
    "codec",
    "bandwidth_mbps",
    "bandwidth_mode",
    "method",
    "preset",
    "warmup_frames",
    "measured_frames",
    "psnr_db",
    "ssim",
    "lpips",
    "ws_psnr_db",
    "actual_bitrate_mbps",
    "measured_payload_bytes",
    "measured_payload_mb",
    "mean_frame_size_bytes",
    "complete_file_bytes",
    "complete_file_mb",
    "processing_fps_this_quality_run",
    "source_kind",
    "encoded_path",
    "events_path",
    "allocation_audit",
]


def save_progress(root, results):
    rows = stable_rows(results)
    dump(root / "FINAL_QUALITY_RESULTS_PARTIAL.json", rows)
    write_csv(root / "FINAL_QUALITY_RESULTS_PARTIAL.csv", rows, FINAL_FIELDS)


# ---------------------------------------------------------------------------
# Final 360-SATC allocation
# ---------------------------------------------------------------------------

def normalize_scores(scores):
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.shape != (45,) or not np.isfinite(s).all():
        raise ValueError("Expected exactly 45 finite saliency scores.")
    if (s < -1e-12).any():
        raise ValueError("Saliency scores must be non-negative.")
    total = float(np.sum(s))
    if total <= 1e-12:
        return np.full(45, 1.0 / 45.0, dtype=np.float64)
    return s / total


def final_tile_qp(scores):
    s = normalize_scores(scores)
    order = np.argsort(-s, kind="stable")
    q = np.zeros(45, dtype=np.int8)
    q[order[:6]] = -2
    q[order[6:12]] = -1
    return q, {
        "qp_minus2_tiles": 6,
        "qp_minus1_tiles": 6,
        "qp_zero_tiles": 33,
        "protected_saliency_mass": float(np.sum(s[q != 0])),
    }


class Final360SATCAllocator:
    """Frozen paper-facing 360-SATC allocation used in the final quality run."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.records = []
        self.next_frame = 0

    def make_map(self, scores, raw, codec, frame_id):
        if codec != self.geometry.codec:
            raise ValueError(f"Codec mismatch: {codec} != {self.geometry.codec}")
        if frame_id != self.next_frame:
            raise ValueError(f"Non-contiguous frame ID: {frame_id} != {self.next_frame}")

        q, meta = final_tile_qp(scores)
        expanded = self.geometry.expand(q)
        self.records.append({
            "frame_id": int(frame_id),
            "qp_crc32": int(zlib.crc32(expanded.tobytes())),
            "tile_deltas_hex": q.tobytes().hex(),
            **meta,
        })
        self.next_frame += 1
        return expanded

    def audit(self, frame_events_csv, out_dir, warmup=WARMUP):
        sent = read_csv(frame_events_csv)
        if len(sent) != len(self.records):
            raise RuntimeError(
                f"Allocation audit frame count mismatch: {len(sent)} vs {len(self.records)}"
            )

        for a, b in zip(sent, self.records):
            if int(a["frame_id"]) != b["frame_id"]:
                raise RuntimeError("Allocation audit frame sequence mismatch.")
            if int(a["qp_crc32"]) != b["qp_crc32"]:
                raise RuntimeError("Allocation audit QP CRC mismatch.")
            q = np.frombuffer(
                bytes.fromhex(b["tile_deltas_hex"]), dtype=np.int8
            )
            if q.size != 45:
                raise RuntimeError("Allocation audit tile count mismatch.")
            if not np.isin(q, [-2, -1, 0]).all():
                raise RuntimeError("Unexpected final 360-SATC QP delta.")
            if int(np.sum(q == -2)) != 6:
                raise RuntimeError("Expected exactly six QP -2 tiles.")
            if int(np.sum(q == -1)) != 6:
                raise RuntimeError("Expected exactly six QP -1 tiles.")
            if int(np.sum(q == 0)) != 33:
                raise RuntimeError("Expected exactly thirty-three QP 0 tiles.")

        measured = self.records[warmup:]
        report = {
            "status": "PASS",
            "paper_method_name": "360-SATC",
            "measured_frames": len(measured),
            "qp_minus2_tiles": 6,
            "qp_minus1_tiles": 6,
            "qp_zero_tiles": 33,
            "positive_qp_offsets": False,
            "neighbor_ring": False,
            "current_frame_continuous_saliency": True,
            "mean_protected_saliency_mass": float(
                np.mean([r["protected_saliency_mass"] for r in measured])
            ),
        }
        out = Path(out_dir)
        dump(out / "FINAL_360SATC_ALLOCATION_AUDIT.json", report)
        with (out / "final_360satc_allocation_frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = list(self.records[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.records)
        return report


# ---------------------------------------------------------------------------
# Reuse exact-compatible London baseline rows
# ---------------------------------------------------------------------------

def try_reuse_london_baselines(old_root):
    old_root = Path(old_root)
    reused = {}
    report = {
        "requested_root": str(old_root),
        "eligible_methods": ["CBR", "MUC", "PC", "AUC", "360-ST"],
        "reused": [],
        "rejected": [],
    }

    proto = old_root / "QUALITY_PROTOCOL.json"
    summary = old_root / "QUALITY_SUMMARY.csv"
    if not proto.is_file() or not summary.is_file():
        report["rejected"].append("Old quality protocol/summary not found.")
        return reused, report

    try:
        p = read_json(proto)
        if int(p.get("measured_frames", -1)) != MEASURED:
            raise RuntimeError("measured_frames mismatch")
        if int(p.get("warmup_frames", -1)) != WARMUP:
            raise RuntimeError("warmup_frames mismatch")
        if str(p.get("preset", "")).lower() != "p7":
            raise RuntimeError("preset mismatch")
        if sorted(int(x) for x in p.get("rates_mbps", [])) != [12, 35]:
            raise RuntimeError("rate set mismatch")
    except Exception as e:
        report["rejected"].append(f"Old quality protocol is not compatible: {e}")
        return reused, report

    rows = read_csv(summary)
    label = {
        "cbr": "CBR",
        "muc": "MUC",
        "pc": "PC",
        "auc": "AUC",
        "360st": "360-ST",
    }

    for r in rows:
        if r.get("video") != "london_tower":
            continue
        raw_method = r.get("method", "")
        if raw_method not in label:
            continue

        codec = r["codec"]
        rate = int(r["target_mbps"])
        method = label[raw_method]
        key = result_key("london_tower", codec, rate, method)

        try:
            for metric_field in ("psnr_db", "ssim", "lpips_mean", "ws_psnr_mean_db"):
                if str(r.get(metric_field, "")) in ("", "None", "nan"):
                    raise RuntimeError(f"missing {metric_field}")

            enc = Path(r["encoded"])
            if not enc.is_file():
                raise RuntimeError(f"encoded file missing: {enc}")

            event_candidates = [
                enc.parent / "frame_events.csv",
                enc.parent / "baseline_events.csv",
            ]
            events = next((x for x in event_candidates if x.is_file()), None)
            if events is None:
                raise RuntimeError("frame/baseline event CSV missing")

            b = measured_byte_stats(events)
            row = {
                "video": "london_tower",
                "codec": codec,
                "bandwidth_mbps": rate,
                "bandwidth_mode": "Low" if rate == 12 else "High",
                "method": method,
                "preset": "p7",
                "warmup_frames": WARMUP,
                **b,
                "psnr_db": float(r["psnr_db"]),
                "ssim": float(r["ssim"]),
                "lpips": float(r["lpips_mean"]),
                "ws_psnr_db": float(r["ws_psnr_mean_db"]),
                "complete_file_bytes": int(enc.stat().st_size),
                "complete_file_mb": float(enc.stat().st_size / 1e6),
                "processing_fps_this_quality_run": None,
                "source_kind": "reused_exact_london_p7_baseline",
                "encoded_path": str(enc),
                "events_path": str(events),
                "allocation_audit": None,
            }
            reused[key] = row
            report["reused"].append(key)
        except Exception as e:
            report["rejected"].append(f"{key}: {e}")

    return reused, report


# ---------------------------------------------------------------------------
# SATC runtime: final 360-SATC and zero-QP CBR
# ---------------------------------------------------------------------------

def prepare_satc_runtime(master, base, helpers, root, videos):
    satc_root = root / "satc_p7"
    modified = root / "final_360satc_runner.py"

    if not modified.is_file():
        frozen = Path(helpers["satc"]).read_text(encoding="utf-8")
        modified.write_text(
            base.patched_satc_source(frozen, "p7"),
            encoding="utf-8",
        )

    satc = load_py(modified, "final_quality_satc")
    base.install_guard(satc, "p7")

    encoder = satc_root / "build" / "nvenc_uncapped"
    code = satc_root / "code"

    if not encoder.is_file() or not code.is_dir():
        if satc_root.exists():
            # Preserve only completed run directories if a previous build died.
            saved_runs = None
            runs = satc_root / "runs"
            if runs.is_dir():
                saved_runs = root / "_saved_satc_runs"
                if saved_runs.exists():
                    shutil.rmtree(saved_runs)
                shutil.move(str(runs), str(saved_runs))
            shutil.rmtree(satc_root, ignore_errors=True)

            _, model, _, _ = master.setup_satc_module(
                satc, satc_root, "p7", videos, MEASURED, WARMUP
            )
            if saved_runs is not None and saved_runs.is_dir():
                target = satc_root / "runs"
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(saved_runs), str(target))
            return satc, model, satc_root

        _, model, _, _ = master.setup_satc_module(
            satc, satc_root, "p7", videos, MEASURED, WARMUP
        )
        return satc, model, satc_root

    # Resume with the already-built native bridge.
    sys.path.insert(0, str(code))
    from live_model import LiveModel

    runtime_meta = satc_root / "model_runtime.json"
    if runtime_meta.is_file():
        model_path = Path(read_json(runtime_meta)["model"])
    else:
        model_path = satc.find_model(None)

    model = LiveModel(
        model_path,
        satc_root,
        provider="auto",
        map_backend="compact-cpu",
    )
    return satc, model, satc_root


def satc_run_paths(satc_root, jid, codec):
    run = satc_root / "runs" / jid
    return (
        run,
        run / f"encoded.{EXT[codec]}",
        run / "frame_events.csv",
        run / "summary.json",
    )


def ensure_satc_encode(
    satc,
    base,
    satc_root,
    model,
    source,
    video,
    codec,
    rate,
    method,
):
    jid = f"{video}_{codec}_{rate}_{'360satc' if method == '360-SATC' else 'cbr'}"
    run, enc, events, summary_path = satc_run_paths(satc_root, jid, codec)

    if summary_path.is_file() and enc.is_file() and events.is_file():
        try:
            s = read_json(summary_path)
            if s.get("status") == "VALID":
                return s, run, enc, events, None, "reused_completed_encode"
        except Exception:
            pass

    if run.exists():
        shutil.rmtree(run)

    model.reset()
    allocator = None
    if method == "360-SATC":
        allocator = Final360SATCAllocator(base.Geometry(codec=codec))
        satc.ALLOCATION_POLICY = allocator
        native_method = "roi"
    elif method == "CBR":
        satc.ALLOCATION_POLICY = None
        native_method = "uniform"
    else:
        raise ValueError(method)

    job = {
        "id": jid,
        "video_name": video,
        "video": str(source),
        "codec": codec,
        "target_mbps": rate,
        "method": native_method,
    }
    r = satc.run_one(
        SimpleNamespace(warmup_frames=WARMUP, frames=MEASURED),
        satc_root,
        job,
        model,
    )
    if r.get("status") != "VALID":
        raise RuntimeError(
            f"SATC/CBR encode failed: {video}/{codec}/{rate}/{method}: "
            f"{r.get('error')}"
        )

    audit = None
    if allocator is not None:
        audit = allocator.audit(events, run, WARMUP)

    return r, run, enc, events, audit, "fresh_encode"


# ---------------------------------------------------------------------------
# Caruso-family P7 runtime
# ---------------------------------------------------------------------------

def prepare_caruso_runtime(master, car, root, args):
    car_root = root / "caruso_p7"
    code = car_root / "code"
    encoder = car_root / "build" / "nvenc_dynamic"

    car.WARMUP = WARMUP
    car.MEASURED = MEASURED
    car.CODECS = CODECS
    car.RATES = RATES
    car.METHODS = CARUSO_METHODS

    if not code.is_dir():
        car_root.mkdir(parents=True, exist_ok=True)
        code.mkdir(parents=True, exist_ok=False)
        car.extract_payload(code)
        master.patch_preset(code / "nvenc_dynamic.cpp", "p7")
        master.patch_caruso_bitstream_producer(code / "baseline_producer.py")

    carg = SimpleNamespace(
        london_video=args.london_video,
        shared_mat=args.shared_mat,
        trace_index=args.trace_index,
    )
    videos, trace, _ = car.preflight(
        car_root,
        master.real_user_home()[1],
        carg,
    )

    pp = car_root / "protocol.json"
    if pp.is_file():
        p = read_json(pp)
        p.setdefault("encoder", {})["preset"] = "p7"
        p["quality_override"] = {
            "warmup_frames": WARMUP,
            "measured_frames": MEASURED,
            "purpose": "final fixed-rate per-video reconstruction quality",
        }
        dump(pp, p)

    if not encoder.is_file():
        encoder = car.build_encoder(car_root, code)

    return car_root, code, encoder, videos, trace


def ensure_caruso_encode(
    car,
    car_root,
    code,
    encoder,
    videos,
    trace,
    video,
    codec,
    rate,
    raw_method,
):
    job_root = car_root / "runs" / f"{video}_{codec}_{rate}Mbps_{raw_method}"

    # Reuse a completed attempt if present.
    for attempt in (1, 2):
        run = job_root / f"attempt_{attempt}"
        result = run / "result.json"
        enc = run / f"encoded.{EXT[codec]}"
        events = run / "baseline_events.csv"
        if result.is_file() and enc.is_file() and events.is_file():
            try:
                r = read_json(result)
                if r.get("status") == "VALID":
                    return r, run, enc, events, "reused_completed_encode"
            except Exception:
                pass

    if job_root.exists():
        shutil.rmtree(job_root)

    r = car.run_one(
        car_root,
        code,
        encoder,
        videos,
        trace,
        video,
        codec,
        rate,
        raw_method,
        1,
    )
    if r.get("status") != "VALID":
        time.sleep(2)
        r = car.run_one(
            car_root,
            code,
            encoder,
            videos,
            trace,
            video,
            codec,
            rate,
            raw_method,
            2,
        )
    if r.get("status") != "VALID":
        raise RuntimeError(
            f"Baseline encode failed: {video}/{codec}/{rate}/{raw_method}"
        )

    attempt = int(r.get("attempt", 1))
    run = job_root / f"attempt_{attempt}"
    enc = run / f"encoded.{EXT[codec]}"
    events = run / "baseline_events.csv"
    return r, run, enc, events, "fresh_encode"


# ---------------------------------------------------------------------------
# Full-reference metrics and row creation
# ---------------------------------------------------------------------------

def metric_result_path(root, video, codec, rate, method):
    safe = method.lower().replace("-", "").replace(" ", "_")
    return root / "metrics" / f"{video}_{codec}_{rate}_{safe}" / "ROW.json"


def compute_or_load_row(
    master,
    master_path,
    root,
    source,
    video,
    codec,
    rate,
    method,
    run_summary,
    enc,
    events,
    source_kind,
    allocation_audit=None,
):
    row_path = metric_result_path(root, video, codec, rate, method)
    if row_path.is_file():
        try:
            row = read_json(row_path)
            required = (
                "psnr_db", "ssim", "lpips", "ws_psnr_db",
                "measured_payload_bytes", "complete_file_bytes",
            )
            if all(row.get(k) is not None for k in required):
                return row
        except Exception:
            pass

    enc = Path(enc)
    events = Path(events)
    if not enc.is_file() or not events.is_file():
        raise FileNotFoundError(f"Missing encoded/events: {enc} / {events}")

    out = row_path.parent
    out.mkdir(parents=True, exist_ok=True)

    metrics = master.metric_pair(
        Path(master_path).resolve(),
        Path(source),
        enc,
        WARMUP,
        MEASURED,
        out,
        True,
    )
    if any(metrics.get(k) is None for k in ("psnr_db", "ssim", "lpips_mean", "ws_psnr_mean_db")):
        raise RuntimeError(
            f"Incomplete metrics for {video}/{codec}/{rate}/{method}: {metrics}"
        )

    b = measured_byte_stats(events)
    processing_fps = None
    if run_summary:
        measurement = run_summary.get("measurement") or {}
        processing_fps = measurement.get("fps", run_summary.get("fps"))
        if processing_fps is not None:
            processing_fps = float(processing_fps)

    row = {
        "video": video,
        "codec": codec,
        "bandwidth_mbps": int(rate),
        "bandwidth_mode": "Low" if int(rate) == 12 else "High",
        "method": method,
        "preset": "p7",
        "warmup_frames": WARMUP,
        **b,
        "psnr_db": float(metrics["psnr_db"]),
        "ssim": float(metrics["ssim"]),
        "lpips": float(metrics["lpips_mean"]),
        "ws_psnr_db": float(metrics["ws_psnr_mean_db"]),
        "complete_file_bytes": int(enc.stat().st_size),
        "complete_file_mb": float(enc.stat().st_size / 1e6),
        "processing_fps_this_quality_run": processing_fps,
        "source_kind": source_kind,
        "encoded_path": str(enc),
        "events_path": str(events),
        "allocation_audit": allocation_audit,
    }
    dump(row_path, row)
    return row


# ---------------------------------------------------------------------------
# Results / summary / package
# ---------------------------------------------------------------------------

def build_summary(rows):
    summary = {
        "planned_rows": 144,
        "completed_rows": len(rows),
        "matrix_complete": len(rows) == 144,
        "by_method": {},
        "by_codec": {},
        "by_bandwidth": {},
    }
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method]
        summary["by_method"][method] = {
            "n": len(rs),
            "mean_psnr_db": float(np.mean([r["psnr_db"] for r in rs])) if rs else None,
            "mean_ssim": float(np.mean([r["ssim"] for r in rs])) if rs else None,
            "mean_lpips": float(np.mean([r["lpips"] for r in rs])) if rs else None,
            "mean_ws_psnr_db": float(np.mean([r["ws_psnr_db"] for r in rs])) if rs else None,
            "mean_measured_payload_mb": float(np.mean([r["measured_payload_mb"] for r in rs])) if rs else None,
        }
    for codec in CODECS:
        summary["by_codec"][codec] = sum(r["codec"] == codec for r in rows)
    for rate in RATES:
        summary["by_bandwidth"][str(rate)] = sum(
            int(r["bandwidth_mbps"]) == rate for r in rows
        )
    return summary


def package_results(root):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{root.name}_RESULTS.zip"
    temp = dest.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()

    skip_parts = {
        "build", "trt_cache", "__pycache__", "map_audit",
    }
    skip_ext = {".h264", ".hevc", ".ivf", ".f32", ".npy"}

    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part in skip_parts for part in rel.parts):
                continue
            if p.suffix.lower() in skip_ext:
                continue
            if p.stat().st_size > 5_000_000:
                continue
            z.write(p, str(rel))

    os.replace(temp, dest)
    return dest


# ---------------------------------------------------------------------------
# Self-test / plan
# ---------------------------------------------------------------------------

def self_test(args):
    master = load_py(args.master, "final_quality_master_self")
    base = load_py(args.base_script, "final_quality_base_self")

    for name in (
        "extract_helpers", "setup_satc_module", "metric_pair",
        "patch_preset", "patch_caruso_bitstream_producer",
        "video_paths", "real_user_home",
    ):
        if not hasattr(master, name):
            raise RuntimeError(f"Master missing required function: {name}")

    helpers_dir = Path("/tmp/360satc_final_quality_selftest")
    shutil.rmtree(helpers_dir, ignore_errors=True)
    helpers_dir.mkdir(parents=True)
    helpers = master.extract_helpers(helpers_dir)

    frozen = Path(helpers["satc"]).read_text(encoding="utf-8")
    patched = base.patched_satc_source(frozen, "p7")
    compile(patched, "patched_satc", "exec")

    for codec in CODECS:
        g = base.Geometry(codec=codec)
        q, meta = final_tile_qp(np.linspace(1, 45, 45, dtype=np.float64))
        expanded = g.expand(q)
        assert q.shape == (45,)
        assert int(np.sum(q == -2)) == 6
        assert int(np.sum(q == -1)) == 6
        assert int(np.sum(q == 0)) == 33
        assert expanded.dtype == np.int8
        assert set(np.unique(expanded).tolist()).issubset({-2, -1, 0})

    planned = {
        "status": "PASS",
        "version": VERSION,
        "final_rows": 144,
        "videos": list(VIDEOS),
        "codecs": list(CODECS),
        "bandwidths_mbps": list(RATES),
        "methods": list(METHODS),
        "warmup_frames": WARMUP,
        "measured_frames": MEASURED,
        "metrics": ["PSNR", "SSIM", "LPIPS AlexNet v0.1", "WS-PSNR", "payload/file size"],
        "final_360satc_allocation": {
            "qp_minus2_tiles": 6,
            "qp_minus1_tiles": 6,
            "qp_zero_tiles": 33,
            "positive_qp_offsets": False,
            "neighbor_ring": False,
        },
        "default_old_london_reuse_root": str(args.reuse_quality_root),
        "note": "Self-test does not start GPU encoding.",
    }
    print(json.dumps(planned, indent=2))
    return 0


def show_plan(args):
    reusable = 0
    if not args.no_reuse_london:
        reused, _ = try_reuse_london_baselines(args.reuse_quality_root)
        reusable = len(reused)

    print(json.dumps({
        "final_matrix_rows": 144,
        "reusable_exact_london_baseline_rows_detected": reusable,
        "fresh_rows_if_started_now": 144 - reusable,
        "fresh_360SATC_rows": 24,
        "fresh_CBR_rows": 24 - sum(
            1 for k in (try_reuse_london_baselines(args.reuse_quality_root)[0] if not args.no_reuse_london else {})
            if k.endswith("|CBR")
        ),
        "note": "Only exact-compatible P7/240+896 London baseline rows are reused. Old 360-SATC is never reused.",
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Main final experiment
# ---------------------------------------------------------------------------

def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError(
            "Need at least 12 GiB free disk space before the final quality run."
        )

    master = load_py(args.master, "final_quality_master")
    base = load_py(args.base_script, "final_quality_base")
    videos = master.video_paths(args.london_video)

    # Save a copy of the exact runner used.
    try:
        shutil.copy2(Path(__file__).resolve(), root / Path(__file__).name)
    except Exception:
        pass

    helpers_dir = root / "embedded_helpers"
    if not helpers_dir.is_dir():
        helpers = master.extract_helpers(helpers_dir)
    else:
        helpers = {
            "satc": helpers_dir / "SATC_Uncapped_P4_v1.py",
            "cbr": helpers_dir / "SATC_CBR_FullWorkload_P4_v1.py",
            "caruso": helpers_dir / "RUN_CARUSO_BASELINES_ALL_IN_ONE_v3.py",
            "v14": helpers_dir / "SATC_Integrated_Real5G_60FPS_v14.py",
        }

    protocol = {
        "version": VERSION,
        "purpose": "final per-video fixed-rate full-reference quality matrix",
        "resolution": [WIDTH, HEIGHT],
        "media_fps": FPS,
        "videos": {k: str(v) for k, v in videos.items()},
        "codecs": list(CODECS),
        "bandwidths_mbps": list(RATES),
        "bandwidth_labels": {"12": "Low", "35": "High"},
        "methods": list(METHODS),
        "encoder": {
            "api": "NVIDIA NVENC",
            "preset": "p7",
            "rate_control": "CBR",
            "gop": 240,
            "b_frames": 0,
        },
        "warmup_frames": WARMUP,
        "measured_frames": MEASURED,
        "metrics": [
            "PSNR",
            "SSIM",
            "LPIPS AlexNet v0.1",
            "WS-PSNR",
            "actual measured bitrate",
            "measured payload",
            "mean measured frame size",
            "complete encoded file size",
        ],
        "final_360satc_policy": {
            "paper_method_name": "360-SATC",
            "tile_grid": [5, 9],
            "qp_minus2_tiles": 6,
            "qp_minus1_tiles": 6,
            "qp_zero_tiles": 33,
            "positive_qp_offsets": False,
            "neighbor_ring": False,
            "score_source": "continuous MoST-Sal spatial scores after existing ERP/EMA processing",
        },
        "baseline_note": (
            "360-ST is the adapted/reproduced viewport-weighted four-zone policy "
            "used in the controlled NVENC comparison, not the unavailable original trained checkpoint."
        ),
        "master_script": str(args.master),
        "master_sha256": sha256(args.master),
        "base_script": str(args.base_script),
        "base_sha256": sha256(args.base_script),
        "state": "RUNNING",
    }

    existing_proto = root / "FINAL_QUALITY_PROTOCOL.json"
    if existing_proto.is_file():
        oldp = read_json(existing_proto)
        immutable = (
            "resolution", "media_fps", "codecs", "bandwidths_mbps",
            "methods", "warmup_frames", "measured_frames",
            "final_360satc_policy",
        )
        for k in immutable:
            if oldp.get(k) != protocol.get(k):
                raise RuntimeError(
                    f"Refusing to resume incompatible experiment: protocol field {k} changed."
                )
    dump(existing_proto, protocol)

    # Load checkpointed rows.
    results = {}
    partial = root / "FINAL_QUALITY_RESULTS_PARTIAL.json"
    if partial.is_file():
        try:
            for row in read_json(partial):
                key = result_key(
                    row["video"], row["codec"], row["bandwidth_mbps"], row["method"]
                )
                results[key] = row
        except Exception:
            results = {}

    # Exact-compatible old London baseline reuse.
    reuse_report = {
        "disabled": bool(args.no_reuse_london),
        "reused": [],
        "rejected": [],
    }
    if not args.no_reuse_london:
        reusable, reuse_report = try_reuse_london_baselines(args.reuse_quality_root)
        for key, row in reusable.items():
            if key not in results:
                results[key] = row
    dump(root / "REUSE_REPORT.json", reuse_report)
    save_progress(root, results)

    missing_now = sorted(expected_keys() - set(results))
    dump(root / "RUN_PLAN.json", {
        "final_rows": 144,
        "already_completed_or_reused": len(results),
        "remaining_at_start": len(missing_now),
        "remaining_keys": missing_now,
    })

    print("\n============================================================")
    print("FINAL 360-SATC QUALITY MATRIX")
    print("============================================================")
    print(f"Completed/reused at start: {len(results)} / 144")
    print(f"Remaining:                 {144 - len(results)}")
    print("This runner does not execute Mininet, WebRTC, GCC, RT-MPC, or RTT tests.")
    print("============================================================\n", flush=True)

    # -------------------- 360-SATC + CBR --------------------
    satc, model, satc_root = prepare_satc_runtime(
        master, base, helpers, root, videos
    )

    satc_plan = []
    for v in VIDEOS:
        for c in CODECS:
            for rate in RATES:
                satc_plan.append((v, c, rate, "360-SATC"))
                satc_plan.append((v, c, rate, "CBR"))

    ordinal = 0
    total_fresh_plan = sum(
        result_key(v, c, rate, method) not in results
        for v, c, rate, method in satc_plan
    )

    for v, c, rate, method in satc_plan:
        key = result_key(v, c, rate, method)
        if key in results:
            continue
        ordinal += 1
        print(
            f"[SATC/CBR {ordinal}/{total_fresh_plan}] "
            f"{v} {c} {rate} Mbps {method}",
            flush=True,
        )

        r, run_dir, enc, events, audit, source_kind = ensure_satc_encode(
            satc, base, satc_root, model, videos[v], v, c, rate, method
        )
        row = compute_or_load_row(
            master,
            args.master,
            root,
            videos[v],
            v,
            c,
            rate,
            method,
            r,
            enc,
            events,
            source_kind,
            allocation_audit=audit,
        )
        results[key] = row
        save_progress(root, results)
        print(
            f"    PSNR={row['psnr_db']:.4f} | "
            f"SSIM={row['ssim']:.6f} | "
            f"LPIPS={row['lpips']:.6f} | "
            f"WS-PSNR={row['ws_psnr_db']:.4f} | "
            f"payload={row['measured_payload_mb']:.3f} MB",
            flush=True,
        )

    # -------------------- MUC / PC / AUC / 360-ST --------------------
    car = master.load_module(helpers["caruso"], "final_quality_caruso")
    car_root, code, encoder, car_videos, trace = prepare_caruso_runtime(
        master, car, root, args
    )

    car_plan = [
        (v, c, rate, raw)
        for v in VIDEOS
        for c in CODECS
        for rate in RATES
        for raw in CARUSO_METHODS
    ]
    car_remaining = sum(
        result_key(v, c, rate, CARUSO_LABEL[raw]) not in results
        for v, c, rate, raw in car_plan
    )
    ordinal = 0

    for v, c, rate, raw_method in car_plan:
        method = CARUSO_LABEL[raw_method]
        key = result_key(v, c, rate, method)
        if key in results:
            continue
        ordinal += 1
        print(
            f"[BASELINE {ordinal}/{car_remaining}] "
            f"{v} {c} {rate} Mbps {method}",
            flush=True,
        )

        r, run_dir, enc, events, source_kind = ensure_caruso_encode(
            car,
            car_root,
            code,
            encoder,
            car_videos,
            trace,
            v,
            c,
            rate,
            raw_method,
        )
        row = compute_or_load_row(
            master,
            args.master,
            root,
            videos[v],
            v,
            c,
            rate,
            method,
            r,
            enc,
            events,
            source_kind,
            allocation_audit=None,
        )
        results[key] = row
        save_progress(root, results)
        print(
            f"    PSNR={row['psnr_db']:.4f} | "
            f"SSIM={row['ssim']:.6f} | "
            f"LPIPS={row['lpips']:.6f} | "
            f"WS-PSNR={row['ws_psnr_db']:.4f} | "
            f"payload={row['measured_payload_mb']:.3f} MB",
            flush=True,
        )

    # -------------------- Final matrix audit --------------------
    missing = sorted(expected_keys() - set(results))
    extra = sorted(set(results) - expected_keys())
    if missing or extra:
        dump(root / "FINAL_MATRIX_ERROR.json", {
            "missing": missing,
            "extra": extra,
        })
        raise RuntimeError(
            f"Final matrix incomplete: missing={len(missing)}, extra={len(extra)}"
        )

    rows = stable_rows(results)

    # Strict final audit.
    for row in rows:
        for k in ("psnr_db", "ssim", "lpips", "ws_psnr_db"):
            if row.get(k) is None or not np.isfinite(float(row[k])):
                raise RuntimeError(f"Non-finite {k}: {row}")
        if int(row["measured_frames"]) != MEASURED:
            raise RuntimeError(f"Measured frame mismatch: {row}")
        if int(row["bandwidth_mbps"]) not in RATES:
            raise RuntimeError(f"Unexpected bandwidth: {row}")
        if row["method"] not in METHODS:
            raise RuntimeError(f"Unexpected method: {row}")

    dump(root / "FINAL_QUALITY_RESULTS.json", rows)
    write_csv(root / "FINAL_QUALITY_RESULTS.csv", rows, FINAL_FIELDS)

    heatmap_fields = [
        "video", "codec", "bandwidth_mbps", "bandwidth_mode", "method",
        "psnr_db", "ssim", "lpips", "ws_psnr_db",
        "actual_bitrate_mbps", "measured_payload_mb",
        "mean_frame_size_bytes", "complete_file_mb", "source_kind",
    ]
    write_csv(root / "HEATMAP_READY.csv", rows, heatmap_fields)

    summary = build_summary(rows)
    dump(root / "FINAL_QUALITY_SUMMARY.json", summary)

    protocol["state"] = "COMPLETE"
    protocol["completed_rows"] = 144
    dump(root / "FINAL_QUALITY_PROTOCOL.json", protocol)

    upload = package_results(root)

    print("\n============================================================")
    print("FINAL QUALITY EXPERIMENT COMPLETE")
    print("============================================================")
    print("Final matrix: 144 / 144 rows")
    print(f"CSV:    {root / 'FINAL_QUALITY_RESULTS.csv'}")
    print(f"Heatmap:{root / 'HEATMAP_READY.csv'}")
    print(f"Upload: {upload}")
    print("============================================================\n")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Final 360-SATC per-video fixed-rate full-reference quality experiment."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")

    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--base-script", type=Path, default=DEFAULT_BASE)
    p.add_argument("--london-video", type=Path, default=DEFAULT_LONDON)
    p.add_argument("--shared-mat", type=Path, default=None)
    p.add_argument("--trace-index", type=int, default=0)
    p.add_argument("--reuse-quality-root", type=Path, default=DEFAULT_OLD_QUALITY)
    p.add_argument("--no-reuse-london", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = p.parse_args()

    if args.self_test:
        return self_test(args)
    if args.plan:
        return show_plan(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
