#!/usr/bin/env python3
"""Rerun the Figure 4 uniform CBR baseline without saliency processing.

Requires the existing prepared videos, FFmpeg with CUDA decoding, CUDA Toolkit,
Video Codec SDK 13, CMake, and the NVIDIA driver. Uses only Python's standard
library. No model, ONNX Runtime, network emulation, or sudo is required.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import datetime
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile
import zlib

VERSION = "CBR-EncoderOnly-P4-v1"
SOURCE_COMMIT = "17459a99c00c677e76d1e088dac8b703d9716ac6"
RETAINED_SHA256 = "0b3cada79161cb086224e8625854c4bf44cd2de8450e24dd64a658063ec76f50"
RETAINED_PATH = "experiments/helpers/SATC_CBR_FullWorkload_P4_v1.py"
RETAINED_URL = f"https://raw.githubusercontent.com/publioelon/360-SATC/{SOURCE_COMMIT}/{RETAINED_PATH}"
LABELS = ("basketball", "rollercoaster", "ballet", "london")
CODECS = ("h264", "hevc", "av1")
RATES = (12, 35)
NATIVE_FILES = ("CMakeLists.txt", "nvenc_dynamic.cpp", "shared_frame.py", "shared_input.h")


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def materialize(code, retained=None):
    """Reuse the exact historical native bridge and raw-frame source.

    Extract source text without importing or executing the historical runner.
    The new worker consumes Source directly; its saliency stages are absent.
    """
    candidate = Path(retained) if retained else Path(__file__).resolve().parent / "helpers" / Path(RETAINED_PATH).name
    if candidate.is_file():
        raw = candidate.read_bytes()
    elif retained:
        raise ValueError(f"Retained source not found: {candidate}")
    else:
        with urllib.request.urlopen(RETAINED_URL, timeout=45) as response:
            raw = response.read(200_001)
    if hashlib.sha256(raw).hexdigest() != RETAINED_SHA256:
        raise ValueError("Retained source checksum mismatch; refusing to change the encoder or source silently")
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    payload = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_PAYLOAD_B85" for t in node.targets))
    with zipfile.ZipFile(io.BytesIO(zlib.decompress(base64.b85decode(payload)))) as archive:
        for name in NATIVE_FILES:
            (code / name).write_bytes(archive.read(name))
        producer = archive.read("producer.py").decode("utf-8")
    selected = []
    constants = {"W", "H", "FPS", "FRAME_BYTES", "PERIOD_NS", "QUEUE_FRAMES", "NATIVE_IN", "NATIVE_OUT"}
    for node in ast.parse(producer).body:
        if isinstance(node, ast.Import) and all(alias.name != "numpy" for alias in node.names):
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module in {"__future__", "pathlib", "shared_frame"}:
            selected.append(node)
        elif isinstance(node, ast.Assign) and {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)} <= constants:
            selected.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {"Source", "read_exact", "write_all"}:
            selected.append(node)
    support = "\n\n".join(ast.get_source_segment(producer, n) for n in selected) + "\n"
    compile(support, "cbr_support.py", "exec")
    (code / "cbr_support.py").write_text(support, encoding="utf-8")
    (code / "historical_cbr_source.py").write_bytes(raw)
    shutil.copy2(Path(__file__).resolve(), code / "cbr_runner.py")
    dump(code / "SHA256SUMS.json", {p.name: digest(p) for p in sorted(code.iterdir()) if p.is_file()})


def worker_main(argv=None):
    # cbr_support contains only the retained Source, framing, and raw I/O code.
    from cbr_support import Source, read_exact, write_all, FRAME_BYTES, NATIVE_IN, NATIVE_OUT, W, H
    from shared_frame import SharedFrame
    import queue

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--codec", choices=CODECS, required=True)
    parser.add_argument("--bitrate", type=int, choices=RATES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=240)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.warmup < 1:
        parser.error("Frame counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    total = args.warmup + args.frames
    bitrate = args.bitrate * 1_000_000
    block = {"h264": 16, "hevc": 32, "av1": 64}[args.codec]
    # A single constant buffer preserves the historical NVENC interface. There
    # is no per-frame spatial allocation, saliency map, or model preprocessing.
    qp = bytes(((W + block - 1) // block) * ((H + block - 1) // block))
    qp_crc = zlib.crc32(qp)
    fields = ["frame_id", "source_read_start_unix_ns", "source_available_unix_ns",
              "encode_submit_unix_ns", "encode_begin_mono_ns", "encode_end_mono_ns",
              "completion_unix_ns", "completion_mono_ns", "target_bitrate_bps",
              "applied_bitrate_bps", "encoded_bytes", "qp_bytes", "qp_crc32"]
    stop, gate = threading.Event(), threading.Event()
    source = shared = encoder = None
    start = end = None
    with (args.output / "producer_events.csv").open("w", newline="", buffering=1) as events, \
            (args.output / "nvenc_dynamic.log").open("w", buffering=1) as encoder_log:
        writer = csv.DictWriter(events, fieldnames=fields)
        writer.writeheader()
        try:
            shared = SharedFrame(FRAME_BYTES)
            encoder = subprocess.Popen([str(args.encoder), args.codec, str(bitrate), str(shared.fd)],
                                       pass_fds=(shared.fd,), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=encoder_log, bufsize=0)
            if read_exact(encoder.stdout, 4) != b"RDY1":
                raise RuntimeError("NVENC bridge not ready")
            source = Source(args.video, False, stop, gate, args.warmup, total)
            for expected in range(total):
                deadline = time.monotonic() + 60
                while True:
                    if source.error:
                        raise RuntimeError(source.error)
                    if encoder.poll() is not None:
                        raise RuntimeError("Encoder exited before all frames completed")
                    try:
                        fid, raw, _, read_start, available = source.q.get(timeout=.2)
                        break
                    except queue.Empty:
                        if source.done.is_set() or time.monotonic() > deadline:
                            raise RuntimeError("Source ended or stalled before all frames completed")
                if fid != expected:
                    raise RuntimeError("Noncontiguous input frame IDs")
                submit = time.time_ns()
                shared.copy_from(raw)
                write_all(encoder.stdin, NATIVE_IN.pack(b"FRM2", fid, len(raw), len(qp), bitrate))
                write_all(encoder.stdin, qp)
                magic, echoed, begin, finish, size, applied = NATIVE_OUT.unpack(read_exact(encoder.stdout, NATIVE_OUT.size))
                if magic != b"AU02" or echoed != fid or finish < begin or applied != bitrate or not 0 < size < 20_000_000:
                    raise RuntimeError("Invalid completed NVENC frame record")
                read_exact(encoder.stdout, size)
                complete = time.monotonic_ns()
                writer.writerow({"frame_id": fid, "source_read_start_unix_ns": read_start,
                                 "source_available_unix_ns": available, "encode_submit_unix_ns": submit,
                                 "encode_begin_mono_ns": begin, "encode_end_mono_ns": finish,
                                 "completion_unix_ns": time.time_ns(), "completion_mono_ns": complete,
                                 "target_bitrate_bps": bitrate, "applied_bitrate_bps": applied,
                                 "encoded_bytes": size, "qp_bytes": len(qp), "qp_crc32": qp_crc})
                if fid == args.warmup - 1:
                    start = time.monotonic_ns()
                    gate.set()
                if fid == total - 1:
                    end = time.monotonic_ns()
            encoder.stdin.close()
            encoder.wait(timeout=30)
            if encoder.returncode:
                raise RuntimeError("Encoder failed at final flush")
            if source.error:
                raise RuntimeError(source.error)
        finally:
            stop.set()
            gate.set()
            try:
                if source:
                    source.close()
                    source.proc.stdout.close()
            finally:
                try:
                    if encoder:
                        if encoder.stdin and not encoder.stdin.closed:
                            encoder.stdin.close()
                        if encoder.poll() is None:
                            encoder.terminate()
                            try:
                                encoder.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                encoder.kill()
                                encoder.wait(timeout=5)
                        encoder.stdout.close()
                finally:
                    if shared:
                        shared.close()
    if start is None or end is None or end <= start:
        raise RuntimeError("Incomplete benchmark")
    elapsed = (end - start) / 1e9
    result = {"status": "COMPLETE_PENDING_AUDIT", "codec": args.codec,
              "target_mbps": args.bitrate, "measured_frames": args.frames,
              "warmup_frames": args.warmup, "start_mono_ns": start, "end_mono_ns": end,
              "elapsed_seconds": elapsed, "fps": args.frames / elapsed,
              "measured_inference_calls": 0, "spatial_allocation": "none; constant zero QP deltas",
              "timing_start": "after final warmup AU and its event row, before opening the source gate",
              "timing_end": "after final measured AU and its event row"}
    dump(args.output / "benchmark.json", result)
    print(json.dumps(result), flush=True)
    return 0


def audit_events(path, benchmark):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    warm, frames = benchmark["warmup_frames"], benchmark["measured_frames"]
    if [int(r["frame_id"]) for r in rows] != list(range(warm + frames)):
        raise ValueError("Completed frame coverage mismatch")
    block = {"h264": 16, "hevc": 32, "av1": 64}[benchmark["codec"]]
    size = ((4096 + block - 1) // block) * ((2048 + block - 1) // block)
    expected_crc = zlib.crc32(bytes(size))
    rate = benchmark["target_mbps"] * 1_000_000
    for r in rows:
        if (int(r["qp_bytes"]) != size or int(r["qp_crc32"]) != expected_crc
                or int(r["target_bitrate_bps"]) != rate or int(r["applied_bitrate_bps"]) != rate
                or int(r["encoded_bytes"]) <= 0
                or int(r["encode_end_mono_ns"]) < int(r["encode_begin_mono_ns"])):
            raise ValueError(f"Uniform CBR audit failed at frame {r['frame_id']}")
    start, end = benchmark["start_mono_ns"], benchmark["end_mono_ns"]
    measured = rows[warm:]
    times = [int(r["completion_mono_ns"]) for r in measured]
    if not (start < times[0] <= times[-1] <= end) or times != sorted(set(times)):
        raise ValueError("Invalid measured completion timestamps")
    if int(rows[warm - 1]["completion_mono_ns"]) > start:
        raise ValueError("Warmup output overlaps measurement")
    elapsed = (end - start) / 1e9
    if not math.isclose(benchmark["elapsed_seconds"], elapsed, rel_tol=1e-12) or not math.isclose(benchmark["fps"], frames / elapsed, rel_tol=1e-12):
        raise ValueError("Reported throughput disagrees with raw timing")
    return {"status": "PASS", "completed_frames": len(rows), "measured_frames": len(measured),
            "zero_qp_frames": len(rows), "missing_or_dropped_frames": 0}, measured


def write_windows(path, benchmark, measured):
    start, end = benchmark["start_mono_ns"], benchmark["end_mono_ns"]
    count = max(1, (end - start + 999_999_999) // 1_000_000_000)
    buckets = [0] * count
    for row in measured:
        index = min((int(row["completion_mono_ns"]) - start) // 1_000_000_000, count - 1)
        buckets[index] += 1
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["window", "start_mono_ns", "end_mono_ns", "duration_seconds", "completed_frames", "fps", "full_second"])
        writer.writeheader()
        for i, n in enumerate(buckets):
            begin, finish = start + i * 1_000_000_000, min(start + (i + 1) * 1_000_000_000, end)
            duration = (finish - begin) / 1e9
            writer.writerow({"window": i, "start_mono_ns": begin, "end_mono_ns": finish,
                             "duration_seconds": duration, "completed_frames": n,
                             "fps": n / duration, "full_second": finish - begin == 1_000_000_000})


def validate_video(path, total):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True, timeout=60)
    stream = json.loads(result.stdout)["streams"][0]
    if (stream.get("width"), stream.get("height"), stream.get("pix_fmt")) != (4096, 2048, "yuv420p"):
        raise ValueError(f"{path}: requires the prepared 4096x2048 8 bit YUV420 video")
    if Fraction(stream.get("avg_frame_rate", "0")) != 60 or Fraction(stream.get("r_frame_rate", "0")) != 60:
        raise ValueError(f"{path}: expected a constant 60 FPS prepared input")
    if stream.get("color_space") != "bt709" or stream.get("color_range") != "tv":
        raise ValueError(f"{path}: expected BT.709 limited range")
    if not str(stream.get("nb_frames", "")).isdigit() or int(stream["nb_frames"]) < total + 64:
        raise ValueError(f"{path}: needs at least {total + 64} input frames, with no loop inside the measured batch")
    return stream


def discover(roots):
    matches = {name: [] for name in LABELS}
    skipped = {".git", ".cache", "node_modules", "venvs", "venv", "__pycache__", "build"}
    seen = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        for folder, dirs, files in os.walk(root):
            rel = Path(folder).relative_to(root)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in skipped and len(rel.parts) < 7)
            for filename in sorted(files):
                if not filename.lower().endswith(".mp4"):
                    continue
                label = next((name for name in LABELS if name in filename.lower()), None)
                if label is None and "tower" in filename.lower() and "bridge" in filename.lower():
                    label = "london"
                if label is None:
                    continue
                path = (Path(folder) / filename).resolve()
                if path in seen:
                    continue
                seen.add(path)
                if label != "london" and filename != f"{label}_4096x2048_60fps.mp4":
                    continue
                if label == "london" and not any(x in filename.lower() for x in ("60fps", "4096x2048")):
                    continue
                matches[label].append(path)
    return matches


def choose_inputs(args):
    videos = {}
    config = {}
    if args.config:
        config = json.loads(args.config.expanduser().read_text())
    prepared = args.prepared_dir or config.get("prepared_dir") or os.environ.get("SATC_PREPARED")
    london = args.london_video or config.get("london_video") or os.environ.get("SATC_LONDON")
    if prepared:
        for label in LABELS[:3]:
            videos[label] = Path(prepared).expanduser() / f"{label}_4096x2048_60fps.mp4"
    if london:
        videos["london"] = Path(london).expanduser()
    if args.video:
        videos = {}
        for value in args.video:
            label, separator, path = value.partition("=")
            if not separator or label not in LABELS or label in videos or not path:
                raise ValueError("Use --video basketball=/path.mp4 (or rollercoaster, ballet, london), with each label once")
            videos[label] = Path(path).expanduser()
    else:
        roots = args.search_root or [Path.home() / "Downloads", Path.home() / "snap/firefox/common", Path.cwd(), Path("/data")]
        candidates = discover(roots) if args.discover or set(videos) != set(LABELS) else {name: [] for name in LABELS}
        if args.discover:
            print(json.dumps({k: [str(p) for p in v] for k, v in candidates.items()}, indent=2))
            return {}, config
        for label in LABELS:
            if label not in videos:
                if len(candidates[label]) != 1:
                    raise ValueError(f"{label}: found {len(candidates[label])} candidates. Run --discover, then supply exact paths using --prepared-dir and --london-video or repeated --video LABEL=PATH")
                videos[label] = candidates[label][0]
    videos = {name: path.resolve() for name, path in videos.items()}
    if not videos or len(set(videos.values())) != len(videos):
        raise ValueError("Supply distinct prepared video files")
    for path in videos.values():
        if not path.is_file():
            raise ValueError(f"Prepared video not found: {path}")
    return videos, config


def command(args, log, timeout=600, env=None):
    """Keep the exact invocation and partial output even on interruption."""
    with Path(log).open("w") as stream:
        stream.write(json.dumps([str(a) for a in args]) + "\n")
        stream.flush()
        process = subprocess.Popen([str(a) for a in args], stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            # Stop only the process group created for this command, including
            # its FFmpeg/NVENC children. Keep completed run directories intact.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise
        # A failed worker must not leave its decoder or encoder running during
        # the next condition. The session leader has exited, so any surviving
        # member of this command's group is a descendant that needs cleanup.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return result


def summarize(root, plan, records):
    fields = ["run", "video", "codec", "target_mbps", "repeat", "status", "fps", "elapsed_seconds", "measured_frames", "measured_inference_calls", "error"]
    with (root / "SUMMARY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in plan:
            record = records.get(job["run"], {})
            writer.writerow({**{key: job[key] for key in ("run", "video", "codec", "target_mbps", "repeat")},
                             "status": record.get("status", "NOT_RUN"),
                             **{key: record.get(key) for key in fields[6:]}})
    result = {"planned_runs": len(plan), "valid_runs": sum(r.get("status") == "VALID" for r in records.values()),
              "selection": "Every planned condition is retained; no retry or exclusion based on FPS"}
    dump(root / "VERDICT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--london-video", type=Path)
    parser.add_argument("--video", action="append", metavar="LABEL=PATH")
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retained-runner", type=Path)
    parser.add_argument("--search-root", type=Path, action="append")
    parser.add_argument("--discover", action="store_true", help="List candidate video paths without running experiments")
    parser.add_argument("--plan", action="store_true", help="Validate video metadata and print the run matrix without GPU execution")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    videos, config = choose_inputs(args)
    if args.discover:
        return 0
    metadata = {name: {"path": str(path), "probe": validate_video(path, 1440)} for name, path in videos.items()}
    plan = []
    for repeat in range(1, args.repeats + 1):
        for name, path in videos.items():
            for codec in CODECS:
                for rate in RATES:
                    plan.append({"run": f"{len(plan)+1:03d}_{name}_{codec}_{rate}_r{repeat}", "video": name,
                                 "path": str(path), "codec": codec, "target_mbps": rate, "repeat": repeat})
    if args.plan:
        print(json.dumps({"inputs": metadata, "planned_runs": len(plan), "matrix": plan}, indent=2))
        return 0
    if os.geteuid() == 0:
        raise ValueError("Run as the normal Ubuntu user, without sudo")
    for executable in ("ffmpeg", "ffprobe", "cmake", "c++", "nvidia-smi"):
        if not shutil.which(executable):
            raise ValueError(f"Required existing program is missing: {executable}")
    sdk = args.sdk or config.get("sdk") or os.environ.get("SATC_SDK") or Path.home() / "nvCodecSDK/samples/Video_Codec_SDK_13.0.19"
    sdk = Path(sdk).expanduser().resolve()
    if not (sdk / "Samples/NvCodec/NvEncoder/NvEncoderCuda.cpp").is_file():
        raise ValueError("Supply --sdk /path/to/Video_Codec_SDK_13.0.19")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    root = args.output.expanduser().resolve() if args.output else Path.home() / "Downloads" / f"SATC_CBR_ENCODER_ONLY_P4_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    print("RESULTS:", root, flush=True)
    records = {}
    dump(root / "frozen_run_order.json", plan)
    summarize(root, plan, records)
    try:
        code = root / "code"
        code.mkdir()
        materialize(code, args.retained_runner)
        for name, path in videos.items():
            metadata[name]["sha256"] = digest(path)
        dump(root / "protocol.json", {"version": VERSION, "command": sys.argv, "runner_sha256": digest(__file__),
             "retained_source_commit": SOURCE_COMMIT, "retained_runner_sha256": RETAINED_SHA256,
             "native_source": "unchanged from the retained full-workload CBR runner",
             "inputs": metadata, "python": sys.version, "python_executable": sys.executable,
             "platform": platform.platform(), "sdk": str(sdk),
             "sdk_header_sha256": digest(sdk / "Interface/nvEncodeAPI.h"),
             "preset": "p4", "tuning": "low_latency", "rate_control": "CBR",
             "media_fps": 60, "gop": 240, "b_frames": 0, "lookahead": 0,
             "adaptive_quantization": False, "multipass": "disabled", "vbv_seconds": .25,
             "warmup_frames": 240, "measured_frames": 1200, "pacing": "none",
             "qp_mode": "constant zero delta buffer; no spatial allocation",
             "model_loaded": False, "inference_calls": 0,
             "timing_start": "after final warmup AU and its event row, before opening the source gate",
             "timing_end": "after final measured AU and its event row",
             "includes": "source CUDA decode/read, NV12 frame preparation, shared-memory copy, GPU upload, NVENC, completed AU receipt, event CSV logging, pipeline fill/drain",
             "excludes": "saliency preprocessing, model inference, saliency smoothing, tile ranking, spatial QP allocation, initialization, warmup, network, receiver, display",
             "bitstream_persistence": "payloads consumed as in the retained CBR benchmark; encoded video files are not saved",
             "window_definition": "nonoverlapping wall-clock windows from the measurement gate; partial final window marked separately"})
        for name, cmd in [("gpu_before.txt", ["nvidia-smi", "-q"]), ("ffmpeg_version.txt", ["ffmpeg", "-version"]),
                          ("ffprobe_version.txt", ["ffprobe", "-version"]), ("cmake_version.txt", ["cmake", "--version"])]:
            if command(cmd, root / name):
                raise RuntimeError(f"Preflight command failed: {name}")
        for name, cmd in [("build_configure.log", ["cmake", "-S", code, "-B", root / "build", f"-DSDK_TOP={sdk}", "-DCMAKE_BUILD_TYPE=Release"]),
                          ("build_compile.log", ["cmake", "--build", root / "build", "-j", str(min(os.cpu_count() or 2, 4))])]:
            if command(cmd, root / name):
                raise RuntimeError(f"Native build failed; see {name}")
        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "SATC_AV1_ANNEXB": "0"})
        # Keep every condition, including execution failures and low throughput.
        for job in plan:
            run = root / "runs" / job["run"]
            run.mkdir(parents=True)
            record = {"status": "ERROR"}
            print(f"{job['run']} ({len(records)+1}/{len(plan)})", flush=True)
            try:
                cmd = [sys.executable, code / "cbr_runner.py", "--worker", "--video", job["path"],
                       "--encoder", root / "build/nvenc_dynamic", "--codec", job["codec"],
                       "--bitrate", str(job["target_mbps"]), "--output", run]
                return_code = command(cmd, run / "console.log", env=environment)
                if return_code:
                    raise RuntimeError(f"Worker returned {return_code}; see console.log")
                record = json.loads((run / "benchmark.json").read_text())
                if record.get("codec") != job["codec"] or record.get("target_mbps") != job["target_mbps"] or record.get("warmup_frames") != 240 or record.get("measured_frames") != 1200 or record.get("measured_inference_calls") != 0:
                    raise ValueError("Worker result disagrees with the frozen condition")
                audit, measured = audit_events(run / "producer_events.csv", record)
                dump(run / "integrity_audit.json", audit)
                write_windows(run / "one_second_throughput.csv", record, measured)
                record.update({"status": "VALID", "integrity_audit": audit,
                               "encoded_media_mbps": sum(int(r["encoded_bytes"]) for r in measured) * 8 / (1200 / 60) / 1e6})
            except (Exception, KeyboardInterrupt) as error:
                (run / "ERROR.txt").write_text(traceback.format_exc())
                record.update({"status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR", "error": str(error) or type(error).__name__})
                dump(run / "result.json", record)
                records[job["run"]] = record
                summarize(root, plan, records)
                if isinstance(error, KeyboardInterrupt):
                    raise
            dump(run / "result.json", record)
            records[job["run"]] = record
            summarize(root, plan, records)
            print(f"  {record['status']}" + (f" | {record['fps']:.3f} FPS" if record.get("status") == "VALID" else " | see ERROR.txt"), flush=True)
        command(["nvidia-smi", "-q"], root / "gpu_after.txt")
    except (Exception, KeyboardInterrupt):
        (root / "ERROR.txt").write_text(traceback.format_exc())
        raise
    finally:
        verdict = summarize(root, plan, records)
        archive = root.with_name(root.name + "_UPLOAD.zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.relative_to(root).parts[0] != "build" and "__pycache__" not in path.parts:
                    bundle.write(path, Path(root.name) / path.relative_to(root))
        print("UPLOAD:", archive, flush=True)
        print(f"Valid runs: {verdict['valid_runs']}/{verdict['planned_runs']}", flush=True)
    return 0 if verdict["valid_runs"] == verdict["planned_runs"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(worker_main() if "--worker" in sys.argv[1:] else main())
    except (Exception, KeyboardInterrupt) as error:
        traceback.print_exc()
        raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 2)
