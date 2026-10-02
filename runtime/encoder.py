#!/usr/bin/env python3
"""Retained NVENC producer and frame-accounting code from the final quality run.

Use satc.py encode for the portable entry point and final six/six/33 policy.
Encoding throughput excludes transport, headset decoding, and presentation.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import selectors
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
import zlib

VERSION = "SATC-Final-Policy-Runtime"
PRESET = "p7"
W, H, MEDIA_FPS = 4096, 2048, 60
FRAME_BYTES = W * H * 3 // 2
QUEUE_FRAMES = 4
GUARD_FRAMES = 64
INPUT = struct.Struct("<4sQII")
OUTPUT = struct.Struct("<4sQQQI")
CODECS = ("h264", "hevc", "av1")
PREPARED_FOLDER = "SATC_ThreeVideos_60FPS_20260920_205952/prepared_inputs"
MODEL_NAME = "FlowSal_R192_T20_C8_v1_1_Robust1000_static_opset16.onnx"
SCOPE = "Unpaced sender processing; no WebRTC, Mininet, RT-MPC, receiver or HMD"


def dump(path, obj):
    """Atomic checkpoint, with strict finite JSON values."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    if not values:
        return None
    v = sorted(values)
    p = (len(v) - 1) * .95
    lo = int(p)
    hi = min(lo + 1, len(v) - 1)
    return {"count": len(v), "mean_ms": statistics.fmean(v),
            "median_ms": statistics.median(v),
            "p95_ms": v[lo] + (p - lo) * (v[hi] - v[lo]), "max_ms": v[-1]}


def metric(count, begin, end):
    if count <= 0 or end <= begin:
        raise ValueError("invalid completed-frame measurement")
    elapsed = (end - begin) / 1e9
    fps = count / elapsed
    return {"completed_frames": count, "elapsed_seconds": elapsed,
            "fps": fps, "strictly_above_60": fps > 60.0}


def csv_rows(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def command(args, log, timeout=300):
    with Path(log).open("w", encoding="utf-8") as f:
        f.write(json.dumps([str(a) for a in args]) + "\n")
        f.flush()
        p = subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout, check=False)
    if p.returncode:
        raise RuntimeError(f"{Path(str(args[0])).name} returned {p.returncode}; see {log}")


def read_exact(stream, length, timeout=60):
    """Finite, cancellable native/decoder wait. Timeout is not frame pacing."""
    data = bytearray(length)
    view = memoryview(data)
    pos = 0
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        sel.register(stream, selectors.EVENT_READ)
        while pos < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not sel.select(max(0, remaining)):
                raise TimeoutError("no complete native/decoder record within watchdog interval")
            n = stream.readinto(view[pos:])
            if not n:
                raise EOFError(f"stream ended at byte {pos}/{length}")
            pos += n
    return data


def write_all(stream, data):
    view = memoryview(data)
    while view:
        n = stream.write(view)
        if not n:
            raise BrokenPipeError("native input stopped")
        view = view[n:]


def stop_process(p):
    if p is None:
        return
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
    for name in ("stdin", "stdout"):
        f = getattr(p, name, None)
        if f:
            f.close()


def put_wait(q, item, stopped):
    while not stopped.is_set():
        try:
            q.put(item, timeout=.2)
            return True
        except queue.Full:
            pass
    return False


def materialize(code):
    """Copy the readable bundled runtime into a fresh result directory."""
    source = Path(__file__).resolve().parent
    for name in ("core.py", "live_model.py", "map_projection.py", "shared_frame.py", "shared_input.h",
                 "nvenc_uncapped.cpp", "CMakeLists.txt", "audit_optimization.py"):
        shutil.copy2(source / name, code / name)
    dump(code / "SHA256SUMS.json", {p.name: digest(p) for p in sorted(code.iterdir()) if p.is_file()})


def find_sdk(explicit):
    roots = [explicit] if explicit else []
    env = os.environ.get("SATC_SDK")
    if env:
        roots.append(Path(env))
    roots += [Path.home() / "nvCodecSDK/samples/Video_Codec_SDK_13.0.19"]
    for p in roots:
        p = Path(p).expanduser().resolve()
        h = p / "Samples/NvCodec/NvEncoder/NvEncoder.h"
        if h.is_file() and "NvEncOutputFrame" in h.read_text(errors="replace"):
            return p
    raise RuntimeError("SDK 13 not found. Supply --sdk /path/to/Video_Codec_SDK_13.0.19")


def find_model(explicit):
    candidates = [explicit] if explicit else []
    candidates += [Path("/data/D-SAV360/flowsal/models") / MODEL_NAME,
                   Path.home() / "D-SAV360/flowsal/models" / MODEL_NAME]
    for p in candidates:
        if Path(p).expanduser().is_file():
            return Path(p).expanduser().resolve()
    raise RuntimeError("Existing MoST-Sal ONNX model not found; supply --model /path/to/model.onnx")


def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, text=True, timeout=60, check=True)
    return json.loads(p.stdout)


def validate_video(path, total):
    d = probe(path)
    s = d["streams"][0]
    if (s.get("width"), s.get("height")) != (W, H):
        raise ValueError(f"{path.name}: use the existing prepared 4096x2048 input, not the original download")
    n = s.get("nb_frames")
    if not n or n == "N/A":
        raise ValueError(f"{path.name}: exact input frame count unavailable; supply the prepared MP4")
    if int(n) < total + GUARD_FRAMES:
        raise ValueError(f"{path.name}: needs {total + GUARD_FRAMES} unique input frames, has {n}; no looping/duplication is substituted")
    if s.get("pix_fmt") != "yuv420p" or s.get("color_space") != "bt709" or s.get("color_range") != "tv":
        raise ValueError(f"{path.name}: expected prepared 8-bit BT.709 limited-range YUV420")
    return d


def qp_from_classes(classes, codec):
    import numpy as np
    from core import tile_edges
    block = {"h264": 16, "hevc": 32, "av1": 64}[codec]
    xe, ye = tile_edges(W, H)
    x = np.minimum(np.arange((W + block - 1) // block) * block + block // 2, W - 1)
    y = np.minimum(np.arange((H + block - 1) // block) * block + block // 2, H - 1)
    tx = np.searchsorted(xe[1:], x, side="right")
    ty = np.searchsorted(ye[1:], y, side="right")
    labels = np.asarray(classes)[ty[:, None], tx[None, :]]
    return np.ascontiguousarray(np.array([2, 0, -2], dtype=np.int8)[labels])


class Source:
    def __init__(self, video, run, total, warmup, gate, stop):
        self.q = queue.Queue(QUEUE_FRAMES)
        self.done = threading.Event()
        self.error = None
        self.events = []
        self.log = (run / "source_decode.log").open("w")
        # Guard frames prevent a decoder EOF/loop boundary inside the measured input.
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(video),
                "-map", "0:v:0", "-an", "-sn", "-frames:v", str(total + GUARD_FRAMES),
                "-vf", "hwdownload,format=nv12", "-fps_mode", "passthrough",
                "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1"]
        dump(run / "source_command.json", args)
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)

        def work():
            try:
                for fid in range(total):
                    if stop.is_set():
                        return
                    if fid == warmup:
                        while not gate.wait(.2):
                            if stop.is_set():
                                return
                    started = time.monotonic_ns()
                    raw = read_exact(self.proc.stdout, FRAME_BYTES)
                    available = time.monotonic_ns()
                    self.events.append({"frame_id": fid, "read_start_ns": started,
                                        "raw_available_ns": available})
                    if not put_wait(self.q, (fid, raw, started, available), stop):
                        return
            except BaseException:
                if not stop.is_set():
                    self.error = traceback.format_exc()
            finally:
                self.done.set()
        self.thread = threading.Thread(target=work, name="raw-source", daemon=True)
        self.thread.start()

    def close(self):
        # This FFmpeg child was created by this session. No unrelated process is killed.
        if self.proc.poll() is None:
            self.proc.terminate()
        self.thread.join(timeout=5)
        stop_process(self.proc)
        self.log.close()
        if self.thread.is_alive():
            raise RuntimeError("source worker failed to terminate")


class Inference:
    def __init__(self, source, model, codec, total, stop):
        import numpy as np
        from core import history_ids
        from live_model import preprocess_nv12, normalize_model_frame
        self.q = queue.Queue(QUEUE_FRAMES)
        self.done = threading.Event()
        self.error = None
        self.events, self.labels = [], []
        self.raw_maps = np.empty((total, 144, 192), np.float32)
        self.scores = np.empty((total, 5, 9), np.float64)
        model.reset()

        def work():
            history = {}
            try:
                for expected in range(total):
                    while not stop.is_set():
                        try:
                            item = source.q.get(timeout=.2)
                            break
                        except queue.Empty:
                            if source.error:
                                raise RuntimeError(source.error)
                            if source.done.is_set():
                                raise RuntimeError("source ended before all input frames")
                    else:
                        return
                    fid, raw, read_start, available = item
                    if fid != expected:
                        raise RuntimeError("source frame gap before live inference")
                    prep = time.monotonic_ns()
                    history[fid] = normalize_model_frame(preprocess_nv12(raw, W, H))
                    ids = history_ids(fid)
                    if ALLOCATION_POLICY is not None and hasattr(ALLOCATION_POLICY, "prepare"):
                        ALLOCATION_POLICY.prepare(model.ema, raw, codec, fid)
                    classes, _, stamps = model.infer([history[i] for i in ids], ids)
                    history.pop(fid - 160, None)
                    qp = (ALLOCATION_POLICY.make_map(model.ema, raw, codec, fid)
                          if ALLOCATION_POLICY is not None else qp_from_classes(classes, codec))
                    ready = time.monotonic_ns()
                    self.raw_maps[fid] = model.last_raw
                    self.scores[fid] = model.last_scores
                    self.labels.append({"frame_id": fid, "classes_5x9_hex": classes.tobytes().hex()})
                    self.events.append({"frame_id": fid, "preprocess_start_ns": prep,
                                        **stamps, "codec_qp_ready_ns": ready})
                    if not put_wait(self.q, (fid, raw, read_start, available, qp), stop):
                        return
            except BaseException:
                if not stop.is_set():
                    self.error = traceback.format_exc()
            finally:
                self.done.set()
        self.thread = threading.Thread(target=work, name="live-most-sal", daemon=True)
        self.thread.start()


class Bitstream:
    def __init__(self, path, codec, total):
        self.f = path.open("wb", buffering=1024 * 1024)
        self.codec, self.total = codec, total
        self.ivf_from_sdk = None

    def write(self, packet, fid):
        if self.codec == "av1":
            if self.ivf_from_sdk is None:
                self.ivf_from_sdk = packet[:4] == b"DKIF"
                if not self.ivf_from_sdk:
                    self.f.write(struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32,
                                             b"AV01", W, H, MEDIA_FPS, 1, self.total, 0))
            if not self.ivf_from_sdk:
                self.f.write(struct.pack("<IQ", len(packet), fid))
        self.f.write(packet)

    def close(self):
        self.f.close()


def verify_stream(path, expected, run):
    """Independent decoder count AFTER timing. Does not inflate headline FPS."""
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-xerror",
            "-hwaccel", "cuda", "-i", str(path), "-map", "0:v:0", "-an", "-sn",
            "-fps_mode", "passthrough", "-progress", "pipe:1", "-f", "null", "-"]
    with (run / "verification_decode_errors.log").open("w") as log:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=log, text=True,
                           timeout=600, check=False)
    (run / "verification_decode_progress.txt").write_text(p.stdout, encoding="utf-8")
    frames = [int(l.split("=", 1)[1].strip()) for l in p.stdout.splitlines() if l.startswith("frame=")]
    count = frames[-1] if frames else 0
    metadata = probe(path)
    s = metadata["streams"][0]
    expected_codec = {".h264": "h264", ".hevc": "hevc", ".ivf": "av1"}[path.suffix]
    report = {"return_code": p.returncode, "decoded_frames": count,
              "expected_frames": expected, "width": s.get("width"), "height": s.get("height"),
              "codec_name": s.get("codec_name"), "expected_codec": expected_codec,
              "measurement": "postrun count, not timed receiver FPS"}
    dump(run / "decoder_verification.json", report)
    if (p.returncode or count != expected or (s.get("width"), s.get("height")) != (W, H)
            or s.get("codec_name") != expected_codec):
        raise RuntimeError(f"independent decoder count/geometry check failed: {report}")
    return report


def check_workers(source, inference):
    for worker in (source, inference):
        if worker is not None and worker.error:
            raise RuntimeError(worker.error)


def run_one(args, root, job, model):
    import numpy as np
    from shared_frame import SharedFrame
    run = root / "runs" / job["id"]
    run.mkdir(parents=True)
    video, codec, method, rate = Path(job["video"]), job["codec"], job["method"], job["target_mbps"]
    total = args.warmup_frames + args.frames
    summary = {**job, "status": "ERROR", "scope": SCOPE, "preset": PRESET,
               "media_fps": MEDIA_FPS, "pacing": "none", "drop_policy": "none; bounded queues block",
               "warmup_frames": args.warmup_frames, "planned_measured_frames": args.frames,
               "processing_queue_frames_per_stage": QUEUE_FRAMES}
    dump(run / "config.json", summary)
    events = []
    source = inference = encoder = shared = bitstream = None
    enc_log = (run / "nvenc.log").open("w")
    stop, gate = threading.Event(), threading.Event()
    t0 = t1 = None
    failure = None
    stream_path = run / ("encoded." + {"h264": "h264", "hevc": "hevc", "av1": "ivf"}[codec])
    print(f"\n{job['id']}: {PRESET}, {method}, {rate} Mbit/s media rate, UNPACED", flush=True)
    try:
        shared = SharedFrame(FRAME_BYTES)
        encoder = subprocess.Popen([str(root / "build/nvenc_uncapped"), codec,
                                    str(rate * 1000000), str(shared.fd)],
                                   pass_fds=(shared.fd,), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=enc_log, bufsize=0)
        if read_exact(encoder.stdout, 4) != b"RDY1":
            raise RuntimeError("native encoder did not become ready")
        bitstream = Bitstream(stream_path, codec, total)
        source = Source(video, run, total, args.warmup_frames, gate, stop)
        if method == "roi":
            inference = Inference(source, model, codec, total, stop)
        block = {"h264": 16, "hevc": 32, "av1": 64}[codec]
        zero = np.zeros(((H + block - 1) // block, (W + block - 1) // block), np.int8)
        for expected in range(total):
            if expected == args.warmup_frames:
                # Every warmup output is complete. No measured-frame inference
                # has started: the source gate is still closed.
                t0 = time.monotonic_ns()
                gate.set()
            q = inference.q if inference else source.q
            deadline = time.monotonic() + 60
            while True:
                check_workers(source, inference)
                if encoder.poll() is not None:
                    raise RuntimeError("native encoder exited; see nvenc.log")
                try:
                    item = q.get(timeout=.2)
                    break
                except queue.Empty:
                    worker = inference if inference else source
                    if worker.done.is_set() or time.monotonic() > deadline:
                        raise RuntimeError("pipeline ended or stalled before all frames completed")
            if inference:
                fid, raw, read_start, available, qp = item
            else:
                fid, raw, read_start, available = item
                qp = zero
            if fid != expected:
                raise RuntimeError("noncontiguous input IDs; refusing to report inflated FPS")
            submit = time.monotonic_ns()
            shared.copy_from(raw)
            write_all(encoder.stdin, INPUT.pack(b"FRM1", fid, len(raw), qp.nbytes))
            write_all(encoder.stdin, qp.tobytes())
            magic, echoed, enc_start, enc_end, size = OUTPUT.unpack(read_exact(encoder.stdout, OUTPUT.size))
            if magic != b"AU01" or echoed != fid or not 0 < size < 20000000 or enc_end < enc_start:
                raise RuntimeError("invalid native completed frame record")
            packet = read_exact(encoder.stdout, size)
            bitstream.write(packet, fid)
            done = time.monotonic_ns()
            events.append({"frame_id": fid, "source_read_start_ns": read_start,
                           "raw_available_ns": available, "bridge_submit_ns": submit,
                           "gpu_upload_start_ns": enc_start, "encoded_ns": enc_end,
                           "bitstream_complete_ns": done, "encoded_bytes": size,
                           "qp_bytes": qp.nbytes, "qp_crc32": zlib.crc32(qp.tobytes())})
            if fid + 1 == total:
                t1 = done
            if fid >= args.warmup_frames and (fid + 1 - args.warmup_frames) % 300 == 0:
                print(f"  completed {fid + 1 - args.warmup_frames}/{args.frames} measured frames", flush=True)
        encoder.stdin.close()
        encoder.wait(timeout=30)
        if encoder.returncode:
            raise RuntimeError("encoder failed at final flush")
        check_workers(source, inference)
    except BaseException:
        failure = traceback.format_exc()
        if isinstance(sys.exc_info()[1], KeyboardInterrupt):
            summary["interrupted"] = True
    finally:
        stop.set(); gate.set()
        cleanup_errors = []
        if source:
            try:
                source.close()
            except Exception:
                cleanup_errors.append(traceback.format_exc())
        if inference:
            inference.thread.join(timeout=30)
            if inference.thread.is_alive():
                cleanup_errors.append("live inference worker did not terminate")
        stop_process(encoder)
        if shared:
            shared.close()
        if bitstream:
            bitstream.close()
        enc_log.close()
        if cleanup_errors:
            failure = (failure or "") + "\n".join(cleanup_errors)
            summary["unsafe_to_continue"] = True
    if events:
        csv_rows(run / "frame_events.csv", list(events[0]), events)
    if source and source.events:
        csv_rows(run / "source_events.csv", list(source.events[0]), source.events)
    if inference and inference.events:
        csv_rows(run / "model_events.csv", list(inference.events[0]), inference.events)
        csv_rows(run / "inference_classes.csv", ["frame_id", "classes_5x9_hex"], inference.labels)
    if not failure:
        try:
            expected_ids = list(range(total))
            if [r["frame_id"] for r in events] != expected_ids:
                raise RuntimeError("completed-frame coverage mismatch")
            if [r["frame_id"] for r in source.events] != expected_ids:
                raise RuntimeError("source coverage mismatch")
            if inference and [r["frame_id"] for r in inference.events] != expected_ids:
                raise RuntimeError("one live inference per encoded frame was not verified")
            measured = events[args.warmup_frames:]
            summary["measurement"] = metric(len(measured), t0, t1)
            summary["measured_input_frames"] = len(measured)
            summary["all_completed_frames"] = total
            summary["missing_or_dropped_source_frames"] = 0
            summary["measured_inference_calls"] = args.frames if inference else 0
            summary["encoded_media_mbps"] = sum(r["encoded_bytes"] for r in measured) * 8 / (args.frames / MEDIA_FPS) / 1e6
            summary["bitstream_wall_output_mbps"] = sum(r["encoded_bytes"] for r in measured) * 8 / summary["measurement"]["elapsed_seconds"] / 1e6
            summary["upload_and_encode"] = distribution([(r["encoded_ns"] - r["gpu_upload_start_ns"]) / 1e6 for r in measured])
            summary["source_read_to_bitstream"] = distribution([(r["bitstream_complete_ns"] - r["source_read_start_ns"]) / 1e6 for r in measured])
            if inference:
                model_rows = inference.events[args.warmup_frames:]
                summary["inference_call"] = distribution([(r["inference_end_ns"] - r["inference_start_ns"]) / 1e6 for r in model_rows])
                summary["preprocess_model_map"] = distribution([(r["codec_qp_ready_ns"] - r["preprocess_start_ns"]) / 1e6 for r in model_rows])
            print(f"  measured throughput = {summary['measurement']['fps']:.3f} FPS; validating output", flush=True)
            summary["decoder_verification"] = verify_stream(stream_path, total, run)
            if inference:
                print("  auditing every saliency map after timing", flush=True)
                audit_dir = run / "map_audit"
                audit_dir.mkdir()
                inference.raw_maps.tofile(audit_dir / "raw_saliency.f32")
                np.save(audit_dir / "compact_scores.npy", inference.scores, allow_pickle=False)
                from audit_optimization import audit_run
                summary["map_audit"] = audit_run(run)
                if summary["map_audit"]["status"] != "PASS":
                    raise RuntimeError("optimized saliency decisions differ from reference")
            summary["status"] = "VALID"
        except BaseException:
            failure = traceback.format_exc()
            if isinstance(sys.exc_info()[1], KeyboardInterrupt):
                summary["interrupted"] = True
    if failure:
        (run / "ERROR.txt").write_text(failure, encoding="utf-8")
        summary["error"] = failure.splitlines()[-1]
        print(f"  ERROR: {summary['error']}", flush=True)
    dump(run / "summary.json", summary)
    return summary


def summarize(root, plan, records):
    rows = []
    for job in plan:
        r = records.get(job["id"], {})
        m = r.get("measurement") or {}
        rows.append({"run": job["id"], "video": job["video_name"], "codec": job["codec"],
                     "method": job["method"], "target_media_mbps": job["target_mbps"],
                     "status": r.get("status", "NOT_RUN"), "completed_frames": m.get("completed_frames"),
                     "elapsed_seconds": m.get("elapsed_seconds"), "measured_fps": m.get("fps"),
                     "valid_and_above_60": r.get("status") == "VALID" and m.get("strictly_above_60", False),
                     "error": r.get("error", "")})
    csv_rows(root / "SUMMARY.csv", list(rows[0]), rows)
    roi = [r for r in rows if r["method"] == "roi"]
    result = {"planned_runs": len(rows), "valid_runs": sum(r["status"] == "VALID" for r in rows),
              "roi_runs": len(roi), "all_roi_valid_and_above_60": all(r["valid_and_above_60"] for r in roi),
              "scope": SCOPE, "no_selection": "Every planned run appears, including errors and FPS below 60"}
    dump(root / "VERDICT.json", result)
    lines = [VERSION, SCOPE, "", "Complete measured frames / actual wall seconds; no pacing.",
             "The 60 FPS field in the encoder is a media timebase, not an input/output limiter.",
             "No measured result is rounded up for the >60 decision.",
             "No maximum-FPS claim about the GPU is made; these are this implementation's measured throughputs.",
             "", "video          codec method  Mbps  status      FPS"]
    for r in rows:
        v = f"{r['measured_fps']:.3f}" if r["measured_fps"] is not None else "NA"
        lines.append(f"{r['video']:14} {r['codec']:5} {r['method']:7} {r['target_media_mbps']:4}  {r['status']:9} {v}")
    lines += ["", json.dumps(result, indent=2), "",
              "Stage/queue latency here is measured under unpaced saturation, not geographic/network latency.",
              "Network trace replay, RT-MPC decisions, delivered FPS, PSNR, SSIM and LPIPS are not measured by this script.",
              "Raw streams and all saliency audits remain local; the upload includes logs and frame timestamps."]
    (root / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def package(root):
    final = root / "UPLOAD_UNCAPPED_P4_RESULTS.zip"
    temp = root / "UPLOAD_UNCAPPED_P4_RESULTS.zip.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if not p.is_file() or p in (final, temp):
                continue
            if any(x in rel.parts for x in ("build", "trt_cache", "__pycache__", "map_audit")):
                continue
            if p.suffix in (".h264", ".hevc", ".ivf"):
                continue
            z.write(p, str(rel))
    os.replace(temp, final)
    return final


def self_test():
    assert metric(1200, 0, 20_000_000_000)["strictly_above_60"] is False
    assert metric(1200, 0, 19_000_000_000)["strictly_above_60"] is True
    assert metric(1200, 0, 21_000_000_000)["strictly_above_60"] is False
    assert distribution([1., 2., 3.])["p95_ms"] == 2.9
    assert INPUT.size == 20 and OUTPUT.size == 32
    source = Path(__file__).resolve().parent
    for name in ("core.py", "live_model.py", "shared_frame.py", "map_projection.py", "audit_optimization.py"):
        compile((source / name).read_bytes(), name, "exec")
    native = (source / "nvenc_uncapped.cpp").read_text()
    assert "NV_ENC_PRESET_P7_GUID" in native
    assert "NV_ENC_MULTI_PASS_DISABLED" in native
    with tempfile.TemporaryDirectory(prefix="satc_uncapped_selftest_") as d:
        p = Path(d) / "test.ivf"
        s = Bitstream(p, "av1", 2)
        s.write(b"payload0", 0); s.write(b"payload1", 1); s.close()
        assert p.read_bytes()[:4] == b"DKIF" and p.stat().st_size == 72
    print("PASS: accounting, strict >60 decision, embedded Python syntax, IVF framing, frozen native settings")
    print("These checks do not exercise CUDA, NVENC, ONNX Runtime, or the user's GPU.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retained encoder integrity checks")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use python satc.py encode --help to run the encoder")
    self_test()
