#!/usr/bin/env python3
"""
RUN_360SATC_VIEWPORT_PILOT_v4.py

Pilot: real-viewer viewport quality for a selected VR-EyeTracking video.

Pipeline
--------
1) Prepare the selected video as 4096x2048 @ 60 FPS, matching the paper-facing SATC runtime.
2) Reuse the final paper-facing 360-SATC allocator:
      5x9 logical grid
      top 6 tiles:   QP delta -2
      next 6 tiles:  QP delta -1
      remaining 33:  QP delta 0
   and encode H.264 P7 at 12 Mbit/s for:
      - 360-SATC
      - uniform CBR
3) Resample the real head-position traces from the original 25-FPS dataset to
   the 60-FPS encoded timeline by timestamp.
4) Extract rectilinear head-centered viewports on the RTX GPU.
5) Compute per-viewer RGB PSNR and Gaussian-window RGB SSIM for CBR and SATC.
6) Save per-viewer CSV, per-frame CSV, summary JSON/TXT, and a few visual triptychs.

This is a PILOT. One video can show whether the expected effect exists, but it
is not a content-level significance experiment.

Run with the SATC venv, e.g.
  /home/mininet-ovs/venvs/satc/bin/python RUN_360SATC_VIEWPORT_PILOT_v4.py

Useful smoke test:
  .../python RUN_360SATC_VIEWPORT_PILOT_v4.py --eval-frames 120

Full pilot:
  .../python RUN_360SATC_VIEWPORT_PILOT_v4.py --eval-frames 896
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


VIDEO_ROOT = Path("/data/VR-EyeTracking/VR Eye-Tracking Dataset/videos/videos")
TRACE_ROOT = Path("/data/VR-EyeTracking/VR Eye-Tracking Dataset/Gaze_txt_files/Gaze_txt_files")

MASTER_PATH = Path("/home/mininet-ovs/Downloads/files_to_upload/RUN_360SATC_ALL_EXPERIMENTS_v6.py")
FINAL_PATH = Path("/home/mininet-ovs/Downloads/files_to_upload/RUN_360SATC_FINAL_QUALITY_MATRIX_v1.py")
BASE_PATH = Path("/home/mininet-ovs/Downloads/files_to_upload/360SATC_Saliency_Solution.py")

HELPER_CANDIDATES = [
    Path("/home/mininet-ovs/Downloads/SATC_FINAL_QUALITY_EXP09/embedded_helpers/SATC_Integrated_Real5G_60FPS_v14.py"),
    Path("/home/mininet-ovs/snap/firefox/common/SATC_Integrated_Real5G_60FPS_v14.py"),
]

# Prefer the already-patched runner from the final paper-quality campaign.  This
# avoids trying to patch an embedded helper that may itself already be modified.
FINAL_RUNNER_CANDIDATES = [
    Path("/home/mininet-ovs/Downloads/SATC_FINAL_QUALITY_EXP09/final_360satc_runner.py"),
]

DEFAULT_OUT_PARENT = Path("/home/mininet-ovs/snap/firefox/common")

ENC_FPS = 60.0
W = 4096
H = 2048
WARMUP = 240
MEASURED = 896
CODEC = "h264"
RATE = 12


def die(msg: str):
    raise RuntimeError(msg)


def load_py(path: Path, name: str):
    if not path.is_file():
        die(f"Missing Python source: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        die(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)

    # Python 3.12 dataclasses (and some other import-time machinery) expect the
    # module to already be registered in sys.modules while its body executes.
    # importlib.util.module_from_spec() does not register it automatically.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Do not leave a half-imported module behind after a failed import.
        if sys.modules.get(name) is mod:
            del sys.modules[name]
        raise
    return mod


def run(cmd, *, log: Path | None = None):
    print("+", " ".join(map(str, cmd)), flush=True)
    if log is None:
        subprocess.run(list(map(str, cmd)), check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, check=True)


def ffprobe_video(path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(p.stdout)["streams"][0]


def ratio_to_float(value: str) -> float:
    s = str(value)
    if "/" in s:
        a, b = s.split("/", 1)
        b = float(b)
        return 0.0 if b == 0.0 else float(a) / b
    return float(s)


def prepare_reference(source: Path, prepared: Path):
    if prepared.is_file():
        info = ffprobe_video(prepared)
        if int(info["width"]) == W and int(info["height"]) == H:
            print(f"[reuse] prepared reference: {prepared}")
            return

    prepared.parent.mkdir(parents=True, exist_ok=True)
    tmp = prepared.with_suffix(".tmp.mp4")
    tmp.unlink(missing_ok=True)

    # High-quality common intermediate. It is the same reference input to both
    # SATC and CBR; the measured comparison is between those two encodes.
    cmd_nvenc = [
        "ffmpeg", "-nostdin", "-hide_banner", "-y",
        "-i", str(source),
        "-map", "0:v:0", "-an",
        "-vf", f"scale={W}:{H}:flags=lanczos,fps={int(ENC_FPS)},setsar=1",
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-tune", "hq",
        "-rc", "constqp",
        "-qp", "10",
        "-g", "240",
        "-bf", "0",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        str(tmp),
    ]
    try:
        run(cmd_nvenc)
    except subprocess.CalledProcessError:
        print("[warn] NVENC reference preparation failed; falling back to libx264.")
        tmp.unlink(missing_ok=True)
        cmd_cpu = [
            "ffmpeg", "-nostdin", "-hide_banner", "-y",
            "-i", str(source),
            "-map", "0:v:0", "-an",
            "-vf", f"scale={W}:{H}:flags=lanczos,fps={int(ENC_FPS)},setsar=1",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "8",
            "-g", "240",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr",
            str(tmp),
        ]
        run(cmd_cpu)

    tmp.replace(prepared)
    info = ffprobe_video(prepared)
    print("[prepared]", json.dumps(info, indent=2))


def find_compatible_helper(base) -> Path:
    candidates = list(HELPER_CANDIDATES)

    # Also discover copies left by earlier campaigns.
    for root in (
        Path("/home/mininet-ovs/Downloads"),
        Path("/home/mininet-ovs/snap/firefox/common"),
    ):
        if root.is_dir():
            candidates.extend(root.glob("**/SATC_Integrated_Real5G_60FPS_v14.py"))

    seen = set()
    failures = []
    for p in candidates:
        p = Path(p)
        if p in seen:
            continue
        seen.add(p)
        if not p.is_file():
            continue
        try:
            frozen = p.read_text(encoding="utf-8")
            # Validate in memory before handing it to prepare_satc_runtime().
            base.patched_satc_source(frozen, "p7")
            print(f"[compatible-helper] {p}")
            return p
        except Exception as e:
            failures.append(f"{p}: {type(e).__name__}: {e}")

    details = "\n  ".join(failures[-10:]) if failures else "(no candidate files found)"
    die(
        "Could not find an unmodified SATC v14 helper compatible with "
        "patched_satc_source(). Last checks:\n  " + details
    )


def seed_final_paper_runner(root: Path) -> Path | None:
    """Copy an already-patched final campaign runner into this pilot root."""
    target = root / "final_360satc_runner.py"
    if target.is_file():
        txt = target.read_text(encoding="utf-8", errors="replace")
        if "def run_one" in txt and "ALLOCATION_POLICY" in txt:
            print(f"[reuse-final-runner] {target}")
            return target
        target.unlink()

    candidates = list(FINAL_RUNNER_CANDIDATES)
    dl = Path("/home/mininet-ovs/Downloads")
    if dl.is_dir():
        candidates.extend(dl.glob("SATC_FINAL_QUALITY_EXP*/final_360satc_runner.py"))
        candidates.extend(dl.glob("**/final_360satc_runner.py"))

    # Prefer the newest valid runner if several campaigns are present.
    valid = []
    seen = set()
    for p in candidates:
        p = Path(p)
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "def run_one" not in txt:
            continue
        if "ALLOCATION_POLICY" not in txt:
            continue
        valid.append(p)

    if not valid:
        return None

    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = valid[0]
    shutil.copy2(chosen, target)
    print(f"[seed-final-runner] {chosen} -> {target}")
    return target


def encode_pair(root: Path, prepared: Path, video_id: str):
    master = load_py(MASTER_PATH, "satc_master_v6")
    final = load_py(FINAL_PATH, "satc_final_quality_v1")
    base = load_py(BASE_PATH, "satc_saliency_solution")

    # Defensive protocol checks: abort instead of silently running another policy.
    assert int(final.WARMUP) == WARMUP
    assert int(final.MEASURED) == MEASURED
    assert int(final.FPS) == int(ENC_FPS)
    assert int(final.WIDTH) == W
    assert int(final.HEIGHT) == H

    root.mkdir(parents=True, exist_ok=True)
    video_key = f"vreye{video_id}"
    videos = {video_key: prepared}

    # Best path: reuse the exact already-patched final paper runner.  The final
    # harness checks root/final_360satc_runner.py first and skips re-patching if
    # it is present.
    seeded = seed_final_paper_runner(root)

    if seeded is not None:
        helpers = {"satc": seeded}  # not read when the seeded runner exists
    else:
        # Fallback: find a truly unmodified v14 helper that the patch function
        # accepts.  Do not blindly use the first file named v14.
        helper = find_compatible_helper(base)
        helpers = {"satc": helper}

    satc, model, satc_root = final.prepare_satc_runtime(
        master, base, helpers, root, videos
    )

    satc_summary, satc_run, satc_enc, satc_events, satc_audit, satc_status = \
        final.ensure_satc_encode(
            satc, base, satc_root, model, prepared,
            video_key, CODEC, RATE, "360-SATC"
        )

    cbr_summary, cbr_run, cbr_enc, cbr_events, _, cbr_status = \
        final.ensure_satc_encode(
            satc, base, satc_root, model, prepared,
            video_key, CODEC, RATE, "CBR"
        )

    meta = {
        "video_id": video_id,
        "prepared_reference": str(prepared),
        "satc_encoded": str(satc_enc),
        "cbr_encoded": str(cbr_enc),
        "satc_status": satc_status,
        "cbr_status": cbr_status,
        "satc_summary": satc_summary,
        "cbr_summary": cbr_summary,
        "satc_audit": satc_audit,
        "protocol": {
            "resolution": [W, H],
            "fps": ENC_FPS,
            "codec": CODEC,
            "target_mbps": RATE,
            "warmup_frames": WARMUP,
            "measured_frames": MEASURED,
            "tile_grid": [5, 9],
            "qp_minus2_tiles": 6,
            "qp_minus1_tiles": 6,
            "qp_zero_tiles": 33,
        }
    }
    (root / "encode_pair.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return Path(satc_enc), Path(cbr_enc)


def remux_for_opencv(raw: Path, out: Path):
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-y",
        "-fflags", "+genpts",
        "-r", str(int(ENC_FPS)),
        "-i", str(raw),
        "-map", "0:v:0",
        "-an",
        "-c:v", "copy",
        str(out),
    ])
    return out


def trace_files(video_id: str):
    files = sorted(TRACE_ROOT.glob(f"p*/{video_id}.*_ori_0.txt"))
    if not files:
        die(f"No video-{video_id} traces found below {TRACE_ROOT}")
    return files


def load_trace(path: Path):
    idx, hx, hy = [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 8 or p[0] != "frame":
                continue
            try:
                idx.append(int(p[1]))
                hx.append(float(p[3]))
                hy.append(float(p[4]))
            except ValueError:
                continue
    if len(idx) < 2:
        die(f"Too few valid rows in {path}")
    return (
        np.asarray(idx, dtype=np.int32),
        np.asarray(hx, dtype=np.float64),
        np.asarray(hy, dtype=np.float64),
    )


def resample_head(trace_path: Path, query_frame_ids: np.ndarray, source_fps: float):
    idx, hx, hy = load_trace(trace_path)
    t = (idx.astype(np.float64) - 1.0) / source_fps
    tq = query_frame_ids.astype(np.float64) / ENC_FPS

    # Horizontal coordinate is circular. Unwrap in angular space before interpolation.
    ang = np.unwrap(hx * 2.0 * np.pi)
    aq = np.interp(tq, t, ang)
    xq = np.mod(aq / (2.0 * np.pi), 1.0)

    yq = np.interp(tq, t, hy)
    yq = np.clip(yq, 0.0, 1.0)

    valid = (tq >= t[0]) & (tq <= t[-1])
    return xq.astype(np.float32), yq.astype(np.float32), valid


def make_local_rays(size: int, hfov_deg: float, device):
    # Square viewport. For a square image, vfov == hfov.
    f = math.tan(math.radians(hfov_deg) / 2.0)
    xx = (2.0 * (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size - 1.0) * f
    yy = (1.0 - 2.0 * (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size) * f
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    z = torch.ones_like(x)
    inv = torch.rsqrt(x*x + y*y + z*z)
    return x*inv, y*inv, z*inv


def make_grid(base_rays, yaw, pitch, image_h: int, image_w: int):
    # yaw,pitch: [B] radians. Camera forward is +Z, right +X, up +Y.
    x, y, z = base_rays
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    z = z.unsqueeze(0)

    sp = torch.sin(pitch)[:, None, None]
    cp = torch.cos(pitch)[:, None, None]
    sy = torch.sin(yaw)[:, None, None]
    cy = torch.cos(yaw)[:, None, None]

    # Rx(-pitch), then Ry(yaw).
    x1 = x
    y1 = cp * y + sp * z
    z1 = -sp * y + cp * z

    x2 = cy * x1 + sy * z1
    y2 = y1
    z2 = -sy * x1 + cy * z1

    lon = torch.atan2(x2, z2)                       # [-pi, pi]
    lat = torch.asin(torch.clamp(y2, -1.0, 1.0))  # [-pi/2, pi/2]

    u = torch.remainder((lon + math.pi) / (2.0 * math.pi), 1.0)
    v_bottom = torch.clamp((lat + math.pi/2.0) / math.pi, 0.0, 1.0)

    # Horizontal circular pad will add one pixel at each side.
    # Pixel-center coordinates use u*W - 0.5, then +1 for the left circular pad.
    xpix_pad = u * image_w - 0.5 + 1.0
    ypix = torch.clamp((1.0 - v_bottom) * image_h - 0.5, 0.0, image_h - 1.0)

    padded_w = image_w + 2
    xnorm = 2.0 * xpix_pad / (padded_w - 1.0) - 1.0
    ynorm = 2.0 * ypix / (image_h - 1.0) - 1.0
    return torch.stack((xnorm, ynorm), dim=-1)


def frame_to_tensor(frame_bgr: np.ndarray, device):
    # OpenCV BGR -> RGB float [0,1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    return t.permute(2, 0, 1).unsqueeze(0).div_(255.0)


def extract_batch(img, grid):
    # Circular horizontal padding; vertical coordinates are clamped in make_grid().
    p = F.pad(img, (1, 1, 0, 0), mode="circular")
    b = grid.shape[0]
    return F.grid_sample(
        p.expand(b, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def gaussian_kernel(device, channels=3, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x*x) / (2.0 * sigma * sigma))
    g = g / g.sum()
    k2 = (g[:, None] * g[None, :]).contiguous()
    return k2[None, None].repeat(channels, 1, 1, 1)


def filt_reflect(x, kernel):
    p = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"),
                    kernel, groups=x.shape[1])


def ssim_batch(a, b, kernel):
    # Gaussian-window RGB SSIM; one value per viewport.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ma = filt_reflect(a, kernel)
    mb = filt_reflect(b, kernel)
    va = filt_reflect(a*a, kernel) - ma*ma
    vb = filt_reflect(b*b, kernel) - mb*mb
    cab = filt_reflect(a*b, kernel) - ma*mb
    num = (2.0*ma*mb + c1) * (2.0*cab + c2)
    den = (ma*ma + mb*mb + c1) * (va + vb + c2)
    return (num / torch.clamp(den, min=1e-12)).mean(dim=(1, 2, 3))


def psnr_batch(a, b):
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        die(f"OpenCV could not open {path}")
    return cap


def save_triptych(path: Path, ref, cbr, satc, label: str):
    # tensors [3,H,W], RGB [0,1]
    imgs = []
    for x, name in ((ref, "Reference"), (cbr, "CBR 12 Mb/s"), (satc, "360-SATC 12 Mb/s")):
        arr = (x.detach().clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy())
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.putText(arr, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(arr, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,0,0), 1, cv2.LINE_AA)
        imgs.append(arr)
    canvas = np.concatenate(imgs, axis=1)
    cv2.putText(canvas, label, (8, canvas.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, canvas.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def evaluate(root: Path, prepared: Path, satc_raw: Path, cbr_raw: Path,
             video_id: str, source_fps: float,
             viewport_size: int, hfov: float, eval_frames: int, max_viewers: int):
    if not torch.cuda.is_available():
        die("CUDA is required for this pilot. torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    print("[gpu]", torch.cuda.get_device_name(0))

    satc_mp4 = remux_for_opencv(satc_raw, root / "decode" / "satc.mp4")
    cbr_mp4 = remux_for_opencv(cbr_raw, root / "decode" / "cbr.mp4")

    traces = trace_files(video_id)
    if max_viewers > 0:
        traces = traces[:max_viewers]
    viewers = [p.parent.name for p in traces]
    print(f"[viewers] {len(viewers)}: {', '.join(viewers)}")

    n_eval = min(eval_frames, MEASURED)
    query_ids = np.arange(WARMUP, WARMUP + n_eval, dtype=np.int32)

    hx_all, hy_all, valid_all = [], [], []
    for p in traces:
        hx, hy, valid = resample_head(p, query_ids, source_fps)
        hx_all.append(hx); hy_all.append(hy); valid_all.append(valid)
    hx_all = np.stack(hx_all)
    hy_all = np.stack(hy_all)
    valid_all = np.stack(valid_all)

    # Samples outside an individual viewer trace are explicitly skipped.
    if not valid_all.all():
        print("[warn] Some viewer/frame samples fall outside trace coverage; they will be skipped.")

    base_rays = make_local_rays(viewport_size, hfov, device)
    kernel = gaussian_kernel(device)

    ref_cap = open_video(prepared)
    cbr_cap = open_video(cbr_mp4)
    satc_cap = open_video(satc_mp4)

    # Per-viewer accumulators and detailed rows.
    sums = {
        "cbr_psnr": np.zeros(len(viewers), np.float64),
        "satc_psnr": np.zeros(len(viewers), np.float64),
        "cbr_ssim": np.zeros(len(viewers), np.float64),
        "satc_ssim": np.zeros(len(viewers), np.float64),
        "count": np.zeros(len(viewers), np.int64),
    }
    detail_rows = []

    example_viewer = viewers.index("p004") if "p004" in viewers else 0
    example_frames = {0, max(0, n_eval//2), max(0, n_eval-1)}

    total_needed = WARMUP + n_eval
    t0 = time.time()

    for fid in range(total_needed):
        ok0, fref = ref_cap.read()
        ok1, fcbr = cbr_cap.read()
        ok2, fsatc = satc_cap.read()
        if not (ok0 and ok1 and ok2):
            die(f"Decode ended early at frame {fid}: ref={ok0}, cbr={ok1}, satc={ok2}")

        if fid < WARMUP:
            continue

        j = fid - WARMUP
        valid = valid_all[:, j]
        ids = np.flatnonzero(valid)
        if ids.size == 0:
            continue

        yaw = torch.from_numpy(hx_all[ids, j]).to(device) * (2.0*math.pi) - math.pi
        pitch = torch.from_numpy(hy_all[ids, j]).to(device) * math.pi - (math.pi/2.0)

        grid = make_grid(base_rays, yaw, pitch, H, W)

        tref = frame_to_tensor(fref, device)
        tcbr = frame_to_tensor(fcbr, device)
        tsatc = frame_to_tensor(fsatc, device)

        vref = extract_batch(tref, grid)
        vcbr = extract_batch(tcbr, grid)
        vsatc = extract_batch(tsatc, grid)

        cbr_p = psnr_batch(vref, vcbr)
        satc_p = psnr_batch(vref, vsatc)
        cbr_s = ssim_batch(vref, vcbr, kernel)
        satc_s = ssim_batch(vref, vsatc, kernel)

        cbr_p_np = cbr_p.detach().cpu().numpy()
        satc_p_np = satc_p.detach().cpu().numpy()
        cbr_s_np = cbr_s.detach().cpu().numpy()
        satc_s_np = satc_s.detach().cpu().numpy()

        for k, viewer_idx in enumerate(ids):
            sums["cbr_psnr"][viewer_idx] += float(cbr_p_np[k])
            sums["satc_psnr"][viewer_idx] += float(satc_p_np[k])
            sums["cbr_ssim"][viewer_idx] += float(cbr_s_np[k])
            sums["satc_ssim"][viewer_idx] += float(satc_s_np[k])
            sums["count"][viewer_idx] += 1
            detail_rows.append({
                "viewer": viewers[viewer_idx],
                "encoded_frame_id": int(fid),
                "time_s": float(fid / ENC_FPS),
                "head_x": float(hx_all[viewer_idx, j]),
                "head_y": float(hy_all[viewer_idx, j]),
                "cbr_psnr_db": float(cbr_p_np[k]),
                "satc_psnr_db": float(satc_p_np[k]),
                "delta_psnr_db": float(satc_p_np[k] - cbr_p_np[k]),
                "cbr_ssim": float(cbr_s_np[k]),
                "satc_ssim": float(satc_s_np[k]),
                "delta_ssim": float(satc_s_np[k] - cbr_s_np[k]),
            })

        if j in example_frames and valid[example_viewer]:
            where = int(np.where(ids == example_viewer)[0][0])
            save_triptych(
                root / "examples" / f"{viewers[example_viewer]}_frame_{fid:04d}.png",
                vref[where], vcbr[where], vsatc[where],
                f"{viewers[example_viewer]} | t={fid/ENC_FPS:.3f}s | HFOV={hfov:.1f} deg"
            )

        if (j + 1) % 30 == 0 or j + 1 == n_eval:
            elapsed = time.time() - t0
            fps = (j + 1) / max(elapsed, 1e-9)
            eta = (n_eval - (j + 1)) / max(fps, 1e-9)
            print(f"[eval] {j+1}/{n_eval} frames | {fps:.2f} encoded-frames/s | ETA {eta/60:.1f} min", flush=True)

    ref_cap.release(); cbr_cap.release(); satc_cap.release()

    # Save detailed frame/viewer table.
    detail_csv = root / "VIEWPORT_FRAME_METRICS.csv"
    if detail_rows:
        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader(); w.writerows(detail_rows)

    per_viewer = []
    for i, viewer in enumerate(viewers):
        n = int(sums["count"][i])
        if n == 0:
            continue
        cbr_p = sums["cbr_psnr"][i] / n
        satc_p = sums["satc_psnr"][i] / n
        cbr_s = sums["cbr_ssim"][i] / n
        satc_s = sums["satc_ssim"][i] / n
        per_viewer.append({
            "viewer": viewer,
            "frames": n,
            "cbr_psnr_db": cbr_p,
            "satc_psnr_db": satc_p,
            "delta_psnr_db": satc_p - cbr_p,
            "cbr_ssim": cbr_s,
            "satc_ssim": satc_s,
            "delta_ssim": satc_s - cbr_s,
        })

    pv_csv = root / "VIEWPORT_PER_VIEWER.csv"
    with pv_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_viewer[0].keys()))
        w.writeheader(); w.writerows(per_viewer)

    dpsnr = np.asarray([r["delta_psnr_db"] for r in per_viewer], np.float64)
    dssim = np.asarray([r["delta_ssim"] for r in per_viewer], np.float64)

    summary = {
        "video": video_id,
        "viewers": len(per_viewer),
        "evaluated_encoded_frames": n_eval,
        "encoded_frame_range": [WARMUP, WARMUP + n_eval - 1],
        "time_range_s": [WARMUP/ENC_FPS, (WARMUP+n_eval-1)/ENC_FPS],
        "source_trace_fps": source_fps,
        "encoded_fps": ENC_FPS,
        "viewport": {
            "projection": "rectilinear perspective",
            "center": "recorded head position (not gaze)",
            "horizontal_fov_deg": hfov,
            "vertical_fov_deg": hfov,
            "output_pixels": [viewport_size, viewport_size],
        },
        "metrics": "RGB PSNR and Gaussian-window RGB SSIM",
        "mean_delta_psnr_db_across_viewers": float(dpsnr.mean()),
        "median_delta_psnr_db_across_viewers": float(np.median(dpsnr)),
        "viewers_positive_psnr": int((dpsnr > 0).sum()),
        "viewers_nonpositive_psnr": int((dpsnr <= 0).sum()),
        "mean_delta_ssim_across_viewers": float(dssim.mean()),
        "median_delta_ssim_across_viewers": float(np.median(dssim)),
        "viewers_positive_ssim": int((dssim > 0).sum()),
        "viewers_nonpositive_ssim": int((dssim <= 0).sum()),
        "important_interpretation": (
            "This one-video pilot measures within-video viewer behavior only. "
            "Do not treat viewers as independent content samples for a paper-level "
            "claim of generalization across videos."
        ),
    }

    (root / "VIEWPORT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = (
        f"VR-EyeTracking video {video_id} viewport pilot\n"
        f"Viewers: {summary['viewers']}\n"
        f"Frames evaluated: {n_eval}\n"
        f"HFOV/VFOV: {hfov:.1f}/{hfov:.1f} deg\n"
        f"Viewport: {viewport_size}x{viewport_size}\n\n"
        f"Mean SATC-CBR PSNR delta: {summary['mean_delta_psnr_db_across_viewers']:+.4f} dB\n"
        f"Median SATC-CBR PSNR delta: {summary['median_delta_psnr_db_across_viewers']:+.4f} dB\n"
        f"Viewers with positive PSNR delta: {summary['viewers_positive_psnr']}/{summary['viewers']}\n\n"
        f"Mean SATC-CBR SSIM delta: {summary['mean_delta_ssim_across_viewers']:+.6f}\n"
        f"Median SATC-CBR SSIM delta: {summary['median_delta_ssim_across_viewers']:+.6f}\n"
        f"Viewers with positive SSIM delta: {summary['viewers_positive_ssim']}/{summary['viewers']}\n\n"
        "Interpretation: pilot only; one content item does not establish across-content significance.\n"
    )
    (root / "VIEWPORT_SUMMARY.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"[saved] {pv_csv}")
    print(f"[saved] {detail_csv}")
    print(f"[saved] {root/'VIEWPORT_SUMMARY.json'}")
    print(f"[saved] {root/'examples'}")


def main():
    print("[pilot] RUN_360SATC_VIEWPORT_PILOT_v4", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", default="033",
                    help="VR-EyeTracking numeric video ID, e.g. 033 or 034.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory; default is VIEWPORT_PILOT_<video-id> under Firefox common.")
    ap.add_argument("--viewport-size", type=int, default=256,
                    help="Square perspective viewport resolution used for the pilot.")
    ap.add_argument("--hfov", type=float, default=100.0,
                    help="Pilot horizontal/vertical FOV in degrees. Use a documented dataset/HMD FOV for paper results.")
    ap.add_argument("--eval-frames", type=int, default=896,
                    help="Number of measured 60-FPS frames to evaluate (<=896).")
    ap.add_argument("--max-viewers", type=int, default=0,
                    help="0 = all viewers; positive integer = first N viewers for smoke test.")
    ap.add_argument("--skip-encode", action="store_true",
                    help="Reuse existing encoded streams under the output directory.")
    args = ap.parse_args()

    if not 1 <= args.eval_frames <= MEASURED:
        die(f"--eval-frames must be between 1 and {MEASURED}")
    if args.viewport_size < 64:
        die("--viewport-size must be >=64")
    if not 20.0 <= args.hfov < 170.0:
        die("--hfov must be in [20,170) degrees")

    video_id = str(args.video_id).strip()
    if not video_id.isdigit():
        die("--video-id must be numeric, e.g. 033")
    video_id = video_id.zfill(3)

    source = VIDEO_ROOT / f"{video_id}.mp4"
    root = (args.output if args.output is not None
            else DEFAULT_OUT_PARENT / f"VIEWPORT_PILOT_{video_id}").resolve()
    root.mkdir(parents=True, exist_ok=True)

    for p in (source, MASTER_PATH, FINAL_PATH, BASE_PATH):
        if not p.is_file():
            die(f"Missing required file: {p}")

    source_info = ffprobe_video(source)
    source_fps = ratio_to_float(source_info.get("r_frame_rate", "0/1"))
    if source_fps <= 0:
        source_fps = ratio_to_float(source_info.get("avg_frame_rate", "0/1"))
    if source_fps <= 0:
        die(f"Could not determine source FPS for {source}")

    print(
        f"[source] video={video_id} "
        f"{source_info.get('width')}x{source_info.get('height')} "
        f"fps={source_fps:.6f} frames={source_info.get('nb_frames')} "
        f"duration={source_info.get('duration')}"
    )

    preflight_traces = trace_files(video_id)
    print(f"[trace-preflight] video {video_id}: {len(preflight_traces)} viewer traces")

    prepared = root / "prepared" / f"vreye{video_id}_4096x2048_60fps_qp10.mp4"
    prepare_reference(source, prepared)

    video_key = f"vreye{video_id}"
    if args.skip_encode:
        satc_candidates = list(
            (root / "satc_p7" / "runs").glob(
                f"{video_key}_h264_12_360satc/encoded.*"
            )
        )
        cbr_candidates = list(
            (root / "satc_p7" / "runs").glob(
                f"{video_key}_h264_12_cbr/encoded.*"
            )
        )
        if not satc_candidates or not cbr_candidates:
            die("--skip-encode requested but completed streams were not found.")
        satc_raw, cbr_raw = satc_candidates[0], cbr_candidates[0]
    else:
        satc_raw, cbr_raw = encode_pair(root, prepared, video_id)

    print(f"[SATC] {satc_raw}")
    print(f"[CBR ] {cbr_raw}")

    evaluate(
        root, prepared, satc_raw, cbr_raw,
        video_id=video_id,
        source_fps=source_fps,
        viewport_size=args.viewport_size,
        hfov=args.hfov,
        eval_frames=args.eval_frames,
        max_viewers=args.max_viewers,
    )


if __name__ == "__main__":
    main()
