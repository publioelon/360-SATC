#!/usr/bin/env python3
"""
Batch driver for the VR-EyeTracking viewport validation.

Default behavior:
  - starts at video ID 041
  - discovers numeric MP4s with matching viewer traces
  - selects the next 13 eligible videos
  - prepares exactly 1200 4K60 frames directly from the original source
    in ONE QP10 preparation encode
  - requires at least 1136 source-derived 60-FPS frames so cloned tail
    frames can occur only after the evaluated interval
  - runs the already-tested RUN_360SATC_VIEWPORT_PILOT_v4.py with
    896 evaluated frames and all viewers
  - skips videos that already have a complete 896-frame summary
  - continues to the next video if one video fails
  - writes aggregate CSV/JSON/TXT summaries at the end

The SATC/CBR algorithm and encoding parameters are NOT retuned here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

VIDEO_ROOT = Path("/data/VR-EyeTracking/VR Eye-Tracking Dataset/videos/videos")
TRACE_ROOT = Path("/data/VR-EyeTracking/VR Eye-Tracking Dataset/Gaze_txt_files/Gaze_txt_files")
OUT_PARENT = Path("/home/mininet-ovs/snap/firefox/common")
DEFAULT_PILOT = OUT_PARENT / "RUN_360SATC_VIEWPORT_PILOT_v4.py"

W = 4096
H = 2048
FPS = 60
WARMUP = 240
MEASURED = 896
EVAL_END_EXCLUSIVE = WARMUP + MEASURED  # 1136
REQUIRED_RUNNER_FRAMES = 1200             # includes 64 post-measurement guard frames
MIN_UNIQUE_SECONDS = EVAL_END_EXCLUSIVE / FPS


def run_capture(cmd):
    p = subprocess.run(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return p.stdout


def ffprobe(path: Path) -> dict:
    out = run_capture([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(path),
    ])
    payload = json.loads(out)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream: {path}")
    info = dict(streams[0])
    info["_format_duration"] = (payload.get("format") or {}).get("duration")
    return info


def ratio_to_float(value) -> float:
    s = str(value or "0")
    if "/" in s:
        a, b = s.split("/", 1)
        b = float(b)
        return 0.0 if b == 0 else float(a) / b
    return float(s)


def trace_count(video_id: str) -> int:
    return len(list(TRACE_ROOT.glob(f"p*/{video_id}.*_ori_0.txt")))


def summary_path(video_id: str) -> Path:
    return OUT_PARENT / f"VIEWPORT_PILOT_{video_id}" / "VIEWPORT_SUMMARY.json"


def summary_is_complete(video_id: str) -> bool:
    p = summary_path(video_id)
    if not p.is_file():
        return False
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        return (
            str(s.get("video", "")).zfill(3) == video_id
            and int(s.get("evaluated_encoded_frames", 0)) == MEASURED
            and int(s.get("viewers", 0)) > 0
        )
    except Exception:
        return False


def prepared_path(video_id: str) -> Path:
    return (
        OUT_PARENT
        / f"VIEWPORT_PILOT_{video_id}"
        / "prepared"
        / f"vreye{video_id}_4096x2048_60fps_qp10.mp4"
    )


def prepared_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        s = ffprobe(path)
        fps = ratio_to_float(s.get("avg_frame_rate") or s.get("r_frame_rate"))
        return (
            int(s.get("width", 0)) == W
            and int(s.get("height", 0)) == H
            and abs(fps - FPS) < 1e-6
            and int(s.get("nb_frames", 0)) >= REQUIRED_RUNNER_FRAMES
        )
    except Exception:
        return False


def source_is_eligible(video_id: str, source: Path):
    """Return (eligible, info/reason)."""
    try:
        s = ffprobe(source)
    except Exception as e:
        return False, f"ffprobe failed: {e}"

    fps = ratio_to_float(s.get("avg_frame_rate") or s.get("r_frame_rate"))
    n = int(s.get("nb_frames") or 0)

    # Some VR-EyeTracking MP4 files omit stream-level duration even though the
    # container duration and/or frame count are valid. Use the strongest
    # available duration source in this order:
    #   1) video-stream duration
    #   2) container/format duration
    #   3) nb_frames / fps
    duration = float(s.get("duration") or 0.0)
    if duration <= 0:
        duration = float(s.get("_format_duration") or 0.0)
    if duration <= 0 and n > 0 and fps > 0:
        duration = n / fps

    # Duration is what matters after deterministic 60-FPS resampling.
    # We require the entire evaluated interval (frames 0..1135) to be
    # source-derived. Tail cloning is allowed only in the guard region.
    if duration + 1e-3 < MIN_UNIQUE_SECONDS:
        return False, (
            f"too short or unreadable duration: duration={duration:.3f}s < "
            f"{MIN_UNIQUE_SECONDS:.3f}s required for 1136 source-derived 60-FPS frames"
        )

    tc = trace_count(video_id)
    if tc <= 0:
        return False, "no matching viewer traces"

    return True, {
        "duration": duration,
        "source_fps": fps,
        "source_frames": n,
        "traces": tc,
    }


def prepare_exact_1200(video_id: str, source: Path, dest: Path):
    if prepared_is_valid(dest):
        info = ffprobe(dest)
        print(
            f"[reuse-prepared] {video_id}: "
            f"{info.get('nb_frames')} frames @ {info.get('avg_frame_rate')}"
        )
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".batchtmp.mp4")
    tmp.unlink(missing_ok=True)

    # tpad is intentionally present for short-but-eligible clips. Because
    # eligibility requires >=1136 source-derived frames, any cloned tail can
    # only fall after the evaluated interval. We always stop at frame 1200.
    filt = (
        f"scale={W}:{H}:flags=lanczos,"
        f"fps={FPS},setsar=1,"
        "tpad=stop_mode=clone:stop_duration=2"
    )

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-y",
        "-i", str(source),
        "-map", "0:v:0", "-an",
        "-vf", filt,
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-tune", "hq",
        "-rc", "constqp",
        "-qp", "10",
        "-g", "240",
        "-bf", "0",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-frames:v", str(REQUIRED_RUNNER_FRAMES),
        str(tmp),
    ]

    print(f"[prepare] video {video_id}: creating exactly 1200 frames from original")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg preparation failed for video {video_id}")

    info = ffprobe(tmp)
    got = int(info.get("nb_frames", 0))
    fps = ratio_to_float(info.get("avg_frame_rate") or info.get("r_frame_rate"))
    if (
        int(info.get("width", 0)) != W
        or int(info.get("height", 0)) != H
        or abs(fps - FPS) > 1e-6
        or got != REQUIRED_RUNNER_FRAMES
    ):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Prepared validation failed for {video_id}: "
            f"{info}"
        )

    tmp.replace(dest)
    print(f"[prepared-ok] {video_id}: 4096x2048, 60 FPS, 1200 frames")


def tee_process(cmd, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(str(x) for x in cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            [str(x) for x in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return p.wait()


def discover(start_id: int, count: int):
    candidates = []
    skipped = []

    files = []
    for p in VIDEO_ROOT.glob("*.mp4"):
        try:
            n = int(p.stem)
        except ValueError:
            continue
        if n < start_id:
            continue
        files.append((n, p))
    files.sort()

    for n, p in files:
        vid = f"{n:03d}"

        if summary_is_complete(vid):
            skipped.append((vid, "already complete"))
            continue

        ok, info = source_is_eligible(vid, p)
        if not ok:
            skipped.append((vid, str(info)))
            continue

        candidates.append((vid, p, info))
        if len(candidates) >= count:
            break

    return candidates, skipped


def collect_summaries():
    rows = []
    for d in sorted(OUT_PARENT.glob("VIEWPORT_PILOT_*")):
        p = d / "VIEWPORT_SUMMARY.json"
        if not p.is_file():
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            if int(s.get("evaluated_encoded_frames", 0)) != MEASURED:
                continue
            rows.append({
                "video": str(s.get("video", "")).zfill(3),
                "viewers": int(s.get("viewers", 0)),
                "mean_delta_psnr_db": float(s["mean_delta_psnr_db_across_viewers"]),
                "median_delta_psnr_db": float(s["median_delta_psnr_db_across_viewers"]),
                "viewers_positive_psnr": int(s["viewers_positive_psnr"]),
                "mean_delta_ssim": float(s["mean_delta_ssim_across_viewers"]),
                "median_delta_ssim": float(s["median_delta_ssim_across_viewers"]),
                "viewers_positive_ssim": int(s["viewers_positive_ssim"]),
            })
        except Exception:
            continue

    rows.sort(key=lambda r: int(r["video"]))
    return rows


def save_aggregate(rows, selected, failures):
    batch_dir = OUT_PARENT / "VIEWPORT_BATCH"
    batch_dir.mkdir(parents=True, exist_ok=True)

    csv_path = batch_dir / "VIEWPORT_ALL_VIDEO_SUMMARIES.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    psnr = [r["mean_delta_psnr_db"] for r in rows]
    mean = statistics.mean(psnr) if psnr else float("nan")
    sd = statistics.stdev(psnr) if len(psnr) >= 2 else float("nan")
    se = sd / math.sqrt(len(psnr)) if len(psnr) >= 2 else float("nan")

    payload = {
        "completed_full_videos": len(rows),
        "video_ids": [r["video"] for r in rows],
        "unweighted_mean_of_video_mean_delta_psnr_db": mean,
        "between_video_sd_delta_psnr_db": sd,
        "standard_error_delta_psnr_db": se,
        "total_viewer_video_observations": sum(r["viewers"] for r in rows),
        "total_positive_psnr_viewer_video_observations": sum(
            r["viewers_positive_psnr"] for r in rows
        ),
        "selected_this_batch": selected,
        "failures_this_batch": failures,
        "note": (
            "Video is the independent content unit. Viewer-video counts are descriptive "
            "and must not be treated as independent content samples."
        ),
    }

    json_path = batch_dir / "VIEWPORT_BATCH_SUMMARY.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "360-SATC viewport batch summary",
        f"Completed full videos: {len(rows)}",
        f"Video IDs: {', '.join(r['video'] for r in rows)}",
    ]
    if psnr:
        lines.append(f"Mean of video mean PSNR deltas: {mean:+.4f} dB")
    if len(psnr) >= 2:
        lines.append(f"Between-video SD: {sd:.4f} dB")
        lines.append(f"Standard error: {se:.4f} dB")
    lines += [
        f"Viewer-video observations: {payload['total_viewer_video_observations']}",
        f"Positive PSNR viewer-video observations: "
        f"{payload['total_positive_psnr_viewer_video_observations']}",
        "",
        "Failures this batch:",
    ]
    if failures:
        lines.extend(f"  {k}: {v}" for k, v in failures.items())
    else:
        lines.append("  none")

    txt_path = batch_dir / "VIEWPORT_BATCH_SUMMARY.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("\n".join(lines))
    print(f"[aggregate] {csv_path}")
    print(f"[aggregate] {json_path}")
    print(f"[aggregate] {txt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=13,
                    help="Number of NEW eligible videos to run (default: 13).")
    ap.add_argument("--start-id", type=int, default=41,
                    help="First numeric video ID to consider (default: 41).")
    ap.add_argument("--pilot", type=Path, default=DEFAULT_PILOT,
                    help="Path to RUN_360SATC_VIEWPORT_PILOT_v4.py")
    ap.add_argument("--python", default="/home/mininet-ovs/venvs/satc/bin/python",
                    help="Python interpreter for the tested SATC environment.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only show which videos would be selected.")
    args = ap.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if not args.pilot.is_file():
        raise SystemExit(f"Missing pilot runner: {args.pilot}")
    if not VIDEO_ROOT.is_dir():
        raise SystemExit(f"Missing video root: {VIDEO_ROOT}")
    if not TRACE_ROOT.is_dir():
        raise SystemExit(f"Missing trace root: {TRACE_ROOT}")

    selected, skipped = discover(args.start_id, args.count)

    print("[batch] requested new videos:", args.count)
    print("[batch] selected:", ", ".join(v for v, _, _ in selected) or "(none)")
    if skipped:
        print("[batch] skipped during discovery:")
        for vid, reason in skipped:
            print(f"  {vid}: {reason}")

    if len(selected) < args.count:
        print(
            f"[warn] only {len(selected)} eligible new videos were found "
            f"from ID {args.start_id} onward."
        )

    if args.dry_run:
        print("[batch] eligible selected videos:")
        for vid, src, info in selected:
            print(
                f"  {vid}: traces={info['traces']} "
                f"duration={info['duration']:.3f}s source={src}"
            )
        return

    failures = {}
    chosen_ids = [v for v, _, _ in selected]
    t0 = time.time()

    for i, (vid, source, info) in enumerate(selected, 1):
        print("\n" + "#" * 78)
        print(
            f"[batch {i}/{len(selected)}] VIDEO {vid} | "
            f"traces={info['traces']} | duration={info['duration']:.3f}s"
        )
        print("#" * 78)

        try:
            prep = prepared_path(vid)
            prepare_exact_1200(vid, source, prep)

            log = (
                OUT_PARENT
                / f"VIEWPORT_PILOT_{vid}"
                / f"BATCH_VIDEO_{vid}.log"
            )
            cmd = [
                args.python,
                str(args.pilot),
                "--video-id", vid,
                "--eval-frames", str(MEASURED),
            ]
            rc = tee_process(cmd, log)
            if rc != 0:
                failures[vid] = f"pilot runner exited with code {rc}; see {log}"
                print(f"[FAILED] video {vid}: {failures[vid]}")
                continue

            if not summary_is_complete(vid):
                failures[vid] = "runner exited 0 but no complete 896-frame summary was found"
                print(f"[FAILED] video {vid}: {failures[vid]}")
                continue

            s = json.loads(summary_path(vid).read_text(encoding="utf-8"))
            print(
                f"[DONE] {vid}: "
                f"mean ΔPSNR={s['mean_delta_psnr_db_across_viewers']:+.4f} dB, "
                f"positive={s['viewers_positive_psnr']}/{s['viewers']}"
            )

        except KeyboardInterrupt:
            print("\n[stopped] KeyboardInterrupt. Existing completed videos are preserved.")
            failures[vid] = "interrupted by user"
            break
        except Exception as e:
            failures[vid] = f"{type(e).__name__}: {e}"
            print(f"[FAILED] video {vid}: {failures[vid]}")
            continue

    rows = collect_summaries()
    save_aggregate(rows, chosen_ids, failures)

    elapsed = time.time() - t0
    print(f"[batch elapsed] {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
