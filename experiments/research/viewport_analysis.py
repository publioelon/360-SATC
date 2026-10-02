#!/usr/bin/env python3
"""
Final statistical analysis for the 360-SATC real-viewer viewport experiment.

Primary inferential unit: VIDEO (n = 20 independent content items).
Viewer-video observations are reported descriptively only and are NOT treated
as independent content samples.

Expected completed videos:
033, 034, 035, 037, 038, 039, 040, 041, 042, 043, 044, 045, 046, 047,
048, 049, 050, 051, 052, 053
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except Exception as e:
    raise SystemExit(
        "SciPy is required. Run with /home/mininet-ovs/venvs/satc/bin/python. "
        f"Import error: {e}"
    )

EXPECTED_VIDEOS = [
    "033", "034", "035", "037", "038", "039", "040",
    "041", "042", "043", "044", "045", "046", "047",
    "048", "049", "050", "051", "052", "053",
]

DEFAULT_ROOT = Path("/home/mininet-ovs/snap/firefox/common")
DEFAULT_OUT = DEFAULT_ROOT / "VIEWPORT_BATCH" / "FINAL_20"

MEASURED_FRAMES = 896
ALPHA = 0.05
BOOTSTRAP_N = 100000
BOOTSTRAP_SEED = 3602027


def first_present(d: dict, names):
    for k in names:
        if k in d:
            return d[k]
    raise KeyError(f"None of these fields found: {names}")


def ffloat(d, names):
    return float(first_present(d, names))


def iint(d, names):
    return int(first_present(d, names))


def load_video_summary(root: Path, vid: str) -> dict:
    p = root / f"VIEWPORT_PILOT_{vid}" / "VIEWPORT_SUMMARY.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing summary for video {vid}: {p}")

    s = json.loads(p.read_text(encoding="utf-8"))

    video_field = str(s.get("video", vid)).zfill(3)
    if video_field != vid:
        raise RuntimeError(
            f"Video ID mismatch for {p}: expected {vid}, got {video_field}"
        )

    frames = iint(s, ["evaluated_encoded_frames", "frames_evaluated", "eval_frames"])
    if frames != MEASURED_FRAMES:
        raise RuntimeError(
            f"Video {vid}: expected {MEASURED_FRAMES} evaluated frames, got {frames}"
        )

    # v4 stores viewport geometry inside the nested "viewport" object.
    vp = s.get("viewport") or {}
    if not isinstance(vp, dict):
        raise RuntimeError(f"Video {vid}: invalid 'viewport' object in {p}")

    output_pixels = vp.get("output_pixels")
    if not (
        isinstance(output_pixels, (list, tuple))
        and len(output_pixels) == 2
    ):
        raise RuntimeError(
            f"Video {vid}: viewport.output_pixels is missing or invalid in {p}"
        )

    return {
        "video": vid,
        "viewers": iint(s, ["viewers"]),
        "frames": frames,
        "hfov_deg": float(first_present(
            vp, ["horizontal_fov_deg", "hfov_deg", "hfov"]
        )),
        "vfov_deg": float(first_present(
            vp, ["vertical_fov_deg", "vfov_deg", "vfov"]
        )),
        "viewport_width": int(output_pixels[0]),
        "viewport_height": int(output_pixels[1]),
        "mean_delta_psnr_db": ffloat(
            s, ["mean_delta_psnr_db_across_viewers", "mean_delta_psnr_db"]
        ),
        "median_delta_psnr_db": ffloat(
            s, ["median_delta_psnr_db_across_viewers", "median_delta_psnr_db"]
        ),
        "positive_psnr_viewers": iint(
            s, ["viewers_positive_psnr", "positive_psnr_viewers"]
        ),
        "mean_delta_ssim": ffloat(
            s, ["mean_delta_ssim_across_viewers", "mean_delta_ssim"]
        ),
        "median_delta_ssim": ffloat(
            s, ["median_delta_ssim_across_viewers", "median_delta_ssim"]
        ),
        "positive_ssim_viewers": iint(
            s, ["viewers_positive_ssim", "positive_ssim_viewers"]
        ),
        "summary_path": str(p),
    }


def load_all_viewers(root: Path, vids):
    rows = []
    for vid in vids:
        p = root / f"VIEWPORT_PILOT_{vid}" / "VIEWPORT_PER_VIEWER.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing per-viewer CSV for video {vid}: {p}")
        with p.open("r", newline="", encoding="utf-8") as f:
            local = list(csv.DictReader(f))
        if not local:
            raise RuntimeError(f"Per-viewer CSV is empty for video {vid}: {p}")
        for r in local:
            out = dict(r)
            out["video"] = vid
            rows.append(out)
    return rows


def mean_ci_t(x: np.ndarray, alpha=0.05):
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return mean, sd, se, mean - crit * se, mean + crit * se


def bootstrap_mean_ci(x: np.ndarray, n_boot=100000, seed=3602027):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot, dtype=np.float64)
    chunk = 10000
    pos = 0
    while pos < n_boot:
        m = min(chunk, n_boot - pos)
        idx = rng.integers(0, n, size=(m, n))
        means[pos:pos + m] = x[idx].mean(axis=1)
        pos += m
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def analyze_metric(x: np.ndarray, name: str):
    n = len(x)
    mean, sd, se, ci_lo, ci_hi = mean_ci_t(x, ALPHA)

    t_res = stats.ttest_1samp(x, popmean=0.0, alternative="two-sided")

    try:
        w_res = stats.wilcoxon(
            x,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_stat = float(w_res.statistic)
        wilcoxon_p = float(w_res.pvalue)
    except Exception:
        wilcoxon_stat = float("nan")
        wilcoxon_p = float("nan")

    positive = int(np.sum(x > 0))
    negative = int(np.sum(x < 0))
    zero = int(np.sum(x == 0))
    nonzero = positive + negative

    if nonzero > 0:
        sign_p = float(
            stats.binomtest(
                positive, n=nonzero, p=0.5, alternative="two-sided"
            ).pvalue
        )
    else:
        sign_p = 1.0

    cohen_dz = mean / sd if sd > 0 else float("inf")
    J = 1.0 - 3.0 / (4.0 * n - 5.0)
    hedges_g = J * cohen_dz

    boot_lo, boot_hi = bootstrap_mean_ci(
        x, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED
    )

    try:
        sh = stats.shapiro(x)
        shapiro_w = float(sh.statistic)
        shapiro_p = float(sh.pvalue)
    except Exception:
        shapiro_w = float("nan")
        shapiro_p = float("nan")

    return {
        "metric": name,
        "n_videos": n,
        "mean": mean,
        "median": float(np.median(x)),
        "sd_between_videos": sd,
        "standard_error": se,
        "ci95_t_lower": ci_lo,
        "ci95_t_upper": ci_hi,
        "bootstrap95_mean_lower": boot_lo,
        "bootstrap95_mean_upper": boot_hi,
        "min_video_mean": float(np.min(x)),
        "max_video_mean": float(np.max(x)),
        "positive_videos": positive,
        "negative_videos": negative,
        "zero_videos": zero,
        "t_statistic": float(t_res.statistic),
        "t_df": n - 1,
        "t_p_two_sided": float(t_res.pvalue),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "sign_test_p_two_sided": sign_p,
        "cohen_dz": float(cohen_dz),
        "hedges_g": float(hedges_g),
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
    }


def fmt_p(p):
    if math.isnan(p):
        return "NA"
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    print("[analysis] FINAL_20 v2: reading nested viewport metadata", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    video_rows = [load_video_summary(root, v) for v in EXPECTED_VIDEOS]

    if len(video_rows) != 20:
        raise RuntimeError(f"Expected 20 videos, loaded {len(video_rows)}")

    for field in ["frames", "hfov_deg", "vfov_deg", "viewport_width", "viewport_height"]:
        vals = {r[field] for r in video_rows}
        if len(vals) != 1:
            raise RuntimeError(
                f"Protocol mismatch across videos for {field}: {sorted(vals)}"
            )

    psnr = np.array(
        [r["mean_delta_psnr_db"] for r in video_rows], dtype=np.float64
    )
    ssim = np.array(
        [r["mean_delta_ssim"] for r in video_rows], dtype=np.float64
    )

    psnr_stats = analyze_metric(psnr, "Viewport PSNR delta (dB)")
    ssim_stats = analyze_metric(ssim, "Viewport SSIM delta")

    viewer_rows = load_all_viewers(root, EXPECTED_VIDEOS)
    total_viewers = sum(r["viewers"] for r in video_rows)
    total_pos_psnr = sum(r["positive_psnr_viewers"] for r in video_rows)
    total_pos_ssim = sum(r["positive_ssim_viewers"] for r in video_rows)

    if len(viewer_rows) != total_viewers:
        raise RuntimeError(
            f"Per-viewer row count mismatch: CSVs contain {len(viewer_rows)} rows, "
            f"summaries report {total_viewers}"
        )

    aggregate = {
        "analysis_unit": "video",
        "n_independent_videos": len(video_rows),
        "video_ids": EXPECTED_VIDEOS,
        "frames_per_video": video_rows[0]["frames"],
        "hfov_deg": video_rows[0]["hfov_deg"],
        "vfov_deg": video_rows[0]["vfov_deg"],
        "viewport_width": video_rows[0]["viewport_width"],
        "viewport_height": video_rows[0]["viewport_height"],
        "bootstrap_resamples": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "psnr": psnr_stats,
        "ssim": ssim_stats,
        "descriptive_viewer_video_counts": {
            "viewer_video_observations": total_viewers,
            "positive_psnr": total_pos_psnr,
            "positive_psnr_fraction": total_pos_psnr / total_viewers,
            "positive_ssim": total_pos_ssim,
            "positive_ssim_fraction": total_pos_ssim / total_viewers,
            "warning": (
                "Viewer-video observations are clustered within videos and are "
                "descriptive only; inferential tests use the 20 video-level means."
            ),
        },
    }

    write_csv(out / "FINAL_PER_VIDEO_RESULTS.csv", video_rows)
    write_csv(out / "FINAL_ALL_VIEWER_RESULTS.csv", viewer_rows)

    (out / "FINAL_STATISTICS.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    txt = []
    txt.append("360-SATC FINAL REAL-VIEWER VIEWPORT STATISTICS")
    txt.append("=" * 62)
    txt.append("Primary independent unit: video")
    txt.append(f"Independent videos: {len(video_rows)}")
    txt.append(f"Video IDs: {', '.join(EXPECTED_VIDEOS)}")
    txt.append(
        f"Protocol: {video_rows[0]['frames']} measured frames/video, "
        f"{video_rows[0]['hfov_deg']:.1f}x{video_rows[0]['vfov_deg']:.1f} deg FOV, "
        f"{video_rows[0]['viewport_width']}x{video_rows[0]['viewport_height']} viewport"
    )
    txt.append("")

    def add_metric(label, st, unit=""):
        txt.append(label)
        txt.append("-" * len(label))
        txt.append(f"Mean: {st['mean']:+.6f}{unit}")
        txt.append(f"Median: {st['median']:+.6f}{unit}")
        txt.append(f"Between-video SD: {st['sd_between_videos']:.6f}{unit}")
        txt.append(f"SE: {st['standard_error']:.6f}{unit}")
        txt.append(
            f"95% t CI: [{st['ci95_t_lower']:+.6f}, "
            f"{st['ci95_t_upper']:+.6f}]{unit}"
        )
        txt.append(
            f"95% bootstrap CI for mean: [{st['bootstrap95_mean_lower']:+.6f}, "
            f"{st['bootstrap95_mean_upper']:+.6f}]{unit}"
        )
        txt.append(
            f"Range of video means: [{st['min_video_mean']:+.6f}, "
            f"{st['max_video_mean']:+.6f}]{unit}"
        )
        txt.append(
            f"Positive/negative/zero videos: "
            f"{st['positive_videos']}/{st['negative_videos']}/{st['zero_videos']}"
        )
        txt.append(
            f"One-sample t test vs 0: t({st['t_df']})="
            f"{st['t_statistic']:.6f}, two-sided p={fmt_p(st['t_p_two_sided'])}"
        )
        txt.append(
            f"Wilcoxon signed-rank vs 0: W={st['wilcoxon_statistic']:.6f}, "
            f"two-sided p={fmt_p(st['wilcoxon_p_two_sided'])}"
        )
        txt.append(
            f"Exact sign test vs 50%: two-sided p="
            f"{fmt_p(st['sign_test_p_two_sided'])}"
        )
        txt.append(f"Cohen's dz: {st['cohen_dz']:.6f}")
        txt.append(f"Hedges' g: {st['hedges_g']:.6f}")
        txt.append(
            f"Shapiro-Wilk on video means: W={st['shapiro_w']:.6f}, "
            f"p={fmt_p(st['shapiro_p'])}"
        )
        txt.append("")

    add_metric("VIEWPORT PSNR DELTA (360-SATC - CBR)", psnr_stats, " dB")
    add_metric("VIEWPORT SSIM DELTA (360-SATC - CBR)", ssim_stats, "")

    txt.append("DESCRIPTIVE VIEWER-VIDEO COUNTS")
    txt.append("--------------------------------")
    txt.append(f"Viewer-video observations: {total_viewers}")
    txt.append(
        f"Positive PSNR viewer-video observations: "
        f"{total_pos_psnr}/{total_viewers} "
        f"({100 * total_pos_psnr / total_viewers:.2f}%)"
    )
    txt.append(
        f"Positive SSIM viewer-video observations: "
        f"{total_pos_ssim}/{total_viewers} "
        f"({100 * total_pos_ssim / total_viewers:.2f}%)"
    )
    txt.append(
        "These viewer-video counts are descriptive only. Statistical inference "
        "uses the 20 video-level means."
    )
    txt.append("")
    txt.append("INTERPRETATION")
    txt.append("--------------")
    txt.append(
        "For PSNR, evidence of an average improvement over CBR is supported when "
        "the video-level mean is positive, its 95% CI excludes zero, and the "
        "video-level inferential tests reject the zero-difference null."
    )
    txt.append(
        "This supports the specific claim that 360-SATC improves head-centered "
        "viewport PSNR relative to uniform CBR under the evaluated configuration. "
        "It is not a universal claim that every saliency method or every metric "
        "is superior under all conditions."
    )

    (out / "FINAL_STATISTICS.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8"
    )

    p = psnr_stats
    s = ssim_stats

    latex = (
        "% Auto-generated from the 20 completed VIEWPORT_SUMMARY.json files.\n"
        "% Primary independent unit: video (n=20).\n\n"
        "Across 20 independent 360-degree videos, 360-SATC increased "
        "head-centered viewport PSNR relative to uniform CBR by "
        f"${p['mean']:.3f}$~dB on average "
        f"(95\\% CI: ${p['ci95_t_lower']:.3f}$--${p['ci95_t_upper']:.3f}$~dB; "
        f"$t(19)={p['t_statistic']:.2f}$, $p={p['t_p_two_sided']:.3e}$). "
        "The nonparametric Wilcoxon signed-rank test gave "
        f"$W={p['wilcoxon_statistic']:.0f}$, "
        f"$p={p['wilcoxon_p_two_sided']:.3e}$, and "
        f"{p['positive_videos']} of 20 video-level mean differences were positive "
        f"(exact two-sided sign-test $p={p['sign_test_p_two_sided']:.3e}$). "
        f"The standardized video-level effect was $d_z={p['cohen_dz']:.2f}$. "
        "Head trajectories were used only for offline viewport extraction and "
        "quality evaluation and were not available to the encoder.\n\n"
        f"Viewport SSIM changed by {s['mean']:+.6f} on average "
        f"(95\\% CI: {s['ci95_t_lower']:+.6f} to {s['ci95_t_upper']:+.6f}; "
        f"$t(19)={s['t_statistic']:.2f}$, "
        f"$p={s['t_p_two_sided']:.3e}$).\n"
    )

    (out / "FINAL_LATEX_RESULTS.txt").write_text(latex, encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = np.arange(len(video_rows))
        vals = np.array([r["mean_delta_psnr_db"] for r in video_rows])

        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.bar(xs, vals)
        ax.axhline(0.0, linewidth=1)
        ax.axhline(psnr_stats["mean"], linestyle="--", linewidth=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [r["video"] for r in video_rows], rotation=45, ha="right"
        )
        ax.set_xlabel("VR-EyeTracking video")
        ax.set_ylabel("Viewport PSNR difference (dB)")
        ax.set_title(
            "360-SATC minus uniform CBR: head-centered viewport PSNR"
        )
        ax.text(
            0.99, 0.97,
            f"Mean = {psnr_stats['mean']:+.3f} dB\n"
            f"95% CI [{psnr_stats['ci95_t_lower']:+.3f}, "
            f"{psnr_stats['ci95_t_upper']:+.3f}] dB",
            transform=ax.transAxes,
            va="top",
            ha="right",
        )
        fig.tight_layout()
        fig.savefig(out / "FINAL_PSNR_PER_VIDEO.pdf", bbox_inches="tight")
        fig.savefig(out / "FINAL_PSNR_PER_VIDEO.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warning] plot generation failed: {e}")

    print("\n".join(txt))
    print("")
    print("[saved]", out / "FINAL_PER_VIDEO_RESULTS.csv")
    print("[saved]", out / "FINAL_ALL_VIEWER_RESULTS.csv")
    print("[saved]", out / "FINAL_STATISTICS.json")
    print("[saved]", out / "FINAL_STATISTICS.txt")
    print("[saved]", out / "FINAL_LATEX_RESULTS.txt")
    if (out / "FINAL_PSNR_PER_VIDEO.pdf").is_file():
        print("[saved]", out / "FINAL_PSNR_PER_VIDEO.pdf")
        print("[saved]", out / "FINAL_PSNR_PER_VIDEO.png")


if __name__ == "__main__":
    main()
