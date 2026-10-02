#!/usr/bin/env python3
r"""
360-SATC: SALIENCY-AWARE ALLOCATION USING ESTIMATED DISTORTION
Algorithm, executable Python code, integration and a small validation protocol
Prepared 25 September 2026

This .txt is also a Python program. You can read it as text or execute it
directly with Python; there is no need to extract code or change its extension.
The implementation starts after this report's closing triple quotation mark.

WHAT THIS CAN HONESTLY DELIVER

Keep MoST-Sal, the 5x9 tile grid, standard NVENC encoding and saliency-driven
spatial compression. Replace the fixed three-level allocation with a bounded
allocation that considers both saliency and a cheap estimate of distortion
risk. Overlap the small CPU calculation with the existing GPU inference.

This is an implemented, locally tested research candidate, NOT an established
solution to the measured quality deficit. I cannot truthfully guarantee a
considerable PSNR/SSIM/LPIPS gain, unchanged file size and unchanged real-time
throughput without running the actual encoder on representative videos.
The provided program makes that test concrete and small. It does not fabricate
a win, change the comparison metric, or select the best encoded frame at runtime.

At a fixed bit budget, giving extra quality to one region can reduce quality
elsewhere. If another allocation already minimizes a particular distortion,
no different saliency allocation can be guaranteed to improve that distortion
strictly on every possible input. The same applies more strongly to demanding
simultaneous improvement in four different quality metrics.

1. FINDINGS FROM THE ACTUAL CODE AND LOGS

I inspected the v6 master and the SATC, NVENC and baseline sources inside:
360SATC_ALL_EXPERIMENTS_20260925_000038_UPLOAD(4).zip
I also inspected RUN_360SATC_QP_ABLATION_v2.py, not just the prose handoff.

The existing SATC path:
  causal MoST-Sal -> normalized map -> ERP latitude weighting -> tile means
  -> EMA (0.7 previous + 0.3 current) -> top five tiles + neighbor ring
  -> offsets {-2, 0, +2} -> NVENC CBR.

The source exposes continuous scores through model.ema before quantizing them
into classes. That is the useful information this proposal retains.

For London Tower / H.264 / 12 Mbps, the 896 measured class maps have these
mean TILE-count fractions, not exact pixel-area fractions:
  low importance (+2):    51.4038%
  medium importance (0):  37.4851%
  high importance (-2):   11.1111%
Only about 0.1813% of tiles change label between consecutive measured frames.
Thus rapid class flicker is not an evidenced main explanation for that run.
Roughly half the tiles receive the positive offset; changing that allocation
is a more relevant hypothesis than adding much more temporal smoothing.

The code uses CBR, P7 for quality, a 240-frame GOP, no B-frames, no lookahead,
and disabled spatial/temporal NVENC AQ. The candidate keeps these settings.
The allocation is a policy above the codec, not a new bitstream format.

Two comparison details need to remain explicit:
  * 360-ST here is an adapted four-zone policy, not the original A2C agent.
  * SATC uses block-aligned logical tile boundaries; the baseline source uses
    linspace boundaries. They are both 5x9 but are not exactly the same grid.
    The candidate retains SATC's geometry. A later harmonized-grid experiment
    must give both methods the same grid and disclose the change.

The existing master also computes WS-PSNR inside its expensive LPIPS pass.
Its --skip-lpips path leaves WS-PSNR absent. This program separates a cheap
screening WS calculation from LPIPS, using the same full-resolution RGB
definition and a declared temporal sample interval.

Source fingerprints inspected:
  RUN_360SATC_ALL_EXPERIMENTS_v6.py
  SHA256 58b1d76a126ef3ea11fb7ee6e9b77225ae280c5d8bda63f673bc278c86199880
  SATC P7 live_model.py
  SHA256 495d89c1f04ecd4e50989d6bb2415adb4cacf9fd0c95a6dc4ff5652058078202
  SATC P7 nvenc_uncapped.cpp
  SHA256 d92c6c89509888734cab97c391e28a607a4e363cea00f9f62bfa6348e23218d6

2. THE PROPOSED CHANGE

Use continuous saliency to guide a bounded allocation with a background
quality floor and an explicit, approximate cost model. Retain 45 logical tiles.
Do not add a learned viewport predictor, another neural network, optical flow,
an additional live encode or a live full-reference decoder.

The default candidate is named "guarded" in the code. Its steps are:

  a. Use the previous completed MoST-Sal EMA scores. Keep the existing model
     running on every frame. At frame t, these scores contain information only
     through frame t-1; there is no future input or ground-truth gaze access.

  b. Recover saliency density from the already latitude-weighted tile scores.
     Otherwise multiplying by latitude again would silently count it twice.
     Normalize density by area and bound extreme relative values.

  c. Sample 72x128 positions in the current input luma plane, plus their right
     and lower adjacent pixels. Estimate tile texture using local gradients
     and variance. This creates no floating-point 4K intermediate image.

  d. Form a positive distortion-risk proxy and a rate-share proxy. These are
     deliberately simple, uncalibrated estimates. Texture is not the same as
     reconstruction error or coding rate; the program does not pretend it is.

  e. Consider integer QP deltas {-3,-2,-1,0,+1}. Background gets at most +1
     relative to NVENC's rate-control QP. This is NOT an absolute decoded
     quality guarantee: the encoder may change its base QP.

  f. Solve a small approximate discrete allocation using saliency-weighted
     estimated distortion, an estimated rate budget, and global/spherical
     distortion checks. A small change penalty discourages gratuitous changes.
     Use a neutral map if no non-neutral candidate meets the estimated checks.

  g. Submit exactly one QP map and one source frame to NVENC. Actual bytes and
     decoded quality determine whether the policy is useful, not its estimates.

Saliency remains an explicit nonzero term. A stronger predicted saliency value
increases the value of preserving that tile, other factors being equal.
Unlike the original policy, a costly background tile can also receive bits
when ignoring it would cause a large estimated global-quality penalty.

This could help the particular observed failure pattern: avoid sacrificing
too much background quality while still protecting salient content. It may
also fail if the activity proxy predicts distortion poorly, saliency misses
the useful regions, or NVENC's internal decisions invalidate the proxy.
Those are testable hypotheses, not hidden implementation details.

3. THE MATHEMATICAL MODEL AND ITS LIMITS

For tile i:
  a_i = actual pixel fraction owned by that tile after codec-block mapping.
  l_i = mean ERP cosine-latitude weight for its actual pixels.
  s_i = relative saliency density, normalized to an area-weighted mean of one.
  v_i = normalized, clipped luma-activity proxy, in [0.25, 4].

The default spatial importance is proportional to:
  w_i = [0.5 + 0.5*s_i] * [0.85 + 0.15*l_i / sum_j(a_j*l_j)]
and is normalized so sum_i(a_i*w_i) = 1.

The simple prior models are:
  d_i = v_i^0.65
  r_i = a_i*v_i^0.25 / sum_j(a_j*v_j^0.25)
  R_hat_i(delta) = r_i * exp(-kR*delta)
  D_hat_i(delta) = a_i*d_i * exp(kD*delta)
  kR = ln(2)/6; kD = ln(2)/3.

These exponential functions are local engineering surrogates, not measured
NVENC rate-distortion curves. In particular, the rate slope must not be treated
as a universal codec law. AV1 sensitivity is uncalibrated: the runner requires
--ack-av1-proxy to make an AV1 exploration explicit. That flag does not calibrate
AV1 or establish equivalent QP strength across codecs.

The allocator seeks small sum_i[w_i*D_hat_i(delta_i)] while enforcing:
  sum_i R_hat_i(delta_i) <= 1
  estimated global MSE <= 1.005 * estimated neutral-map global MSE
  estimated spherical MSE <= 1.005 * estimated neutral-map spherical MSE.

The 0.5% predicted MSE allowance is approximately 0.022 dB. It is a declared
surrogate tolerance, not an allowed measured quality loss against 360-ST.

Implementation: three Lagrangian global-distortion weights {0,1,8}, 17 bisection
steps, bounded integer refinement, then explicit constraint checks. Always
include the neutral map. This is approximate discrete optimization, not a proof
of the global optimum of a multiple-choice knapsack problem.

What is actually guaranteed by the local numerical checks:
  * finite valid input, bounded signed integer offsets and correct map geometry;
  * feasibility of the chosen map under the declared estimated budget/checks;
  * at most one allocation job in flight, with no waiting for an unfinished job
    in the live frame path;
  * unchanged NVENC bitrate target and one encode per frame.

What is NOT guaranteed:
  * actual bitstream size from the estimated rate constraint;
  * actual PSNR, SSIM, LPIPS or WS-PSNR from estimated distortion constraints;
  * equal speed on a different CPU/GPU or under thread contention;
  * superiority on unseen videos, even after a development sample passes.

Zero-mean QP offsets would not provide an exact bitrate guarantee either.
NVENC's CBR loop remains responsible for actual rate control. The empirical
acceptance test additionally rejects larger measured payloads AND larger
complete output files. Predicted bits are never substituted for actual bytes.

4. HOW THE SPEED INTEGRATION WORKS

The default --allocation-mode overlap starts one CPU job before current-frame
GPU inference, using the previous completed saliency map and current luma.
After inference, the sender checks whether a completed allocation is ready.
It never waits for an unfinished allocation. If needed it reuses the latest
valid map. No pending job queue is allowed to grow.

Normally this introduces a one-frame saliency lag (about 16.7 ms at 60 fps),
not another frame of display buffering. If saliency information is older than
four frames, the implementation sends a neutral map and records that fallback.
This lag and fallback behavior must be included in the evaluation. Causality
alone does not prove that the lag is harmless for quality.

--allocation-mode serial is available for an ablation using current-frame
saliency. It puts allocation on the frame path and may be slower.
--refresh-frames 2 or 4 optionally reduces allocation frequency; the default
is 1. These options keep MoST-Sal inference at every frame and can increase
spatial allocation staleness. Do not silently change them after validation.

The CPU-only test here used real saved London saliency scores with synthetic
luma, 500 timed calls after warm-up, NumPy 2.3.5 on this x86_64 environment:
  synchronous allocation each frame: mean 0.6466 ms, p95 0.8134 ms;
  synchronous allocation every fourth frame: mean 0.1940 ms per input frame.
These are allocator microbenchmarks, NOT RTX 4060 throughput results and NOT
quality evidence. They motivated overlapping this work with existing inference.
The nonblocking design avoids waiting for allocation; CPU/GIL contention and
NVENC coding decisions can still affect real throughput.

5. EXACT INTEGRATION WITH YOUR V6 HARNESS

The runner extracts the frozen helper into a NEW output directory and changes
two verified Python anchors around LiveModel.infer and qp_from_classes. The
original master, original campaign and original result files are not edited.

The original three-class computation remains available for the control and
for the existing saliency-projection audit. Candidate allocation reads the
continuous EMA scores; it does not derive its QPs from those three labels.

The native defensive guard is changed only to permit [-3,+2]. +2 remains
necessary for the ORIGINAL SATC control; the candidate itself is bounded at
+1. The guard patch is essential: the previous bridge rejected other deltas.
No arbitrary offsets are permitted and the compiled bridge still uses CBR.

For each candidate, a separate allocation log records the exact tile deltas,
expanded-map CRC, source frame, saliency age, predicted constraints and fallback.
The post-run audit compares those CRCs with the maps actually submitted to the
native bridge. The old projection audit is explicitly labeled as an audit of
the old scores/classes, not proof of equality to the candidate QP policy.

No network, Mininet, RT-MPC or RTT experiments are launched by this program.
P7 is the default quality preset; P4 is an explicit throughput option.

6. COMMANDS: FIRST TEST ONLY

Dependencies: your existing Linux v6 setup, NVENC SDK, MoST-Sal ONNX model,
prepared source videos, and /home/mininet-ovs/venvs/satc/bin/python.
The code itself uses NumPy and the Python standard library. Full metrics use
your existing video-metrics environment through the master.

Copy the downloaded file once, if the browser placed it in Downloads:

mkdir -p /home/mininet-ovs/Downloads/files_to_upload
cp -n -- /home/mininet-ovs/Downloads/360SATC_Saliency_Solution.txt /home/mininet-ovs/Downloads/files_to_upload/360SATC_Saliency_Solution.txt

Check the algorithm and the exact source anchors without running experiments:

/home/mininet-ovs/venvs/satc/bin/python /home/mininet-ovs/Downloads/files_to_upload/360SATC_Saliency_Solution.txt --self-test

Measure only allocation cost on your CPU:

/home/mininet-ovs/venvs/satc/bin/python /home/mininet-ovs/Downloads/files_to_upload/360SATC_Saliency_Solution.txt --benchmark

FIRST actual development test: Basketball, H.264, 12 Mbps, one candidate.
It uses 240 warm-up + 120 contiguous measured frames and fresh, identically
windowed original SATC, uniform CBR and 360-ST-adaptation controls:

/home/mininet-ovs/venvs/satc/bin/python /home/mininet-ovs/Downloads/files_to_upload/360SATC_Saliency_Solution.txt --screen --video-name basketball --codec h264 --mbps 12 --frames 120 --candidates guarded --output /home/mininet-ovs/Downloads/SATC_ALLOC_DEV01

This is four short encodes, not a 12/18-run quality sweep. Initial metrics are
PSNR on the entire measured window and full-resolution RGB WS-PSNR every eighth
measured frame. SSIM and LPIPS are deferred. Model initialization, native build,
source decoding and existing full-map audits still have a cost; no wall-time
promise is made. The v6 preflight expects the existing four source videos and
the same head-trace dataset used by the adapted baseline.

The program automatically writes a small diagnostic ZIP under:
/home/mininet-ovs/Downloads/files_to_upload/SATC_ALLOC_DEV01_RESULTS.zip
Raw streams remain in the experiment directory. Upload the diagnostic ZIP to
inspect actual results. If a native/API/preflight error occurs, it fails with
logs rather than silently changing codecs, presets, providers or criteria.

The new Basketball quality window has no completed Experiment-2 baseline in
the handoff, so fresh matched controls are needed. This does not rerun the
completed London campaign. For additional candidates on this exact window,
one call with --candidates guarded gentle saliency_only shares these controls.
Do not start that broader call until the first result has been inspected.
Output directories must be new; this script does not implement resumable runs.

7. DEVELOPMENT, PROMOTION AND HELD-OUT CONFIRMATION

Predeclare the sequence and retain failures. A 120-frame sample is suitable
for rejecting an obviously bad candidate; it is not sufficient to establish
generalization or even performance throughout the 240-frame GOP.

Suggested sequence:
  1. Run the single short Basketball case above. Inspect encoding correctness,
     actual bytes, decoded PSNR/WS-PSNR, allocation activity and saliency age.
  2. If promising, test the SAME candidate on short Rollercoaster and London
     diagnostic windows. London has already been inspected and is development
     material, not a pristine holdout. Fix easy correctness errors before any
     further search. Do not respond to every miss by trying more random QPs.
  3. For at most two surviving configurations, extend development windows to
     at least 480 measured frames (two GOPs), all codec/rate conditions, with
     --full-metrics. AV1 additionally needs codec-specific proxy validation.
  4. Lock one complete configuration, including weights, cadence, lag policy,
     encoder settings, source ranges and metric definitions.
  5. Confirm on Ballet and preferably additional previously uninspected videos.
     Reserve these before looking at their candidate quality. Run at least the
     original 896-frame window, reporting all methods and every failed case.
  6. Run paired original/candidate live P4 throughput measurements, preferably
     three repeats with alternating order and 1200 measured frames per repeat.
     Compare sustained FPS and latency distributions, not just allocator time.

Full metrics for a development finalist, for example:

/home/mininet-ovs/venvs/satc/bin/python /home/mininet-ovs/Downloads/files_to_upload/360SATC_Saliency_Solution.txt --screen --video-name basketball --codec h264 --mbps 12 --frames 480 --full-metrics --candidates guarded --output /home/mininet-ovs/Downloads/SATC_ALLOC_DEV_LONG01

The code's first-screen criteria require:
  * PSNR and sampled WS-PSNR both above the same-window 360-ST adaptation;
  * measured payload no larger than either original SATC or 360-ST;
  * complete file no larger than either control;
  * non-neutral allocation on at least 10% of measured frames;
  * short-run processing FPS within 1% of original SATC.

The 10% activity threshold is only a basic check against collapsing to uniform
CBR. It is not proof of saliency's contribution. Report full activity and add
a saliency-disabled ablation before attributing any gain specifically to saliency.
The 1% speed tolerance is an exploratory noise allowance, not a promise of
zero slowdown. Final no-speed-sacrifice claims need the paired live timing.

With --full-metrics, the criteria also require SSIM above and LPIPS below the
360-ST adaptation, plus a predeclared substantial-gain target: either at least
+0.25 dB PSNR or at least 5% lower LPIPS versus that comparator. These are goals,
NOT forecasts. Primary quality metrics remain the measured existing metrics.

Even if all checks pass, the result label is only PROMISING_SINGLE_WINDOW.
The script never announces a publication-ready win or deploys a new policy.
If it reports DO_NOT_PROMOTE, retain the result. A small surrogate improvement
or a nicer saliency-weighted score is not sufficient to override a measured loss.

For paper claims involving MUC/AUC/PC, evaluate them on the final locked windows
as well. This first runner focuses on the requested 360-ST deficit.

8. WHAT HAS BEEN TESTED HERE

PASS: 450 randomized allocation cases over all three codec map geometries,
checking finite input, integer QP bounds, surrogate rate/global/spherical checks,
neutral behavior on flat input, saliency responsiveness and block-map expansion.

PASS: nonblocking delayed-worker behavior, one-frame causal saliency age and
fallback when the available decision becomes too old.

PASS: the exact v6 source anchors compile as Python; the extracted native source
retains P7/CBR/GOP/AQ/lookahead settings with only the intended QP guard extension.
The geometry is checked independently against the frozen core.py implementation.

PASS: sparse WS sampling on independently constructed lossless RGB videos.
Requested frame IDs 2,5,8 produced the independently expected mean of
33.40070351172823 dB, with zero numerical discrepancy in that test.

NOT TESTED HERE: native CUDA compilation, TensorRT inference, real-video NVENC
quality, real-time end-to-end throughput, or actual bitrate/file-size gains for
the new candidate. This environment has neither your GPU nor your video/model
assets. The CPU tests do not establish any of those results.

9. REFERENCES AND IMPLEMENTATION BASIS

The supplied handoff and the inspected experiment sources are the basis for
the configuration, numerical results and integration points above. The new
allocation formula, solver, overlap integration and screening code are a
proposed implementation, not an asserted reproduction of a published method.

NVIDIA's NVENC programming guide documents CBR target-rate configuration,
QP-map-related features and optional encoded block statistics. Actual output
still needs measurement; this proposal does not use iterative re-encoding or
emphasis mode to obtain its quality gain:
https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/

NVIDIA's AQ discussion explains that perceptual bit redistribution can reduce
PSNR. It does not establish that this candidate improves LPIPS or any other
metric, and it is not a justification for changing the reported objective:
https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html

Read the code below as a concrete candidate to evaluate. Its estimated-budget
constraints are useful engineering safeguards, not a substitute for actual
decoded comparisons against 360-ST.

"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VERSION = "360SATC_ALLOCATION_20260925_v1"
W, H, FPS = 4096, 2048, 60
DEFAULT_MASTER = Path('/home/mininet-ovs/Downloads/files_to_upload/RUN_360SATC_ALL_EXPERIMENTS_v6.py')
DEFAULT_EXISTING = Path('/home/mininet-ovs/Downloads/360SATC_ALL_EXPERIMENTS_20260925_000038')
DEFAULT_LONDON = Path('/home/mininet-ovs/Videos/360_london_tower_60fps_120s.mp4')
EXT = {'h264': 'h264', 'hevc': 'hevc', 'av1': 'ivf'}


def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_py(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load Python module: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Settings:
    saliency_strength: float = 0.5
    sphere_strength: float = 0.15
    distortion_strength: float = 0.65
    rate_activity_strength: float = 0.25
    min_delta: int = -3
    max_delta: int = 1
    rate_slope: float = math.log(2.0) / 6.0
    distortion_slope: float = math.log(2.0) / 3.0
    global_proxy_limit: float = 1.005
    spherical_proxy_limit: float = 1.005
    smoothness_penalty: float = 0.001
    refresh_frames: int = 1
    sample_height: int = 72
    sample_width: int = 128

    def validate(self):
        if not (0 <= self.saliency_strength <= 1 and 0 <= self.sphere_strength <= 1):
            raise ValueError('Saliency and sphere strengths must be in [0,1].')
        if not (0 <= self.distortion_strength <= 1 and 0 <= self.rate_activity_strength <= 1):
            raise ValueError('Activity strengths must be in [0,1].')
        if not (-3 <= self.min_delta <= 0 <= self.max_delta <= 1):
            raise ValueError('Candidate offsets must stay within [-3,+1].')
        if not (self.rate_slope > 0 and self.distortion_slope > 0):
            raise ValueError('Surrogate slopes must be positive.')
        if min(self.global_proxy_limit, self.spherical_proxy_limit) < 1:
            raise ValueError('The neutral map must remain feasible.')
        if self.smoothness_penalty < 0 or self.refresh_frames not in (1, 2, 4):
            raise ValueError('Invalid smoothing or refresh interval.')
        if self.sample_height < 10 or self.sample_width < 18:
            raise ValueError('Feature sampling is too sparse.')


def candidate_settings(name, refresh=1):
    base = Settings(refresh_frames=refresh)
    if name == 'guarded':
        return base
    if name == 'saliency_only':
        return Settings(distortion_strength=0.0, rate_activity_strength=0.0,
                        min_delta=-2, refresh_frames=refresh)
    if name == 'gentle':
        return Settings(saliency_strength=0.35, distortion_strength=0.4,
                        min_delta=-2, refresh_frames=refresh)
    raise ValueError(f'Unknown candidate {name!r}')


class Geometry:
    """Exactly matches frozen SATC's aligned 5x9 boundaries and block centers."""
    def __init__(self, width=W, height=H, codec='h264', sample=(72, 128)):
        if codec not in EXT:
            raise ValueError(codec)
        self.width, self.height, self.codec = int(width), int(height), codec
        tw = max(16, (width // 9 // 16) * 16)
        th = max(16, (height // 5 // 16) * 16)
        if tw * 8 >= width or th * 4 >= height or width % 2 or height % 2:
            raise ValueError('Unsupported frame geometry.')
        self.xe = np.array([i * tw for i in range(9)] + [width])
        self.ye = np.array([i * th for i in range(5)] + [height])
        block = {'h264': 16, 'hevc': 32, 'av1': 64}[codec]
        x = np.minimum(np.arange((width + block - 1) // block) * block + block // 2, width - 1)
        y = np.minimum(np.arange((height + block - 1) // block) * block + block // 2, height - 1)
        tx = np.searchsorted(self.xe[1:], x, side='right')
        ty = np.searchsorted(self.ye[1:], y, side='right')
        self.block_ids = np.ascontiguousarray(ty[:, None] * 9 + tx[None, :])
        px = np.repeat(tx, block)[:width]
        py = np.repeat(ty, block)[:height]
        nx = np.bincount(px, minlength=9)
        ny = np.bincount(py, minlength=5)
        self.area = np.outer(ny, nx).ravel().astype(float)
        self.area /= self.area.sum()
        lat = np.sin(np.pi * (np.arange(height) + 0.5) / height)
        # Mean latitude weight over actual codec-block ownership, for distortion.
        row_lat = np.bincount(py, weights=lat, minlength=5) / ny
        self.latitude = np.repeat(row_lat, 9)
        self.sphere_area = self.area * self.latitude
        self.sphere_area /= self.sphere_area.sum()
        # Existing model scores were averaged on LOGICAL boundaries. Undo that
        # latitude factor before applying the explicitly chosen objective weight.
        self.score_latitude = np.repeat([lat[a:b].mean() for a, b in zip(self.ye[:-1], self.ye[1:])], 9)
        sy, sx = sample
        self.sample_y = np.minimum(((np.arange(sy) + 0.5) * height / sy).astype(int), height - 2)
        self.sample_x = np.minimum(((np.arange(sx) + 0.5) * width / sx).astype(int), width - 2)
        self.sample_ids = (py[self.sample_y, None] * 9 + px[self.sample_x][None, :]).ravel()
        self.counts = np.bincount(self.sample_ids, minlength=45)
        if (self.counts == 0).any():
            raise ValueError('Every tile needs feature samples.')

    def means(self, values):
        return np.bincount(self.sample_ids, weights=np.asarray(values).ravel(), minlength=45) / self.counts

    def expand(self, q):
        q = np.asarray(q)
        if q.size != 45 or not np.isfinite(q).all() or not np.equal(q, np.rint(q)).all():
            raise ValueError('Expected 45 finite integer QP offsets.')
        if q.min() < -127 or q.max() > 127:
            raise ValueError('QP offsets do not fit signed bytes.')
        return np.ascontiguousarray(q.reshape(-1)[self.block_ids], dtype=np.int8)

    def activity(self, raw):
        # No full-frame float allocation, blur, OpenCV, optical flow or GPU copy.
        y = np.frombuffer(raw, np.uint8, self.width * self.height).reshape(self.height, self.width)
        iy, ix = self.sample_y[:, None], self.sample_x[None, :]
        a = y[iy, ix].astype(np.float64) / 255.0
        dx = y[iy, ix + 1].astype(np.float64) / 255.0 - a
        dy = y[iy + 1, ix].astype(np.float64) / 255.0 - a
        mean = self.means(a)
        variance = np.maximum(0.0, self.means(a * a) - mean * mean)
        detail = self.means(dx * dx + dy * dy)
        # A deliberately bounded proxy, NOT measured reconstruction distortion.
        activity = 1e-4 + detail + 0.05 * variance
        return np.clip(activity / np.dot(self.area, activity), 0.25, 4.0)


class Allocator:
    """Conservative saliency-aware discrete allocation under a SURROGATE budget."""
    def __init__(self, geometry, settings=Settings(), record=True):
        settings.validate()
        self.g, self.cfg, self.record = geometry, settings, record
        self.levels = np.arange(settings.min_delta, settings.max_delta + 1, dtype=np.int8)
        self.rf = np.exp(-settings.rate_slope * self.levels)
        self.df = np.exp(settings.distortion_slope * self.levels)
        self.previous = np.zeros(45, np.int8)
        self.last_map = None
        self.records = []
        self.last_report = {}
        self.next_frame = 0

    def allocate(self, smoothed_scores, activity):
        c, g = self.cfg, self.g
        s = np.asarray(smoothed_scores, float).reshape(-1)
        v = np.asarray(activity, float).reshape(-1)
        if s.shape != (45,) or v.shape != (45,) or not np.isfinite(s).all() or not np.isfinite(v).all():
            raise ValueError('Invalid saliency/activity; refusing to silently replace input.')
        if (s < -1e-9).any() or (v <= 0).any():
            raise ValueError('Saliency must be nonnegative and activity positive.')
        density = np.maximum(s, 0.0) / g.score_latitude
        mean = float(np.dot(g.area, density))
        relative = np.ones(45) if mean <= 1e-12 else np.clip(density / mean, 0.25, 4.0)
        relative /= np.dot(g.area, relative)
        sal = (1 - c.saliency_strength) + c.saliency_strength * relative
        sphere = (1 - c.sphere_strength) + c.sphere_strength * g.latitude / np.dot(g.area, g.latitude)
        weight = sal * sphere
        weight /= np.dot(g.area, weight)
        d0 = np.power(np.clip(v, 0.25, 4.0), c.distortion_strength)
        r0 = g.area * np.power(np.clip(v, 0.25, 4.0), c.rate_activity_strength)
        r0 /= r0.sum()
        R = r0[:, None] * self.rf[None, :]
        Dglobal = g.area[:, None] * d0[:, None] * self.df[None, :]
        Dsphere = g.sphere_area[:, None] * d0[:, None] * self.df[None, :]
        Dsal = Dglobal * weight[:, None]
        base_global = float(np.dot(g.area, d0))
        base_sphere = float(np.dot(g.sphere_area, d0))
        base_sal = float(np.dot(g.area * weight, d0))
        # Three Lagrangian weights explore solutions closer to global quality.
        # Finite discrete optimization is approximate; no global-optimum claim.
        mu = np.array([0.0, 1.0, 8.0])
        cost = Dsal[None, :, :] / base_sal + mu[:, None, None] * Dglobal[None, :, :] / base_global
        cost += c.smoothness_penalty * g.area[None, :, None] * np.square(
            self.levels[None, None, :] - self.previous[None, :, None])
        row = np.arange(45)
        lo = np.zeros(3)
        hi = np.full(3, 256.0)
        indices = np.argmin(cost + hi[:, None, None] * R, axis=2)
        if np.any(R[row[None, :], indices].sum(axis=1) > 1 + 1e-12):
            raise RuntimeError('Failed to bracket surrogate budget.')
        for _ in range(17):
            mid = (lo + hi) * 0.5
            idx = np.argmin(cost + mid[:, None, None] * R, axis=2)
            too_large = R[row[None, :], idx].sum(axis=1) > 1 + 1e-12
            lo = np.where(too_large, mid, lo)
            hi = np.where(too_large, hi, mid)
        indices = np.argmin(cost + hi[:, None, None] * R, axis=2)
        # Resolve discrete ties and spend some residual surrogate budget. A dual
        # search alone can jump between many tied tiles and waste available rate.
        # This is bounded greedy refinement, not an exact knapsack solver.
        for k in range(len(indices)):
            idx = indices[k].copy()
            for _ in range(len(self.levels) - 1):
                eligible = np.flatnonzero(idx > 0)
                if not len(eligible):
                    break
                dr = R[eligible, idx[eligible] - 1] - R[eligible, idx[eligible]]
                benefit = cost[k, eligible, idx[eligible]] - cost[k, eligible, idx[eligible] - 1]
                order = np.argsort(-(benefit / dr), kind='stable')
                remaining = 1.0 - float(R[row, idx].sum())
                changed = False
                for j in order:
                    if benefit[j] > 0 and dr[j] <= remaining + 1e-12:
                        idx[eligible[j]] -= 1
                        remaining -= float(dr[j])
                        changed = True
                if not changed:
                    break
            indices[k] = idx
        candidates = [np.zeros(45, np.int8)] + [self.levels[x] for x in indices]
        best = candidates[0]
        best_objective = base_sal
        best_report = {'predicted_rate_ratio': 1.0, 'predicted_global_mse_ratio': 1.0,
                       'predicted_spherical_mse_ratio': 1.0, 'predicted_saliency_mse_ratio': 1.0}
        for q in candidates[1:]:
            ii = q.astype(int) - c.min_delta
            r = float(R[row, ii].sum())
            dg = float(Dglobal[row, ii].sum())
            ds = float(Dsphere[row, ii].sum())
            dw = float(Dsal[row, ii].sum())
            if (r <= 1 + 1e-12 and dg <= c.global_proxy_limit * base_global
                    and ds <= c.spherical_proxy_limit * base_sphere and dw < best_objective - 1e-10):
                best, best_objective = q.copy(), dw
                best_report = {'predicted_rate_ratio': r, 'predicted_global_mse_ratio': dg / base_global,
                               'predicted_spherical_mse_ratio': ds / base_sphere,
                               'predicted_saliency_mse_ratio': dw / base_sal}
        self.previous = best.copy()
        best_report['neutral_fallback'] = bool(not np.any(best))
        best_report['nonzero_tile_fraction'] = float(np.mean(best != 0))
        return best, best_report

    def make_map(self, scores, raw, codec, frame_id):
        if codec != self.g.codec or frame_id != self.next_frame:
            raise ValueError('Codec/frame sequence mismatch. Create a new allocator for each run.')
        start = time.perf_counter_ns()
        refreshed = self.last_map is None or frame_id % self.cfg.refresh_frames == 0
        if refreshed:
            activity = self.g.activity(raw)
            q, self.last_report = self.allocate(scores, activity)
            self.last_map = self.g.expand(q)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if self.record:
            self.records.append({'frame_id': frame_id, 'allocation_ms': elapsed, 'refreshed': refreshed,
                                 'qp_crc32': zlib.crc32(self.last_map.tobytes()),
                                 'tile_deltas_hex': self.previous.tobytes().hex(), **self.last_report})
        self.next_frame += 1
        # Maps are never modified in-place. Queued frames can retain this array.
        return self.last_map

    def audit(self, frame_csv, out, warmup):
        sent = read_csv(frame_csv)
        if len(sent) != len(self.records):
            raise RuntimeError('Allocation audit missing frame records.')
        for a, b in zip(sent, self.records):
            if int(a['frame_id']) != b['frame_id'] or int(a['qp_crc32']) != b['qp_crc32']:
                raise RuntimeError('Transmitted QP map did not match allocation log.')
            q = np.frombuffer(bytes.fromhex(b['tile_deltas_hex']), dtype=np.int8)
            if q.size != 45 or q.min() < self.cfg.min_delta or q.max() > self.cfg.max_delta:
                raise RuntimeError('Recorded tile deltas violate policy bounds.')
            expanded = self.g.expand(q)
            if zlib.crc32(expanded.tobytes()) != b['qp_crc32'] or int(a['qp_bytes']) != expanded.nbytes:
                raise RuntimeError('Recorded tile deltas do not expand to the submitted QP map.')
            if not (b['predicted_rate_ratio'] <= 1 + 1e-12
                    and b['predicted_global_mse_ratio'] <= self.cfg.global_proxy_limit + 1e-12
                    and b['predicted_spherical_mse_ratio'] <= self.cfg.spherical_proxy_limit + 1e-12):
                raise RuntimeError('Recorded surrogate constraint violation.')
        measured = self.records[warmup:]
        timing = np.array([r['allocation_ms'] for r in measured])
        report = {'status': 'PASS', 'scope': 'submitted QP CRC/sequence, bounds and surrogate constraints only',
                  'frames': len(sent), 'settings': asdict(self.cfg),
                  'allocation_mean_ms': float(timing.mean()), 'allocation_p95_ms': float(np.percentile(timing, 95)),
                  'active_frame_fraction': float(np.mean([not r['neutral_fallback'] for r in measured])),
                  'mean_nonzero_tile_fraction': float(np.mean([r['nonzero_tile_fraction'] for r in measured]))}
        write_csv(Path(out) / 'allocation_frames.csv', self.records)
        dump(Path(out) / 'ALLOCATION_AUDIT.json', report)
        return report


class OverlappedAllocator(Allocator):
    """One CPU worker, one in-flight job, no wait in the video frame path.

    Allocation uses previous completed MoST-Sal scores and current luma. It
    overlaps current-frame GPU inference. This is a causal one-frame saliency
    lag, not zero-lag current-map allocation. Stale fallback events are logged.
    """
    def __init__(self, geometry, settings=Settings(), max_age_frames=4):
        super().__init__(geometry, settings)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='satc-allocation')
        self.future = None
        self.job_frame = None
        self.score_frame = -1
        self.zero_map = geometry.expand(np.zeros(45, np.int8))
        self.ready_map = self.zero_map
        self.ready_q = np.zeros(45, np.int8)
        self.ready_report = {'predicted_rate_ratio': 1.0, 'predicted_global_mse_ratio': 1.0,
                             'predicted_spherical_mse_ratio': 1.0, 'predicted_saliency_mse_ratio': 1.0,
                             'neutral_fallback': True, 'nonzero_tile_fraction': 0.0}
        self.max_age_frames = max_age_frames
        self.work_times = []
        self.prepared_frame = -1

    def _compute(self, scores, raw, fid):
        started = time.perf_counter_ns()
        q, report = self.allocate(scores, self.g.activity(raw))
        qm = self.g.expand(q)
        return fid, q, qm, report, (time.perf_counter_ns() - started) / 1e6

    def _collect(self):
        if self.future is not None and self.future.done():
            # done() is checked first: result() never waits for unfinished work.
            fid, q, qm, report, ms = self.future.result()
            self.ready_q, self.ready_map, self.ready_report = q, qm, report
            self.score_frame = fid - 1
            self.work_times.append(ms)
            self.future = None

    def prepare(self, previous_scores, raw, codec, frame_id):
        if codec != self.g.codec or frame_id != self.next_frame:
            raise ValueError('Allocation preparation frame mismatch.')
        self.prepared_frame = frame_id
        self._collect()
        if previous_scores is not None and self.future is None and frame_id % self.cfg.refresh_frames == 0:
            # A bytes reference keeps current raw input alive; no full-frame copy.
            self.future = self.worker.submit(self._compute, np.asarray(previous_scores).copy(), raw, frame_id)
            self.job_frame = frame_id

    def make_map(self, current_scores, raw, codec, frame_id):
        started = time.perf_counter_ns()
        if codec != self.g.codec or frame_id != self.next_frame or frame_id != self.prepared_frame:
            raise ValueError('Overlapped allocator must be prepared before inference for every frame.')
        self._collect()
        age = frame_id - self.score_frame if self.score_frame >= 0 else None
        stale = age is None or age > self.max_age_frames
        if stale:
            qm, q = self.zero_map, np.zeros(45, np.int8)
            report = {'predicted_rate_ratio': 1.0, 'predicted_global_mse_ratio': 1.0,
                      'predicted_spherical_mse_ratio': 1.0, 'predicted_saliency_mse_ratio': 1.0,
                      'neutral_fallback': True, 'nonzero_tile_fraction': 0.0}
        else:
            qm, q, report = self.ready_map, self.ready_q, self.ready_report
        self.records.append({'frame_id': frame_id, 'allocation_ms': (time.perf_counter_ns() - started) / 1e6,
                             'refreshed': self.score_frame == frame_id - 1,
                             'qp_crc32': zlib.crc32(qm.tobytes()), 'tile_deltas_hex': q.tobytes().hex(),
                             'saliency_source_frame': self.score_frame, 'saliency_age_frames': age,
                             'stale_fallback': stale, **report})
        self.next_frame += 1
        return qm

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        if self.future is not None and not self.future.cancelled():
            self._collect()

    def audit(self, frame_csv, out, warmup):
        report = super().audit(frame_csv, out, warmup)
        measured = self.records[warmup:]
        ages = [r['saliency_age_frames'] for r in measured if r['saliency_age_frames'] is not None]
        report.update({'allocation_mode': 'CPU worker overlaps live GPU inference',
                       'worker_completed_jobs': len(self.work_times),
                       'worker_mean_ms': float(np.mean(self.work_times)) if self.work_times else None,
                       'saliency_age_p95_frames': float(np.percentile(ages, 95)) if ages else None,
                       'stale_fallback_fraction': float(np.mean([r['stale_fallback'] for r in measured])),
                       'allocation_mean_ms_note': 'finish/poll path only; use live throughput to assess total overhead'})
        dump(Path(out) / 'ALLOCATION_AUDIT.json', report)
        return report


ANCHOR = '                    qp = qp_from_classes(classes, codec)'
REPLACEMENT = '''                    qp = (ALLOCATION_POLICY.make_map(model.ema, raw, codec, fid)
                          if ALLOCATION_POLICY is not None else qp_from_classes(classes, codec))'''
PREPARE_ANCHOR = '                    classes, _, stamps = model.infer([history[i] for i in ids], ids)'
PREPARE_REPLACEMENT = '''                    if ALLOCATION_POLICY is not None and hasattr(ALLOCATION_POLICY, "prepare"):
                        ALLOCATION_POLICY.prepare(model.ema, raw, codec, fid)
                    classes, _, stamps = model.infer([history[i] for i in ids], ids)'''
GUARD = 'for (auto v:qp) require(v==-2 || v==0 || v==2,"unexpected QP delta");'


def patched_satc_source(source, preset):
    if source.count(ANCHOR) != 1 or source.count(PREPARE_ANCHOR) != 1:
        raise RuntimeError('Frozen SATC source changed: allocation anchor must occur exactly once.')
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    source = source.replace(PREPARE_ANCHOR, PREPARE_REPLACEMENT, 1)
    source = source.replace('"preset": "p4"', f'"preset": "{preset}"')
    source = source.replace(': p4, {method}', f': {preset}, {{method}}')
    return source + '\nALLOCATION_POLICY = None\n'


def install_guard(satc, preset):
    original = satc.materialize
    def materialize(code):
        original(code)
        p = Path(code) / 'nvenc_uncapped.cpp'
        src = p.read_text(encoding='utf-8')
        if src.count(GUARD) != 1:
            raise RuntimeError('Native QP validator changed; refusing an ambiguous patch.')
        src = src.replace(GUARD, 'for (auto v:qp) require(v>=-3 && v<=2,"unexpected allocation QP delta");', 1)
        src = src.replace('preset=p4', 'preset=' + preset)
        p.write_text(src, encoding='utf-8')
    satc.materialize = materialize


def byte_stats(path, warmup, frames):
    rows = read_csv(path)
    wanted = [r for r in rows if warmup <= int(r['frame_id']) < warmup + frames]
    if [int(r['frame_id']) for r in wanted] != list(range(warmup, warmup + frames)):
        raise RuntimeError('Incorrect measured frame range for byte comparison.')
    total = sum(int(r['encoded_bytes']) for r in wanted)
    return {'encoded_bytes': total, 'bitrate_mbps': total * 8 * FPS / frames / 1e6}


def sampled_ws_psnr(source, encoded, warmup, frames, step, out):
    """Matches the master's full-resolution RGB WS formula, without LPIPS.

    Sparse sampling is for rejection only. This is not Y-WS-PSNR, not a
    downsampled metric, and not comparable to the old 896-frame mean unless
    warmup/frames/step also match. Both streams are decoded continuously.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    end = warmup + frames
    filt = (f'[0:v]trim=start_frame={warmup}:end_frame={end},setpts=PTS-STARTPTS,'
            f"select='not(mod(n,{step}))'[r];"
            f'[1:v]trim=start_frame={warmup}:end_frame={end},setpts=PTS-STARTPTS,'
            f"select='not(mod(n,{step}))'[d];[r][d]hstack=inputs=2,format=rgb24")
    n = (frames + step - 1) // step
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-v', 'error', '-i', str(source), '-i', str(encoded),
           '-filter_complex', filt, '-frames:v', str(n), '-fps_mode', 'passthrough',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    weights = np.sin(np.pi * (np.arange(H) + 0.5) / H)
    weights /= weights.sum()
    values = []
    with (out / 'ws_decode.log').open('wb') as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, bufsize=0)
        try:
            for i in range(n):
                raw = bytearray(H * W * 2 * 3)
                view, pos = memoryview(raw), 0
                while pos < len(raw):
                    got = p.stdout.readinto(view[pos:])
                    if not got:
                        raise RuntimeError(f'WS metric EOF at sample {i}; see {out / "ws_decode.log"}')
                    pos += got
                a = np.frombuffer(raw, np.uint8).reshape(H, W * 2, 3)
                # Stripe evaluation bounds peak CPU memory; numerical definition is unchanged.
                row_mse = np.empty(H, dtype=np.float64)
                for y in range(0, H, 64):
                    diff = a[y:y+64, :W].astype(np.float32) - a[y:y+64, W:].astype(np.float32)
                    row_mse[y:y+64] = np.mean(diff * diff, axis=(1, 2), dtype=np.float64)
                mse = float(np.dot(weights, row_mse))
                score = 99.0 if mse <= 1e-12 else 10 * math.log10(255.0 ** 2 / mse)
                values.append({'frame_id': warmup + i * step, 'rgb_ws_psnr_db': score})
            if p.wait(timeout=60):
                raise RuntimeError('FFmpeg WS decoding failed.')
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            if p.stdout:
                p.stdout.close()
    write_csv(out / 'ws_samples.csv', values)
    return float(np.mean([x['rgb_ws_psnr_db'] for x in values]))


def metrics(master, master_path, source, encoded, warm, frames, out, full):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if full:
        return master.metric_pair(master_path, source, encoded, warm, frames, out, True)
    return {'psnr_db': master.ffmpeg_metric(source, encoded, warm, frames, 'psnr', out / 'psnr.log'),
            'ssim': None, 'lpips_mean': None,
            'ws_psnr_mean_db': sampled_ws_psnr(source, encoded, warm, frames, 8, out),
            'ws_sample_step': 8}


def strict_verdict(candidate, old, st, full):
    criteria = {
        'no_more_bytes_than_original_and_360st': candidate['encoded_bytes'] <= min(old['encoded_bytes'], st['encoded_bytes']),
        'no_larger_complete_file': candidate['file_bytes'] <= min(old['file_bytes'], st['file_bytes']),
        'psnr_exceeds_360st': candidate['psnr_db'] > st['psnr_db'],
        'ws_psnr_exceeds_360st': candidate['ws_psnr_mean_db'] > st['ws_psnr_mean_db'],
        'saliency_allocation_active': candidate.get('active_frame_fraction', 0) >= 0.1,
        'within_1pct_short_run_processing_rate': candidate['processing_fps'] >= 0.99 * old['processing_fps'],
    }
    if full:
        criteria['ssim_exceeds_360st'] = candidate['ssim'] > st['ssim']
        criteria['lpips_lower_than_360st'] = candidate['lpips_mean'] < st['lpips_mean']
        criteria['substantial_gain_target'] = (candidate['psnr_db'] - st['psnr_db'] >= 0.25
                                               or candidate['lpips_mean'] <= 0.95 * st['lpips_mean'])
    return {'status': ('PROMISING_SINGLE_WINDOW' if all(criteria.values()) else 'DO_NOT_PROMOTE'),
            'criteria': criteria,
            'interpretation': 'A single window is development evidence only. No automatic deployment or test-set win claim.',
            'speed_note': '1% is a screening tolerance, not proof of zero slowdown. Paired repeated live timing is still required.'}


def screen(args):
    if args.frames < 120 or args.warmup < 240:
        raise ValueError('Keep at least 240 warmup + 120 measured contiguous frames.')
    if args.codec == 'av1' and not args.ack_av1_proxy:
        raise ValueError('AV1 surrogate slopes are uncalibrated. Pass --ack-av1-proxy only for an explicitly exploratory run.')
    if args.full_metrics and args.frames < 480:
        raise ValueError('Full-metric confirmation needs at least 480 measured frames (two GOPs).')
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f'Choose a new output directory: {root}')
    root.mkdir(parents=True)
    master = load_py(args.master, 'satc_master_for_allocation')
    videos = master.video_paths(args.london_video)
    source = videos[args.video_name]
    helpers = master.extract_helpers(root / 'embedded_helpers')
    original = Path(helpers['satc']).read_text(encoding='utf-8')
    modified = root / 'satc_runner_allocation.py'
    modified.write_text(patched_satc_source(original, args.preset), encoding='utf-8')
    satc = load_py(modified, 'satc_candidate_runner')
    install_guard(satc, args.preset)
    protocol = {'version': VERSION, 'source': str(source), 'source_sha256': sha(source),
                'master_sha256': sha(args.master), 'script_sha256': sha(__file__),
                'codec': args.codec, 'mbps': args.mbps, 'preset': args.preset,
                'warmup': args.warmup, 'frames': args.frames, 'video_name': args.video_name,
                'candidates': args.candidates, 'refresh_frames': args.refresh_frames,
                'allocation_mode': args.allocation_mode,
                'scope': 'development screening; live MoST-Sal and one NVENC encoding per frame',
                'metrics': 'full' if args.full_metrics else 'PSNR and sparse full-resolution RGB WS-PSNR',
                'heldout_claim': False, 'state': 'RUNNING'}
    dump(root / 'PROTOCOL.json', protocol)
    sr = root / 'satc'
    _, model, _, _ = master.setup_satc_module(satc, sr, args.preset, {args.video_name: source}, args.frames, args.warmup)
    protocol['materialized_code_sha256'] = {p.name: sha(p) for p in sorted((sr / 'code').iterdir())
                                           if p.is_file() and p.suffix in ('.cpp', '.h', '.py')}
    dump(root / 'PROTOCOL.json', protocol)
    results = []
    # Exactly one base configuration and at most three candidates, one codec/rate.
    # Controls are fresh so shortened windows never inherit 896-frame aggregates.
    choices = ['original', 'uniform'] + args.candidates
    for name in choices:
        allocator = None
        if name not in ('original', 'uniform'):
            settings = candidate_settings(name, args.refresh_frames)
            cls = OverlappedAllocator if args.allocation_mode == 'overlap' else Allocator
            allocator = cls(Geometry(codec=args.codec, sample=(settings.sample_height, settings.sample_width)), settings)
        satc.ALLOCATION_POLICY = allocator
        job = {'id': f'{args.video_name}_{args.codec}_{args.mbps}_{name}', 'video_name': args.video_name,
               'video': str(source), 'codec': args.codec, 'target_mbps': args.mbps,
               'method': 'uniform' if name == 'uniform' else 'roi'}
        try:
            r = satc.run_one(SimpleNamespace(warmup_frames=args.warmup, frames=args.frames), sr, job, model)
        finally:
            if allocator is not None and hasattr(allocator, 'close'):
                allocator.close()
        run = sr / 'runs' / job['id']
        r['preset_actual'] = args.preset
        if allocator and r.get('status') == 'VALID':
            r['allocation_audit'] = allocator.audit(run / 'frame_events.csv', run, args.warmup)
        # Original projection audit still audits original scores/classes only.
        if 'map_audit' in r:
            r['map_audit']['scope_note'] = 'Original saliency projection/classification only; candidate QP map has a separate CRC audit.'
        dump(run / 'summary.json', r)
        if r.get('status') != 'VALID':
            raise RuntimeError(f'Encoding failed: {r.get("error")}; inspect {run}')
        enc = run / ('encoded.' + EXT[args.codec])
        values = metrics(master, args.master, source, enc, args.warmup, args.frames, root / 'metrics' / name, args.full_metrics)
        row = {'method': name, 'processing_fps': r['measurement']['fps'], 'file_bytes': enc.stat().st_size, **values,
               **byte_stats(run / 'frame_events.csv', args.warmup, args.frames)}
        if allocator:
            row.update({k: r['allocation_audit'][k] for k in ('active_frame_fraction', 'mean_nonzero_tile_fraction', 'allocation_mean_ms', 'allocation_p95_ms')})
            if isinstance(allocator, OverlappedAllocator):
                row.update({k: r['allocation_audit'][k] for k in ('worker_mean_ms', 'saliency_age_p95_frames', 'stale_fallback_fraction')})
        results.append(row)
        dump(root / 'RESULTS.json', results)
        print(json.dumps(row, indent=2), flush=True)
    # Generate only the adapted 360-ST comparator, with exactly the same window.
    car = master.load_module(helpers['caruso'], 'caruso_for_allocation')
    car.CODECS, car.RATES, car.METHODS = (args.codec,), (args.mbps,), ('360st',)
    carg = SimpleNamespace(quality_warmup=args.warmup, quality_frames=args.frames,
                          london_video=args.london_video, shared_mat=args.shared_mat, trace_index=args.trace_index)
    if args.preset != 'p7':
        # The master hard-codes p7 for quality. Throughput comparisons need p4 too.
        original_patch = master.patch_preset
        master.patch_preset = lambda p, _: original_patch(p, args.preset)
    cr = root / '360st'
    rr = master.run_quality_caruso(car, cr, carg, {args.video_name: source})
    if len(rr) != 1:
        raise RuntimeError('Comparator unexpectedly expanded to a larger campaign.')
    attempt = int(rr[0].get('attempt', 1))
    run = cr / 'runs' / f'{args.video_name}_{args.codec}_{args.mbps}Mbps_360st' / f'attempt_{attempt}'
    enc = run / ('encoded.' + EXT[args.codec])
    st = {'method': '360st_adaptation', 'file_bytes': enc.stat().st_size,
          **metrics(master, args.master, source, enc, args.warmup, args.frames,
                                               root / 'metrics' / '360st', args.full_metrics),
          **byte_stats(run / 'baseline_events.csv', args.warmup, args.frames)}
    results.append(st)
    dump(root / 'RESULTS.json', results)
    old = next(x for x in results if x['method'] == 'original')
    verdicts = {x['method']: strict_verdict(x, old, st, args.full_metrics) for x in results if x['method'] in args.candidates}
    dump(root / 'VERDICTS.json', verdicts)
    protocol['state'] = 'COMPLETE'
    dump(root / 'PROTOCOL.json', protocol)
    # Package small diagnostics, never the large streams/model weights.
    import zipfile
    upload = Path('/home/mininet-ovs/Downloads/files_to_upload')
    upload.mkdir(parents=True, exist_ok=True)
    destination = upload / (root.name + '_RESULTS.zip')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.csv', '.log') and p.stat().st_size <= 2_000_000:
                z.write(p, p.relative_to(root))
    print(f'Results to upload: {destination}', flush=True)
    print(json.dumps(verdicts, indent=2), flush=True)


def self_test(master_path=None):
    rng = np.random.default_rng(73921)
    tested = 0
    for codec in EXT:
        g = Geometry(codec=codec)
        assert abs(g.area.sum() - 1) < 1e-12
        assert abs(g.sphere_area.sum() - 1) < 1e-12
        assert np.all(g.area > 0)
        a = Allocator(g)
        for _ in range(150):
            scores = rng.lognormal(0, 1, 45) * g.score_latitude
            activity = rng.uniform(0.25, 4, 45)
            q, r = a.allocate(scores, activity)
            assert q.dtype == np.int8 and q.min() >= -3 and q.max() <= 1
            assert r['predicted_rate_ratio'] <= 1 + 1e-12
            assert r['predicted_global_mse_ratio'] <= 1.005 + 1e-12
            assert r['predicted_spherical_mse_ratio'] <= 1.005 + 1e-12
            assert r['predicted_saliency_mse_ratio'] <= 1 + 1e-12
            expected_shape = {'h264': (128, 256), 'hevc': (64, 128), 'av1': (32, 64)}[codec]
            m = g.expand(q)
            assert m.shape == expected_shape and m.flags.c_contiguous
            tested += 1
        # No spherical weighting and flat density/activity -> neutral optimum.
        flat = Allocator(g, Settings(sphere_strength=0))
        q, _ = flat.allocate(g.score_latitude, np.ones(45))
        assert np.array_equal(q, np.zeros(45, np.int8))
        try:
            a.allocate(np.full(45, np.nan), np.ones(45))
            raise AssertionError('Invalid input did not fail closed.')
        except ValueError:
            pass
    # Stronger saliency must affect allocation when activity is identical.
    g = Geometry()
    a = Allocator(g, Settings(sphere_strength=0, global_proxy_limit=1.1, spherical_proxy_limit=1.1))
    density = np.ones(45)
    density[20:25] = 4
    q, r = a.allocate(density * g.score_latitude, np.ones(45))
    assert q[20:25].mean() < np.delete(q, np.arange(20, 25)).mean(), 'Saliency signal was lost.'
    # Check content sampling, constant input, sequence and CRC independent of NVENC.
    raw = bytes([100]) * (W * H * 3 // 2)
    assert np.allclose(g.activity(raw), np.ones(45))
    live = Allocator(g, Settings(refresh_frames=2))
    m0 = live.make_map(density * g.score_latitude, raw, 'h264', 0)
    m1 = live.make_map(density * g.score_latitude, raw, 'h264', 1)
    assert m0 is m1 and len(live.records) == 2
    # Known unfinished future exercises the nonblocking fallback without a
    # scheduling-dependent sleep. Await one real job only in this unit test.
    from concurrent.futures import Future
    overlap = OverlappedAllocator(g)
    overlap.prepare(None, raw, 'h264', 0)
    assert not np.any(overlap.make_map(None, raw, 'h264', 0))
    overlap.prepare(density * g.score_latitude, raw, 'h264', 1)
    overlap.future.result(timeout=5)
    overlap.make_map(None, raw, 'h264', 1)
    assert overlap.records[-1]['saliency_age_frames'] == 1
    overlap.future = Future()
    for fid in range(2, 7):
        overlap.prepare(density * g.score_latitude, raw, 'h264', fid)
        overlap.make_map(None, raw, 'h264', fid)
    assert overlap.records[-1]['stale_fallback']
    overlap.future.cancel()
    overlap.close()
    source_checked = False
    if master_path and Path(master_path).is_file():
        master = load_py(master_path, 'allocation_selftest_master')
        with tempfile.TemporaryDirectory(prefix='satc_allocation_test_') as t:
            root = Path(t)
            helpers = master.extract_helpers(root / 'helpers')
            source = Path(helpers['satc']).read_text(encoding='utf-8')
            text = patched_satc_source(source, 'p7')
            compile(text, 'patched_satc.py', 'exec')
            p = root / 'patched_satc.py'
            p.write_text(text, encoding='utf-8')
            mod = load_py(p, 'allocation_selftest_runner')
            install_guard(mod, 'p7')
            code = root / 'code'
            code.mkdir()
            mod.materialize(code)
            master.patch_preset(code / 'nvenc_uncapped.cpp', 'p7')
            cpp = (code / 'nvenc_uncapped.cpp').read_text(encoding='utf-8')
            assert 'NV_ENC_PRESET_P7_GUID' in cpp and 'v>=-3 && v<=2' in cpp
            assert 'NV_ENC_PARAMS_RC_CBR' in cpp and 'cfg.gopLength=240' in cpp
            assert 'cfg.rcParams.enableAQ=0' in cpp and 'cfg.rcParams.enableLookahead=0' in cpp
            # Independent cross-check against the frozen mapper's actual block geometry.
            core = load_py(code / 'core.py', 'allocation_selftest_core')
            xe, ye = core.tile_edges(W, H)
            assert np.array_equal(xe, g.xe) and np.array_equal(ye, g.ye)
            labels = np.arange(45).reshape(5, 9)
            assert np.array_equal(core.block_classes(labels, W, H), g.block_ids)
            source_checked = True
    report = {'status': 'PASS', 'randomized_surrogate_cases': tested,
              'frozen_v6_source_integration_checked': source_checked,
              'asynchronous_causality_and_late_worker_tested': True,
              'real_NVENC_quality_or_speed_tested': False}
    print(json.dumps(report, indent=2))
    return report


def benchmark(iterations=500):
    g = Geometry()
    rng = np.random.default_rng(1701)
    raw = rng.integers(0, 256, W * H * 3 // 2, dtype=np.uint8).tobytes()
    scores = rng.lognormal(0, 1, 45) * g.score_latitude
    a = Allocator(g, record=False)
    measured = []
    for fid in range(iterations + 30):
        t = time.perf_counter_ns()
        a.make_map(scores, raw, 'h264', fid)
        elapsed = (time.perf_counter_ns() - t) / 1e6
        if fid >= 30:
            measured.append(elapsed)
    result = {'scope': 'CPU allocation only; synthetic input; excludes MoST-Sal/NVENC/video metrics',
              'platform': platform.platform(), 'processor': platform.processor(), 'numpy': np.__version__,
              'iterations': iterations, 'mean_ms': float(np.mean(measured)),
              'median_ms': float(np.median(measured)), 'p95_ms': float(np.percentile(measured, 95))}
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description='Experimental distortion-aware SATC allocation; read the report before running.')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--benchmark', action='store_true')
    mode.add_argument('--screen', action='store_true')
    p.add_argument('--master', type=Path, default=DEFAULT_MASTER)
    p.add_argument('--london-video', type=Path, default=DEFAULT_LONDON)
    p.add_argument('--video-name', choices=('basketball', 'rollercoaster', 'ballet', 'london_tower'), default='basketball')
    p.add_argument('--codec', choices=tuple(EXT), default='h264')
    p.add_argument('--mbps', type=int, choices=(12, 35), default=12)
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--warmup', type=int, default=240)
    p.add_argument('--preset', choices=('p4', 'p7'), default='p7')
    p.add_argument('--candidates', nargs='+', choices=('guarded', 'saliency_only', 'gentle'), default=['guarded'])
    p.add_argument('--refresh-frames', type=int, choices=(1, 2, 4), default=1)
    p.add_argument('--allocation-mode', choices=('overlap', 'serial'), default='overlap')
    p.add_argument('--full-metrics', action='store_true')
    p.add_argument('--ack-av1-proxy', action='store_true')
    p.add_argument('--shared-mat', type=Path, default=None)
    p.add_argument('--trace-index', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('/home/mininet-ovs/Downloads/SATC_ALLOC_DEV_H26412_BASKETBALL'))
    args = p.parse_args()
    if args.self_test:
        self_test(args.master)
    elif args.benchmark:
        benchmark()
    else:
        args.candidates = list(dict.fromkeys(args.candidates))
        screen(args)


if __name__ == '__main__':
    main()
