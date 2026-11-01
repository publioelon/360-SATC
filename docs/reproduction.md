# Reproduce the experiments

Start with the local encoder example in the README. It is the smallest test of
the included model, final spatial policy, and native encoder.

## Final reconstruction-quality matrix

The recovered final runner evaluates four videos × three codecs × two target
bitrates × six methods: **144 conditions**, with 240 warm-up and 896 measured
frames per condition. It uses p7, CBR, GOP 240, no B-frames, and the final −2/−1/0
six/six/33 SATC policy.

```bash
cp configs/experiments.example.json configs/experiments.local.json
# Edit the paths to your prepared videos, SDK, Python environments, and metadata.
python experiments/reproduce.py quality \
  --config configs/experiments.local.json --output outputs/quality --plan
python experiments/reproduce.py quality \
  --config configs/experiments.local.json --output outputs/quality
```

Required prepared filenames are `basketball_4096x2048_60fps.mp4`,
`rollercoaster_4096x2048_60fps.mp4`, and `ballet_4096x2048_60fps.mp4`.
London Tower has a separate configurable path. Provide the original metadata
`.mat` used by the network/baseline runner; it contains trace/trajectory inputs
needed by the relevant methods. Videos, datasets, and this metadata are not in
the source-candidate ZIP and are not fabricated or substituted.

Outputs include partial and final CSV/JSON tables, per-condition logs,
encoded streams, frame checks, and allocation audits. The final runner disables
reuse of historical London results by default through this entry point.
PC and 360-ST use common trajectories. The retained 360-ST baseline is an adapted
four-zone implementation, not an original trained A2C checkpoint.

## Throughput, network, and RTT

```bash
python experiments/reproduce.py throughput --config configs/experiments.local.json --output outputs/throughput
python experiments/reproduce.py network --config configs/experiments.local.json --output outputs/network
python experiments/reproduce.py rtt --config configs/experiments.local.json --output outputs/rtt
```

These commands expose the retained v6 campaign. **Its historical neighbor-ring
policy is preserved**, so its outputs must not be relabeled as measurements of
the final six/six/33 policy. For a local final-policy throughput run, use
`satc.py encode --preset p4` and explicitly select its measured frame count.

Network/RTT runs require Mininet, Open vSwitch, GStreamer, the original metadata,
and elevated privileges for network namespace construction. The script requests
`sudo` for that stage. Run on an isolated experiment machine. Encoder processing
FPS, received media FPS, and HMD refresh rate have different measurement boundaries.

## Uniform CBR throughput rerun for Figure 4

`experiments/benchmark_cbr.py` measures uniform CBR without MoST-Sal inference,
model preprocessing, saliency smoothing, tile ranking, or spatial QP allocation.
It uses Python's standard library and the existing FFmpeg/CUDA/NVENC installation;
it requires no model, Python GPU packages, transport setup, or administrator access.

Use the same prepared Basketball, Rollercoaster, Ballet, and London Tower videos
and the same notebook used for the original throughput measurements:

```bash
# List candidate files if the existing paths are unknown.
python3 experiments/benchmark_cbr.py --discover

# The existing local configuration supplies prepared_dir, london_video, and sdk.
python3 experiments/benchmark_cbr.py --config configs/experiments.local.json --plan
python3 experiments/benchmark_cbr.py --config configs/experiments.local.json
```

Alternatively, supply `--prepared-dir /path/to/prepared_inputs`,
`--london-video /path/to/the/prepared/london.mp4`, and
`--sdk /path/to/Video_Codec_SDK_13.0.19`. Repeated `--video LABEL=/exact/path.mp4`
arguments select explicit inputs, where LABEL is basketball, rollercoaster, ballet,
or london. When no input paths are given, discovery requires exactly one candidate
per video; ambiguous files must be selected explicitly. Inputs must be 4096×2048,
60 FPS, 8 bit BT.709 limited range YUV420, with at least 1,504 frames. The runner
does not prepare or substitute videos. `--plan` validates metadata without running
CUDA or NVENC. Close other GPU workloads before the measurements.

Four videos × three codecs × two rates give **24 runs** per repetition. The
defaults are one repetition, 240 warmup frames, and 1,200 measured frames per run;
`--repeats N` repeats the entire matrix in the recorded order. The native P4
bridge and raw-frame Source are extracted unchanged from the retained
`SATC_CBR_FullWorkload_P4_v1.py` after checking its SHA256. If the script is run
outside the repository, it downloads that source from the pinned commit, checks
the same hash, and records the source with the results. The historical runner is
preserved for provenance; it remains a different workload and should not be used
as the corrected uniform CBR throughput baseline.

P4, low latency, CBR, GOP 240, no B frames, no lookahead or AQ, single pass, and
the 0.25 second VBV setting are retained. A constant zero delta buffer preserves
the NVENC interface; no saliency map or spatial allocation is calculated. Necessary
CUDA source decoding, NV12 preparation, copies, NVENC encoding, completed access
unit receipt, and event logging remain within the processing workload. The timer
starts after the last warmup output and its event row, before the measured source
gate opens, and stops after the final measured output and its event row. This
includes pipeline fill and drain. Encoded payloads are consumed, as in the retained
CBR runner, rather than saved as video files. Network and display are excluded.

Each run records completed frame IDs, timings, encoded byte counts, target and
applied bitrates, and a checksum of the constant QP buffer. Invalid runs remain
in `SUMMARY.csv`. Low FPS does not cause a retry or invalidate a completed run.
`one_second_throughput.csv` records nonoverlapping wall-clock windows starting
at the measurement gate; use rows with `full_second=True` for Figure 4 and retain
the marked final partial interval. Whole-batch FPS is reported separately.

Results go into a new `SATC_CBR_ENCODER_ONLY_P4_...` directory under Downloads,
or the new directory selected by `--output`. Existing results are never replaced.
The final printed `_UPLOAD.zip` contains the run order, summary, frame logs,
integrity audits, software/GPU information, input hashes, and source snapshots.
The checks and runner do not supply measured GPU results. Keep MoST-Sal enabled
for 360-SATC, and compare results only when its inputs, P4 settings, hardware,
and timing boundary match. Update Figure 4 and the manuscript after reviewing
the actual rerun output.

## Saliency and viewport evaluation

The original saliency scripts are in `saliency/research/`; FlowSal is the working
name used for MoST-Sal in those files. They cover VR-EyeTracking, PanoNUT360,
Sports-360, and the balanced-anchor study. They retain original dataset paths and
training/evaluation assumptions: inspect and configure them for your local dataset
layout before running. The frozen ONNX artifact is sufficient for deployment;
these recipes are not a new one-command training release.

`experiments/research/viewport_pilot.py`, `viewport_batch.py`, and
`viewport_analysis.py` retain the head-centered evaluation and statistics. They
require the original VR-EyeTracking videos and head trajectories and retain
notebook paths. These trajectories are evaluation inputs, never SATC runtime inputs.

## Record the implementation used for each result

Keep the command, configuration, source and model hashes, software versions,
and result files together. [Provenance documentation](provenance.md) separates
the recovered source snapshot from evidence for an individual experimental run
and provides a record template. Missing historical values must be recovered
from original logs rather than filled with the current notebook's versions.

## Local checks

```bash
python -m unittest discover -s tests -v
python runtime/encoder.py --self-test
python experiments/research/final_quality_matrix.py --self-test
```

These verify software structure and invariants. Reproducing the numerical results
requires the original inputs, metadata, environment, and target hardware.
