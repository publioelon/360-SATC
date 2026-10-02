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

## Local checks

```bash
python -m unittest discover -s tests -v
python runtime/encoder.py --self-test
python experiments/research/final_quality_matrix.py --self-test
```

These verify software structure and invariants. Reproducing the numerical results
requires the original inputs, metadata, environment, and target hardware.
