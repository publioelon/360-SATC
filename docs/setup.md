# Setup

## Existing notebook environment

Use the Python environment that already runs your SATC experiments:

```bash
source /path/to/venvs/satc/bin/activate
export SATC_SDK="/path/to/Video_Codec_SDK_13.0.19"
python satc.py doctor
```

`doctor` checks the model checksum, Python modules, CUDA availability, native build
tools, and SDK location. Add `--streaming` to inspect GStreamer and encoder plugins.
TensorRT is optional for the portable encoder; CUDA execution remains required.
Use `--provider tensorrt` when the TensorRT provider must be enforced.

## New installation

System tools:

```bash
sudo apt update
sudo apt install -y python3-venv ffmpeg cmake build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Install an NVIDIA driver and CUDA toolkit compatible with the chosen PyTorch,
ONNX Runtime, and GPU. `nvcc` must be on `PATH`. Download **Video Codec SDK
13.0.19** from NVIDIA and point `SATC_SDK` at its top-level directory, which
contains `Samples/` and `Interface/`. The SDK and driver libraries are not bundled.

```bash
export SATC_SDK="/path/to/Video_Codec_SDK_13.0.19"
python satc.py doctor
```

The first encoding run builds the native bridge in its output directory. The
frozen ONNX model is automatically decompressed and verified. TensorRT engine
creation and warm-up are initialization costs, not measured processing throughput.

For quality evaluation, install PyTorch/torchvision compatible with your metrics
environment, then `python -m pip install -r requirements-metrics.txt`. LPIPS may
need to download its pretrained backbone on first use. A separate metrics Python
can be selected in the experiment configuration.

The uploaded notebook environment is recorded in
[environment_original.txt](environment_original.txt). It includes TensorRT
10.13.3.9, ONNX Runtime GPU 1.23.2, and PyTorch 2.8.0+cu126; this is a source
snapshot, not a guarantee that every version combination is portable.

## If a check fails

| Symptom | Action |
|---|---|
| CUDA provider unavailable | Verify `onnxruntime-gpu`, driver, and CUDA/cuDNN library paths; avoid installing CPU and GPU ONNX Runtime together |
| SDK not found | Set `SATC_SDK` to the directory containing `Samples/NvCodec/NvEncoder/` |
| Encoder build fails | Read `configure.log` and `build.log` in the run directory |
| Clip rejected | Prepare it first; the benchmark checks resolution, pixel format, color metadata, and frame count |
| TensorRT initialization fails | Try `--provider cuda` to isolate TensorRT setup |
