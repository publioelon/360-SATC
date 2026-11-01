# Source versions and experiment records

Source identity, an environment snapshot, and evidence for a particular
experimental run are different records. This page identifies the records
already present and describes how to attach missing provenance to existing
results.

## Recorded source snapshot

| Item | Recorded identity |
|---|---|
| Implementation snapshot described by the new standards and interface documentation | [360-SATC commit `79026354849fbd90338cf1f56d951d688462fc7f`](https://github.com/publioelon/360-SATC/tree/79026354849fbd90338cf1f56d951d688462fc7f) |
| Receiver submodule in that snapshot | [QGXS commit `26f94a5a4fb8a9b6055e01a56e2614514f4422a5`](https://github.com/publioelon/QGXS---quest-gstreamer-xr-streaming-/tree/26f94a5a4fb8a9b6055e01a56e2614514f4422a5) |
| Uncompressed MoST-Sal ONNX SHA-256 | `51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa` |
| Recovered source paths and file hashes | [source_manifest.json](source_manifest.json) |

The implementation snapshot above is a documentation reference. It is not
evidence that every manuscript figure was produced by that Git commit.
The source manifest distinguishes original source hashes from packaged hashes,
including files whose paths or configuration were adapted.

## Recorded software information

| Record | Values or scope |
|---|---|
| [Original Python environment](environment_original.txt) | PyTorch 2.8.0+cu126, ONNX Runtime GPU 1.23.2, TensorRT 10.13.3.9, NumPy 2.2.6, and the other listed distributions |
| [Portable runtime requirements](../requirements.txt) | NumPy 2.2.6, OpenCV headless 4.12.0.88, ONNX Runtime GPU 1.23.2 |
| [Metric requirements](../requirements-metrics.txt) | Version constraints for the metric environment |
| [SDK setup](setup.md) | NVIDIA Video Codec SDK 13.0.19 |
| [Streaming release instructions](streaming.md) | Unity 2022.3.45f1 is the documented release editor |

The Python snapshot does not record the NVIDIA driver, CUDA toolkit, GStreamer
core and plugin build identities, or the receiver binary used for every run.
A package version can also differ from the toolkit installed on the host.
Recover the applicable values from the original run logs, build records, and
release checksums before assigning them to a result.

## Attach provenance to an existing result

Use [run_provenance.example.json](../configs/run_provenance.example.json) as a
local record alongside the original outputs. It is a documentation template;
no runtime entry point consumes it.

For each figure, table, or run group, record:

1. The result files, input manifest, command, and configuration.
2. The code commit, any local changes, receiver source commit and build identity,
   and model checksum.
3. The Python environment, NVIDIA driver, CUDA toolkit, SDK, and GStreamer core
   and plugin versions, as applicable.
4. The codec settings, spatial policy, model sampling stride and inference
   provider, and measurement boundaries.

Keep unknown values as `null` and list their names in `unresolved_fields`.
Set `record_status` to `partial` until the record is supported by evidence.
A record completed today from historical logs should state both the capture
date and the original run date. Do not substitute the current notebook's
versions for missing historical values.

Preserve the existing distinction between the final quality policy,
historical network and throughput policies, and the different streaming
entry points. See [reproduction.md](reproduction.md) and
[implementation.md](implementation.md).

## Inspect the current Ubuntu environment

The commands below inspect source and software versions. They do not encode,
stream, train, or execute a performance experiment. Run them from the repository
with the Python environment used by the relevant run activated.

```bash
record_dir="outputs/provenance/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$record_dir"

git rev-parse HEAD > "$record_dir/repository-commit.txt"
git status --porcelain > "$record_dir/working-tree.txt"
git submodule status --cached > "$record_dir/submodules.txt"
python --version > "$record_dir/python-version.txt" 2>&1
python -m pip list --format=json > "$record_dir/python-packages.json"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$record_dir/gpu-driver.txt" 2>&1
nvcc --version > "$record_dir/cuda-toolkit.txt" 2>&1
gst-inspect-1.0 --version > "$record_dir/gstreamer-version.txt" 2>&1

for element in webrtcbin rtph264pay rtph265pay rtpav1pay rtpgccbwe nvh264enc nvh265enc nvav1enc; do
    gst-inspect-1.0 "$element" > "$record_dir/gst-$element.txt" 2>&1
done
```

A missing tool or plugin leaves an error in its corresponding record. Review
these files before using them as evidence of available functionality.
Add the SDK version and receiver release or binary checksum from the actual
installation. Store the capture with local experiment outputs; it describes
the environment when the commands are run, not a new measurement or proof of
the software used by an earlier experiment.

