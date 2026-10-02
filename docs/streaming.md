# GXRStream

The paper calls the transport **GXRStream**. Its upstream repository is currently
named **QGXS**. The Ubuntu sender recovered from the uploaded notebook is included
here. Unity and Quest receiver assets are pinned as `GXRStream/receiver-source`.

## Receiver

```bash
git submodule update --init GXRStream/receiver-source
```

Use the Quest setup and native build instructions in
`GXRStream/receiver-source/receiver/quest3/README.md`. The complete Unity project
is at `GXRStream/receiver-source/receiver/unity/QSXRReceiver`. Requirements include
Unity 2022.3 LTS, Android ARM64 tooling, GStreamer Android libraries, and a Quest 3
in Developer Mode. The upstream release link and source-build instructions remain
in its README; this repository does not bundle a newly built APK.

Connect the sender and headset to the same network. Start the receiver and use
its IP address and signaling port (default 9001). Sender/receiver dimensions must
match: 4096×2048 for the examples below.

## Ubuntu sender

```bash
bash GXRStream/ubuntu/scripts/linux/install_sender_deps_ubuntu.sh
/usr/bin/python3 satc.py doctor --streaming-only
```

The distribution packages install the base GStreamer stack. NVIDIA `nvcodec`
plugins and `rtpgccbwe` may require your existing custom GStreamer build.
A stock Ubuntu install alone does not guarantee these elements are available.
Use a Python interpreter that can import `gi` and load your GStreamer installation.

Test delivery with a prepared video:

```bash
/usr/bin/python3 satc.py stream --video /path/to/prepared.mp4 --host QUEST_IP
```

This tests the regular GStreamer encoder/transport/receiver path. It does not
apply MoST-Sal or the SATC QP policy. H.264, H.265, and AV1 are available when the
corresponding sender plugins and receiver decoder are present.

## Send the SATC-encoded panorama

After `satc.py encode` completes, replay its H.264 output directly through the
retained external-bitstream FIFO path, without re-encoding:

```bash
/usr/bin/python3 GXRStream/stream_encoded.py \
  --input outputs/satc/runs/encode/encoded.h264 --host QUEST_IP
```

This preserves SATC's spatial quality allocation and tests headset delivery.
It is playback of a completed encode, not a measurement of live end-to-end
saliency inference or RT-MPC adaptation. The retained FIFO path accepts H.264
only; the regular streaming path supports the other codecs separately.

The original integrated network experiments are described in
[reproduction.md](reproduction.md). Their receiver is an instrumented GStreamer
client for transport measurements, distinct from Unity display performance.
