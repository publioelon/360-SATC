# Encoder and streaming interfaces

This document describes the interfaces in the retained source. See
[standards.md](standards.md) for specification references and
[provenance.md](provenance.md) for source and software records.

## Entry points

| Entry point | Input and output | Spatial quality |
|---|---|---|
| `python satc.py encode` | Prepared video to encoded file and local result logs | MoST-Sal and the selected SATC policy, or uniform CBR |
| `python satc.py stream` | Video to the regular GXRStream encoder and WebRTC receiver | The regular streaming encoder; this command does not apply SATC QP maps |
| `python GXRStream/stream_encoded.py` | Completed H.264 file to RTP/WebRTC through a FIFO | Preserves the quality allocation already present in the encoded video |
| `python experiments/reproduce.py` | Configured experiment inputs to campaign outputs | Selects retained experiment runners; see [reproduction.md](reproduction.md) |

The H.264 replay entry point does not re-encode the video. It does not provide
the corresponding external encoder path for HEVC or AV1. The regular GXRStream
path supports those codecs separately.

## Portable SATC encoder

Dimensions below use width by height. The ONNX model tensor dimensions in
[implementation.md](implementation.md) use height by width.

| Setting | Source configuration |
|---|---|
| Prepared video | 4096 by 2048, 8 bit YUV 4:2:0, BT.709, limited range |
| Encoder input | NV12 raw frames |
| Media frame rate | 60 frames per second |
| H.264 | Main Profile, Level 5.2 |
| HEVC | Main Profile; no explicit level override in the native bridge |
| AV1 | Main Profile; no explicit level override in the native bridge |
| Rate control | CBR, with target bitrate supplied by the caller |
| Preset | `p7` by default; `satc.py encode --preset p4` selects the throughput variant |
| GOP | 240 frames; no B frames |
| VBV size | Target bitrate divided by 4, corresponding to 0.25 seconds |
| QP map spacing | 16 by 16 for H.264, 32 by 32 for HEVC, 64 by 64 for AV1 |
| Stored output | H.264 and HEVC elementary streams; AV1 in an IVF wrapper |

These settings come from [encoder.py](../runtime/encoder.py),
[nvenc_uncapped.cpp](../runtime/nvenc_uncapped.cpp), and the
[`satc.py` options](../satc.py). They identify the portable encoder path,
not every historical experiment or the standalone GXRStream encoder.
`satc.py encode --help` lists its selectable options.

Frame preparation and QP maps pass between local processes through shared
memory and pipes. The `FRM1`, `AU01`, and `RDY1` markers in the native
bridge belong to this local process interface. They are not inserted into the
video RTP payload.

## Media interface in GXRStream

[The pipeline builder](../GXRStream/ubuntu/sender/webrtc_sender_backend/pipeline_builder.py)
selects the parser and RTP packetizer for each codec:

| Codec | RTP encoding name | Clock rate | Parser and packetizer |
|---|---|---|---|
| H.264 | `H264` | 90000 Hz | `h264parse`, `rtph264pay` |
| H.265 | `H265` | 90000 Hz | `h265parse`, `rtph265pay` |
| AV1 | `AV1` | 90000 Hz | `av1parse`, `rtpav1pay` |

The sender configures dynamic payload type 96 and an RTP packetizer MTU of
1200 bytes. Payload type 96 is a session mapping, not a globally assigned
codec identifier. The session's SDP describes the codec and payload mapping.
`webrtcbin` manages WebRTC session negotiation and media transport.

GStreamer buffer PTS and duration use nanoseconds. The RTP clock is 90000 Hz
for these video paths. A monotonic timestamp in a local log is not
automatically a timestamp synchronized with the remote host.

## TCP session signaling

The sender opens a TCP connection to the receiver. Port 9001 is the default
used in the streaming guide and can be configured. Each message is one UTF-8
line terminated by LF. The Python reader also ignores CR characters.

| Message | Direction | Fields |
|---|---|---|
| `OFFER|<base64_sdp>` | Sender to receiver | Base64 of the UTF-8 SDP offer |
| `ANSWER|<base64_sdp>` | Receiver to sender | Base64 of the UTF-8 SDP answer |
| `ICE|<mline_index>|<base64_candidate>` | Either direction | Decimal media line index and base64 of the UTF-8 ICE candidate |

These are the existing application envelopes, not a new RTP payload or an
IETF signaling format. Base64 is an encoding, not encryption. The retained
signaling socket is plain TCP. Deployments must account for how they trust
the peer and the exchanged SDP.

The implementation is in
[signaling.py](../GXRStream/ubuntu/sender/webrtc_sender_backend/signaling.py),
[the sender callbacks](../GXRStream/ubuntu/sender/webrtc_sender_backend/gst_app.py),
and [the Linux receiver](../GXRStream/ubuntu/receiver/linux/webrtc_receiver.py).
The pinned native Quest receiver uses the same offer, answer, and candidate
envelopes.

## Optional receiver diagnostics

The Ubuntu sender listens for receiver diagnostics on TCP port
`signaling_port + 100` by default, hence 9101 when signaling uses 9001.
A configured receiver feedback port overrides that default.

The Linux receiver connects to this listener and publishes lines of the form:

```text
RECEIVER_METRICS|receiver_alive=1|codec=h264|received_fps=60.0|jitter_ms=0.0
```

This line is a syntax example, not an experimental observation. Fields are
separated by `|`; each field is split at its first `=`. The Linux publisher
replaces pipe characters with slashes, replaces line breaks with spaces,
trims text, and limits each value to 180 characters.

| Fields | Meaning and units in the Linux publisher |
|---|---|
| `receiver_alive`, `texture_attached`, `stall` | Status flags, 0 or 1; `texture_attached` denotes an assembled video chain in this receiver |
| `received_fps`, `decoded_fps`, `displayed_fps` | Frames per second at the respective receiver stages |
| `unity_fps`, `copied_fps` | Compatibility names for Linux displayed and decoded FPS; they do not measure a Unity render loop |
| `bitrate_mbps` | Receiver incoming rate in decimal Mbit/s |
| `received_frames_total`, `decoded_frames_total`, `displayed_frames_total` | Cumulative frame counters |
| `last_frame_id` | Linux displayed frame count, not a transmitted source frame identifier |
| `packets_received`, `bytes_received`, `packets_lost` | Packet and byte counters; unavailable statistics can be `n/a` |
| `packet_loss_pct` | Loss percentage |
| `jitter_ms`, `rtt_ms`, `last_frame_age_ms` | Milliseconds; unavailable statistics can be `n/a` |
| `queue_backlog` | Number of buffers queued at the display stage |
| `dropped_frames` | Derived count: decoded minus displayed minus queued, clipped at zero |
| `codec`, `decoder`, `resolution`, `display_mode`, `last_error` | Text identifiers or diagnostics |

The sender accepts additional fields and compatibility aliases. See
`handle_receiver_feedback_line` in
[gst_app.py](../GXRStream/ubuntu/sender/webrtc_sender_backend/gst_app.py).
Diagnostic availability and counter definitions depend on the receiver build;
the table above describes the Linux publisher specifically.

The sender's `METRICS|...` output is a stdout dashboard record. It is distinct
from receiver diagnostics and from RTCP feedback used by GCC. Receiver
diagnostics do not contain viewer head coordinates, gaze, or viewport positions.
These messages are an application format, not CMCD or CMSD.

