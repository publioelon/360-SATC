# Coding and transport specifications

360-SATC assigns spatial quality through encoder QP maps. NVENC produces a
single panoramic video stream. The receiver needs a decoder for the selected
codec and the panoramic display configuration; it does not need MoST-Sal weights,
saliency maps, or the encoder's tile priorities.

## Specification map

| Layer | Specification | Implementation |
|---|---|---|
| AVC coding | [ISO/IEC 14496-10](https://www.iso.org/standard/87574.html) / ITU-T H.264 | NVENC H.264 encoder |
| HEVC coding | [ISO/IEC 23008-2](https://www.iso.org/standard/93906.html) / ITU-T H.265 | NVENC HEVC encoder |
| AV1 coding | [AOMedia AV1 specifications](https://aomedia.org/specifications/av1/) | NVENC AV1 encoder |
| H.264 RTP payload | [RFC 6184](https://www.rfc-editor.org/rfc/rfc6184.html) | GStreamer `rtph264pay` and `rtph264depay` |
| H.265 RTP payload | [RFC 7798](https://www.rfc-editor.org/rfc/rfc7798.html) | GStreamer `rtph265pay` and `rtph265depay` |
| AV1 RTP payload | [AOMedia AV1 RTP specification, v1.0.0](https://aomediacodec.github.io/av1-rtp-spec/v1.0.0.html) | GStreamer `rtpav1pay` and `rtpav1depay` |
| WebRTC media transport | [RFC 8834](https://www.rfc-editor.org/rfc/rfc8834.html), with the protocol suite described in [RFC 8825](https://www.rfc-editor.org/rfc/rfc8825.html) | GStreamer `webrtcbin` at the sender and receiver |

The references identify the formats and protocols used by the implementation.
They do not establish conformance of every output or every WebRTC endpoint
requirement. Record the installed plugin versions and negotiated session
parameters when identifying a particular run. AV1 is specified by AOMedia and
is not an MPEG codec.

## Spatial quality and decoder requirements

The 5 by 9 grid consists of logical regions used for saliency ranking. Their
priorities become signed QP offsets on the NVENC block grid. These regions
remain within one coded panoramic picture; they are not independently encoded
or transmitted codec tiles.

[The native encoder](../runtime/nvenc_uncapped.cpp) supplies
`NV_ENC_QP_MAP_DELTA` and `NV_ENC_PIC_PARAMS::qpDeltaMap` to NVENC. The resulting
video uses the codec's existing syntax. See
[NVIDIA's SDK 13.0 programming guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html)
for the encoder API and [implementation notes](implementation.md) for the model
and ranking policy.

## Application signaling and presentation

Session signaling carries SDP offers, SDP answers, and ICE candidates in
application messages over TCP. Receiver diagnostics use a separate optional
TCP connection. The message grammar, connection roles, and units are specified
in [interfaces.md](interfaces.md). RFC 8825 section 3 leaves the signaling
protocol choice outside the WebRTC protocol suite.

The Unity receiver is configured to display equirectangular frames on a sphere.
This deployment does not provide an OMAF package or establish OMAF conformance.
[OMAF, ISO/IEC 23090-2](https://www.iso.org/standard/84636.html), defines a separate
omnidirectional media format. The logical tile grid and ERP image dimensions
do not by themselves identify an OMAF presentation.

## Configuration scope

The portable SATC encoder, completed H.264 replay, regular GXRStream streaming,
and historical network experiments are different entry points. Their
settings and measurement boundaries are described in
[interfaces.md](interfaces.md) and [reproduction.md](reproduction.md).
The standalone GXRStream encoder has its own preset and GOP settings; do not
infer them from the SATC encoder configuration.

The bitrate controller chooses encoder settings from network feedback.
Its quality utility and selection policy are application decisions, not MPEG
or IETF mandated algorithms. The saliency model and controller do not add a new
video RTP payload.

Source snapshots and recorded software versions are listed in
[provenance.md](provenance.md). The codec and transport descriptions here do not
add codecs to the reported experimental comparison.

