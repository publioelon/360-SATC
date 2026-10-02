from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

from .config import (
    SenderConfig,
    gop_size,
    is_av1,
    is_h264,
    is_image_folder_mode,
    is_video_file_mode,
    normalize_path_for_gstreamer,
)


# ============================================================
# Generic helpers
# ============================================================

def escape_gst_string(value: str) -> str:
    """
    Escape a string used inside a quoted gst_parse_launch property.
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _caps_video_raw(config: SenderConfig) -> str:
    return (
        f"video/x-raw,width={config.width},height={config.height},"
        f"framerate={config.fps}/1"
    )


@lru_cache(maxsize=128)
def gst_element_exists(factory_name: str) -> bool:
    """
    Best-effort element availability check.

    If gst-inspect-1.0 is not available, return False. The caller decides
    whether to fall back or to keep the desired high-performance element.
    """
    gst_inspect = shutil.which("gst-inspect-1.0") or shutil.which("gst-inspect-1.0.exe")

    if not gst_inspect:
        return False

    try:
        result = subprocess.run(
            [gst_inspect, factory_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _encoder_preference() -> str:
    """
    Encoder selection policy.

    Default is nvenc because the research goal is high-FPS 360 streaming.

    Override examples:
        QGXS_ENCODER=nvenc      force NVIDIA NVENC elements
        QGXS_ENCODER=software   force x264enc/x265enc
        QGXS_ENCODER=auto       prefer NVENC, fall back to software
    """
    value = os.environ.get("QGXS_ENCODER", "nvenc").strip().lower()

    aliases = {
        "nvidia": "nvenc",
        "nv": "nvenc",
        "cpu": "software",
        "sw": "software",
        "x264": "software",
        "x265": "software",
    }

    return aliases.get(value, value)


def _strict_nvenc_required() -> bool:
    return os.environ.get("QGXS_REQUIRE_NVENC", "1").strip() not in ("0", "false", "False", "no")


# ============================================================
# Source builders
# ============================================================

def build_video_file_source_fragment(config: SenderConfig) -> str:
    """
    Build the source section for normal video files.

    Default behavior is conservative and keeps the original QGXS Windows path:
        filesrc -> qtdemux -> decodebin

    For broader Linux container formats, you can test:
        QGXS_VIDEO_SOURCE=uridecodebin
    """
    gst_path = escape_gst_string(normalize_path_for_gstreamer(config.input_path))
    source_mode = os.environ.get("QGXS_VIDEO_SOURCE", "qtdemux").strip().lower()

    parts: list[str] = []

    if source_mode in ("uri", "uridecodebin", "auto"):
        # Use pathlib-like file URI without importing GLib here.
        # Keep raw path normalized because Windows paths may still be used.
        if gst_path.startswith("file://"):
            uri = gst_path
        elif os.name == "nt" or ":/" in gst_path[:4]:
            uri = "file:///" + gst_path
        else:
            uri = "file://" + gst_path

        parts.append(f'uridecodebin uri="{escape_gst_string(uri)}"')
        return " ".join(parts)

    parts.append(f'filesrc location="{gst_path}"')
    parts.append("! qtdemux name=demux demux.video_0")
    parts.append("! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0")
    parts.append("! decodebin")

    return " ".join(parts)


def build_image_folder_source_fragment(config: SenderConfig) -> str:
    """
    Build the source section for arbitrary image folders.

    Actual image loading/decoding is done in Python inside gst_app.py using
    appsrc. This is good for functionality, but video-file mode is usually the
    first choice for 4K/120 tests because Python image loading may bottleneck.
    """
    parts: list[str] = []

    parts.append(
        "appsrc name=frame_source "
        "is-live=true "
        "format=time "
        "do-timestamp=false "
        "block=true "
        f'caps="video/x-raw,format=RGB,width={config.width},height={config.height},framerate={config.fps}/1"'
    )

    parts.append("! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0")

    return " ".join(parts)


def build_source_fragment(config: SenderConfig) -> str:
    if is_image_folder_mode(config):
        return build_image_folder_source_fragment(config)

    if is_video_file_mode(config):
        return build_video_file_source_fragment(config)

    raise ValueError(f"Unsupported input mode: {config.input_mode}")


# ============================================================
# Common raw-video processing
# ============================================================

def build_common_processing_fragment(config: SenderConfig) -> str:
    parts: list[str] = []

    parts.append("! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0")
    parts.append("! videoconvert")
    parts.append("! videoscale")
    parts.append("! videorate")
    parts.append(f"! {_caps_video_raw(config)}")

    # Live pacing and EOS interception point.
    parts.append("! identity name=loop_guard sync=true")
    parts.append("! videoconvert")

    return " ".join(parts)


# ============================================================
# Codec / encoder builders
# ============================================================

def _nvh264_encoder(config: SenderConfig) -> str:
    return (
        "! video/x-raw,format=NV12 "
        f"! nvh264enc name=video_encoder bitrate={config.bitrate_kbps} "
        f"gop-size={gop_size(config)} "
        "zerolatency=true "
        "bframes=0 "
        "rc-lookahead=0 "
        "tune=ultra-low-latency "
        "preset=p1"
    )


def _nvh265_encoder(config: SenderConfig) -> str:
    return (
        "! video/x-raw,format=NV12 "
        f"! nvh265enc name=video_encoder bitrate={config.bitrate_kbps} "
        f"gop-size={gop_size(config)} "
        "zerolatency=true "
        "bframes=0 "
        "rc-lookahead=0 "
        "tune=ultra-low-latency "
        "preset=p1"
    )


def _x264_encoder(config: SenderConfig) -> str:
    return (
        "! video/x-raw,format=I420 "
        f"! x264enc name=video_encoder bitrate={config.bitrate_kbps} "
        f"key-int-max={gop_size(config)} "
        "tune=zerolatency "
        "speed-preset=ultrafast "
        "bframes=0 "
        "byte-stream=true"
    )


def _x265_encoder(config: SenderConfig) -> str:
    return (
        "! video/x-raw,format=I420 "
        f"! x265enc name=video_encoder bitrate={config.bitrate_kbps} "
        f"key-int-max={gop_size(config)} "
        "tune=zerolatency "
        "speed-preset=ultrafast"
    )


def _choose_h264_encoder(config: SenderConfig) -> str:
    pref = _encoder_preference()

    if pref == "software":
        return _x264_encoder(config)

    if pref == "auto":
        if gst_element_exists("nvh264enc"):
            return _nvh264_encoder(config)
        return _x264_encoder(config)

    if pref != "nvenc":
        raise ValueError(f"Unsupported QGXS_ENCODER for H.264: {pref}. Use nvenc, auto, or software.")

    if _strict_nvenc_required() and not gst_element_exists("nvh264enc"):
        raise ValueError(
            "nvh264enc was not found. Install GStreamer nvcodec support/NVIDIA driver, "
            "or run with QGXS_REQUIRE_NVENC=0 QGXS_ENCODER=auto/software for a non-120-FPS fallback."
        )

    return _nvh264_encoder(config)


def _choose_h265_encoder(config: SenderConfig) -> str:
    pref = _encoder_preference()

    if pref == "software":
        return _x265_encoder(config)

    if pref == "auto":
        if gst_element_exists("nvh265enc"):
            return _nvh265_encoder(config)
        return _x265_encoder(config)

    if pref != "nvenc":
        raise ValueError(f"Unsupported QGXS_ENCODER for H.265: {pref}. Use nvenc, auto, or software.")

    if _strict_nvenc_required() and not gst_element_exists("nvh265enc"):
        raise ValueError(
            "nvh265enc was not found. Install GStreamer nvcodec support/NVIDIA driver, "
            "or run with QGXS_REQUIRE_NVENC=0 QGXS_ENCODER=auto/software for a non-120-FPS fallback."
        )

    return _nvh265_encoder(config)


def build_h264_fragment(config: SenderConfig) -> str:
    parts: list[str] = []

    parts.append(_choose_h264_encoder(config))
    parts.append("! h264parse config-interval=-1")
    parts.append("! rtph264pay name=rtp_pay pt=96 config-interval=1 mtu=1200")
    parts.append(
        "! application/x-rtp,media=video,encoding-name=H264,"
        "payload=96,clock-rate=90000"
    )

    return " ".join(parts)


def build_h265_fragment(config: SenderConfig) -> str:
    parts: list[str] = []

    parts.append(_choose_h265_encoder(config))
    parts.append("! h265parse config-interval=-1")
    parts.append("! rtph265pay name=rtp_pay pt=96 config-interval=1 mtu=1200")
    parts.append(
        "! application/x-rtp,media=video,encoding-name=H265,"
        "payload=96,clock-rate=90000"
    )

    return " ".join(parts)


def _nvav1_encoder(config: SenderConfig) -> str:
    if _strict_nvenc_required() and not gst_element_exists("nvav1enc"):
        raise ValueError(
            "nvav1enc was not found. Install GStreamer nvcodec support/NVIDIA driver, "
            "or run with another codec."
        )

    # Keep approximately two encoded frames in the NVENC VBV buffer.
    # This constrains instantaneous bitrate bursts without introducing
    # a large latency reservoir.
    fps = max(1, int(config.fps))
    vbv_buffer_kbits = max(
        1,
        int(round((config.bitrate_kbps / fps) * 2.0)),
    )

    return (
        "! video/x-raw,format=NV12 "
        f"! nvav1enc name=video_encoder bitrate={config.bitrate_kbps} "
        f"gop-size={gop_size(config)} "
        f"vbv-buffer-size={vbv_buffer_kbits} "
        "rc-mode=cbr "
        "tune=ultra-low-latency "
        "multi-pass=disabled "
        "zerolatency=true "
        "bframes=0 "
        "rc-lookahead=0 "
        "preset=p1"
    )


def build_av1_fragment(config: SenderConfig) -> str:
    parts: list[str] = []

    parts.append(_nvav1_encoder(config))
    parts.append("! av1parse")
    parts.append("! video/x-av1,stream-format=obu-stream,alignment=tu,parsed=true")
    parts.append("! rtpav1pay name=rtp_pay pt=96 mtu=1200")
    parts.append(
        "! application/x-rtp,media=video,encoding-name=AV1,"
        "payload=96,clock-rate=90000"
    )

    return " ".join(parts)


def build_codec_fragment(config: SenderConfig) -> str:
    if is_h264(config.codec):
        return build_h264_fragment(config)

    if is_av1(config.codec):
        return build_av1_fragment(config)

    return build_h265_fragment(config)


# ============================================================
# External NVIDIA SDK encoded source
# ============================================================

def external_h264_fifo_path() -> str:
    """
    Return the optional FIFO carrying an externally encoded H.264
    byte stream generated by the custom NVIDIA Video Codec SDK encoder.
    """
    return os.environ.get("QGXS_EXTERNAL_H264_FIFO", "").strip()


def build_external_h264_fragment(config: SenderConfig, fifo_path: str) -> str:
    """
    Read an Annex-B H.264 byte stream from a named pipe and send it
    directly through the existing RTP/WebRTC path.

    This bypasses raw-video conversion and the GStreamer encoder.
    """
    if not is_h264(config.codec):
        raise ValueError(
            "QGXS_EXTERNAL_H264_FIFO currently supports H.264 only."
        )

    gst_path = escape_gst_string(
        normalize_path_for_gstreamer(fifo_path)
    )

    parts: list[str] = []

    parts.append(
        "appsrc "
        "name=external_h264_source "
        "is-live=true "
        "format=time "
        "do-timestamp=false "
        "block=true "
        "max-bytes=4194304 "
        f'caps="video/x-h264,'
        f'stream-format=byte-stream,'
        f'framerate={config.fps}/1"'
    )

    parts.append(
        "! queue "
        "max-size-buffers=16 "
        "max-size-bytes=0 "
        "max-size-time=0"
    )

    # Supply the configured frame rate to the parser. The custom encoder
    # outputs Annex-B byte-stream H.264.
    parts.append(
        f'! video/x-h264,'
        f'stream-format=byte-stream,'
        f'framerate={config.fps}/1'
    )

    parts.append(
        "! h264parse "
        "name=external_h264_parse "
        "config-interval=-1 "
        "disable-passthrough=true"
    )

    parts.append(
        f'! video/x-h264,'
        f'stream-format=byte-stream,'
        f'alignment=au,'
        f'framerate={config.fps}/1'
    )

    # Synchronize AU delivery against the GStreamer clock.
    parts.append(
        "! identity "
        "name=external_h264_pacer "
        "sync=true"
    )

    parts.append(
        "! rtph264pay "
        "name=rtp_pay "
        "pt=96 "
        "config-interval=1 "
        "mtu=1200 "
        "aggregate-mode=zero-latency"
    )

    parts.append(
        "! application/x-rtp,"
        "media=video,"
        "encoding-name=H264,"
        "payload=96,"
        "clock-rate=90000"
    )

    return " ".join(parts)


# ============================================================
# Full sender pipeline builder
# ============================================================

def build_sender_pipeline_description(config: SenderConfig) -> str:
    """
    Build the full GStreamer sender pipeline.

    When QGXS_EXTERNAL_H264_FIFO is set, the pipeline bypasses the
    GStreamer encoder and consumes the H.264 stream generated by the
    custom NVIDIA Video Codec SDK encoder.
    """
    parts: list[str] = []

    parts.append(
        "webrtcbin name=sender_webrtc "
        "bundle-policy=max-bundle "
        "latency=100"
    )

    external_fifo = external_h264_fifo_path()

    if external_fifo:
        print(
            "[pipeline] External NVIDIA SDK H.264 source: "
            f"{external_fifo}",
            flush=True,
        )
        parts.append(
            build_external_h264_fragment(
                config,
                external_fifo,
            )
        )
    else:
        parts.append(build_source_fragment(config))
        parts.append(build_common_processing_fragment(config))
        parts.append(build_codec_fragment(config))

    parts.append(
        "! identity "
        "name=rtp_counter "
        "silent=true "
        "signal-handoffs=true"
    )

    parts.append("! sender_webrtc.")

    return " ".join(parts)
