#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-/home/$USER/Videos/london_tower_4k_120.mp4}"
UNITY_HOST="${2:-127.0.0.1}"
BITRATE_KBPS="${3:-50000}"

export QGXS_USE_SYSTEM_GSTREAMER=1
export QGXS_ENCODER=nvenc
export QGXS_REQUIRE_NVENC=1

python3 -u ./sender/webrtc_sender.py h265 \
  --input-mode video-file \
  --input "$VIDEO_PATH" \
  --image-format auto \
  --loop \
  "$UNITY_HOST" 9001 4096 2048 120 "$BITRATE_KBPS"
