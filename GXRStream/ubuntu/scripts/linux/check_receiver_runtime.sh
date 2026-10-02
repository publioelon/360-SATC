#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstSdp', '1.0')
gi.require_version('GstWebRTC', '1.0')
from gi.repository import Gst, GstSdp, GstWebRTC
Gst.init(None)
print('Python GStreamer WebRTC bindings: OK')
PY

required=(
  webrtcbin
  queue
  videoconvert
  rtph264depay
  h264parse
  rtph265depay
  h265parse
  rtpav1depay
  av1parse
)

printf '\n===== REQUIRED ELEMENTS =====\n'
for element in "${required[@]}"; do
  if gst-inspect-1.0 "$element" >/dev/null 2>&1; then
    printf 'FOUND   %s\n' "$element"
  else
    printf 'MISSING %s\n' "$element"
    exit 1
  fi
done

printf '\n===== DECODERS =====\n'
for element in \
  nvh264dec vah264dec qsvh264dec avdec_h264 \
  nvh265dec vah265dec qsvh265dec avdec_h265 \
  nvav1dec vaav1dec qsvav1dec dav1ddec av1dec
 do
  if gst-inspect-1.0 "$element" >/dev/null 2>&1; then
    printf 'FOUND   %s\n' "$element"
  else
    printf 'MISSING %s\n' "$element"
  fi
done

printf '\n===== VIDEO SINKS =====\n'
for element in glimagesink waylandsink ximagesink autovideosink fakesink; do
  if gst-inspect-1.0 "$element" >/dev/null 2>&1; then
    printf 'FOUND   %s\n' "$element"
  else
    printf 'MISSING %s\n' "$element"
  fi
done

printf '\n===== VERSIONS =====\n'
gst-launch-1.0 --version
python3 --version
