#!/usr/bin/env bash
set -uo pipefail

check() {
  echo
  echo "$ $*"
  "$@"
  local rc=$?
  echo "[exit $rc]"
  return $rc
}

check gst-inspect-1.0 --version
check gst-inspect-1.0 webrtcbin
check gst-inspect-1.0 rtph264pay
check gst-inspect-1.0 rtph265pay
check gst-inspect-1.0 h264parse
check gst-inspect-1.0 h265parse
check gst-inspect-1.0 nvh264enc
check gst-inspect-1.0 nvh265enc

check python3 - <<'PY'
import gi
gi.require_version('Gst','1.0')
gi.require_version('GstSdp','1.0')
gi.require_version('GstWebRTC','1.0')
from gi.repository import Gst, GstSdp, GstWebRTC
Gst.init(None)
print('Python GStreamer WebRTC OK')
PY

echo

echo "If nvh264enc/nvh265enc failed, the functional sender may still work with:"
echo "  QGXS_REQUIRE_NVENC=0 QGXS_ENCODER=auto ..."
echo "but the high-FPS target should use NVENC."
