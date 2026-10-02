#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${1:-9001}"
SINK="${2:-auto}"
DECODER="${3:-auto}"

export QGXS_USE_SYSTEM_GSTREAMER=1
export GST_DEBUG_NO_COLOR=1

exec python3 -u \
  "$ROOT/receiver/linux/webrtc_receiver.py" \
  --listen-host 0.0.0.0 \
  --port "$PORT" \
  --feedback-port "$((PORT + 100))" \
  --feedback-interval-ms 500 \
  --latency-ms 100 \
  --sink "$SINK" \
  --decoder "$DECODER"
