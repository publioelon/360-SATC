#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip python3-tk \
  python3-gi python3-gi-cairo \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gir1.2-gst-plugins-bad-1.0 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  gstreamer1.0-nice \
  python3-pil

echo

echo "Base Linux sender dependencies installed."
echo "Now run: ./scripts/linux/check_sender_runtime.sh"
echo "For 120 FPS H.264/H.265, gst-inspect-1.0 nvh264enc and nvh265enc must succeed."
