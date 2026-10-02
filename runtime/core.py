"""Frame identity, region definitions and measurement arithmetic.

No hardware or GStreamer imports: these rules are independently testable.
"""
from __future__ import annotations
import math
import re
import struct
import zlib
import numpy as np

UUID = bytes.fromhex('81a445588f294a1caa92aa720bb6085a')
START = re.compile(b'\x00\x00(?:\x00)?\x01')
INPUT = struct.Struct('<4sQII')       # magic, source ID, raw size, QP size
OUTPUT = struct.Struct('<4sQQQI')    # magic, source ID, encode start/end ns, AU size


def read_exact(stream, length):
    result = bytearray(length)
    view = memoryview(result)
    offset = 0
    while offset < length:
        n = stream.readinto(view[offset:])
        if not n:
            if not offset:
                raise EOFError('stream ended')
            raise EOFError(f'truncated record: {offset}/{length} bytes')
        offset += n
    return result


def write_all(stream, data):
    view = memoryview(data)
    while view:
        n = stream.write(view)
        if not n:
            raise BrokenPipeError('short pipe write')
        view = view[n:]


def rbsp_escape(raw):
    out = bytearray()
    zeros = 0
    for value in raw:
        if zeros >= 2 and value <= 3:
            out.append(3)
            zeros = 0
        out.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return bytes(out)


def rbsp_unescape(raw):
    out = bytearray()
    zeros = 0
    for value in raw:
        if zeros >= 2 and value == 3:
            zeros = 0
            continue
        out.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return bytes(out)


def nal_units(annexb):
    matches = list(START.finditer(annexb))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(annexb)
        nal = annexb[match.end():end]
        if nal:
            yield match.start(), nal


def insert_frame_id(annexb, frame_id):
    value = struct.pack('>Q', frame_id)
    payload = UUID + value + struct.pack('>I', zlib.crc32(value))
    sei = b'\x00\x00\x00\x01\x06' + rbsp_escape(bytes([5, len(payload)]) + payload + b'\x80')
    for offset, nal in nal_units(annexb):
        if nal[0] & 31 in (1, 5):
            return annexb[:offset] + sei + annexb[offset:]
    raise ValueError('NVENC packet contains no H.264 picture')


def extract_frame_ids(annexb):
    """Return every valid 360-SATC frame identifier in Annex B order."""
    ids = []
    for _, nal in nal_units(annexb):
        if nal[0] & 31 != 6:
            continue
        data = rbsp_unescape(nal[1:])
        i = 0
        while i < len(data) and data[i:] != b'\x80':
            kind = size = 0
            while i < len(data) and data[i] == 255:
                kind += 255
                i += 1
            if i >= len(data):
                break
            kind += data[i]
            i += 1
            while i < len(data) and data[i] == 255:
                size += 255
                i += 1
            if i >= len(data):
                break
            size += data[i]
            i += 1
            payload = data[i:i + size]
            i += size
            if kind == 5 and len(payload) == 28 and payload[:16] == UUID:
                value = payload[16:24]
                if zlib.crc32(value) != struct.unpack('>I', payload[24:])[0]:
                    raise ValueError('frame identifier CRC mismatch')
                ids.append(struct.unpack('>Q', value)[0])
    return ids


def extract_frame_id(annexb):
    ids = extract_frame_ids(annexb)
    if len(set(ids)) > 1:
        raise ValueError('multiple source frames in one access unit')
    return ids[0] if ids else None


def history_ids(frame_id, count=20, stride=8):
    return [max(0, frame_id - stride * i) for i in reversed(range(count))]


def tile_edges(width, height, rows=5, cols=9):
    tw = max(16, (width // cols // 16) * 16)
    th = max(16, (height // rows // 16) * 16)
    if tw * (cols - 1) >= width or th * (rows - 1) >= height:
        raise ValueError('frame is too small for the block aligned logical grid')
    return np.array([i * tw for i in range(cols)] + [width]), np.array([i * th for i in range(rows)] + [height])


def classify(scores, fraction=.10):
    scores = np.asarray(scores)
    if scores.shape != (5, 9) or not np.isfinite(scores).all():
        raise ValueError('expected finite 5 by 9 importance scores')
    # C++ std::round for positive x; Python round uses a different tie rule.
    k = max(1, int(math.floor(scores.size * fraction + .5)))
    threshold = np.partition(scores.ravel(), -k)[-k]
    high = scores >= threshold  # ties are included and resulting areas are recorded
    cls = np.zeros(scores.shape, dtype=np.uint8)
    for r, c in zip(*np.where(high)):
        for dr in (-1, 0, 1):
            if 0 <= r + dr < 5:
                for dc in (-1, 0, 1):
                    cls[r + dr, (c + dc) % 9] = 1
    cls[high] = 2
    return cls


def block_classes(classes, width, height):
    xe, ye = tile_edges(width, height)
    x = np.minimum(np.arange((width + 15) // 16) * 16 + 8, width - 1)
    y = np.minimum(np.arange((height + 15) // 16) * 16 + 8, height - 1)
    tx = np.searchsorted(xe[1:], x, side='right')
    ty = np.searchsorted(ye[1:], y, side='right')
    return np.asarray(classes)[ty[:, None], tx[None, :]]


def pixel_classes(classes, width, height):
    return np.repeat(np.repeat(block_classes(classes, width, height), 16, 0), 16, 1)[:height, :width]


def area_weights(height):
    # cos(latitude) = sin(colatitude), with pixel centers rather than row edges.
    return np.sin(np.pi * (np.arange(height, dtype=np.float64) + .5) / height)


def regional_errors(reference, reconstructed, classes):
    ref, rec = np.asarray(reference), np.asarray(reconstructed)
    if ref.shape != rec.shape or ref.ndim != 2 or ref.dtype != np.uint8 or rec.dtype != np.uint8:
        raise ValueError('quality requires aligned, equally sized uint8 luma planes')
    h, w = ref.shape
    masks = pixel_classes(classes, w, h)
    error = (ref.astype(np.float64) - rec.astype(np.float64)) ** 2
    weights = np.broadcast_to(area_weights(h)[:, None], (h, w))
    result = {}
    for name, label in [('high', 2), ('medium', 1), ('low', 0), ('full', None)]:
        keep = np.ones((h, w), bool) if label is None else masks == label
        result[name] = (float((error[keep] * weights[keep]).sum()), float(weights[keep].sum()))
    return result


def psnr(weighted_error, weighted_area):
    if weighted_area <= 0:
        return None
    if weighted_error == 0:
        return math.inf
    return 10 * math.log10(255.0 ** 2 * weighted_area / weighted_error)


def fps_summary(events, start_ns, end_ns):
    if end_ns <= start_ns:
        raise ValueError('measurement duration must be positive')
    # Keep only the first observation of each identity, including those before
    # the window. A duplicate from warm-up cannot become a measurement frame.
    first = {}
    for frame_id, ns in events:
        if frame_id is not None:
            first[frame_id] = min(first.get(frame_id, ns), ns)
    times = sorted(t for t in first.values() if start_ns <= t < end_ns)
    duration = (end_ns - start_ns) / 1e9
    bins = []
    for i in range(math.ceil(duration)):
        lo = start_ns + int(i * 1e9)
        hi = min(end_ns, lo + 1_000_000_000)
        bins.append(sum(lo <= t < hi for t in times) * 1e9 / (hi - lo))
    gaps = np.diff([start_ns, *times, end_ns]) / 1e6
    return {'unique_frames': len(times), 'fps': len(times) / duration,
            'fps_by_second': bins, 'largest_gap_ms_including_edges': float(max(gaps)),
            'duplicate_events_total': len(events) - len(first)}
