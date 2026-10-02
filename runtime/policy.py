"""Final six/six/33 policy recovered from the completed quality campaign."""
import numpy as np
from core import tile_edges


def tile_qp(scores):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != (45,) or not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError('Expected 45 finite, nonnegative saliency scores.')
    order = np.argsort(-values, kind='stable')
    offsets = np.zeros(45, dtype=np.int8)
    offsets[order[:6]] = -2
    offsets[order[6:12]] = -1
    return offsets


class PaperPolicy:
    def __init__(self, codec, width=4096, height=2048):
        block = {'h264': 16, 'hevc': 32, 'av1': 64}[codec]
        xe, ye = tile_edges(width, height)
        x = np.minimum(np.arange((width + block - 1) // block) * block + block // 2, width - 1)
        y = np.minimum(np.arange((height + block - 1) // block) * block + block // 2, height - 1)
        tx = np.searchsorted(xe[1:], x, side='right')
        ty = np.searchsorted(ye[1:], y, side='right')
        self.block_ids = ty[:, None] * 9 + tx[None, :]
        self.codec = codec

    def make_map(self, scores, raw, codec, frame_id):
        if codec != self.codec:
            raise ValueError('Policy/encoder codec mismatch.')
        return np.ascontiguousarray(tile_qp(scores)[self.block_ids], dtype=np.int8)
