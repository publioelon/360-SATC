from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from core import area_weights, tile_edges, classify, block_classes
from map_projection import projection_matrices, reference_scores, compact_scores

EXPECTED_SHA256 = '51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa'


def normalize_model_frame(frame):
    """Convert one model input frame to the exact BGR/255 representation.

    Source frames are normalized once when they enter the causal history. This
    removes repeated conversion of the same history frames at every inference.
    """
    values = np.asarray(frame)
    if values.shape != (3, 144, 192) or values.dtype != np.uint8:
        raise RuntimeError(f'Unexpected model history frame: {values.shape}, {values.dtype}')
    normalized = np.empty(values.shape, np.float32)
    np.divide(values, np.float32(255), out=normalized, casting='unsafe')
    return normalized


def history_cache_slots(history_ids, cache_size=192):
    """Return cache slots and reject aliasing within one causal window."""
    ids = np.asarray(history_ids, dtype=np.int64)
    if ids.shape != (20,):
        raise RuntimeError('MoST-Sal requires exactly 20 causal history identifiers.')
    slots = ids % cache_size
    slot_owners = {}
    for frame_id, slot in zip(ids.tolist(), slots.tolist()):
        if slot in slot_owners and slot_owners[slot] != frame_id:
            raise RuntimeError('GPU history cache aliases distinct causal frames.')
        slot_owners[slot] = frame_id
    return ids, slots


def normalize_saliency_map(scores):
    """Map finite raw scores to [0, 1], following the supplied deployment.

    A raw score is not a probability and may be negative. Constant maps
    become zero maps, as in normalize_01 in the supplied evaluation script.
    Normalize before resizing and ERP weighting in this live implementation.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise RuntimeError(f'Unexpected saliency map shape: {values.shape}')
    if not np.isfinite(values).all():
        raise RuntimeError('MoST-Sal returned nonfinite values.')
    minimum, maximum = float(values.min()), float(values.max())
    if maximum == minimum:
        normalized = np.zeros(values.shape, np.float32)
    else:
        normalized = ((values - minimum) / (maximum - minimum)).astype(np.float32)
    return normalized, minimum, maximum


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def preprocess_nv12(raw, width, height):
    """BGR / 255, matching the supplied deployment's actual channel order.

    Bilinear reduction to 320x240 then 192x144. Explicit BT.709 limited-range
    conversion avoids OpenCV's implicit BT.601 YUV conversion. Chroma is
    resized from its subsampled plane; this convention is part of this runner.
    """
    import cv2
    y = np.frombuffer(raw, np.uint8, width * height).reshape(height, width)
    uv = np.frombuffer(raw, np.uint8, width * height // 2,
                       offset=width * height).reshape(height // 2, width // 2, 2)
    y = cv2.resize(y, (320, 240), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    uv = cv2.resize(uv, (320, 240), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    yy = (y - 16) * (255 / 219)
    u, v = (uv[:, :, 0] - 128) * (255 / 224), (uv[:, :, 1] - 128) * (255 / 224)
    bgr = np.stack([yy + 1.8556 * u, yy - .187324 * u - .468124 * v,
                    yy + 1.5748 * v], axis=-1)
    bgr = np.clip(np.rint(bgr), 0, 255).astype(np.uint8)
    low = cv2.resize(bgr, (192, 144), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(low.transpose(2, 0, 1))


class LiveModel:
    def __init__(self, model_path, out, provider='auto', map_backend='compact-cpu'):
        import cv2
        self.cv2 = cv2
        cv2.setNumThreads(2)
        self.digest = sha256(model_path)
        if self.digest != EXPECTED_SHA256:
            raise RuntimeError('The model differs from the uploaded inventory; refusing to assume its input convention.')
        # Loading torch first supplies the CUDA/cuDNN libraries installed in
        # the existing environment. No dependency is downloaded or upgraded.
        import torch
        self.torch = torch
        self.stream = torch.cuda.Stream(device=0)
        self.map_backend = map_backend
        self.audit_raw_maps = map_backend != 'reference'
        self.out = Path(out)
        import onnxruntime as ort
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        available = ort.get_available_providers()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        providers = []
        if provider != 'cuda' and 'TensorrtExecutionProvider' in available:
            cache = Path(out) / 'trt_cache'
            cache.mkdir(exist_ok=True)
            providers.append(('TensorrtExecutionProvider', {
                'device_id': 0, 'trt_fp16_enable': True,
                'user_compute_stream': str(self.stream.cuda_stream),
                'trt_engine_cache_enable': True, 'trt_engine_cache_path': str(cache),
                'trt_max_workspace_size': 1073741824}))
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError('CUDAExecutionProvider is unavailable in the selected Python environment.')
        providers += [('CUDAExecutionProvider', {'device_id': 0,
                       'user_compute_stream': str(self.stream.cuda_stream)}), 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        actual = self.session.get_providers()
        if actual[0] == 'CPUExecutionProvider':
            raise RuntimeError('ONNX Runtime fell back to CPU; no live experiment was started.')
        if provider == 'tensorrt' and actual[0] != 'TensorrtExecutionProvider':
            raise RuntimeError('The requested TensorRT provider was not activated.')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 20, 3, 144, 192] or inputs[0].type != 'tensor(float)':
            raise RuntimeError('Unexpected MoST-Sal input interface.')
        if outputs[0].shape != [1, 20, 1, 144, 192]:
            raise RuntimeError('Unexpected MoST-Sal output interface.')
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.ema = None
        self.weights = area_weights(2048).astype(np.float32)
        self.xe, self.ye = tile_edges(4096, 2048)
        self.matrices = projection_matrices()
        self.map_host = torch.empty((144,192), dtype=torch.float32, pin_memory=True)
        # A causal window spans 152 source frames. A 192-slot cache therefore
        # retains every history frame until its final possible use.
        self.history_cache_size = 192
        self.history_cache_ids = np.full(self.history_cache_size, -1, np.int64)
        self.history_indices_host = torch.empty((20,), dtype=torch.int64, pin_memory=True)
        self.history_indices_numpy = self.history_indices_host.numpy()
        with torch.cuda.stream(self.stream):
            self.input_gpu = torch.empty((1,20,3,144,192), dtype=torch.float32, device='cuda:0')
            self.output_gpu = torch.empty((1,20,1,144,192), dtype=torch.float32, device='cuda:0')
            self.history_gpu = torch.empty((self.history_cache_size,3,144,192),
                                           dtype=torch.float32, device='cuda:0')
            self.history_indices_gpu = torch.empty((20,), dtype=torch.int64, device='cuda:0')
            self.left_gpu = torch.as_tensor(self.matrices[0], dtype=torch.float64, device='cuda:0')
            self.right_gpu = torch.as_tensor(self.matrices[1], dtype=torch.float64, device='cuda:0')
        self.stream.synchronize()
        self.binding = self.session.io_binding()
        self.binding.bind_input(self.input_name,'cuda',0,np.float32,
                                tuple(self.input_gpu.shape),self.input_gpu.data_ptr())
        self.binding.bind_output(self.output_name,'cuda',0,np.float32,
                                 tuple(self.output_gpu.shape),self.output_gpu.data_ptr())
        sample = np.zeros((1, 20, 3, 144, 192), np.float32)
        warmup = []
        for _ in range(10):
            start = time.monotonic_ns()
            self.session.run(None, {self.input_name: sample})
            warmup.append((time.monotonic_ns() - start) / 1e6)
        metadata = {
            'model': str(model_path), 'sha256': self.digest,
            'onnxruntime': ort.__version__, 'providers': actual,
            'provider_options': self.session.get_provider_options(),
            'channel_order': 'BGR as implemented in supplied zero-shot deployment',
            'input_scale': 'uint8 / 255', 'input_shape': inputs[0].shape,
            'history_stride_source_frames': 8, 'history_frames': 20,
            'map_policy': 'one causal inference per admitted ROI frame; inference for the next frame overlaps encoding',
            'map_backend': map_backend,
            'optimization_version': 'gpu-history-cache-compact-projection-v5',
            'map_implementation': 'fixed linear resize/blur/latitude/tile operators composed into two small matrices; temporal smoothing and discrete class rules unchanged',
            'decision_validation': 'all actual raw maps retained locally; every inferred class map checked offline against full-resolution reference calculation before accepting the session',
            'output_normalization': '(last_map - min(last_map)) / (max(last_map) - min(last_map)); constant maps become zero; nonfinite values are errors',
            'output_processing_order': 'last output map, min-max normalization, bilinear resize, Gaussian blur, ERP latitude weighting, tile means, temporal smoothing, importance classes',
            'normalization_reference': 'supplied provenance/047_Run_FlowSal_v1_1_VREyeTracking_ZeroShot_v1.sh, normalize_01; this runner normalizes the last low resolution map before resizing',
            'input_preparation': 'each 3x144x192 uint8 BGR frame is converted to float32/255 once when added to causal history; normalized frames are cached on the GPU for their complete causal lifetime and gathered into the bound 20-frame input tensor',
            'timing': 'synchronized run_with_iobinding wall time including input transfer and last-map copy to host; includes provider fallback; not isolated GPU kernel time',
            'warmup_call_ms': warmup,
        }
        (Path(out) / 'model_runtime.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f"MoST-Sal: {actual[0]}, input [1,20,3,144,192], BGR/255", flush=True)

    def reset(self):
        self.ema = None
        self.history_cache_ids.fill(-1)

    def scores_from_raw(self, raw_map):
        sal, raw_min, raw_max = normalize_saliency_map(raw_map)
        if self.map_backend == 'reference':
            scores = reference_scores(sal)
        elif self.map_backend == 'compact-cpu':
            scores = compact_scores(sal,self.matrices)
        else:
            torch = self.torch
            with torch.cuda.stream(self.stream):
                # Normalize with the same float64 -> float32 convention as
                # the CPU oracle. The GPU result remains in device memory.
                raw = self.output_gpu[0,-1,0].to(torch.float64)
                normalized = torch.zeros_like(raw,dtype=torch.float32) if raw_max == raw_min else (
                    (raw-raw_min)/(raw_max-raw_min)).to(torch.float32)
                scores_gpu = (self.left_gpu @ normalized.to(torch.float64)) @ self.right_gpu
                scores = scores_gpu.cpu().numpy()
            self.stream.synchronize()
        return scores, raw_min, raw_max

    def infer(self, history, history_ids):
        started = time.monotonic_ns()
        if len(history) != 20:
            raise RuntimeError('MoST-Sal requires exactly 20 causal history frames and identifiers.')
        # Repeated frame zero during startup is intentionally represented by
        # one slot; distinct identifiers may not alias.
        ids, slots = history_cache_slots(history_ids, self.history_cache_size)
        self.history_indices_numpy[:] = slots
        torch = self.torch
        with torch.cuda.stream(self.stream), torch.no_grad():
            for frame_id, slot, frame in zip(ids, slots, history):
                if self.history_cache_ids[slot] == frame_id:
                    continue
                values = np.asarray(frame)
                if values.shape != (3,144,192) or values.dtype != np.float32:
                    raise RuntimeError(f'Unexpected normalized history frame: {values.shape}, {values.dtype}')
                self.history_gpu[slot].copy_(torch.from_numpy(values), non_blocking=True)
                self.history_cache_ids[slot] = frame_id
            self.history_indices_gpu.copy_(self.history_indices_host, non_blocking=True)
            torch.index_select(self.history_gpu, 0, self.history_indices_gpu,
                               out=self.input_gpu[0])
            self.stream.synchronize()
        call_start = time.monotonic_ns()
        with torch.cuda.stream(self.stream), torch.no_grad():
            self.session.run_with_iobinding(self.binding)
            self.binding.synchronize_outputs()
            self.map_host.copy_(self.output_gpu[0,-1,0],non_blocking=True)
        self.stream.synchronize()
        # Keep the actual output for offline decision equivalence checks.
        self.last_raw = self.map_host.numpy().copy()
        call_end = time.monotonic_ns()
        scores, raw_min, raw_max = self.scores_from_raw(self.last_raw)
        self.last_scores = scores.copy()
        self.ema = scores if self.ema is None else .7 * self.ema + .3 * scores
        classes = classify(self.ema)
        qp = np.array([2, 0, -2], dtype=np.int8)[block_classes(classes, 4096, 2048)]
        end = time.monotonic_ns()
        return classes, qp, {'input_tensor_start_ns': started, 'inference_start_ns': call_start,
                             'inference_end_ns': call_end, 'map_ready_ns': end,
                             'raw_saliency_min': raw_min, 'raw_saliency_max': raw_max}

    def validate_bound_inference(self, history):
        """Outside timing: same input/model/provider, old vs bound inference."""
        tensor = np.stack(history).astype(np.float32)[None] / 255.0
        reference = self.session.run([self.output_name],{self.input_name:tensor})[0][0,-1,0]
        normalized_history = [normalize_model_frame(frame) for frame in history]
        self.reset()
        classes, qp, stamps = self.infer(normalized_history, [0] * 20)
        actual_input = self.input_gpu.detach().cpu().numpy()
        if not np.array_equal(tensor,actual_input):
            raise RuntimeError('Optimized input preparation changed model input values.')
        error = float(np.max(np.abs(reference.astype(np.float64)-self.last_raw)))
        normalized,_,_=normalize_saliency_map(self.last_raw)
        full_scores=reference_scores(normalized)
        score_error=float(np.max(np.abs(full_scores-self.last_scores)))
        identical=np.array_equal(classify(full_scores),classes)
        report={'raw_output_max_abs_error':error,'score_max_abs_error':score_error,
                'first_frame_class_map_identical':bool(identical),
                'model_input_bitwise_identical':True,
                'raw_output_bitwise_identical':bool(np.array_equal(reference,self.last_raw)),
                'scope':'single actual source frame preflight; every run has an additional complete decision audit'}
        (self.out/'optimization_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if error != 0 or score_error > 1e-6 or not identical:
            raise RuntimeError('Optimization preflight differs from reference; inspect optimization_preflight.json.')
        self.reset()
        return classes,qp,stamps
