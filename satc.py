#!/usr/bin/env python3
"""360-SATC: check dependencies, prepare a clip, encode, and launch streaming."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'saliency/models/most_sal_144x192.onnx'
MODEL_SHA = '51a49cf5cc9ead1bef4a60e9c454d164c1eca6733f0ea2039ab84bdc3bbbbafa'


def unpack_model():
    if not MODEL.exists():
        data = gzip.decompress(MODEL.with_suffix('.onnx.gz').read_bytes())
        if hashlib.sha256(data).hexdigest() != MODEL_SHA:
            raise RuntimeError('Bundled MoST-Sal checksum failed.')
        temp = MODEL.with_suffix('.tmp')
        temp.write_bytes(data)
        temp.replace(MODEL)
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA:
        raise RuntimeError('MoST-Sal model checksum failed.')
    return MODEL


def run(command):
    return subprocess.run([str(x) for x in command], check=True)


def doctor(args):
    failures = []
    checks = ['ffmpeg', 'ffprobe', 'nvidia-smi'] if args.streaming_only else ['ffmpeg', 'ffprobe', 'cmake', 'c++', 'nvcc', 'nvidia-smi']
    if args.streaming or args.streaming_only:
        checks += ['gst-inspect-1.0']
    for name in checks:
        found = shutil.which(name)
        print(f"{'OK' if found else 'MISSING'}  {name}: {found or 'install or add to PATH'}")
        if not found: failures.append(name)
    print('Python:', sys.executable)
    print('Model:', unpack_model())
    for name in (['gi'] if args.streaming_only else ['numpy', 'cv2', 'torch', 'onnxruntime'] + (['gi'] if args.streaming else [])):
        try:
            __import__(name)
            print('OK ', name)
        except Exception as error:
            failures.append(name)
            print('MISSING', name, str(error))
    try:
        if args.streaming_only: raise ImportError
        import torch
        if not torch.cuda.is_available():
            failures.append('CUDA')
        import onnxruntime as ort
        print('ONNX providers:', ', '.join(ort.get_available_providers()))
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            failures.append('CUDAExecutionProvider')
    except ImportError:
        pass
    sdk = args.sdk or os.environ.get('SATC_SDK')
    if not args.streaming_only and (not sdk or not (Path(sdk).expanduser() / 'Samples/NvCodec/NvEncoder/NvEncoderCuda.cpp').is_file()):
        failures.append('Video Codec SDK')
        print('MISSING SDK: set SATC_SDK to your Video_Codec_SDK_13.0.19 directory')
    if (args.streaming or args.streaming_only) and shutil.which('gst-inspect-1.0'):
        for name in ['webrtcbin', 'rtpgccbwe', 'nvh264enc', 'nvh265enc', 'nvav1enc']:
            result = subprocess.run(['gst-inspect-1.0', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('OK' if result.returncode == 0 else 'MISSING', name)
            if result.returncode: failures.append(name)
    return 1 if failures else 0


def prepare(args):
    if not args.input.is_file():
        raise ValueError(f'Input not found: {args.input}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-nostdin', '-n', '-i', args.input, '-an', '-vf',
               'scale=4096:2048:flags=lanczos,fps=60,format=yuv420p',
               '-c:v', 'libx264', '-preset', 'fast', '-qp', '10',
               '-color_range', 'tv', '-colorspace', 'bt709',
               '-color_primaries', 'bt709', '-color_trc', 'bt709']
    if args.frames:
        command += ['-frames:v', str(args.frames)]
    run(command + [args.output])
    print('Prepared:', args.output)


def encode(args):
    if os.geteuid() == 0:
        raise RuntimeError('Run encoding without sudo.')
    if args.frames < 1 or args.warmup < 1:
        raise ValueError('Frame counts must be positive.')
    sys.path.insert(0, str(ROOT / 'runtime'))
    import encoder
    import core
    from policy import PaperPolicy
    from live_model import LiveModel, preprocess_nv12
    encoder.PRESET = args.preset
    original_history = core.history_ids
    core.history_ids = lambda fid: original_history(fid, stride=args.history_stride)
    model_path = args.model or unpack_model()
    sdk = encoder.find_sdk(args.sdk)
    encoder.validate_video(args.video, args.frames + args.warmup)
    args.output.mkdir(parents=True, exist_ok=False)
    code = args.output / 'code'
    code.mkdir()
    encoder.materialize(code)
    native = code / 'nvenc_uncapped.cpp'
    text = native.read_text()
    text = text.replace('NV_ENC_PRESET_P7_GUID', f'NV_ENC_PRESET_{args.preset.upper()}_GUID')
    text = text.replace('preset=p7', 'preset=' + args.preset)
    native.write_text(text)
    encoder.command(['cmake', '-S', code, '-B', args.output / 'build', f'-DSDK_TOP={sdk}', '-DCMAKE_BUILD_TYPE=Release'], args.output / 'configure.log')
    encoder.command(['cmake', '--build', args.output / 'build', '-j', '4'], args.output / 'build.log')
    model = LiveModel(model_path, args.output, provider=args.provider, map_backend='compact-cpu')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(args.video), '-an', '-frames:v', '1', '-pix_fmt', 'nv12', '-f', 'rawvideo', 'pipe:1']
    frame = subprocess.run(command, capture_output=True, check=True, timeout=60).stdout
    if len(frame) != encoder.FRAME_BYTES:
        raise ValueError('Prepared clip geometry mismatch.')
    model.validate_bound_inference([preprocess_nv12(frame, 4096, 2048)] * 20)
    encoder.ALLOCATION_POLICY = PaperPolicy(args.codec) if args.method == 'satc' else None
    job = {'id': 'encode', 'video_name': args.video.stem, 'video': str(args.video.resolve()),
           'codec': args.codec, 'target_mbps': args.bitrate, 'method': 'roi' if args.method == 'satc' else 'uniform'}
    result = encoder.run_one(SimpleNamespace(frames=args.frames, warmup_frames=args.warmup), args.output, job, model)
    if result.get('status') == 'VALID' and args.method == 'satc':
        import csv
        import numpy as np
        import zlib
        run_dir = args.output / 'runs/encode'
        with (run_dir / 'frame_events.csv').open(newline='') as stream:
            events = list(csv.DictReader(stream))
        scores = np.load(run_dir / 'map_audit/compact_scores.npy', allow_pickle=False)
        ema = None
        policy = PaperPolicy(args.codec)
        for fid, (score, event) in enumerate(zip(scores, events)):
            ema = score if ema is None else .7 * ema + .3 * score
            expected = policy.make_map(ema, None, args.codec, fid)
            if zlib.crc32(expected.tobytes()) != int(event['qp_crc32']):
                raise RuntimeError(f'Final-policy QP map mismatch at frame {fid}')
        if len(scores) != len(events):
            raise RuntimeError('Final-policy audit frame count mismatch')
        encoder.dump(run_dir / 'FINAL_POLICY_AUDIT.json', {'status': 'PASS', 'checked_frames': len(events), 'policy': 'six/six/33: -2/-1/0'})
    # Override the retained runner's descriptive p7 field when p4 is selected.
    result['preset'] = args.preset
    result['history_stride'] = args.history_stride
    result['spatial_policy'] = 'six/six/33: -2/-1/0' if args.method == 'satc' else 'zero QP delta'
    encoder.dump(args.output / 'summary.json', result)
    encoder.dump(args.output / 'runs/encode/summary.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result.get('status') == 'VALID' else 1


def stream(args):
    command = [args.python, ROOT / 'GXRStream/ubuntu/sender/webrtc_sender.py', args.codec,
               '--input-mode', 'video-file', '--input', args.video.resolve(), '--no-loop',
               args.host, str(args.port), '4096', '2048', '60', str(args.bitrate * 1000)]
    run(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    d = commands.add_parser('doctor', help='Check the selected Python/GPU environment')
    d.add_argument('--sdk', type=Path)
    d.add_argument('--streaming', action='store_true')
    d.add_argument('--streaming-only', action='store_true', help='Check GStreamer independently of the SATC Python environment')
    commands.add_parser('model', help='Unpack and verify the frozen ONNX model')
    p = commands.add_parser('prepare', help='Prepare a 4K60 BT.709 clip')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int)
    e = commands.add_parser('encode', help='Encode with final SATC or matched uniform CBR')
    e.add_argument('--video', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--sdk', type=Path)
    e.add_argument('--model', type=Path)
    e.add_argument('--codec', choices=['h264', 'hevc', 'av1'], default='h264')
    e.add_argument('--bitrate', type=int, choices=[12, 35], default=12, help='Mbit/s')
    e.add_argument('--method', choices=['satc', 'cbr'], default='satc')
    e.add_argument('--preset', choices=['p4', 'p7'], default='p7')
    e.add_argument('--frames', type=int, default=896)
    e.add_argument('--warmup', type=int, default=240)
    e.add_argument('--history-stride', type=int, choices=[1, 8], default=8)
    e.add_argument('--provider', choices=['auto', 'cuda', 'tensorrt'], default='auto')
    s = commands.add_parser('stream', help='Test GXRStream transport with a video file')
    s.add_argument('--video', type=Path, required=True)
    s.add_argument('--host', required=True, help='Quest/receiver IP')
    s.add_argument('--port', type=int, default=9001)
    s.add_argument('--codec', choices=['h264', 'h265', 'av1'], default='h264')
    s.add_argument('--bitrate', type=int, choices=[12, 35], default=12)
    s.add_argument('--python', default=sys.executable, help='Python with GObject/GStreamer support')
    args = parser.parse_args()
    try:
        if args.command == 'model': print(unpack_model()); return 0
        return {'doctor': doctor, 'prepare': prepare, 'encode': encode, 'stream': stream}[args.command](args) or 0
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print('ERROR:', error, file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
