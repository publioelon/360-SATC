import os
import cv2
import numpy as np
import time
import statistics
import argparse
import torch
from torch.amp import autocast
import onnxruntime as ort

# Configurations
RAFT_HEIGHT = 240
RAFT_WIDTH = 320
DEVICE = torch.device("cuda")

def load_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_frames_from_dir(frames_dir):
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))])
    frames = [cv2.imread(f) for f in frame_files]
    return frames

def run_raft_benchmark(frames, raft_model_path, mixed_precision):
    # 1. Model loading time
    t0 = time.perf_counter()
    sess = ort.InferenceSession(raft_model_path, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    t1 = time.perf_counter()
    model_load_time = t1 - t0

    # PRINT PROVIDERS USED
    print("\nONNX Runtime - Providers available and used for this session:")
    for idx, provider in enumerate(sess.get_providers()):
        print(f"  [{idx+1}] {provider}")
    print(f"Primary inference provider: {sess.get_providers()[0]}\n")

    # 2. Dummy input for warmup and name
    dummy = np.zeros((1,3,RAFT_HEIGHT,RAFT_WIDTH), np.float32)
    inp_names = [i.name for i in sess.get_inputs()]

    # 3. First inference time
    t2 = time.perf_counter()
    sess.run(None, {inp_names[0]: dummy, inp_names[1]: dummy})
    t3 = time.perf_counter()
    first_inference_time = t3 - t2

    # 4. Real inference timings
    per_frame_times = []
    num_frames = len(frames) - 1
    for idx in range(num_frames):
        img1 = cv2.resize(frames[idx], (RAFT_WIDTH, RAFT_HEIGHT))
        img2 = cv2.resize(frames[idx+1], (RAFT_WIDTH, RAFT_HEIGHT))

        inp1 = torch.from_numpy(img1.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0
        inp2 = torch.from_numpy(img2.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0

        # Use autocast for mixed precision inference
        with autocast(device_type='cuda', enabled=mixed_precision):
            t_start = time.perf_counter()
            o1, o2 = inp1.cpu().numpy(), inp2.cpu().numpy()
            sess.run(None, {inp_names[0]: o1, inp_names[1]: o2})
            t_end = time.perf_counter()
            per_frame_times.append(t_end - t_start)

    mean_time = np.mean(per_frame_times)
    std_time = np.std(per_frame_times)
    median_time = np.median(per_frame_times)

    return {
        "model_load_time": model_load_time,
        "first_inference_time": first_inference_time,
        "mean_time": mean_time,
        "std_time": std_time,
        "median_time": median_time,
        "num_frames": num_frames
    }

def print_report(name, stats):
    print(f"\n--- {name} ---")
    print(f"Model loading time:     {stats['model_load_time']:.4f} s")
    print(f"First inference time:   {stats['first_inference_time']:.4f} s")
    print(f"Mean inference time:    {stats['mean_time']*1000:.2f} ms")
    print(f"Std deviation:          {stats['std_time']*1000:.2f} ms")
    print(f"Median inference time:  {stats['median_time']*1000:.2f} ms")
    print(f"Frames processed:       {stats['num_frames']}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark SEA-RAFT ONNX on video vs image frames.")
    parser.add_argument("--video_path", type=str, default="360_video.mp4", help="Input video path")
    parser.add_argument("--frames_dir", type=str, help="Directory of pre-extracted frames (jpg/png)")
    parser.add_argument("--raft_model_path", type=str, required=True, help="Path to SEA-RAFT .onnx model")
    parser.add_argument("--mixed_precision", type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()

    if not os.path.exists(args.raft_model_path):
        raise FileNotFoundError(f"Model not found: {args.raft_model_path}")

    # VIDEO method
    print(f"Loading video: {args.video_path}")
    frames_video = load_frames_from_video(args.video_path)
    stats_video = run_raft_benchmark(frames_video, args.raft_model_path, args.mixed_precision)
    print_report("Video (in-memory decoding)", stats_video)

    # FRAMES method
    if args.frames_dir:
        print(f"\nLoading frames from directory: {args.frames_dir}")
        frames_dir = load_frames_from_dir(args.frames_dir)
        stats_frames = run_raft_benchmark(frames_dir, args.raft_model_path, args.mixed_precision)
        print_report("Frames (pre-extracted images)", stats_frames)
    else:
        print("\nNo frames_dir specified; skipping frame-folder benchmark.")

if __name__ == "__main__":
    main()
