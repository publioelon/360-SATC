import os
import time
import argparse
import torch
from torch.amp import autocast
import numpy as np

RAFT_HEIGHT = 240
RAFT_WIDTH = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sst_sal_model(path):
    model = torch.load(path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE).eval()
    return model

def warm_up_sst_sal(model, mixed_precision):
    inp = torch.zeros((1, 20, 6, RAFT_HEIGHT, RAFT_WIDTH), device=DEVICE)
    with torch.no_grad(), autocast(device_type='cuda', enabled=mixed_precision):
        _ = model(inp)
    torch.cuda.synchronize()

def benchmark_sst_sal(model, n_repeats=50, mixed_precision=True):
    dummy_rgb  = torch.rand((1, 20, 3, RAFT_HEIGHT, RAFT_WIDTH), device=DEVICE)
    dummy_flow = torch.rand((1, 20, 3, RAFT_HEIGHT, RAFT_WIDTH), device=DEVICE)
    inp = torch.cat([dummy_rgb, dummy_flow], dim=2)  # Shape: (1, 20, 6, 240, 320)
    
    times = []
    with torch.no_grad(), autocast(device_type='cuda', enabled=mixed_precision):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(inp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        first_inf = t1 - t0

        # Real benchmarking loop
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            _ = model(inp)
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            times.append(t_end - t_start)

    mean_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    return {
        "first_inference_time": first_inf,
        "mean_time": mean_time,
        "std_time": std_time,
        "median_time": median_time,
        "n_repeats": n_repeats
    }

def print_report(stats):
    print("\n--- SST-SAL Benchmark (random input) ---")
    print(f"First inference time:   {stats['first_inference_time']*1000:.2f} ms")
    print(f"Mean inference time:    {stats['mean_time']*1000:.2f} ms")
    print(f"Std deviation:          {stats['std_time']*1000:.2f} ms")
    print(f"Median inference time:  {stats['median_time']*1000:.2f} ms")
    print(f"Runs:                   {stats['n_repeats']}")

def main():
    parser = argparse.ArgumentParser("Benchmark SST-SAL PyTorch model using random input")
    parser.add_argument("--sst_sal_model_path", type=str, required=True, help="Path to SST-SAL .pth model")
    parser.add_argument("--n_repeats", type=int, default=50, help="Number of inference runs")
    parser.add_argument("--mixed_precision", type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()

    print(f"Loading SST-SAL model from: {args.sst_sal_model_path}")
    t_load0 = time.perf_counter()
    model = load_sst_sal_model(args.sst_sal_model_path)
    t_load1 = time.perf_counter()
    model_load_time = t_load1 - t_load0
    print(f"Model load time: {model_load_time:.4f} s")

    print("Warming up model...")
    warm_up_sst_sal(model, mixed_precision=args.mixed_precision)

    print("Benchmarking SST-SAL inference on random inputs...")
    stats = benchmark_sst_sal(model, n_repeats=args.n_repeats, mixed_precision=args.mixed_precision)
    print_report(stats)

if __name__ == "__main__":
    main()
