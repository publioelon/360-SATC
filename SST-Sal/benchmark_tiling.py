import os
import cv2
import numpy as np
import time
import argparse

IMG_HEIGHT = 240
IMG_WIDTH = 320
TILE_ROWS = 5
TILE_COLS = 9
N_REPEAT = 50

def tile_image(img, tile_rows, tile_cols):
    h, w = img.shape[:2]
    th, tw = h // tile_rows, w // tile_cols
    return img[:tile_rows*th, :tile_cols*tw].reshape(tile_rows, th, tile_cols, tw, 3).transpose(0,2,1,3,4)

def merge_tiles(tiles):
    tile_rows, tile_cols, th, tw, ch = tiles.shape
    return tiles.transpose(0,2,1,3,4).reshape(tile_rows*th, tile_cols*tw, ch)

def compress_tile_jpeg(tile, quality):
    success, buf = cv2.imencode('.jpg', tile, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    comp = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return comp

def benchmark_tiling_compression(n_repeat=50, tile_rows=TILE_ROWS, tile_cols=TILE_COLS, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, jpeg_quality=100):
    times = []
    for _ in range(n_repeat):
        img = np.random.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        tiles = tile_image(img, tile_rows, tile_cols)
        comp_tiles = np.empty_like(tiles)
        for r in range(tile_rows):
            for c in range(tile_cols):
                comp_tiles[r, c] = compress_tile_jpeg(tiles[r, c], jpeg_quality)
        _ = merge_tiles(comp_tiles)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "median_time": np.median(times),
        "n_repeat": n_repeat,
        "tiles_per_img": tile_rows * tile_cols
    }

def print_report(stats):
    print("\n--- Tiling+JPEG Compression Benchmark (random image input) ---")
    print(f"Mean per-image time:    {stats['mean_time']*1000:.2f} ms")
    print(f"Std deviation:          {stats['std_time']*1000:.2f} ms")
    print(f"Median per-image time:  {stats['median_time']*1000:.2f} ms")
    print(f"Images processed:       {stats['n_repeat']}")
    print(f"Tiles per image:        {stats['tiles_per_img']}")

def main():
    parser = argparse.ArgumentParser("Benchmark SATC tiling+compression using random images")
    parser.add_argument("--n_repeat", type=int, default=N_REPEAT, help="How many images to process (default: 50)")
    parser.add_argument("--jpeg_quality", type=int, default=75, help="JPEG quality for compression (default: 75)")
    parser.add_argument("--tile_rows", type=int, default=TILE_ROWS, help="Number of tile rows")
    parser.add_argument("--tile_cols", type=int, default=TILE_COLS, help="Number of tile columns")
    parser.add_argument("--img_height", type=int, default=IMG_HEIGHT, help="Input image height")
    parser.add_argument("--img_width", type=int, default=IMG_WIDTH, help="Input image width")
    args = parser.parse_args()

    stats = benchmark_tiling_compression(
        n_repeat=args.n_repeat,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        img_h=args.img_height,
        img_w=args.img_width,
        jpeg_quality=args.jpeg_quality
    )
    print_report(stats)

if __name__ == "__main__":
    main()
