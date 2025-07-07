import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse
import torch
import warnings
from torch.amp import autocast
import onnxruntime as ort
from utils.flow_viz import flow_to_image
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

RAFT_WIDTH  = 320
RAFT_HEIGHT = 240
TILE_ROWS   = 5
TILE_COLS   = 9

LOW_BANDWIDTH  = (0.21e6, 0.40e6)
HIGH_BANDWIDTH = (0.41e6, 0.70e6)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PATH = r'path/to/outputs'
OVERLAY_PATH = r'path/to/overlay/frames
SST_SAL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'SST-Sal', 'SST_Sal.pth')
RAFT_MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'SEA-RAFT', 'Tartan-C-T-TSKH-kitti432x960-S.onnx')

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(OVERLAY_PATH, exist_ok=True)

def generate_bandwidth_values(num_frames, mode):
    if mode == "low":
        bw = np.linspace(LOW_BANDWIDTH[0]*0.21/72, LOW_BANDWIDTH[1]*0.40/72, num_frames)
        qp = [{"red":100, "green":max(25,75-(i%20)), "black":max(10,50-(i%20))} for i in range(num_frames)]
    else:
        bw = np.linspace(HIGH_BANDWIDTH[0]*0.41/144, HIGH_BANDWIDTH[1]*0.70/144, num_frames)
        qp = [{"red":100, "green":min(100,95+(i%10)), "black":min(95,90+(i%5))} for i in range(num_frames)]
    return list(zip(bw, qp))

def load_sst_sal_model(path):
    m = torch.load(path, map_location=device, weights_only=False)
    m.to(device).eval()
    return m

def warm_up_sst_sal(model, mixed_precision):
    dummy_rgb  = torch.zeros((1,3,RAFT_HEIGHT,RAFT_WIDTH), device=device)
    dummy_flow = torch.zeros((1,3,RAFT_HEIGHT,RAFT_WIDTH), device=device)
    inp = torch.cat([dummy_rgb, dummy_flow], dim=1).unsqueeze(1).repeat(1,20,1,1,1)
    with torch.no_grad(), autocast(device_type='cuda', enabled=mixed_precision):
        _ = model(inp)
    torch.cuda.synchronize()

def adjust_dimensions(img, rows, cols):
    h, w = img.shape[:2]
    return img[: (h//rows)*rows, : (w//cols)*cols]

def tile_image(img, tile_rows, tile_cols):
    h, w = img.shape[:2]
    th, tw = h // tile_rows, w // tile_cols
    return img.reshape(tile_rows, th, tile_cols, tw, 3).transpose(0,2,1,3,4)

def merge_tiles(tiles):
    tile_rows, tile_cols, th, tw, ch = tiles.shape
    return tiles.transpose(0,2,1,3,4).reshape(tile_rows*th, tile_cols*tw, ch)

def calculate_psnr(orig, comp):
    diff = orig.astype(np.float32) - comp.astype(np.float32)
    mse = np.mean(diff*diff)
    return float('inf') if mse == 0.0 else 10 * np.log10((255.0**2) / mse)

def compress_tile_jpeg_time(tile, quality):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buf = cv2.imencode('.jpg', tile, params)
    if not success:
        raise RuntimeError("JPEG encoding failed")
    comp = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return comp

def compress_tile_worker(args):
    tile, quality = args
    comp = compress_tile_jpeg_time(tile, quality)
    psnr = calculate_psnr(tile, comp)
    return comp, psnr

def process_frame_with_saliency(frame, salmap, qp, executor):
    f_adj = adjust_dimensions(frame, TILE_ROWS, TILE_COLS)
    s_adj = adjust_dimensions(salmap, TILE_ROWS, TILE_COLS)
    h, w = s_adj.shape
    th, tw = h // TILE_ROWS, w // TILE_COLS

    sal_tiles = s_adj.reshape(TILE_ROWS, th, TILE_COLS, tw).transpose(0,2,1,3)
    avg_sal   = sal_tiles.mean(axis=(2,3)) / 255.0
    red_mask  = avg_sal > 0.35
    kernel    = np.ones((3,3), dtype=np.uint8)
    dilated   = cv2.dilate(red_mask.astype(np.uint8), kernel, iterations=1)
    green_mask= (dilated==1) & (~red_mask)

    Q_tiles = np.full_like(avg_sal, fill_value=qp["black"], dtype=np.uint8)
    Q_tiles[green_mask] = qp["green"]
    Q_tiles[red_mask]   = qp["red"]

    tiles    = tile_image(f_adj, TILE_ROWS, TILE_COLS)
    flat_tiles = tiles.reshape(-1, th, tw, 3)
    flat_q     = Q_tiles.flatten()

    results    = list(executor.map(compress_tile_worker, zip(flat_tiles, flat_q)))
    flat_comp, flat_psnrs = zip(*results)
    comp_tiles = np.array(flat_comp, dtype=np.uint8).reshape(TILE_ROWS, TILE_COLS, th, tw, 3)
    psnrs      = np.array(flat_psnrs, dtype=np.float32).reshape(TILE_ROWS, TILE_COLS)

    comp = merge_tiles(comp_tiles)
    return comp, psnrs, f_adj, s_adj, Q_tiles

def compute_tile_qualities(sal_map, qp):
    h, w = sal_map.shape
    th, tw = h // TILE_ROWS, w // TILE_COLS
    sal_tiles = sal_map.reshape(TILE_ROWS, th, TILE_COLS, tw).transpose(0,2,1,3)
    avg_sal   = sal_tiles.mean(axis=(2,3)) / 255.0

    red_mask   = avg_sal > 0.35
    dilated    = cv2.dilate(red_mask.astype(np.uint8), np.ones((3,3), dtype=np.uint8), 1)
    green_mask = (dilated==1) & (~red_mask)

    Q = np.full_like(avg_sal, qp["black"], dtype=np.uint8)
    Q[green_mask] = qp["green"]
    Q[red_mask]   = qp["red"]
    return Q

def create_visual_overlay(frame, sal_map, tile_qualities, qp):
    heatmap = cv2.applyColorMap(sal_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.95, heatmap, 0.15, 0.0)
    h, w = frame.shape[:2]
    th, tw = h // TILE_ROWS, w // TILE_COLS
    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            y1, x1 = r*th, c*tw
            y2, x2 = y1+th, x1+tw
            qv = int(tile_qualities[r,c])
            tile_idx = r*TILE_COLS + c
            if qv == qp["red"]:
                color = (0,0,255)
            elif qv == qp["green"]:
                color = (0,255,0)
            else:
                color = (0,0,0)
            label = f"T{tile_idx} {qv}%"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1+3, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return overlay

def onnx_raft_inference(sess, im1, im2):
    o1, o2    = im1.cpu().numpy(), im2.cpu().numpy()
    inp_names = [i.name for i in sess.get_inputs()]
    flow, _   = sess.run(["flow_final","info_final"], {inp_names[0]: o1, inp_names[1]: o2})
    return torch.tensor(flow).to(device)

def str2bool_custom(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser("SATC Pipeline")
    parser.add_argument("--video_path",           required=True, help="Path to input video (.mp4)")
    parser.add_argument("--sst_sal_model_path",   default=SST_SAL_MODEL_PATH)
    parser.add_argument("--raft_model_path",      default=RAFT_MODEL_PATH)
    parser.add_argument("--bw",                   choices=["low","high"], required=True)
    parser.add_argument("--mixed_precision",      type=str2bool_custom, default=True)
    parser.add_argument("--overlay",              type=str2bool_custom, default=True)
    args = parser.parse_args()

    print(f"Reading frames from: {args.video_path}")
    rgb_frames = read_video_frames(args.video_path)
    total_frames = len(rgb_frames) - 1
    bw_qp = generate_bandwidth_values(total_frames, args.bw)
    print(f"Total frames loaded: {len(rgb_frames)}")

    sess = ort.InferenceSession(args.raft_model_path,
                providers=["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"])
    dummy = np.zeros((1,3,RAFT_HEIGHT,RAFT_WIDTH), np.float32)
    inp_names = [i.name for i in sess.get_inputs()]
    sess.run(None, {inp_names[0]: dummy, inp_names[1]: dummy})

    # Setup SST-Sal
    sst = load_sst_sal_model(args.sst_sal_model_path)
    warm_up_sst_sal(sst, mixed_precision=args.mixed_precision)

    executor = ProcessPoolExecutor(max_workers=os.cpu_count())

    for idx in tqdm(range(total_frames), desc="Compressing Frames", ncols=80):
        img1 = rgb_frames[idx]
        img2 = rgb_frames[idx+1]
        r1   = cv2.resize(img1, (RAFT_WIDTH, RAFT_HEIGHT))
        r2   = cv2.resize(img2, (RAFT_WIDTH, RAFT_HEIGHT))

        inp1 = torch.from_numpy(r1.transpose(2,0,1)).unsqueeze(0).float().to(device)/255.0
        inp2 = torch.from_numpy(r2.transpose(2,0,1)).unsqueeze(0).float().to(device)/255.0

        flow = onnx_raft_inference(sess, inp1, inp2)
        flow_img = flow_to_image(flow[0].cpu().numpy().transpose(1,2,0), True)

        tin   = torch.from_numpy(r1.transpose(2,0,1)).unsqueeze(0).float().to(device)/255.0
        tflow = torch.from_numpy(flow_img.transpose(2,0,1)).unsqueeze(0).float().to(device)/255.0
        inp_sal = torch.cat([tin, tflow],1).unsqueeze(1).repeat(1,20,1,1,1)
        with torch.no_grad():
            sm_tensor = sst(inp_sal).mean(dim=1).squeeze()

        sm = sm_tensor.detach().cpu().numpy()
        sm_norm = ((sm - sm.min()) / (sm.max() - sm.min() + 1e-8) * 255).astype(np.uint8)
        sal_big = cv2.resize(sm_norm, img1.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        _, qp = bw_qp[idx]
        comp, psnrs, f_adj, s_adj, Q_tiles = process_frame_with_saliency(img1, sal_big, qp, executor)

        outp = os.path.join(OUTPUT_PATH, f"compressed_{idx:04d}.jpeg")
        _, buf = cv2.imencode('.jpeg', comp, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        with open(outp, 'wb') as f:
            f.write(buf)

        if args.overlay:
            zones   = compute_tile_qualities(s_adj, qp)
            overlay = create_visual_overlay(f_adj, s_adj, zones, qp)
            ov_out  = os.path.join(OVERLAY_PATH, f"overlay_{idx:04d}.png")
            cv2.imwrite(ov_out, overlay)

    executor.shutdown()

if __name__ == "__main__":
    main()
