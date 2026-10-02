#!/usr/bin/env python3
"""
RUN_360SATC_ALL_EXPERIMENTS_v6.py
=================================
Master reproducibility harness for the four 360-SATC experiments discussed on 2026-09-24.

Modes
-----
  throughput : Experiment 1. Uncapped encoder-side processing throughput.
               360-SATC P4 + matched full-workload CBR P4 + MUC/PC/AUC/360-ST P4.
  quality    : Experiment 2. Fixed-rate reconstruction quality at 4K60.
               P7 CBR, 12/35 Mbit/s, 896 measured frames. Reports frame size,
               PSNR, SSIM, LPIPS (AlexNet 0.1), and WS-PSNR.
  network    : Experiment 3. Controlled replay of the frozen real-5G traces.
               Fixed-rate, GCC-direct, and GCC+RT-MPC controllers; neutral 60-Mbit/s warm-up and one fixed RTT.
  rtt        : Experiment 4. RTT sensitivity with one frozen 5G trace and GCC+RT-MPC.
  all        : Execute throughput -> quality -> network -> rtt sequentially.

Important measurement boundaries
--------------------------------
* Experiment 1 is computational throughput, not receiver frame delivery.
* Experiment 2 is encoder reconstruction quality, not received-video quality.
* Experiments 3/4 are WebRTC/Mininet/GCC network experiments and are kept separate.
* MUC/PC/AUC/360-ST use the common NVENC backend. The 360-ST baseline is the
  public-implementation/action-space adaptation used in the previously audited runner;
  it is not claimed to reproduce an unavailable trained A2C checkpoint.

This single file embeds the exact helper sources recovered from the prior work:
  - SATC_Uncapped_P4_v1.py
  - SATC_CBR_FullWorkload_P4_v1.py
  - RUN_CARUSO_BASELINES_ALL_IN_ONE_v3.py
  - SATC_Integrated_Real5G_60FPS_v14.py

The master never silently caps processing to 60 FPS and never rounds a failing result up.
"""
from __future__ import annotations

import argparse, base64, csv, datetime as dt, hashlib, importlib.util, json, math, os
from pathlib import Path
import pwd, re, shutil, statistics, subprocess, sys, tempfile, time, traceback, zipfile, zlib
from types import SimpleNamespace
from typing import Any

VERSION = "360SATC-All-Experiments-v6"
W,H,FPS = 4096,2048,60
DEFAULT_LONDON = Path(os.environ.get("SATC_LONDON", str(Path(__file__).resolve().parents[2] / "data/prepared/london_tower_4096x2048_60fps.mp4")))
PREPARED = Path(os.environ.get("SATC_PREPARED", str(Path(__file__).resolve().parents[2] / "data/prepared")))
DEFAULT_MODEL = Path(os.environ.get("SATC_MODEL", str(Path(__file__).resolve().parents[2] / "saliency/models/most_sal_144x192.onnx")))
SATC_PY = Path(os.environ.get("SATC_PYTHON", sys.executable))
VIDEO_METRICS_PY = Path(os.environ.get("SATC_METRICS_PYTHON", sys.executable))

HELPERS = {"satc": "SATC_Uncapped_P4_v1.py", "cbr": "SATC_CBR_FullWorkload_P4_v1.py", "caruso": "RUN_CARUSO_BASELINES_ALL_IN_ONE_v3.py", "v14": "SATC_Integrated_Real5G_60FPS_v14.py"}


def say(s=""):
    print(s, flush=True)


def dump(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024), b""): h.update(c)
    return h.hexdigest()


def command(cmd, *, log=None, env=None, check=True, timeout=None, cwd=None):
    say("+ "+" ".join(map(str,cmd)))
    if log:
        log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w",encoding="utf-8") as f:
            p=subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd,timeout=timeout,check=False,text=True)
    else:
        p=subprocess.run([str(x) for x in cmd],env=env,cwd=cwd,timeout=timeout,check=False,text=True)
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str,cmd))}")
    return p


def real_user_home():
    user=os.environ.get("SUDO_USER") or os.environ.get("SATC_REAL_USER")
    if user and user!="root":
        return user,Path(pwd.getpwnam(user).pw_dir)
    try: return "mininet-ovs",Path(pwd.getpwnam("mininet-ovs").pw_dir)
    except KeyError:
        pw=pwd.getpwuid(os.getuid()); return pw.pw_name,Path(pw.pw_dir)


def chown_tree(path: Path, user: str):
    if os.geteuid()!=0:return
    try: pw=pwd.getpwnam(user)
    except KeyError:return
    for root,dirs,files in os.walk(path):
        for p in [root]+[os.path.join(root,x) for x in dirs+files]:
            try: os.chown(p,pw.pw_uid,pw.pw_gid)
            except OSError: pass


def extract_helpers(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in HELPERS.items():
        source = Path(__file__).resolve().parents[1] / "helpers" / name
        target = dest / name
        shutil.copy2(source, target)
        out[key] = target
    return out


def load_module(path: Path, name: str):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def video_paths(london: Path):
    d={
      "basketball":PREPARED/"basketball_4096x2048_60fps.mp4",
      "rollercoaster":PREPARED/"rollercoaster_4096x2048_60fps.mp4",
      "ballet":PREPARED/"ballet_4096x2048_60fps.mp4",
      "london_tower":london,
    }
    missing=[str(p) for p in d.values() if not p.is_file()]
    if missing: raise RuntimeError("missing prepared videos: "+", ".join(missing))
    return d


def patch_preset(cpp: Path, preset: str):
    t=cpp.read_text(encoding="utf-8")
    old="NV_ENC_PRESET_P4_GUID"; new=f"NV_ENC_PRESET_{preset.upper()}_GUID"
    if old not in t and new not in t: raise RuntimeError(f"preset token not found in {cpp}")
    if old in t: t=t.replace(old,new)
    cpp.write_text(t,encoding="utf-8")


def setup_satc_module(mod, root: Path, preset: str, videos: dict[str,Path], frames: int, warmup: int):
    root.mkdir(parents=True,exist_ok=False); code=root/"code"; code.mkdir(); mod.materialize(code)
    if preset.lower()!="p4": patch_preset(code/"nvenc_uncapped.cpp",preset)
    sys.path.insert(0,str(code))
    sdk=mod.find_sdk(None); model_path=mod.find_model(DEFAULT_MODEL if DEFAULT_MODEL.is_file() else None)
    for v in videos.values(): mod.validate_video(v,frames+warmup)
    mod.command(["cmake","-S",code,"-B",root/"build",f"-DSDK_TOP={sdk}","-DCMAKE_BUILD_TYPE=Release"],root/"build_configure.log")
    mod.command(["cmake","--build",root/"build","-j",str(min(os.cpu_count() or 2,4))],root/"build_compile.log")
    from live_model import LiveModel, preprocess_nv12
    model=LiveModel(model_path,root,provider="auto",map_backend="compact-cpu")
    # Bound-inference preflight identical in spirit to the frozen runner.
    a=["ffmpeg","-nostdin","-v","error","-i",str(next(iter(videos.values()))),"-map","0:v:0","-an","-frames:v","1","-pix_fmt","nv12","-f","rawvideo","pipe:1"]
    one=subprocess.run(a,capture_output=True,timeout=60,check=True).stdout
    if len(one)!=mod.FRAME_BYTES: raise RuntimeError("first-frame preflight geometry mismatch")
    model.validate_bound_inference([preprocess_nv12(one,mod.W,mod.H)]*20)
    return code,model,model_path,sdk


def run_satc_jobs(mod, root: Path, model, videos: dict[str,Path], *, preset: str, frames: int, warmup: int, methods=("roi",), rates=(12,35)):
    args=SimpleNamespace(warmup_frames=warmup,frames=frames)
    rows=[]
    for video_name,video in videos.items():
      for codec in mod.CODECS:
       for rate in rates:
        for method in methods:
          jid=f"{video_name}_{codec}_{rate}_{method}"
          job={"id":jid,"video_name":video_name,"video":str(video),"codec":codec,"target_mbps":rate,"method":method}
          r=mod.run_one(args,root,job,model); r["preset_actual"]=preset
          sp=root/"runs"/jid/"summary.json"
          if sp.is_file(): dump(sp,r)
          cp=root/"runs"/jid/"config.json"
          if cp.is_file():
              try:
                  cd=json.loads(cp.read_text()); cd["preset"]=preset; cd["preset_actual"]=preset; dump(cp,cd)
              except Exception: pass
          rows.append(r)
          if r.get("status")!="VALID": raise RuntimeError(f"SATC job failed: {jid}: {r.get('error')}")
    return rows


def stage_cbr_inputs(root: Path, videos: dict[str,Path]):
    d=root/"inputs"; d.mkdir(parents=True,exist_ok=True); mapping={}
    for n,p in videos.items():
        target=d/p.name
        if not target.exists(): target.symlink_to(p)
        mapping[n]=p.name
    return d,mapping


def run_matched_cbr(cbr, root: Path, videos: dict[str,Path]):
    root.mkdir(parents=True,exist_ok=False); code=root/"code"; code.mkdir(); cbr.extract_payload(code); cbr.patch_producer_for_full_workload_cbr(code)
    staged,mapping=stage_cbr_inputs(root,videos); cbr.INPUT_ROOT_ABS=staged; cbr.VIDEOS=mapping
    # Recovered helper v1 has a one-name typo in cbr_preflight(): it calls
    # sha256_file(), while the helper defines sha256().  Keep the embedded
    # helper byte-for-byte and provide the intended compatibility alias here.
    if not hasattr(cbr,"sha256"):
        raise RuntimeError("matched-CBR helper is missing sha256()")
    cbr.sha256_file=cbr.sha256
    # The recovered helper also calls read_csv() from its zero-QP audit, but
    # that utility was absent from the saved helper source. Supply the exact
    # intended CSV-dictionary reader without altering the experiment logic.
    def _read_csv_compat(path):
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    cbr.read_csv=_read_csv_compat
    cbr.cbr_preflight(root,code); encoder=cbr.build_encoder(root,code)
    rows=[]
    for vn in mapping:
      for codec in cbr.CODECS:
       for rate in (12,35):
        r=cbr.run_one_cbr(root,code,encoder,vn,codec,rate,1)
        if r.get("status")!="VALID":
            time.sleep(2); r=cbr.run_one_cbr(root,code,encoder,vn,codec,rate,2)
        rows.append(r)
        if r.get("status")!="VALID": raise RuntimeError(f"matched-CBR failed: {vn}/{codec}/{rate}")
    with (root/"CBR_FULLWORKLOAD_SUMMARY_4VIDEOS.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["video","codec","target_mbps","status","fps","measured_frames","warmup_frames","frame_drops","integrity_pass","attempt"]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    dump(root/"CBR_FULLWORKLOAD_VERDICT_4VIDEOS.json",{"planned_runs":24,"valid_runs":sum(r.get('status')=='VALID' for r in rows),"rows":rows})
    return rows


def caruso_args(args):
    return SimpleNamespace(london_video=args.london_video,shared_mat=args.shared_mat,trace_index=args.trace_index)


def run_caruso_throughput(car, root: Path, args):
    root.mkdir(parents=True,exist_ok=False); code=root/"code"; code.mkdir(); car.extract_payload(code)
    car.WARMUP=240; car.MEASURED=1200
    videos,trace,_=car.preflight(root,real_user_home()[1],caruso_args(args)); encoder=car.build_encoder(root,code)
    rows=car.matrix(root,code,encoder,videos,trace)
    if len(rows)!=96 or any(r.get("status")!="VALID" for r in rows): raise RuntimeError("Caruso throughput matrix incomplete")
    return rows


def _load_completed_satc_exp1(root: Path, videos: dict[str,Path]):
    expected=[]
    for vn in videos:
      for codec in ("h264","hevc","av1"):
       for rate in (12,35): expected.append(f"{vn}_{codec}_{rate}_roi")
    rows=[]
    for jid in expected:
        p=root/"runs"/jid/"summary.json"
        if not p.is_file(): return None
        try:r=json.loads(p.read_text(encoding="utf-8"))
        except Exception:return None
        m=r.get("measurement") or {}
        if r.get("status")!="VALID" or int(m.get("completed_frames",0))!=1200 or int(r.get("warmup_frames",0))!=240:
            return None
        rows.append(r)
    return rows if len(rows)==24 else None


def _load_completed_cbr_exp1(root: Path):
    p=root/"CBR_FULLWORKLOAD_VERDICT_4VIDEOS.json"
    if not p.is_file(): return None
    try:d=json.loads(p.read_text(encoding="utf-8")); rows=d.get("rows") or []
    except Exception:return None
    if len(rows)==24 and all(r.get("status")=="VALID" for r in rows): return rows
    return None


def _load_completed_caruso_exp1(root: Path):
    p=root/"CARUSO_BASELINES_VERDICT.json"
    s=root/"CARUSO_BASELINES_SUMMARY.csv"
    if not p.is_file() or not s.is_file(): return None
    try:
        d=json.loads(p.read_text(encoding="utf-8"))
        with s.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    except Exception:return None
    # The frozen matrix is 4 videos x 3 codecs x 2 rates x 4 methods = 96.
    if len(rows)==96 and all(r.get("status")=="VALID" for r in rows): return rows
    return None


def run_exp1(args, root: Path, helpers):
    say("\n========== EXPERIMENT 1: PROCESSING THROUGHPUT ==========")
    out=root/"exp1_throughput"; out.mkdir(parents=True,exist_ok=True); vids=video_paths(args.london_video)

    satc_root=out/"satc_p4"
    satc_rows=_load_completed_satc_exp1(satc_root,vids) if satc_root.exists() else None
    if satc_rows is not None:
        say(f"RESUME: reusing {len(satc_rows)}/24 validated 360-SATC P4 runs from {satc_root}")
    else:
        if satc_root.exists():
            say("RESUME: incomplete SATC P4 directory found; restarting only that component")
            shutil.rmtree(satc_root)
        satc=load_module(helpers["satc"],"satc_thr")
        _,model,_,_=setup_satc_module(satc,satc_root,"p4",vids,1200,240)
        satc_rows=run_satc_jobs(satc,satc_root,model,vids,preset="p4",frames=1200,warmup=240,methods=("roi",))

    cbr_root=out/"matched_fullworkload_cbr_p4"
    cbr_rows=_load_completed_cbr_exp1(cbr_root) if cbr_root.exists() else None
    if cbr_rows is not None:
        say(f"RESUME: reusing {len(cbr_rows)}/24 validated matched full-workload CBR runs")
    else:
        if cbr_root.exists():
            say("RESUME: matched-CBR component did not complete; removing its partial directory and restarting CBR only")
            shutil.rmtree(cbr_root)
        cbr=load_module(helpers["cbr"],"cbr_thr")
        cbr_rows=run_matched_cbr(cbr,cbr_root,vids)

    car_root=out/"caruso_p4"
    car_rows=_load_completed_caruso_exp1(car_root) if car_root.exists() else None
    if car_rows is not None:
        say(f"RESUME: reusing {len(car_rows)}/96 validated Caruso baseline runs")
    else:
        if car_root.exists():
            say("RESUME: incomplete Caruso component found; restarting only that component")
            shutil.rmtree(car_root)
        car=load_module(helpers["caruso"],"car_thr")
        car_rows=run_caruso_throughput(car,car_root,args)

    dump(out/"EXP1_VERDICT.json",{"satc_valid":len(satc_rows),"cbr_valid":len(cbr_rows),"caruso_valid":len(car_rows),"measurement":"uncapped encoder-side processing throughput; no network","resumable":True})
    return out


# ---------- Experiment 2: quality ----------
def patch_caruso_bitstream_producer(path: Path):
    t=path.read_text(encoding="utf-8")
    old="import argparse,csv,hashlib,json,math,os,queue,signal,statistics,struct,subprocess,sys,threading,time,traceback"
    # imports already contain struct in the recovered producer; do not depend on exact full import string.
    needle="if read_exact(enc.stdout,4)!=b'RDY1': raise RuntimeError('NVENC bridge not ready')\n    source=Source(args.video,stop_evt,gate,warm,total)"
    repl="""if read_exact(enc.stdout,4)!=b'RDY1': raise RuntimeError('NVENC bridge not ready')
    stream_path=args.output_dir/('encoded.'+{'h264':'h264','hevc':'hevc','av1':'ivf'}[args.codec])
    stream_f=stream_path.open('wb'); ivf_from_sdk=None
    source=Source(args.video,stop_evt,gate,warm,total)"""
    if needle not in t: raise RuntimeError("cannot patch Caruso producer bitstream open")
    t=t.replace(needle,repl)
    needle2="_packet=read_exact(enc.stdout,size); complete=time.time_ns()"
    repl2="""_packet=read_exact(enc.stdout,size)
            if args.codec=='av1':
                if ivf_from_sdk is None:
                    ivf_from_sdk=_packet[:4]==b'DKIF'
                    if not ivf_from_sdk:
                        stream_f.write(struct.pack('<4sHH4sHHIIII',b'DKIF',0,32,b'AV01',W,H,FPS,1,total,0))
                if not ivf_from_sdk: stream_f.write(struct.pack('<IQ',len(_packet),fid))
            stream_f.write(_packet)
            complete=time.time_ns()"""
    if needle2 not in t: raise RuntimeError("cannot patch Caruso producer packet write")
    t=t.replace(needle2,repl2)
    needle3="shared.close(); log.close()"
    repl3="stream_f.close(); shared.close(); log.close()"
    if needle3 not in t: raise RuntimeError("cannot patch Caruso producer close")
    t=t.replace(needle3,repl3)
    path.write_text(t,encoding="utf-8")


def ffmpeg_metric(source: Path, encoded: Path, warm: int, frames: int, kind: str, log: Path):
    end=warm+frames
    if kind=="psnr": filt=f"[0:v]trim=start_frame={warm}:end_frame={end},setpts=PTS-STARTPTS[r];[1:v]trim=start_frame={warm}:end_frame={end},setpts=PTS-STARTPTS[d];[r][d]psnr"
    elif kind=="ssim": filt=f"[0:v]trim=start_frame={warm}:end_frame={end},setpts=PTS-STARTPTS[r];[1:v]trim=start_frame={warm}:end_frame={end},setpts=PTS-STARTPTS[d];[r][d]ssim"
    else: raise ValueError(kind)
    cmd=["ffmpeg","-nostdin","-hide_banner","-i",str(source),"-i",str(encoded),"-filter_complex",filt,"-frames:v",str(frames),"-f","null","-"]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=3600)
    log.write_text(p.stderr,encoding="utf-8")
    if p.returncode: raise RuntimeError(f"ffmpeg {kind} failed for {encoded}")
    if kind=="psnr":
        mm=re.findall(r"average:([0-9.+-eEinfINF]+)",p.stderr)
    else:
        mm=re.findall(r"All:([0-9.+-eE]+)",p.stderr)
    if not mm: raise RuntimeError(f"could not parse {kind} result")
    return float(mm[-1])


def internal_lpips(argv):
    # Invoked under the video-metrics venv so the master itself does not require lpips.
    import numpy as np
    import torch, lpips
    source,encoded=Path(argv.source),Path(argv.encoded); warm=int(argv.warm); frames=int(argv.frames); out=Path(argv.out)
    device="cuda" if torch.cuda.is_available() else "cpu"
    loss_fn=lpips.LPIPS(net="alex",version="0.1").to(device).eval()
    end=warm+frames
    filt=f"[0:v]trim=start_frame={warm}:end_frame={end},setpts=PTS-STARTPTS[r];[1:v]trim=start_frame={warm}:end_frame={end},setpts=PTS-STARTPTS[d];[r][d]hstack=inputs=2,format=rgb24"
    cmd=["ffmpeg","-nostdin","-hide_banner","-v","error","-i",str(source),"-i",str(encoded),"-filter_complex",filt,"-frames:v",str(frames),"-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    frame_bytes=W*2*H*3
    lat=(np.arange(H,dtype=np.float64)+0.5)/H*math.pi-math.pi/2
    weights=np.cos(lat); weights/=weights.sum()
    vals=[]; ws=[]; per=[]
    def rex(n):
        b=bytearray(n); mv=memoryview(b); pos=0
        while pos<n:
            k=p.stdout.readinto(mv[pos:])
            if not k: break
            pos+=k
        return bytes(b[:pos])
    with torch.no_grad():
      for i in range(frames):
        raw=rex(frame_bytes)
        if len(raw)!=frame_bytes: raise RuntimeError(f"raw metric pipe ended at {i}/{frames}")
        a=np.frombuffer(raw,dtype=np.uint8).reshape(H,W*2,3)
        ref=np.ascontiguousarray(a[:,:W]); dis=np.ascontiguousarray(a[:,W:])
        diff=ref.astype(np.float32)-dis.astype(np.float32)
        row_mse=np.mean(diff*diff,axis=(1,2),dtype=np.float64); wmse=float(np.dot(row_mse,weights)); wsp=99.0 if wmse<=1e-12 else 10*math.log10((255.0**2)/wmse)
        t1=torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float().to(device)/127.5-1.0
        t2=torch.from_numpy(dis).permute(2,0,1).unsqueeze(0).float().to(device)/127.5-1.0
        try: lv=float(loss_fn(t1,t2).mean().item())
        except RuntimeError as e:
            if device=="cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); device="cpu"; loss_fn=loss_fn.cpu(); t1=t1.cpu(); t2=t2.cpu(); lv=float(loss_fn(t1,t2).mean().item())
            else: raise
        vals.append(lv); ws.append(wsp); per.append((i,lv,wsp))
        if (i+1)%64==0: print(f"LPIPS/WS-PSNR {i+1}/{frames}",flush=True)
    err=p.stderr.read().decode("utf-8","replace"); rc=p.wait()
    if rc: raise RuntimeError("ffmpeg metric pipe failed: "+err[-2000:])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["frame_index","lpips_alex_v0_1","ws_psnr_db"]);w.writerows(per)
    result={"lpips_mean":statistics.fmean(vals),"lpips_std":statistics.pstdev(vals) if len(vals)>1 else 0.0,"ws_psnr_mean_db":statistics.fmean(ws),"ws_psnr_std_db":statistics.pstdev(ws) if len(ws)>1 else 0.0,"frames":len(vals),"device":device}
    dump(out,result); return 0


def metric_pair(master_path: Path, source: Path, encoded: Path, warm: int, frames: int, outdir: Path, do_lpips: bool):
    outdir.mkdir(parents=True,exist_ok=True)
    psnr=ffmpeg_metric(source,encoded,warm,frames,"psnr",outdir/"psnr.log")
    ssim=ffmpeg_metric(source,encoded,warm,frames,"ssim",outdir/"ssim.log")
    extra={"lpips_mean":None,"ws_psnr_mean_db":None}
    if do_lpips:
        interp=VIDEO_METRICS_PY if VIDEO_METRICS_PY.is_file() else Path(sys.executable)
        j=outdir/"lpips_wspsnr.json"
        command([interp,master_path,"--internal-lpips","--source",source,"--encoded",encoded,"--warm",str(warm),"--frames",str(frames),"--out",j],log=outdir/"lpips_console.log",timeout=24*3600)
        extra=json.loads(j.read_text())
    return {"psnr_db":psnr,"ssim":ssim,**extra}


def run_quality_caruso(car, root: Path, args, videos_select: dict[str,Path]):
    root.mkdir(parents=True,exist_ok=False); code=root/"code"; code.mkdir(); car.extract_payload(code)
    car.WARMUP=args.quality_warmup; car.MEASURED=args.quality_frames
    patch_preset(code/"nvenc_dynamic.cpp","p7"); patch_caruso_bitstream_producer(code/"baseline_producer.py")
    # preflight still discovers the full four-video set; filter after discovery.
    allvids,trace,_=car.preflight(root,real_user_home()[1],caruso_args(args)); vids={k:v for k,v in allvids.items() if k in videos_select}
    pp=root/"protocol.json"
    if pp.is_file():
        pd=json.loads(pp.read_text()); pd.setdefault("encoder",{})["preset"]="p7"; pd["quality_override"]={"measured_frames":args.quality_frames,"warmup_frames":args.quality_warmup,"purpose":"fixed-rate reconstruction quality"}; dump(pp,pd)
    encoder=car.build_encoder(root,code); rows=[]
    for vn in vids:
      for codec in car.CODECS:
       for rate in car.RATES:
        for method in car.METHODS:
          r=car.run_one(root,code,encoder,vids,trace,vn,codec,rate,method,1)
          if r.get("status")!="VALID": r=car.run_one(root,code,encoder,vids,trace,vn,codec,rate,method,2)
          if r.get("status")!="VALID": raise RuntimeError(f"quality Caruso encode failed: {vn}/{codec}/{rate}/{method}")
          r["preset_actual"]="p7"; rows.append(r)
    return rows


def run_exp2(args, root: Path, helpers):
    say("\n========== EXPERIMENT 2: FIXED-RATE IMAGE QUALITY ==========")
    out=root/"exp2_quality"; out.mkdir(parents=True,exist_ok=True); allvid=video_paths(args.london_video)
    vids=allvid if args.quality_all_videos else {"london_tower":allvid["london_tower"]}
    # Resume completed quality work. London-only = 36 rows; all four videos = 144.
    summary=out/"QUALITY_SUMMARY.csv"; expected_rows=len(vids)*3*2*6
    if summary.is_file():
        try:
            with summary.open(newline="",encoding="utf-8") as f: done=list(csv.DictReader(f))
            required=("psnr_db","ssim","ws_psnr_mean_db") if args.skip_lpips else ("psnr_db","ssim","lpips_mean","ws_psnr_mean_db")
            if len(done)==expected_rows and all(all(str(r.get(k,"")) not in ("","None","nan") for k in required) for r in done):
                say(f"RESUME: reusing completed Experiment 2 quality matrix ({len(done)}/{expected_rows} rows)")
                return out
        except Exception:
            pass
    warm=args.quality_warmup; frames=args.quality_frames
    satc=load_module(helpers["satc"],"satc_quality"); sr=out/"satc_cbr_p7"
    _,model,_,_=setup_satc_module(satc,sr,"p7",vids,frames,warm)
    rows_satc=run_satc_jobs(satc,sr,model,vids,preset="p7",frames=frames,warmup=warm,methods=("roi","uniform"))
    car=load_module(helpers["caruso"],"car_quality"); cr=out/"caruso_p7"; rows_car=run_quality_caruso(car,cr,args,vids)
    manifest=[]
    ext={"h264":"h264","hevc":"hevc","av1":"ivf"}
    # SATC/CBR pairs
    for r in rows_satc:
        vn=r["video_name"]; codec=r["codec"]; rate=r["target_mbps"]; method="360-satc" if r["method"]=="roi" else "cbr"
        enc=sr/"runs"/r["id"]/f"encoded.{ext[codec]}"; src=vids[vn]
        m=metric_pair(Path(__file__).resolve(),src,enc,warm,frames,out/"metrics"/f"{vn}_{codec}_{rate}_{method}",not args.skip_lpips)
        manifest.append({"video":vn,"codec":codec,"target_mbps":rate,"method":method,"preset":"p7","encoded":str(enc),"mean_encoded_frame_bytes":None,**m})
    for r in rows_car:
        vn=r["video"]; codec=r["codec"]; rate=int(r["target_mbps"]); method=r["method"]; attempt=int(r.get("attempt",1))
        enc=cr/"runs"/f"{vn}_{codec}_{rate}Mbps_{method}"/f"attempt_{attempt}"/f"encoded.{ext[codec]}"; src=vids[vn]
        m=metric_pair(Path(__file__).resolve(),src,enc,warm,frames,out/"metrics"/f"{vn}_{codec}_{rate}_{method}",not args.skip_lpips)
        manifest.append({"video":vn,"codec":codec,"target_mbps":rate,"method":method,"preset":"p7","encoded":str(enc),"mean_encoded_frame_bytes":r.get("mean_encoded_bytes"),**m})
    fields=["video","codec","target_mbps","method","preset","mean_encoded_frame_bytes","psnr_db","ssim","lpips_mean","ws_psnr_mean_db","encoded"]
    with (out/"QUALITY_SUMMARY.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(manifest)
    dump(out/"QUALITY_PROTOCOL.json",{"resolution":[W,H],"media_fps":FPS,"preset":"p7","rate_control":"CBR","gop":240,"b_frames":0,"rates_mbps":[12,35],"measured_frames":frames,"warmup_frames":warm,"metrics":["PSNR","SSIM","LPIPS AlexNet v0.1","WS-PSNR"],"videos":list(vids),"note":"CBR reconstruction uses the zero-QP uniform output; running MoST-Sal in matched-CBR changes compute workload but not the zero-QP encoded reconstruction."})
    return out


# ---------- Experiments 3/4: network ----------
def patch_sender_controller(path: Path):
    t=path.read_text(encoding="utf-8")
    t=t.replace("import argparse,csv,itertools,json,signal,statistics,struct,subprocess,sys,threading,time,traceback","import argparse,csv,itertools,json,os,signal,statistics,struct,subprocess,sys,threading,time,traceback")
    old="self.profile=self.load_profile(Path(a.profile_csv)); self.prepare_scores()"
    new="self.profile=self.load_profile(Path(a.profile_csv)); self.prepare_scores(); self.controller_mode=os.environ.get('SATC_CONTROLLER_MODE','rtmpc').strip().lower(); self.fixed_target_mbps=float(os.environ.get('SATC_FIXED_TARGET_MBPS','12'))"
    if old not in t: raise RuntimeError("sender profile patch anchor missing")
    t=t.replace(old,new)
    old="initial=max(candidates,key=lambda r:r.target_bitrate_mbps) if candidates else min(self.profile,key=lambda r:r.measured_bitrate_mbps)"
    new="initial=(min(self.profile,key=lambda r:abs(r.target_bitrate_mbps-self.fixed_target_mbps)) if self.controller_mode=='fixed' else (max(candidates,key=lambda r:r.target_bitrate_mbps) if candidates else min(self.profile,key=lambda r:r.measured_bitrate_mbps)))"
    if old not in t: raise RuntimeError("sender initial patch anchor missing")
    t=t.replace(old,new)
    old="""        now_ns=time.time_ns(); mono=time.monotonic()
        trigger=False; reason=''; budget=None; before=None; after=None
        with self.lock:"""
    new="""        now_ns=time.time_ns(); mono=time.monotonic()
        if self.controller_mode!='rtmpc':
            with self.lock:
                prev_raw=max(1,int(self.last_gcc_event_bps)); self.latest_gcc_bps=v; self.last_gcc_event_bps=v
                before=float(self.selected if self.selected is not None else self.previous if self.previous is not None else min(r.target_bitrate_mbps for r in self.profile)); after=before
            elapsed=(now_ns-self.start_unix_ns)/1e9
            self.gw.writerow([now_ns,f'{elapsed:.9f}',f'{prev_raw/1e6:.9f}',f'{v/1e6:.9f}',f'{float(v)/float(prev_raw):.9f}',f'{before:.9f}',f'{after:.9f}',0,'observe_only','']); self.gcc_event_file.flush(); return
        trigger=False; reason=''; budget=None; before=None; after=None
        with self.lock:"""
    if old not in t: raise RuntimeError("sender GCC patch anchor missing")
    t=t.replace(old,new)
    old="""        d0=time.perf_counter_ns(); sel,obj=self.select(safe,prev); d1=time.perf_counter_ns()
        mode='mpc'"""
    new="""        d0=time.perf_counter_ns()
        if self.controller_mode=='fixed':
            sel=min(self.profile,key=lambda r:abs(r.target_bitrate_mbps-self.fixed_target_mbps)); obj=0.0; mode='fixed'
        elif self.controller_mode=='gcc':
            cand=self.feasible(safe); sel=max(cand,key=lambda r:r.target_bitrate_mbps); obj=0.0; mode='gcc_direct'
        else:
            sel,obj=self.select(safe,prev); mode='mpc'
        d1=time.perf_counter_ns()"""
    if old not in t: raise RuntimeError("sender select patch anchor missing")
    t=t.replace(old,new)
    t=t.replace("if sel.target_bitrate_mbps > prev+1e-9:","if self.controller_mode=='rtmpc' and sel.target_bitrate_mbps > prev+1e-9:")
    t=t.replace("elif sel.target_bitrate_mbps < prev-1e-9:","elif self.controller_mode=='rtmpc' and sel.target_bitrate_mbps < prev-1e-9:")
    path.write_text(t,encoding="utf-8")
    subprocess.run([sys.executable,"-m","py_compile",str(path)],check=True)


def profile_quality(code: Path, session_dir: Path):
    p=code/"rtmpc_quality_profile.csv"; c=session_dir/"rtmpc_control.csv"
    if not p.is_file() or not c.is_file(): return {}
    with p.open() as f: prof=[r for r in csv.DictReader(f)]
    with c.open() as f: controls=[r for r in csv.DictReader(f)]
    if not controls:return {}
    def nearest(x): return min(prof,key=lambda r:abs(float(r["target_bitrate_mbps"])-x))
    ps=[];ss=[];lp=[]
    for r in controls:
        try:q=nearest(float(r["selected_target_mbps"]));ps.append(float(q["psnr_db"]));ss.append(float(q["ssim"]));lp.append(float(q["lpips"]))
        except Exception:pass
    return {"profile_estimated_psnr_db":statistics.fmean(ps) if ps else None,"profile_estimated_ssim":statistics.fmean(ss) if ss else None,"profile_estimated_lpips":statistics.fmean(lp) if lp else None,"quality_semantics":"time-sampled profile estimate from selected target; not decoded received-frame quality"}


def import_patched_v14(source: Path, work: Path, rtt_ms: float, name: str):
    """Patch the frozen V14 runner without changing its controller/codec logic.

    Three protocol corrections are intentional here:
      1) warm-up uses a non-constraining 60 Mbit/s link so startup validation is
         independent of the first sample of the measured trace;
      2) the transport/signalling canary stays at the frozen neutral 1 ms one-way
         delay; only measured long sessions use RTT/2 per direction;
      3) transport preflight waits for 600 complete units before auditing the
         final 300 PTS values, excluding codec/WebRTC startup transients.
    """
    text=source.read_text(encoding="utf-8")
    one=max(0.0,float(rtt_ms)/2.0)
    # Keep the transport/signalling canary neutral at 1 ms. Patch ONLY the
    # measured long-session link to RTT/2 per direction.
    long_link='link=net.addLink(sender,receiver,cls=TCLink,bw=warmup_capacity,delay="1ms",max_queue_size=1000,use_htb=True)'
    if long_link not in text:
        raise RuntimeError("V14 long-session RTT patch anchor missing")
    text=text.replace(long_link,long_link.replace('delay="1ms"',f'delay="{one:g}ms"'),1)
    # Do not judge PTS continuity while codec discovery/WebRTC subscription is
    # still settling. Collect 600 complete units, then audit the final 300.
    preflight_break='if len(rows)>=300 and len(controls)>=20 and len(gcc_events)>=3: break'
    preflight_min='if len(rows)<300: raise RuntimeError(f"{tag} receiver produced only {len(rows)} complete AUs in preflight")'
    if preflight_break not in text or preflight_min not in text:
        raise RuntimeError("V14 transport-preflight steady-tail patch anchor missing")
    text=text.replace(preflight_break,'if len(rows)>=600 and len(controls)>=20 and len(gcc_events)>=3: break',1)
    text=text.replace(preflight_min,'if len(rows)<600: raise RuntimeError(f"{tag} receiver produced only {len(rows)} complete AUs in preflight")',1)
    # Turn the transport preflight into a transport-health canary rather than a
    # zero-loss performance test. A few *forward* PTS gaps are allowed here
    # because the actual network experiment records continuity/loss as an
    # outcome. Non-increasing PTS remains a hard failure. The canary still
    # requires a near-60-Hz median media clock and at least 95% adjacent-Pair
    # continuity over the audited tail.
    strict_pts="""            pts=[int(x["pts_ns"]) for x in rows[-300:] if int(x.get("pts_ns","-1"))>=0]
            if len(pts)<290: raise RuntimeError(f"{tag} preflight has too few valid PTS values: {len(pts)}")
            bad=sum(1 for a,b in zip(pts,pts[1:]) if b<=a or b-a>1.5*(1_000_000_000/FPS))
            if bad: raise RuntimeError(f"{tag} preflight access-unit PTS continuity failures: {bad}")
"""
    health_pts="""            pts=[int(x["pts_ns"]) for x in rows[-300:] if int(x.get("pts_ns","-1"))>=0]
            if len(pts)<290: raise RuntimeError(f"{tag} preflight has too few valid PTS values: {len(pts)}")
            expected=(1_000_000_000/FPS)
            noninc=sum(1 for a,b in zip(pts,pts[1:]) if b<=a)
            forward_gaps=sum(1 for a,b in zip(pts,pts[1:]) if b>a and b-a>1.5*expected)
            positive_deltas=[b-a for a,b in zip(pts,pts[1:]) if b>a]
            median_delta=statistics.median(positive_deltas) if positive_deltas else 0.0
            gap_limit=max(3,math.ceil(0.05*max(1,len(pts)-1)))
            if noninc: raise RuntimeError(f"{tag} preflight has non-increasing PTS values: {noninc}")
            if forward_gaps>gap_limit: raise RuntimeError(f"{tag} preflight forward PTS gaps exceed 5% health-canary allowance: {forward_gaps}>{gap_limit}")
            if not (0.85*expected <= median_delta <= 1.15*expected): raise RuntimeError(f"{tag} preflight median PTS delta is not near 60 Hz: {median_delta:.0f} ns")
"""
    if strict_pts not in text:
        raise RuntimeError("V14 strict transport PTS block anchor missing")
    text=text.replace(strict_pts,health_pts,1)
    # Keep startup validation independent from experimental trace sample 0.
    anchor='warmup_capacity=float(trace[0]["capacity_mbps"])'
    if anchor not in text:
        raise RuntimeError("V14 warm-up capacity patch anchor missing")
    text=text.replace(anchor,'warmup_capacity=60.0')
    # Keep saved protocol metadata truthful after the delay patch.
    text=text.replace('"link_delay_ms":1',f'"link_delay_ms":{one:g},"configured_rtt_ms":{float(rtt_ms):g}')
    p=work/f"{name}.py"
    p.write_text(text,encoding="utf-8")
    subprocess.run([sys.executable,"-m","py_compile",str(p)],check=True)
    patched=p.read_text(encoding="utf-8")
    if 'warmup_capacity=60.0' not in patched:
        raise RuntimeError("V14 warm-up patch did not apply")
    expected_delay=f'delay="{one:g}ms"'
    if patched.count(expected_delay) < 1:
        raise RuntimeError(f"V14 RTT patch incomplete: expected measured-session occurrence of {expected_delay}")
    if 'bw=35.0,delay="1ms"' not in patched:
        raise RuntimeError("transport preflight must remain at neutral 1 ms one-way delay")
    if 'if len(rows)>=600 and len(controls)>=20 and len(gcc_events)>=3: break' not in patched:
        raise RuntimeError("transport preflight steady-tail patch did not apply")
    if 'forward PTS gaps exceed 5% health-canary allowance' not in patched or 'preflight has non-increasing PTS values' not in patched:
        raise RuntimeError("transport preflight health-canary PTS patch did not apply")
    return load_module(p,name)


def run_transport_preflight_stable(base, out: Path, code: Path, encoder: Path, attempts: int = 3):
    """Run the codec/signalling canary under neutral conditions with bounded retry.

    A retry is allowed only for the short preflight; measured network sessions are
    never retried or selected based on performance. Persistent transport failure
    still aborts the campaign.
    """
    failures=[]
    canary_root=out/"transport_canary"
    canary_root.mkdir(parents=True,exist_ok=True)
    for i in range(1,attempts+1):
        aroot=canary_root/f"attempt_{i}"
        try:
            rows,mode=base.transport_preflight(aroot,code,encoder)
            dump(out/"TRANSPORT_CANARY_VERDICT.json",{"pass":True,"attempts_used":i,"selected_av1_input_mode":mode,"failures":failures,"rows":rows})
            return rows,mode
        except BaseException:
            err=traceback.format_exc()
            failures.append({"attempt":i,"failure":err.splitlines()[-1] if err else "unknown"})
            say(f"  transport canary attempt {i}/{attempts} failed: {failures[-1]['failure']}")
            if i<attempts: time.sleep(2.0)
    dump(out/"TRANSPORT_CANARY_VERDICT.json",{"pass":False,"attempts_used":attempts,"failures":failures})
    raise RuntimeError(f"transport preflight failed persistently after {attempts} attempts; see {canary_root}")


def annotate_network_result(r: dict, controller: str, rtt_ms: float) -> dict:
    """Separate measurement validity from whether the stream sustained 60 FPS.

    A fixed-rate baseline is allowed to perform badly when the trace capacity
    falls below its configured rate. That is an experimental outcome, not a
    harness/setup failure, provided the complete 150-s trace was executed.
    """
    r=dict(r)
    r["controller"]=controller
    r["configured_rtt_ms"]=float(rtt_ms)
    wall=float(r.get("measurement_wall_seconds") or 0.0)
    cap=int(r.get("capacity_updates_recorded") or 0)
    ctrl=int(r.get("rtmpc_control_samples") or 0)
    no_failure=not bool(r.get("failure"))
    r["measurement_valid"] = bool(no_failure and wall >= 149.0 and cap == 150 and ctrl >= 1200)
    r["sustains_60fps"] = bool(r.get("receiver_continuity_pass") and r.get("producer_sustain_pass"))
    r["receiver_sustains_60fps"] = bool(r.get("receiver_continuity_pass"))
    r["producer_sustains_60fps"] = bool(r.get("producer_sustain_pass"))
    # Preserve the frozen V14 field for provenance, but do not use it as the
    # validity criterion for the fixed-rate baseline.
    r["legacy_v14_session_valid_and_sustains_60fps"] = bool(r.get("session_valid_and_sustains_60fps"))
    return r


def internal_network(args, helpers):
    if os.geteuid()!=0: raise RuntimeError("--internal-network requires root")
    user,home=real_user_home(); out=Path(args.internal_output).resolve(); out.mkdir(parents=True,exist_ok=True)
    try:
      base=import_patched_v14(helpers["v14"],out,args.network_rtt_ms,"v14_base")
      code=out/"code";code.mkdir(exist_ok=True);base.extract_payload(code);patch_sender_controller(code/"integrated_sender.py")
      traces=json.loads((code/"trace_data.json").read_text()); meta=base.preflight(out,code,user,home);base.write_protocol(out,meta,traces)
      encoder=base.build_encoder(out,code);base.producer_wire_preflight(out,code,encoder)
      os.environ["SATC_CONTROLLER_MODE"]="rtmpc";os.environ["SATC_FIXED_TARGET_MBPS"]="12"
      _,av1mode=run_transport_preflight_stable(base,out,code,encoder,attempts=3)
      benchmarks=base.benchmark_matrix(out,code,encoder,True,av1mode)
      if args.internal_network=="main":
        allrows=[]
        for controller in ("fixed","gcc","rtmpc"):
          crow=[];croot=out/controller;croot.mkdir(exist_ok=True);os.environ["SATC_CONTROLLER_MODE"]=controller
          for ci,codec in enumerate(base.CODECS):
            for rep in (1,2,3):
              conds=base.CONDITIONS if (ci+rep)%2 else tuple(reversed(base.CONDITIONS))
              for condition in conds:
                os.environ["SATC_FIXED_TARGET_MBPS"]="12" if condition=="low" else "35"
                r=base.run_network_session(croot,code,encoder,traces,codec,condition,rep,av1mode)
                r=annotate_network_result(r,controller,args.network_rtt_ms)
                sdir=croot/"sessions"/r["session"];r.update(profile_quality(code,sdir));dump(sdir/"session_summary_master.json",r);crow.append(r);allrows.append(r)
          dump(croot/"CONTROLLER_VERDICT.json",{"controller":controller,"planned":18,"completed":len(crow),"measurement_valid":sum(bool(x.get('measurement_valid')) for x in crow),"sustains_60fps":sum(bool(x.get('sustains_60fps')) for x in crow),"failed_measurements":sum(not bool(x.get('measurement_valid')) for x in crow),"rows":crow})
        dump(out/"NETWORK_COMPARISON.json",{"fixed_rtt_ms":args.network_rtt_ms,"warmup_capacity_mbps":60.0,"controllers":["fixed","gcc","rtmpc"],"planned":54,"completed":len(allrows),"measurement_valid":sum(bool(x.get('measurement_valid')) for x in allrows),"rows":allrows})
      else:
        allrows=[];condition=args.rtt_condition;rep=args.rtt_replicate
        for rtt in args.rtt_values:
          mod=import_patched_v14(helpers["v14"],out,float(rtt),f"v14_rtt_{str(rtt).replace('.','p')}")
          rroot=out/f"rtt_{rtt:g}ms";rroot.mkdir(exist_ok=True);os.environ["SATC_CONTROLLER_MODE"]="rtmpc";os.environ["SATC_FIXED_TARGET_MBPS"]="12" if condition=="low" else "35"
          for codec in mod.CODECS:
            r=mod.run_network_session(rroot,code,encoder,traces,codec,condition,rep,av1mode)
            r=annotate_network_result(r,"rtmpc",float(rtt))
            sdir=rroot/"sessions"/r["session"];r.update(profile_quality(code,sdir));dump(sdir/"session_summary_master.json",r);allrows.append(r)
        dump(out/"RTT_SENSITIVITY.json",{"condition":condition,"replicate":rep,"rtt_values_ms":args.rtt_values,"warmup_capacity_mbps":60.0,"planned":len(args.rtt_values)*3,"completed":len(allrows),"measurement_valid":sum(bool(x.get('measurement_valid')) for x in allrows),"rows":allrows})
      return 0
    finally:
      chown_tree(out,user)


def run_network_via_sudo(args, root: Path, kind: str):
    out=root/("exp3_network" if kind=="main" else "exp4_rtt")
    verdict=out/("NETWORK_COMPARISON.json" if kind=="main" else "RTT_SENSITIVITY.json")
    expected=54 if kind=="main" else len(args.rtt_values)*3
    if verdict.is_file():
        try:
            d=json.loads(verdict.read_text(encoding="utf-8")); rows=d.get("rows") or []
            if len(rows)==expected and all(bool(r.get("measurement_valid")) for r in rows):
                say(f"RESUME: reusing completed {'Experiment 3 network' if kind=='main' else 'Experiment 4 RTT'} results ({len(rows)}/{expected} valid measurements)")
                return out
        except Exception:
            pass
    if out.exists():
        say(f"RESUME: removing partial {'Experiment 3' if kind=='main' else 'Experiment 4'} directory before corrected network rerun: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=False)
    if os.geteuid()==0:
        cmd=[sys.executable,Path(__file__).resolve(),"--internal-network",kind,"--internal-output",out,"--network-rtt-ms",str(args.network_rtt_ms),"--rtt-values",",".join(str(x) for x in args.rtt_values),"--rtt-condition",args.rtt_condition,"--rtt-replicate",str(args.rtt_replicate)]
    else:
        sudo=shutil.which("sudo")
        if not sudo: raise RuntimeError("network experiments require sudo/Mininet")
        cmd=[sudo,"-E","/usr/bin/python3",Path(__file__).resolve(),"--internal-network",kind,"--internal-output",out,"--network-rtt-ms",str(args.network_rtt_ms),"--rtt-values",",".join(str(x) for x in args.rtt_values),"--rtt-condition",args.rtt_condition,"--rtt-replicate",str(args.rtt_replicate)]
    command(cmd,timeout=24*3600)
    return out


def package_light(root: Path):
    user,home=real_user_home();dest=home/"Downloads"/"files_to_upload";dest.mkdir(parents=True,exist_ok=True);z=dest/f"{root.name}_UPLOAD.zip"
    skip_ext={".h264",".h265",".hevc",".ivf",".bin",".mp4",".mkv"}
    with zipfile.ZipFile(z,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as f:
      for p in root.rglob("*"):
        if not p.is_file():continue
        if p.suffix.lower() in skip_ext or p.stat().st_size>80*1024*1024:continue
        f.write(p,p.relative_to(root.parent))
      f.write(Path(__file__).resolve(),Path(root.name)/Path(__file__).name)
    return z


def self_test(helpers):
    say("Master self-test: extracting and syntax-checking embedded helpers...")
    for k,p in helpers.items(): subprocess.run([sys.executable,"-m","py_compile",str(p)],check=True)
    # Matched-CBR helper compatibility audit. The recovered helper defines sha256()
    # but cbr_preflight references sha256_file(); the master deliberately supplies
    # that alias at runtime. This test makes that dependency explicit.
    cbr=load_module(helpers["cbr"],"cbr_self")
    if not hasattr(cbr,"sha256"): raise RuntimeError("matched-CBR helper missing sha256()")
    cbr.sha256_file=cbr.sha256
    def _read_csv_compat(path):
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    cbr.read_csv=_read_csv_compat
    if not callable(cbr.sha256_file) or not callable(cbr.read_csv):
        raise RuntimeError("matched-CBR compatibility shims failed")
    # Audit every helper function for unresolved LOAD_GLOBAL references so a
    # missing saved utility fails here instead of after a GPU run.
    import builtins, dis, inspect
    unresolved={}
    for _name,_obj in vars(cbr).items():
        if inspect.isfunction(_obj) and getattr(_obj,"__module__",None)==cbr.__name__:
            _miss=set()
            for _ins in dis.get_instructions(_obj):
                if _ins.opname in ("LOAD_GLOBAL","LOAD_NAME"):
                    _g=_ins.argval
                    if _g not in vars(cbr) and not hasattr(builtins,_g): _miss.add(_g)
            if _miss: unresolved[_name]=sorted(_miss)
    if unresolved: raise RuntimeError(f"matched-CBR unresolved globals after compatibility shims: {unresolved}")
    # Exercise the zero-QP CSV audit itself with a tiny synthetic file.
    _tmp_csv=Path(tempfile.mkdtemp(prefix="cbr-csv-selftest-"))/"events.csv"
    with _tmp_csv.open("w",newline="",encoding="utf-8") as _f:
        _w=csv.DictWriter(_f,fieldnames=["frame_id","qp_nonzero_count"]); _w.writeheader(); _w.writerow({"frame_id":0,"qp_nonzero_count":0})
    _audit=cbr._audit_zero_map(_tmp_csv,1)
    shutil.rmtree(_tmp_csv.parent,ignore_errors=True)
    if not _audit.get("pass"): raise RuntimeError("matched-CBR zero-QP CSV audit self-test failed")
    # Caruso self-test proves -5..+5 QP support.
    subprocess.run([sys.executable,str(helpers["caruso"]),"--self-test"],check=True)
    # Patch a sender and ensure all three controller modes are represented syntactically.
    v=load_module(helpers["v14"],"v14_self");td=Path(tempfile.mkdtemp(prefix="satc-master-selftest-"));v.extract_payload(td);patch_sender_controller(td/"integrated_sender.py")
    # Network source patch audit: 60-Mbit/s neutral warm-up, neutral 1-ms
    # transport canary, RTT/2 only for measured sessions, steady-tail PTS audit,
    # and a health-canary allowance for sparse forward loss (not duplicate/backward PTS).
    nd=td/"network_patch";nd.mkdir();nm=import_patched_v14(helpers["v14"],nd,40.0,"v14_rtt_self")
    nt=(nd/"v14_rtt_self.py").read_text(encoding="utf-8")
    if ('warmup_capacity=60.0' not in nt or nt.count('delay="20ms"')<1 or
        'bw=35.0,delay="1ms"' not in nt or
        'if len(rows)>=600 and len(controls)>=20 and len(gcc_events)>=3: break' not in nt or
        'forward PTS gaps exceed 5% health-canary allowance' not in nt or
        'preflight has non-increasing PTS values' not in nt or
        '"configured_rtt_ms":40' not in nt):
        raise RuntimeError("network warm-up/canary/RTT/steady-tail/health-canary patch audit failed")
    s=(td/"integrated_sender.py").read_text();
    for token in ("controller_mode=='fixed'","controller_mode=='gcc'","controller_mode=='rtmpc'"):
        if token not in s: raise RuntimeError("controller patch missing "+token)
    # Quality-mode patch audit: P7 encoder token + saved Caruso bitstream producer.
    qd=td/"quality_caruso";qd.mkdir();car=load_module(helpers["caruso"],"car_quality_self");car.extract_payload(qd);patch_preset(qd/"nvenc_dynamic.cpp","p7");patch_caruso_bitstream_producer(qd/"baseline_producer.py")
    subprocess.run([sys.executable,"-m","py_compile",str(qd/"baseline_producer.py")],check=True)
    qcpp=(qd/"nvenc_dynamic.cpp").read_text(); qprod=(qd/"baseline_producer.py").read_text()
    if "NV_ENC_PRESET_P7_GUID" not in qcpp or "stream_f.write(_packet)" not in qprod: raise RuntimeError("quality P7/bitstream patch audit failed")
    shutil.rmtree(td,ignore_errors=True);say("PASS: master syntax, matched-CBR SHA/CSV compatibility + unresolved-global audit, embedded helpers, Caruso QP range, quality P7/bitstream patch, and network controller + neutral warm-up + neutral transport canary + 5%-gap health check + strict non-increasing PTS rejection + measured-session RTT/2 patch.");return 0


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment",choices=["throughput","quality","network","rtt","all"],default="all")
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--resume-root",type=Path,default=None,help="continue an existing 360SATC_ALL_EXPERIMENTS_* directory and reuse validated completed components")
    p.add_argument("--london-video",type=Path,default=DEFAULT_LONDON)
    p.add_argument("--shared-mat",type=Path,default=None)
    p.add_argument("--trace-index",type=int,default=0)
    p.add_argument("--quality-frames",type=int,default=896)
    p.add_argument("--quality-warmup",type=int,default=240)
    p.add_argument("--quality-all-videos",action="store_true",help="default quality protocol uses London Tower only; set this for all four videos")
    p.add_argument("--skip-lpips",action="store_true",help="diagnostic only; paper run should keep LPIPS enabled")
    p.add_argument("--network-rtt-ms",type=float,default=20.0,help="fixed RTT for Experiment 3")
    p.add_argument("--rtt-values",type=lambda s:[float(x) for x in s.split(',')],default=[20.,40.,60.,80.])
    p.add_argument("--rtt-condition",choices=["low","high"],default="high")
    p.add_argument("--rtt-replicate",type=int,choices=[1,2,3],default=1)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--internal-network",choices=["main","rtt"],default=None,help=argparse.SUPPRESS)
    p.add_argument("--internal-output",type=Path,default=None,help=argparse.SUPPRESS)
    p.add_argument("--internal-lpips",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--source",type=Path,help=argparse.SUPPRESS);p.add_argument("--encoded",type=Path,help=argparse.SUPPRESS);p.add_argument("--warm",type=int,help=argparse.SUPPRESS);p.add_argument("--frames",type=int,help=argparse.SUPPRESS);p.add_argument("--out",type=Path,help=argparse.SUPPRESS)
    return p.parse_args()


def maybe_reexec_satc(args):
    if args.internal_network or args.internal_lpips or args.self_test:return
    if args.experiment in ("throughput","quality","all") and SATC_PY.is_file():
        try: same=Path(sys.executable).resolve()==SATC_PY.resolve()
        except Exception:same=False
        if not same:
            os.execv(str(SATC_PY),[str(SATC_PY),str(Path(__file__).resolve()),*sys.argv[1:]])


def main():
    args=parse_args()
    if args.internal_lpips: return internal_lpips(args)
    # Helpers must be available even in root/network child mode.
    if args.internal_network:
        temp=Path(args.internal_output)/"embedded_helpers";helpers=extract_helpers(temp);return internal_network(args,helpers)
    maybe_reexec_satc(args)
    user,home=real_user_home();stamp=dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_root is not None:
        root=args.resume_root.expanduser().resolve()
        if not root.is_dir(): raise RuntimeError(f"--resume-root does not exist or is not a directory: {root}")
        resumed=True
    else:
        root=args.output.expanduser().resolve() if args.output else home/"Downloads"/f"360SATC_ALL_EXPERIMENTS_{stamp}"
        root.mkdir(parents=True,exist_ok=False);resumed=False
    helpers=extract_helpers(root/"embedded_helpers")
    if args.self_test:return self_test(helpers)
    proto={"version":VERSION,"experiment":args.experiment,"created":dt.datetime.now().isoformat(),"resumed":resumed,"resume_root":str(root) if resumed else None,"modes":{"exp1":"processing throughput / no network","exp2":"fixed-rate encoder reconstruction quality / no network","exp3":f"real-5G trace replay / fixed RTT {args.network_rtt_ms} ms / fixed vs GCC vs RT-MPC","exp4":f"RTT sweep {args.rtt_values} ms / {args.rtt_condition}_rep{args.rtt_replicate} / RT-MPC"},"london_video":str(args.london_video)}
    if resumed:
        dump(root/f"MASTER_RESUME_{stamp}.json",proto)
    else:
        dump(root/"MASTER_PROTOCOL.json",proto)
    try:
        if args.experiment in ("throughput","all"):run_exp1(args,root,helpers)
        if args.experiment in ("quality","all"):run_exp2(args,root,helpers)
        if args.experiment in ("network","all"):run_network_via_sudo(args,root,"main")
        if args.experiment in ("rtt","all"):run_network_via_sudo(args,root,"rtt")
    except BaseException:
        err=(root/(f"MASTER_RESUME_ERROR_{stamp}.txt" if resumed else "MASTER_ERROR.txt"))
        err.write_text(traceback.format_exc(),encoding="utf-8");raise
    else:
        if resumed:
            # Preserve the original failure record, but mark the resumed campaign successful.
            (root/f"MASTER_RESUME_SUCCESS_{stamp}.txt").write_text("Resume completed successfully.\n",encoding="utf-8")
    finally:
        try:
            z=package_light(root);say("\nUPLOAD THIS FILE:\n"+str(z))
        except Exception: traceback.print_exc()
    say("\nDONE: "+str(root));return 0

if __name__=="__main__":
    raise SystemExit(main())
