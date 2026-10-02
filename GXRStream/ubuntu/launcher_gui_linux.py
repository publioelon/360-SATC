#!/usr/bin/env python3
from __future__ import annotations

import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional, TextIO

APP_TITLE = "QGXS Linux Sender"
CODECS = ["h265", "h264", "av1"]
INPUT_MODES = ["video-file", "image-folder"]
IMAGE_FORMATS = ["auto", "jpg", "jpeg", "png", "webp", "avif"]


def parse_metrics_line(line: str) -> dict[str, str]:
    """Parse one sender METRICS line into a key/value dictionary."""
    stripped = line.strip()
    if not stripped.startswith("METRICS|"):
        return {}

    metrics: dict[str, str] = {}
    for field in stripped.split("|")[1:]:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        key = key.strip()
        if key:
            metrics[key] = value.strip()
    return metrics


class ProcessHandle:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen[str]] = None
        self.thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class LinuxLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x930")
        self.minsize(1080, 760)

        self.base_dir = Path(__file__).resolve().parent
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.sender = ProcessHandle()
        self.last_metrics: dict[str, str] = {}
        self.metrics_file_handle: Optional[TextIO] = None
        self.metrics_file_path: Optional[Path] = None
        self.metrics_sample_count: int = 0

        self._create_vars()
        self._detect_defaults()
        self._build_ui()
        self._poll_log_queue()
        self._update_status_loop()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_vars(self) -> None:
        self.python_var = tk.StringVar(value=sys.executable)
        self.sender_var = tk.StringVar(value="")
        self.input_mode_var = tk.StringVar(value="video-file")
        self.input_path_var = tk.StringVar(value="")
        self.image_format_var = tk.StringVar(value="auto")
        self.codec_var = tk.StringVar(value="h265")
        self.host_var = tk.StringVar(value="127.0.0.1")
        self.port_var = tk.StringVar(value="9001")
        self.width_var = tk.StringVar(value="1920")
        self.height_var = tk.StringVar(value="960")
        self.fps_var = tk.StringVar(value="30")
        self.bitrate_var = tk.StringVar(value="8000")
        self.loop_var = tk.BooleanVar(value=True)
        self.encoder_var = tk.StringVar(value="nvenc")
        self.require_nvenc_var = tk.BooleanVar(value=True)
        self.gcc_enabled_var = tk.BooleanVar(value=False)
        self.gcc_min_bitrate_var = tk.StringVar(value="1000")
        self.record_metrics_var = tk.BooleanVar(value=False)
        self.metrics_dir_var = tk.StringVar(value=str(self.base_dir / "metrics"))
        self.status_var = tk.StringVar(value="Sender: stopped")

        self.metric_vars: dict[str, tk.StringVar] = {}
        defaults = {
            "connection": "Waiting",
            "sender_codec": "—",
            "sender_resolution": "—",
            "sender_fps": "—",
            "sender_encoder": "—",
            "sender_target_bitrate": "—",
            "sender_actual_bitrate": "—",
            "estimated_bandwidth": "—",
            "adaptation_mode": "Manual",
            "adaptation_state": "Manual bitrate",
            "adaptation_reason": "Automatic adaptation disabled",
            "recording_status": "Off",
            "recording_file": "—",
            "recording_samples": "0",
            "sender_rtt": "—",
            "sender_loss": "—",
            "sender_jitter": "—",
            "pipeline_state": "—",
            "receiver_alive": "Waiting",
            "receiver_mode": "—",
            "receiver_codec": "—",
            "receiver_resolution": "—",
            "receiver_decoder": "—",
            "receiver_received_fps": "—",
            "receiver_decoded_fps": "—",
            "receiver_displayed_fps": "—",
            "receiver_bitrate": "—",
            "receiver_loss": "—",
            "receiver_jitter": "—",
            "receiver_rtt": "—",
            "receiver_backlog": "—",
            "receiver_drops": "—",
            "receiver_stall": "—",
            "receiver_frame_age": "—",
            "receiver_metrics_age": "—",
            "receiver_last_error": "None",
        }
        for key, value in defaults.items():
            self.metric_vars[key] = tk.StringVar(value=value)

    def _detect_defaults(self) -> None:
        sender = self.base_dir / "sender" / "webrtc_sender.py"
        if sender.is_file():
            self.sender_var.set(str(sender))

        python3 = shutil.which("python3")
        if python3:
            self.python_var.set(python3)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root, text=APP_TITLE, font=("TkDefaultFont", 16, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            root,
            text="GStreamer WebRTC sender with live sender, receiver, and network telemetry.",
        )
        subtitle.pack(anchor="w", pady=(2, 10))

        config = ttk.LabelFrame(root, text="Runtime and input", padding=10)
        config.pack(fill=tk.X)

        self._path_row(config, 0, "Python", self.python_var, self._browse_python)
        self._path_row(config, 1, "Sender", self.sender_var, self._browse_sender)
        self._path_row(config, 2, "Input", self.input_path_var, self._browse_input)

        mode_frame = ttk.Frame(config)
        mode_frame.grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(config, text="Input mode").grid(row=3, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(
            mode_frame,
            textvariable=self.input_mode_var,
            values=INPUT_MODES,
            width=16,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Label(mode_frame, text="Image format").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Combobox(
            mode_frame,
            textvariable=self.image_format_var,
            values=IMAGE_FORMATS,
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(mode_frame, text="Loop", variable=self.loop_var).pack(side=tk.LEFT, padx=(18, 0))

        config.columnconfigure(1, weight=1)

        stream = ttk.LabelFrame(root, text="Stream settings", padding=10)
        stream.pack(fill=tk.X, pady=(10, 0))

        self._entry(stream, 0, "Codec", self.codec_var, combo=CODECS)
        self._entry(stream, 0, "Receiver host", self.host_var, col=2)
        self._entry(stream, 0, "Port", self.port_var, col=4, width=8)
        self._entry(stream, 1, "Width", self.width_var, width=8)
        self._entry(stream, 1, "Height", self.height_var, col=2, width=8)
        self._entry(stream, 1, "FPS", self.fps_var, col=4, width=8)
        self._entry(stream, 1, "Bitrate kbps", self.bitrate_var, col=6, width=10)

        enc_frame = ttk.Frame(stream)
        enc_frame.grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Label(stream, text="Encoder").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        ttk.Combobox(
            enc_frame,
            textvariable=self.encoder_var,
            values=["nvenc", "auto", "software"],
            width=12,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            enc_frame,
            text="Require NVENC for high-FPS tests",
            variable=self.require_nvenc_var,
        ).pack(side=tk.LEFT, padx=(12, 0))

        feature_frame = ttk.Frame(stream)
        feature_frame.grid(row=3, column=0, columnspan=8, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            feature_frame,
            text="Automatic GCC/TWCC bitrate adaptation",
            variable=self.gcc_enabled_var,
        ).pack(side=tk.LEFT)
        ttk.Label(feature_frame, text="Minimum kbps").pack(side=tk.LEFT, padx=(12, 5))
        ttk.Entry(
            feature_frame,
            textvariable=self.gcc_min_bitrate_var,
            width=8,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            feature_frame,
            text="Record session metrics to .txt",
            variable=self.record_metrics_var,
        ).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Button(
            feature_frame,
            text="Metrics folder",
            command=self._browse_metrics_dir,
        ).pack(side=tk.LEFT, padx=(8, 0))

        buttons = ttk.Frame(root)
        buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(buttons, text="Check Linux runtime", command=self.check_runtime).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Start sender", command=self.start_sender).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Stop sender", command=self.stop_sender).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(buttons, textvariable=self.status_var).pack(side=tk.RIGHT)

        telemetry = ttk.Frame(root)
        telemetry.pack(fill=tk.X, pady=(10, 0))
        telemetry.columnconfigure(0, weight=1)
        telemetry.columnconfigure(1, weight=1)

        sender_panel = ttk.LabelFrame(telemetry, text="Sender and network", padding=10)
        sender_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        receiver_panel = ttk.LabelFrame(telemetry, text="Receiver", padding=10)
        receiver_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        sender_rows = [
            ("Connection", "connection"),
            ("Codec / resolution", "sender_codec_resolution"),
            ("Actual FPS / encoder", "sender_fps_encoder"),
            ("Target / actual bitrate", "sender_bitrates"),
            ("Estimated bandwidth", "estimated_bandwidth"),
            ("Adaptation", "adaptation_summary"),
            ("Adaptation reason", "adaptation_reason"),
            ("RTT / loss / jitter", "sender_network"),
            ("Metrics recording", "recording_summary"),
            ("Metrics file / samples", "recording_file_samples"),
            ("Pipeline", "pipeline_state"),
        ]
        receiver_rows = [
            ("Status / mode", "receiver_status_mode"),
            ("Codec / resolution", "receiver_codec_resolution"),
            ("Decoder", "receiver_decoder"),
            ("Received / decoded FPS", "receiver_input_decode"),
            ("Displayed FPS / bitrate", "receiver_display_bitrate"),
            ("Loss / jitter / RTT", "receiver_network"),
            ("Backlog / drops", "receiver_queue_drops"),
            ("Stall / frame age", "receiver_stall_age"),
            ("Telemetry age", "receiver_metrics_age"),
            ("Last error", "receiver_last_error"),
        ]

        self.composite_vars: dict[str, tk.StringVar] = {}
        for _, key in sender_rows + receiver_rows:
            self.composite_vars[key] = tk.StringVar(value="—")

        for row, (label, key) in enumerate(sender_rows):
            self._metric_row(sender_panel, row, label, self.composite_vars[key])
        for row, (label, key) in enumerate(receiver_rows):
            self._metric_row(receiver_panel, row, label, self.composite_vars[key])

        self._refresh_composite_metrics()

        log_frame = ttk.LabelFrame(root, text="Sender log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_frame, wrap="word", height=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=yscroll.set)

    def _path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)

    def _entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        col: int = 0,
        width: int = 14,
        combo=None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=4)
        if combo:
            widget = ttk.Combobox(parent, textvariable=var, values=combo, width=width, state="readonly")
        else:
            widget = ttk.Entry(parent, textvariable=var, width=width)
        widget.grid(row=row, column=col + 1, sticky="w", padx=(0, 16), pady=4)

    @staticmethod
    def _metric_row(parent: ttk.Frame, row: int, label: str, value_var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
        ttk.Label(parent, textvariable=value_var, font=("TkDefaultFont", 9, "bold")).grid(
            row=row,
            column=1,
            sticky="w",
            pady=2,
        )
        parent.columnconfigure(1, weight=1)

    def _browse_python(self) -> None:
        path = filedialog.askopenfilename(title="Select python3 executable")
        if path:
            self.python_var.set(path)

    def _browse_sender(self) -> None:
        path = filedialog.askopenfilename(
            title="Select sender/webrtc_sender.py",
            filetypes=[("Python", "*.py"), ("All", "*")],
        )
        if path:
            self.sender_var.set(path)

    def _browse_input(self) -> None:
        if self.input_mode_var.get() == "image-folder":
            path = filedialog.askdirectory(title="Select image folder")
        else:
            path = filedialog.askopenfilename(title="Select video file")
        if path:
            self.input_path_var.set(path)

    def _browse_metrics_dir(self) -> None:
        initial = self.metrics_dir_var.get().strip() or str(self.base_dir)
        path = filedialog.askdirectory(
            title="Select metrics output folder",
            initialdir=initial if Path(initial).exists() else str(self.base_dir),
        )
        if path:
            self.metrics_dir_var.set(path)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _start_metrics_recording(self, cmd: list[str], env: dict[str, str]) -> None:
        self._stop_metrics_recording("new session")
        self.metrics_sample_count = 0

        if not self.record_metrics_var.get():
            self.metric_vars["recording_status"].set("Off")
            self.metric_vars["recording_file"].set("—")
            self.metric_vars["recording_samples"].set("0")
            self._refresh_composite_metrics()
            return

        output_dir = Path(self.metrics_dir_var.get().strip() or (self.base_dir / "metrics"))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        path = output_dir / f"QSXR_metrics_{stamp}.txt"

        handle = path.open("w", encoding="utf-8", buffering=1)
        handle.write("# QSXR sender and receiver network metrics\n")
        handle.write(f"# started_at={self._timestamp()}\n")
        handle.write(f"# command={shlex.join(cmd)}\n")
        handle.write(f"# gcc_enabled={env.get('QGXS_GCC_ENABLED', '0')}\n")
        handle.write(f"# gcc_min_bitrate_kbps={env.get('QGXS_GCC_MIN_BITRATE_KBPS', 'n/a')}\n")
        handle.write(f"# gcc_max_bitrate_kbps={env.get('QGXS_GCC_MAX_BITRATE_KBPS', 'n/a')}\n")
        handle.write(
            f"SESSION_START|timestamp={self._timestamp()}|codec={self.codec_var.get()}|"
            f"resolution={self.width_var.get()}x{self.height_var.get()}|"
            f"fps={self.fps_var.get()}|configured_bitrate_kbps={self.bitrate_var.get()}\n"
        )
        handle.flush()

        self.metrics_file_handle = handle
        self.metrics_file_path = path
        self.metric_vars["recording_status"].set("On")
        self.metric_vars["recording_file"].set(path.name)
        self.metric_vars["recording_samples"].set("0")
        self._refresh_composite_metrics()
        self._append_log(f"[metrics-recorder] Recording to {path}\n")

    def _record_metrics_sample(self, raw_line: str) -> None:
        handle = self.metrics_file_handle
        if handle is None:
            return
        try:
            handle.write(f"timestamp={self._timestamp()}|{raw_line.strip()}\n")
            handle.flush()
            self.metrics_sample_count += 1
            self.metric_vars["recording_samples"].set(str(self.metrics_sample_count))
            self._refresh_composite_metrics()
        except Exception as exc:
            self._append_log(f"[metrics-recorder] Write failed: {exc}\n")
            self._stop_metrics_recording("write error")

    def _stop_metrics_recording(self, reason: str) -> None:
        handle = self.metrics_file_handle
        if handle is not None:
            try:
                handle.write(
                    f"SESSION_STOP|timestamp={self._timestamp()}|"
                    f"reason={str(reason).replace('|', '/')}|"
                    f"samples={self.metrics_sample_count}\n"
                )
                handle.flush()
                handle.close()
            except Exception:
                pass

        self.metrics_file_handle = None
        if self.metrics_file_path is not None:
            self.metric_vars["recording_status"].set("Saved")
            self.metric_vars["recording_file"].set(self.metrics_file_path.name)
        else:
            self.metric_vars["recording_status"].set("Off")
        self.metric_vars["recording_samples"].set(str(self.metrics_sample_count))
        if hasattr(self, "composite_vars"):
            self._refresh_composite_metrics()

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    @staticmethod
    def _metric(metrics: dict[str, str], key: str, default: str = "—") -> str:
        value = metrics.get(key, "").strip()
        return default if not value or value.lower() == "n/a" else value

    @staticmethod
    def _unit(value: str, suffix: str) -> str:
        if value in ("", "—", "n/a"):
            return "—"
        return f"{value} {suffix}"

    def _apply_metrics(self, metrics: dict[str, str]) -> None:
        if not metrics:
            return

        self.last_metrics = metrics
        v = self.metric_vars

        signaling = self._metric(metrics, "signaling_state")
        webrtc = self._metric(metrics, "webrtc_state")
        ice = self._metric(metrics, "ice_state")
        v["connection"].set(f"Signaling {signaling} · WebRTC {webrtc} · ICE {ice}")

        v["sender_codec"].set(self._metric(metrics, "codec"))
        v["sender_resolution"].set(self._metric(metrics, "resolution"))
        v["sender_fps"].set(self._unit(self._metric(metrics, "actual_fps"), "FPS"))
        v["sender_encoder"].set(self._metric(metrics, "encoder"))
        v["sender_target_bitrate"].set(self._unit(self._metric(metrics, "target_bitrate_mbps"), "Mbps"))
        v["sender_actual_bitrate"].set(self._unit(self._metric(metrics, "actual_bitrate_mbps"), "Mbps"))
        v["estimated_bandwidth"].set(self._unit(self._metric(metrics, "estimated_bandwidth_mbps"), "Mbps"))
        v["sender_rtt"].set(self._unit(self._metric(metrics, "rtt_ms"), "ms"))
        v["sender_loss"].set(self._unit(self._metric(metrics, "packet_loss_pct"), "%"))
        v["sender_jitter"].set(self._unit(self._metric(metrics, "jitter_ms"), "ms"))
        v["pipeline_state"].set(self._metric(metrics, "pipeline_state"))

        adaptation_mode = self._metric(metrics, "adaptation_mode", "manual").lower()
        if adaptation_mode in ("automatic", "auto", "gcc"):
            v["adaptation_mode"].set("Automatic")
        else:
            v["adaptation_mode"].set("Manual")
        v["adaptation_state"].set(
            self._metric(
                metrics,
                "adaptation_state",
                "Active" if adaptation_mode == "automatic" else "Manual bitrate",
            )
        )
        v["adaptation_reason"].set(
            self._metric(metrics, "adaptation_reason", "Automatic adaptation disabled")
        )

        alive = self._metric(metrics, "receiver_alive", "no").lower()
        v["receiver_alive"].set("Connected" if alive in ("1", "yes", "true") else "Disconnected")
        v["receiver_mode"].set(self._metric(metrics, "receiver_display_mode"))
        v["receiver_codec"].set(self._metric(metrics, "receiver_codec"))
        v["receiver_resolution"].set(self._metric(metrics, "receiver_resolution"))
        v["receiver_decoder"].set(self._metric(metrics, "receiver_decoder"))
        v["receiver_received_fps"].set(self._unit(self._metric(metrics, "receiver_received_fps"), "FPS"))
        v["receiver_decoded_fps"].set(self._unit(self._metric(metrics, "receiver_decoded_fps"), "FPS"))
        v["receiver_displayed_fps"].set(self._unit(self._metric(metrics, "receiver_displayed_fps"), "FPS"))
        v["receiver_bitrate"].set(self._unit(self._metric(metrics, "receiver_bitrate_mbps"), "Mbps"))
        v["receiver_loss"].set(self._unit(self._metric(metrics, "receiver_packet_loss_pct"), "%"))
        v["receiver_jitter"].set(self._unit(self._metric(metrics, "receiver_jitter_ms"), "ms"))
        v["receiver_rtt"].set(self._unit(self._metric(metrics, "receiver_rtt_ms"), "ms"))
        v["receiver_backlog"].set(self._metric(metrics, "receiver_queue_backlog"))
        v["receiver_drops"].set(self._metric(metrics, "receiver_dropped_frames"))

        stall = self._metric(metrics, "receiver_stall", "0").lower()
        v["receiver_stall"].set("Yes" if stall in ("1", "yes", "true") else "No")
        v["receiver_frame_age"].set(self._unit(self._metric(metrics, "receiver_last_frame_age_ms"), "ms"))
        v["receiver_metrics_age"].set(self._unit(self._metric(metrics, "receiver_metrics_age_ms"), "ms"))
        v["receiver_last_error"].set(self._metric(metrics, "receiver_last_error", "None"))

        self._refresh_composite_metrics()

    def _refresh_composite_metrics(self) -> None:
        v = self.metric_vars
        c = self.composite_vars
        c["sender_codec_resolution"].set(f"{v['sender_codec'].get()} · {v['sender_resolution'].get()}")
        c["sender_fps_encoder"].set(f"{v['sender_fps'].get()} · {v['sender_encoder'].get()}")
        c["sender_bitrates"].set(f"{v['sender_target_bitrate'].get()} / {v['sender_actual_bitrate'].get()}")
        c["estimated_bandwidth"].set(v["estimated_bandwidth"].get())
        c["adaptation_summary"].set(f"{v['adaptation_mode'].get()} · {v['adaptation_state'].get()}")
        c["adaptation_reason"].set(v["adaptation_reason"].get())
        c["sender_network"].set(f"{v['sender_rtt'].get()} / {v['sender_loss'].get()} / {v['sender_jitter'].get()}")
        c["recording_summary"].set(v["recording_status"].get())
        c["recording_file_samples"].set(
            f"{v['recording_file'].get()} / {v['recording_samples'].get()}"
        )
        c["pipeline_state"].set(v["pipeline_state"].get())
        c["connection"].set(v["connection"].get())

        c["receiver_status_mode"].set(f"{v['receiver_alive'].get()} · {v['receiver_mode'].get()}")
        c["receiver_codec_resolution"].set(f"{v['receiver_codec'].get()} · {v['receiver_resolution'].get()}")
        c["receiver_decoder"].set(v["receiver_decoder"].get())
        c["receiver_input_decode"].set(f"{v['receiver_received_fps'].get()} / {v['receiver_decoded_fps'].get()}")
        c["receiver_display_bitrate"].set(f"{v['receiver_displayed_fps'].get()} / {v['receiver_bitrate'].get()}")
        c["receiver_network"].set(f"{v['receiver_loss'].get()} / {v['receiver_jitter'].get()} / {v['receiver_rtt'].get()}")
        c["receiver_queue_drops"].set(f"{v['receiver_backlog'].get()} / {v['receiver_drops'].get()}")
        c["receiver_stall_age"].set(f"{v['receiver_stall'].get()} / {v['receiver_frame_age'].get()}")
        c["receiver_metrics_age"].set(v["receiver_metrics_age"].get())
        c["receiver_last_error"].set(v["receiver_last_error"].get())

    def _reset_live_metrics(self) -> None:
        self.last_metrics = {}
        self.metric_vars["connection"].set("Waiting")
        self.metric_vars["receiver_alive"].set("Waiting")
        if self.gcc_enabled_var.get():
            self.metric_vars["adaptation_mode"].set("Automatic")
            self.metric_vars["adaptation_state"].set("Waiting")
            self.metric_vars["adaptation_reason"].set("Waiting for GCC estimate")
        else:
            self.metric_vars["adaptation_mode"].set("Manual")
            self.metric_vars["adaptation_state"].set("Manual bitrate")
            self.metric_vars["adaptation_reason"].set("Automatic adaptation disabled")
        self._refresh_composite_metrics()

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
                metrics = parse_metrics_line(line)
                if metrics:
                    self._apply_metrics(metrics)
                    self._record_metrics_sample(line)
                if line.startswith("\n[sender exited with code"):
                    self._stop_metrics_recording("sender process exited")
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _update_status_loop(self) -> None:
        self.status_var.set("Sender: running" if self.sender.is_running() else "Sender: stopped")
        self.after(500, self._update_status_loop)

    def _validate_basic(self) -> bool:
        if not Path(self.python_var.get()).exists():
            messagebox.showerror(APP_TITLE, "Python executable does not exist.")
            return False
        if not Path(self.sender_var.get()).is_file():
            messagebox.showerror(APP_TITLE, "Sender script does not exist.")
            return False
        if not Path(self.input_path_var.get()).exists():
            messagebox.showerror(APP_TITLE, "Input path does not exist.")
            return False
        try:
            port = int(self.port_var.get())
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            fps = int(self.fps_var.get())
            bitrate = int(self.bitrate_var.get())
            gcc_min = int(self.gcc_min_bitrate_var.get())
        except ValueError:
            messagebox.showerror(APP_TITLE, "Port, dimensions, FPS, and bitrates must be integers.")
            return False
        if min(port, width, height, fps, bitrate, gcc_min) <= 0:
            messagebox.showerror(APP_TITLE, "Port, dimensions, FPS, and bitrates must be positive.")
            return False
        if self.gcc_enabled_var.get() and gcc_min > bitrate:
            messagebox.showerror(
                APP_TITLE,
                "Minimum GCC bitrate cannot be greater than the configured bitrate ceiling.",
            )
            return False
        return True

    def _build_sender_command(self) -> list[str]:
        cmd = [
            self.python_var.get(),
            "-u",
            self.sender_var.get(),
            self.codec_var.get(),
            "--input-mode",
            self.input_mode_var.get(),
            "--input",
            self.input_path_var.get(),
            "--image-format",
            self.image_format_var.get(),
        ]

        cmd.append("--loop" if self.loop_var.get() else "--no-loop")
        cmd.extend(
            [
                self.host_var.get(),
                self.port_var.get(),
                self.width_var.get(),
                self.height_var.get(),
                self.fps_var.get(),
                self.bitrate_var.get(),
            ]
        )
        return cmd

    def _sender_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["QGXS_USE_SYSTEM_GSTREAMER"] = "1"
        env["QGXS_ENCODER"] = self.encoder_var.get()
        env["QGXS_REQUIRE_NVENC"] = "1" if self.require_nvenc_var.get() else "0"
        env["QGXS_GCC_ENABLED"] = "1" if self.gcc_enabled_var.get() else "0"
        env["QGXS_GCC_MIN_BITRATE_KBPS"] = self.gcc_min_bitrate_var.get().strip()
        env["QGXS_GCC_MAX_BITRATE_KBPS"] = self.bitrate_var.get().strip()
        env.setdefault("QGXS_GCC_HEADROOM", "0.85")
        env.setdefault("GST_DEBUG_NO_COLOR", "1")
        return env

    def check_runtime(self) -> None:
        commands = [
            ["gst-inspect-1.0", "--version"],
            ["gst-inspect-1.0", "webrtcbin"],
            ["gst-inspect-1.0", "rtpgccbwe"],
            ["gst-inspect-1.0", "rtph264pay"],
            ["gst-inspect-1.0", "rtph265pay"],
            ["gst-inspect-1.0", "rtpav1pay"],
            ["gst-inspect-1.0", "nvh264enc"],
            ["gst-inspect-1.0", "nvh265enc"],
            ["gst-inspect-1.0", "nvav1enc"],
            [
                self.python_var.get(),
                "-c",
                "import gi; gi.require_version('Gst','1.0'); gi.require_version('GstSdp','1.0'); gi.require_version('GstWebRTC','1.0'); from gi.repository import Gst, GstSdp, GstWebRTC; Gst.init(None); print('Python GStreamer WebRTC OK')",
            ],
        ]

        self._append_log("\n=== Runtime check ===\n")
        for cmd in commands:
            self._append_log("$ " + " ".join(cmd) + "\n")
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=12,
                )
                output = result.stdout or ""
                self._append_log(output)
                self._append_log(f"[exit {result.returncode}]\n\n")
            except Exception as exc:
                self._append_log(f"[failed] {exc}\n\n")

    def start_sender(self) -> None:
        if self.sender.is_running():
            messagebox.showinfo(APP_TITLE, "Sender is already running.")
            return
        if not self._validate_basic():
            return

        self._reset_live_metrics()
        cmd = self._build_sender_command()
        env = self._sender_env()
        cwd = str(Path(self.sender_var.get()).resolve().parent)

        self._append_log("\n=== Starting sender ===\n")
        self._append_log("$ " + " ".join(cmd) + "\n")
        self._append_log(f"cwd={cwd}\n")
        self._append_log(
            f"QGXS_ENCODER={env.get('QGXS_ENCODER')} "
            f"QGXS_REQUIRE_NVENC={env.get('QGXS_REQUIRE_NVENC')} "
            f"QGXS_GCC_ENABLED={env.get('QGXS_GCC_ENABLED')} "
            f"QGXS_GCC_MIN_BITRATE_KBPS={env.get('QGXS_GCC_MIN_BITRATE_KBPS')} "
            f"QGXS_GCC_MAX_BITRATE_KBPS={env.get('QGXS_GCC_MAX_BITRATE_KBPS')}\n\n"
        )

        try:
            self.sender.process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to start sender:\n{exc}")
            return

        try:
            self._start_metrics_recording(cmd, env)
        except Exception as exc:
            self._append_log(f"[metrics-recorder] Could not start recording: {exc}\n")
            self._stop_metrics_recording("start error")

        self.sender.thread = threading.Thread(target=self._read_sender_output, daemon=True)
        self.sender.thread.start()

    def _read_sender_output(self) -> None:
        process = self.sender.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.log_queue.put(line)
        code = process.wait()
        self.log_queue.put(f"\n[sender exited with code {code}]\n")

    def stop_sender(self) -> None:
        process = self.sender.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        if self.sender.thread is not None and self.sender.thread.is_alive():
            self.sender.thread.join(timeout=2)
        # The reader queues an exit marker after all preceding METRICS lines.
        # _poll_log_queue closes the file only after those samples are written.

    def _on_close(self) -> None:
        self.stop_sender()
        try:
            while True:
                line = self.log_queue.get_nowait()
                metrics = parse_metrics_line(line)
                if metrics:
                    self._record_metrics_sample(line)
        except queue.Empty:
            pass
        self._stop_metrics_recording("launcher closed")
        self.destroy()


if __name__ == "__main__":
    app = LinuxLauncher()
    app.mainloop()
