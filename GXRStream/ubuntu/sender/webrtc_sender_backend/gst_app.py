from __future__ import annotations

import os
import select
import socket
import threading
from pathlib import Path
from typing import Optional, Any

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")

from gi.repository import GLib, Gst, GstSdp, GstWebRTC  # noqa: E402

from .config import (
    SenderConfig,
    codec_label,
    is_image_folder_mode,
    is_video_file_mode,
    list_supported_images,
)
from .pipeline_builder import (
    build_sender_pipeline_description,
    external_h264_fifo_path,
)
from .signaling import SignalingClient, base64_decode_to_string, base64_encode


def initialize_gstreamer(argv: list[str]) -> None:
    # Do not pass application CLI arguments to GStreamer.
    # This avoids GStreamer trying to parse sender-specific flags such as
    # --input-mode/--image-format and also prevents startup crashes observed
    # when different GStreamer runtimes are present on Windows.
    Gst.init(None)


class WebRTCSender:
    def __init__(self, config: SenderConfig):
        self.config = config

        self.pipeline: Optional[Gst.Element] = None
        self.webrtc: Optional[Gst.Element] = None
        self.main_loop: Optional[GLib.MainLoop] = None
        self.initial_offer_requested: bool = False

        self.frame_source: Optional[Gst.Element] = None
        self.image_files: list[Path] = []
        self.image_index: int = 0
        self.frame_number: int = 0
        self.frame_duration_ns: int = 0
        self.image_loader: Optional[Any] = None
        self.loop_image_folder: bool = True

        self.external_h264_source: Optional[Gst.Element] = None
        self.external_h264_parse: Optional[Gst.Element] = None
        self.external_h264_au_probe_id: Optional[int] = None
        self.external_h264_au_number: int = 0
        self.external_h264_frame_duration_ns: int = 0
        self.external_h264_base_pts_ns: Optional[int] = None
        self.external_h264_fifo: str = ""
        self.external_h264_reader_thread: Optional[threading.Thread] = None
        self.external_h264_reader_stop = threading.Event()
        self.external_h264_reader_fd: Optional[int] = None

        self.running = threading.Event()
        self.running.set()

        self.signaling = SignalingClient(self.running)
        self.signaling_thread: Optional[threading.Thread] = None

        self.loop_enabled: bool = bool(getattr(config, "loop", True))
        self.loop_count: int = 0
        self.loop_in_progress: bool = False
        self.loop_guard_probe_id: Optional[int] = None

        self.loop_watchdog_id: Optional[int] = None
        self.loop_watchdog_interval_ms: int = 100

        self.loop_margin_ns: int = int(1.50 * Gst.SECOND)
        self.loop_cooldown_ns: int = int(2.00 * Gst.SECOND)
        self.last_loop_request_monotonic_ns: int = 0

        self.last_watchdog_duration_ns: int = 0
        self.last_watchdog_position_ns: int = 0

        self.debug_loop_messages: bool = False

        self.raw_frame_probe_id: Optional[int] = None
        self.rtp_probe_id: Optional[int] = None

        self.metrics_timer_id: Optional[int] = None
        self.metrics_interval_ms: int = 1000
        self.metrics_last_time_ns: int = GLib.get_monotonic_time() * 1000

        self.metrics_raw_frames: int = 0
        self.metrics_last_raw_frames: int = 0

        self.metrics_bytes_sent: int = 0
        self.metrics_packets_sent: int = 0
        self.metrics_last_bytes_sent: int = 0
        self.metrics_last_packets_sent: int = 0

        self.metrics_actual_fps: float = 0.0
        self.metrics_actual_bitrate_mbps: float = 0.0

        self.metrics_signaling_state: str = "created"
        self.metrics_webrtc_state: str = "unknown"
        self.metrics_ice_state: str = "unknown"
        self.metrics_selected_route: str = "n/a"
        self.metrics_rtt_ms: str = "n/a"
        self.metrics_packet_loss_pct: str = "n/a"
        self.metrics_jitter_ms: str = "n/a"
        self.metrics_estimated_bandwidth_mbps: str = "n/a"
        self.metrics_pipeline_state: str = "unknown"
        self.metrics_last_error: str = "None"

        self.receiver_feedback_host: str = "0.0.0.0"
        self.receiver_feedback_port: int = int(getattr(config, "receiver_feedback_port", int(config.port) + 100))
        self.receiver_feedback_thread: Optional[threading.Thread] = None
        self.receiver_feedback_server_socket: Optional[socket.socket] = None
        self.receiver_feedback_client_socket: Optional[socket.socket] = None
        self.receiver_metrics_lock = threading.Lock()
        self.receiver_metrics: dict[str, str] = {
            "receiver_alive": "0",
            "receiver_display_mode": "n/a",
            "receiver_unity_fps": "n/a",
            "receiver_copied_fps": "n/a",
            "receiver_frame_id": "0",
            "receiver_texture_attached": "0",
            "receiver_stall": "n/a",
            "received_fps": "n/a",
            "decoded_fps": "n/a",
            "displayed_fps": "n/a",
            "bitrate_mbps": "n/a",
            "packets_received": "0",
            "bytes_received": "0",
            "packets_lost": "n/a",
            "packet_loss_pct": "n/a",
            "jitter_ms": "n/a",
            "rtt_ms": "n/a",
            "decoder": "n/a",
            "codec": "n/a",
            "resolution": "n/a",
            "queue_backlog": "0",
            "dropped_frames": "0",
            "last_frame_age_ms": "n/a",
            "last_error": "None",
        }
        self.receiver_metrics_last_seen_ns: int = 0

        # GCC/TWCC bitrate adaptation. Manual mode remains the default so
        # existing command-line behavior is preserved unless the launcher
        # explicitly enables automatic adaptation.
        self.gcc_enabled: bool = self._env_flag("QGXS_GCC_ENABLED", False)
        self.gcc_min_bitrate_kbps: int = self._env_int(
            "QGXS_GCC_MIN_BITRATE_KBPS",
            min(1000, max(1, int(config.bitrate_kbps))),
        )
        self.gcc_max_bitrate_kbps: int = self._env_int(
            "QGXS_GCC_MAX_BITRATE_KBPS",
            max(1, int(config.bitrate_kbps)),
        )
        self.gcc_max_bitrate_kbps = max(
            self.gcc_min_bitrate_kbps,
            self.gcc_max_bitrate_kbps,
        )
        self.gcc_headroom: float = self._env_float(
            "QGXS_GCC_HEADROOM",
            0.85,
            minimum=0.50,
            maximum=1.00,
        )
        self.gcc_increase_hysteresis: float = 1.08
        self.gcc_decrease_hysteresis: float = 0.95
        self.gcc_recovery_step: float = 1.10
        self.gcc_latest_estimate_bps: int = 0
        self.gcc_applied_bitrate_kbps: int = max(
            self.gcc_min_bitrate_kbps,
            min(int(config.bitrate_kbps), self.gcc_max_bitrate_kbps),
        )
        self.gcc_estimators: list[Gst.Element] = []
        self.gcc_lock = threading.Lock()
        self.video_encoder: Optional[Gst.Element] = None
        self.metrics_adaptation_mode: str = "automatic" if self.gcc_enabled else "manual"
        self.metrics_adaptation_state: str = "waiting" if self.gcc_enabled else "manual"
        self.metrics_adaptation_reason: str = (
            "Waiting for GCC estimate" if self.gcc_enabled else "Automatic adaptation disabled"
        )

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in ("0", "false", "no", "off", "")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return int(default)
        try:
            value = int(raw.strip())
        except (TypeError, ValueError):
            return int(default)
        return max(1, value)

    @staticmethod
    def _env_float(
        name: str,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        raw = os.environ.get(name)
        try:
            value = float(raw) if raw is not None else float(default)
        except (TypeError, ValueError):
            value = float(default)
        return max(minimum, min(maximum, value))

    def safe_metric_value(self, value: Any) -> str:
        text = str(value)
        text = text.replace("|", "/")
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.strip()

        if len(text) > 180:
            text = text[:177] + "..."

        return text

    def enum_to_string(self, value: Any) -> str:
        if value is None:
            return "unknown"

        if hasattr(value, "value_nick"):
            return str(value.value_nick)

        if hasattr(value, "value_name"):
            return str(value.value_name)

        return str(value)

    def on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> bool:
        src_name = message.src.get_name() if message.src else "unknown"

        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            error_message = err.message if err else "unknown"

            self.metrics_last_error = f"ERROR from {src_name}: {error_message}"

            print(f"\n[ERROR from {src_name}] {error_message}", flush=True)
            if debug:
                print(f"[DEBUG] {debug}", flush=True)

            if self.main_loop is not None:
                self.main_loop.quit()

        elif message.type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            warning_message = err.message if err else "unknown"

            self.metrics_last_error = f"WARNING from {src_name}: {warning_message}"

            print(f"\n[WARNING from {src_name}] {warning_message}", flush=True)
            if debug:
                print(f"[DEBUG] {debug}", flush=True)

        elif message.type == Gst.MessageType.EOS:
            if self.loop_enabled and is_video_file_mode(self.config):
                if self.debug_loop_messages:
                    print("[bus] EOS reached bus. Attempting fallback loop seek.", flush=True)
                self.schedule_video_loop_seek("bus-eos")
            elif self.loop_enabled and is_image_folder_mode(self.config):
                if self.debug_loop_messages:
                    print("[bus] EOS received from image-folder mode. Attempting fallback loop seek.", flush=True)
                self.schedule_video_loop_seek("image-folder-eos")
            else:
                print("[bus] EOS received. Loop disabled. Stopping sender.", flush=True)
                if self.main_loop is not None:
                    self.main_loop.quit()

        return True

    def add_bus_watch(self, pipeline: Gst.Element) -> None:
        bus = pipeline.get_bus()

        if bus is None:
            print("[bus] Could not get pipeline bus.", flush=True)
            return

        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    def install_video_loop_guard(self) -> None:
        if self.pipeline is None:
            return

        if not is_video_file_mode(self.config):
            return

        loop_guard = self.pipeline.get_by_name("loop_guard")

        if loop_guard is None:
            print("[sender] Loop guard element was not found. Video-file loop will use watchdog/bus fallback.", flush=True)
            return

        src_pad = loop_guard.get_static_pad("src")

        if src_pad is None:
            print("[sender] Loop guard src pad was not found. Video-file loop will use watchdog/bus fallback.", flush=True)
            return

        self.loop_guard_probe_id = src_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            self.on_loop_guard_event,
        )

        print("[sender] Loop playback guard enabled.", flush=True)

    def on_loop_guard_event(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        event = info.get_event()

        if event is None:
            return Gst.PadProbeReturn.OK

        if event.type != Gst.EventType.EOS:
            return Gst.PadProbeReturn.OK

        if not self.loop_enabled:
            if self.debug_loop_messages:
                print("[sender] EOS reached loop guard. Loop disabled, allowing EOS downstream.", flush=True)
            return Gst.PadProbeReturn.OK

        if self.debug_loop_messages:
            print("[sender] EOS intercepted before WebRTC. Scheduling loop seek.", flush=True)

        self.schedule_video_loop_seek("loop-guard-eos")

        return Gst.PadProbeReturn.DROP

    def install_video_loop_watchdog(self) -> None:
        if self.pipeline is None:
            return

        if not is_video_file_mode(self.config):
            return

        if not self.loop_enabled:
            return

        if self.loop_watchdog_id is not None:
            return

        self.loop_watchdog_id = GLib.timeout_add(
            self.loop_watchdog_interval_ms,
            self.on_loop_watchdog_tick,
        )

        print("[sender] Loop playback watchdog enabled.", flush=True)

    def on_loop_watchdog_tick(self) -> bool:
        if not self.running.is_set():
            return False

        if self.pipeline is None:
            return False

        if not self.loop_enabled:
            return False

        if not is_video_file_mode(self.config):
            return False

        if self.loop_in_progress:
            return True

        try:
            ok_pos, position = self.pipeline.query_position(Gst.Format.TIME)
            ok_dur, duration = self.pipeline.query_duration(Gst.Format.TIME)
        except Exception as exc:
            if self.debug_loop_messages:
                print(f"[sender] Loop watchdog query failed: {exc}", flush=True)
            return True

        if not ok_pos or not ok_dur:
            return True

        if duration <= 0 or position < 0:
            return True

        self.last_watchdog_position_ns = int(position)
        self.last_watchdog_duration_ns = int(duration)

        remaining = int(duration) - int(position)

        if remaining <= self.loop_margin_ns:
            if self.debug_loop_messages:
                print(
                    "[sender] Loop watchdog reached end region. "
                    f"position={position / Gst.SECOND:.3f}s "
                    f"duration={duration / Gst.SECOND:.3f}s "
                    f"remaining={remaining / Gst.SECOND:.3f}s",
                    flush=True,
                )

            self.schedule_video_loop_seek("watchdog")

        return True

    def schedule_video_loop_seek(self, source: str) -> None:
        if self.loop_in_progress:
            return

        now_ns = GLib.get_monotonic_time() * 1000

        if now_ns - self.last_loop_request_monotonic_ns < self.loop_cooldown_ns:
            return

        self.last_loop_request_monotonic_ns = now_ns
        self.loop_in_progress = True

        if self.debug_loop_messages:
            print(f"[sender] Scheduling loop seek from {source}.", flush=True)

        GLib.idle_add(self.loop_video_from_start)

    def loop_video_from_start(self) -> bool:
        if self.pipeline is None:
            self.loop_in_progress = False
            return False

        try:
            self.loop_count += 1

            if self.debug_loop_messages:
                print(f"[sender] Seamless loop #{self.loop_count}: non-flush seek back to start.", flush=True)

            success = self.pipeline.seek(
                1.0,
                Gst.Format.TIME,
                Gst.SeekFlags.KEY_UNIT,
                Gst.SeekType.SET,
                0,
                Gst.SeekType.NONE,
                -1,
            )

            if not success:
                if self.debug_loop_messages:
                    print("[sender] Non-flush loop seek failed. Trying seek_simple without FLUSH.", flush=True)

                try:
                    success = self.pipeline.seek_simple(
                        Gst.Format.TIME,
                        Gst.SeekFlags.KEY_UNIT,
                        0,
                    )
                except Exception as seek_simple_exc:
                    if self.debug_loop_messages:
                        print(f"[sender] seek_simple raised exception: {seek_simple_exc}", flush=True)
                    success = False

            if not success:
                self.metrics_last_error = "Loop seek failed"
                print("[sender] Loop seek failed. Stream may stop at end of input.", flush=True)

            return False

        except Exception as exc:
            self.metrics_last_error = f"Loop seek exception: {exc}"
            print(f"[sender] Loop seek raised exception: {exc}", flush=True)
            return False

        finally:
            self.loop_in_progress = False

    def configure_gcc_adaptation(self) -> bool:
        if self.pipeline is None or self.webrtc is None:
            return False

        self.video_encoder = self.pipeline.get_by_name("video_encoder")

        if self.video_encoder is None:
            self.metrics_last_error = "video_encoder element missing"
            print("[gcc] Could not get encoder element named video_encoder.", flush=True)
            return not self.gcc_enabled

        if not self.gcc_enabled:
            self.metrics_adaptation_mode = "manual"
            self.metrics_adaptation_state = "manual"
            self.metrics_adaptation_reason = "Automatic adaptation disabled"
            print(
                f"[gcc] Manual bitrate mode: {self.config.bitrate_kbps} kbps.",
                flush=True,
            )
            return True

        if Gst.ElementFactory.find("rtpgccbwe") is None:
            self.metrics_last_error = "rtpgccbwe element missing"
            self.metrics_adaptation_state = "unavailable"
            self.metrics_adaptation_reason = "rtpgccbwe element is not installed"
            print("[gcc] rtpgccbwe is not available.", flush=True)
            return False

        try:
            self.webrtc.connect("request-aux-sender", self.on_request_aux_sender)
        except Exception as exc:
            self.metrics_last_error = f"GCC request-aux-sender connect error: {exc}"
            self.metrics_adaptation_state = "error"
            self.metrics_adaptation_reason = str(exc)
            print(f"[gcc] Could not connect request-aux-sender: {exc}", flush=True)
            return False

        try:
            self.video_encoder.set_property("bitrate", self.gcc_applied_bitrate_kbps)
        except Exception as exc:
            self.metrics_last_error = f"Initial encoder bitrate error: {exc}"
            print(f"[gcc] Could not set initial encoder bitrate: {exc}", flush=True)
            return False

        self.metrics_adaptation_mode = "automatic"
        self.metrics_adaptation_state = "waiting"
        self.metrics_adaptation_reason = "Waiting for GCC estimate"
        print(
            "[gcc] Automatic GCC/TWCC adaptation enabled: "
            f"min={self.gcc_min_bitrate_kbps} kbps "
            f"max={self.gcc_max_bitrate_kbps} kbps "
            f"headroom={self.gcc_headroom:.2f}.",
            flush=True,
        )
        return True

    def on_request_aux_sender(self, webrtc, dtls_transport):
        estimator = Gst.ElementFactory.make("rtpgccbwe", None)

        if estimator is None:
            self.metrics_last_error = "Could not create rtpgccbwe"
            self.metrics_adaptation_state = "error"
            self.metrics_adaptation_reason = "Could not create rtpgccbwe"
            print("[gcc] Could not create rtpgccbwe.", flush=True)
            return None

        try:
            estimator.set_property(
                "estimated-bitrate",
                int(self.gcc_applied_bitrate_kbps * 1000),
            )
            estimator.set_property(
                "min-bitrate",
                int(self.gcc_min_bitrate_kbps * 1000),
            )
            estimator.set_property(
                "max-bitrate",
                int(self.gcc_max_bitrate_kbps * 1000),
            )
            estimator.connect(
                "notify::estimated-bitrate",
                self.on_gcc_estimated_bitrate,
            )
            self.gcc_estimators.append(estimator)
            print(
                "[gcc] rtpgccbwe auxiliary sender created "
                f"(instance={len(self.gcc_estimators)}).",
                flush=True,
            )
            return estimator
        except Exception as exc:
            self.metrics_last_error = f"rtpgccbwe configuration error: {exc}"
            self.metrics_adaptation_state = "error"
            self.metrics_adaptation_reason = str(exc)
            print(f"[gcc] rtpgccbwe configuration failed: {exc}", flush=True)
            return None

    def on_gcc_estimated_bitrate(self, estimator, _pspec) -> None:
        try:
            estimate_bps = int(estimator.get_property("estimated-bitrate"))
        except Exception as exc:
            self.metrics_last_error = f"GCC estimate read error: {exc}"
            return

        if estimate_bps <= 0:
            return

        with self.gcc_lock:
            self.gcc_latest_estimate_bps = estimate_bps

    def apply_gcc_adaptation(self) -> None:
        if not self.gcc_enabled:
            self.metrics_adaptation_mode = "manual"
            self.metrics_adaptation_state = "manual"
            self.metrics_adaptation_reason = "Automatic adaptation disabled"
            self.metrics_estimated_bandwidth_mbps = "n/a"
            return

        with self.gcc_lock:
            estimate_bps = int(self.gcc_latest_estimate_bps)

        if estimate_bps <= 0:
            self.metrics_adaptation_state = "waiting"
            self.metrics_adaptation_reason = "Waiting for GCC estimate"
            return

        estimate_kbps = max(1, int(round(estimate_bps / 1000.0)))
        desired_kbps = int(round(estimate_kbps * self.gcc_headroom))
        desired_kbps = max(
            self.gcc_min_bitrate_kbps,
            min(self.gcc_max_bitrate_kbps, desired_kbps),
        )
        current_kbps = max(1, int(self.gcc_applied_bitrate_kbps))
        new_kbps = current_kbps
        state = "stable"
        reason = "Estimate is inside the hysteresis window"

        if desired_kbps < int(current_kbps * self.gcc_decrease_hysteresis):
            new_kbps = desired_kbps
            state = "decreasing"
            reason = "Fast decrease to preserve GCC headroom"
        elif desired_kbps > int(current_kbps * self.gcc_increase_hysteresis):
            recovery_limit = max(current_kbps + 100, int(round(current_kbps * self.gcc_recovery_step)))
            new_kbps = min(desired_kbps, recovery_limit)
            state = "recovering"
            reason = "Slow recovery toward available bandwidth"

        self.metrics_estimated_bandwidth_mbps = f"{estimate_bps / 1_000_000.0:.2f}"
        self.metrics_adaptation_mode = "automatic"
        self.metrics_adaptation_state = state
        self.metrics_adaptation_reason = reason

        if new_kbps == current_kbps:
            return

        if self.video_encoder is None:
            self.metrics_adaptation_state = "error"
            self.metrics_adaptation_reason = "video_encoder is unavailable"
            return

        try:
            self.video_encoder.set_property("bitrate", int(new_kbps))

            # AV1 uses an explicit two-frame VBV reservoir in the current
            # pipeline. Keep that latency target aligned with the new bitrate.
            if str(self.config.codec).strip().lower() == "av1":
                try:
                    vbv_kbits = max(
                        1,
                        int(round((new_kbps / max(1, int(self.config.fps))) * 2.0)),
                    )
                    self.video_encoder.set_property("vbv-buffer-size", vbv_kbits)
                except Exception:
                    pass

            self.gcc_applied_bitrate_kbps = int(new_kbps)
            print(
                "[gcc] bitrate update: "
                f"estimate={estimate_kbps} kbps "
                f"desired={desired_kbps} kbps "
                f"applied={new_kbps} kbps "
                f"state={state}",
                flush=True,
            )
        except Exception as exc:
            self.metrics_last_error = f"Encoder bitrate update error: {exc}"
            self.metrics_adaptation_state = "error"
            self.metrics_adaptation_reason = str(exc)
            print(f"[gcc] Encoder bitrate update failed: {exc}", flush=True)

    def install_metrics_probes(self) -> None:
        self.install_raw_frame_probe()
        self.install_rtp_metrics_probe()

    def install_raw_frame_probe(self) -> None:
        if self.pipeline is None:
            return

        loop_guard = self.pipeline.get_by_name("loop_guard")

        if loop_guard is None:
            print("[metrics] Raw frame probe unavailable: loop_guard not found.", flush=True)
            return

        src_pad = loop_guard.get_static_pad("src")

        if src_pad is None:
            print("[metrics] Raw frame probe unavailable: loop_guard src pad not found.", flush=True)
            return

        self.raw_frame_probe_id = src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self.on_raw_frame_buffer,
        )

        print("[metrics] Raw sender FPS probe enabled.", flush=True)

    def on_raw_frame_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()

        if buffer is not None:
            self.metrics_raw_frames += 1

        return Gst.PadProbeReturn.OK

    def find_element_by_factory_names(self, factory_names: set[str]) -> Optional[Gst.Element]:
        if self.pipeline is None:
            return None

        try:
            iterator = self.pipeline.iterate_elements()
        except Exception:
            return None

        while True:
            try:
                result, element = iterator.next()
            except Exception:
                return None

            if result == Gst.IteratorResult.OK:
                if element is None:
                    continue

                try:
                    factory = element.get_factory()
                    if factory is None:
                        continue

                    factory_name = factory.get_name()

                    if factory_name in factory_names:
                        return element
                except Exception:
                    continue

            elif result == Gst.IteratorResult.DONE:
                break

            elif result == Gst.IteratorResult.RESYNC:
                try:
                    iterator.resync()
                except Exception:
                    break

            else:
                break

        return None

    def install_rtp_metrics_probe(self) -> None:
        if self.pipeline is None:
            return

        counter = self.pipeline.get_by_name("rtp_counter")

        if counter is None:
            print("[metrics] RTP identity counter unavailable: rtp_counter not found.", flush=True)
            return

        try:
            counter.connect("handoff", self.on_rtp_identity_handoff)
            print("[metrics] RTP bitrate/packet counter enabled on rtp_counter.", flush=True)
        except Exception as exc:
            self.metrics_last_error = f"RTP counter connect error: {exc}"
            print(f"[metrics] Could not connect RTP identity counter: {exc}", flush=True)

    def on_rtp_identity_handoff(self, identity, buffer, pad=None) -> None:
        if buffer is None:
            return

        try:
            self.metrics_bytes_sent += int(buffer.get_size())
            self.metrics_packets_sent += 1
        except Exception as exc:
            self.metrics_last_error = f"RTP identity counter error: {exc}"
    def install_metrics_timer(self) -> None:
        if self.metrics_timer_id is not None:
            return

        self.metrics_last_time_ns = GLib.get_monotonic_time() * 1000

        self.metrics_timer_id = GLib.timeout_add(
            self.metrics_interval_ms,
            self.on_metrics_tick,
        )

        print("[metrics] Research dashboard metrics enabled.", flush=True)

    def on_metrics_tick(self) -> bool:
        if not self.running.is_set():
            return False

        if self.pipeline is None:
            return False

        now_ns = GLib.get_monotonic_time() * 1000
        elapsed_ns = max(1, now_ns - self.metrics_last_time_ns)
        elapsed_s = elapsed_ns / 1_000_000_000.0

        frame_delta = max(0, self.metrics_raw_frames - self.metrics_last_raw_frames)
        bytes_delta = max(0, self.metrics_bytes_sent - self.metrics_last_bytes_sent)

        self.metrics_actual_fps = frame_delta / elapsed_s
        self.metrics_actual_bitrate_mbps = (bytes_delta * 8.0) / elapsed_s / 1_000_000.0

        self.metrics_last_raw_frames = self.metrics_raw_frames
        self.metrics_last_bytes_sent = self.metrics_bytes_sent
        self.metrics_last_packets_sent = self.metrics_packets_sent
        self.metrics_last_time_ns = now_ns

        self.update_pipeline_state_string()
        self.update_webrtc_state_strings()
        self.update_selected_route_string()
        self.request_webrtc_stats()
        self.apply_gcc_adaptation()

        resolution = f"{self.config.width}x{self.config.height}"
        configured_bitrate_mbps = self.config.bitrate_kbps / 1000.0
        target_bitrate_mbps = (
            self.gcc_applied_bitrate_kbps / 1000.0
            if self.gcc_enabled
            else configured_bitrate_mbps
        )
        receiver_snapshot = self.get_receiver_metrics_snapshot()

        print(
            "METRICS|"
            f"signaling_state={self.safe_metric_value(self.metrics_signaling_state)}|"
            f"webrtc_state={self.safe_metric_value(self.metrics_webrtc_state)}|"
            f"ice_state={self.safe_metric_value(self.metrics_ice_state)}|"
            f"selected_route={self.safe_metric_value(self.metrics_selected_route)}|"
            f"rtt_ms={self.safe_metric_value(self.metrics_rtt_ms)}|"
            f"codec={self.safe_metric_value(self.config.codec)}|"
            f"resolution={self.safe_metric_value(resolution)}|"
            f"target_fps={self.config.fps}|"
            f"actual_fps={self.metrics_actual_fps:.1f}|"
            f"encoder={self.safe_metric_value(self.encoder_name())}|"
            f"target_bitrate_mbps={target_bitrate_mbps:.2f}|"
            f"configured_bitrate_mbps={configured_bitrate_mbps:.2f}|"
            f"actual_bitrate_mbps={self.metrics_actual_bitrate_mbps:.2f}|"
            f"estimated_bandwidth_mbps={self.safe_metric_value(self.metrics_estimated_bandwidth_mbps)}|"
            f"gcc_estimated_bitrate_mbps={self.safe_metric_value(self.metrics_estimated_bandwidth_mbps)}|"
            f"applied_encoder_bitrate_mbps={target_bitrate_mbps:.2f}|"
            f"adaptation_mode={self.safe_metric_value(self.metrics_adaptation_mode)}|"
            f"adaptation_state={self.safe_metric_value(self.metrics_adaptation_state)}|"
            f"adaptation_reason={self.safe_metric_value(self.metrics_adaptation_reason)}|"
            f"adaptation_min_bitrate_mbps={self.gcc_min_bitrate_kbps / 1000.0:.2f}|"
            f"adaptation_max_bitrate_mbps={self.gcc_max_bitrate_kbps / 1000.0:.2f}|"
            f"packets_sent={self.metrics_packets_sent}|"
            f"bytes_sent={self.metrics_bytes_sent}|"
            f"packet_loss_pct={self.safe_metric_value(self.metrics_packet_loss_pct)}|"
            f"jitter_ms={self.safe_metric_value(self.metrics_jitter_ms)}|"
            f"pipeline_state={self.safe_metric_value(self.metrics_pipeline_state)}|"
            f"last_error={self.safe_metric_value(self.metrics_last_error)}|"
            f"receiver_alive={self.safe_metric_value(receiver_snapshot.get('receiver_alive', '0'))}|"
            f"receiver_display_mode={self.safe_metric_value(receiver_snapshot.get('display_mode', receiver_snapshot.get('receiver_display_mode', 'n/a')))}|"
            f"receiver_unity_fps={self.safe_metric_value(receiver_snapshot.get('unity_fps', receiver_snapshot.get('receiver_unity_fps', 'n/a')))}|"
            f"receiver_copied_fps={self.safe_metric_value(receiver_snapshot.get('copied_fps', receiver_snapshot.get('receiver_copied_fps', 'n/a')))}|"
            f"receiver_frame_id={self.safe_metric_value(receiver_snapshot.get('last_frame_id', receiver_snapshot.get('receiver_frame_id', '0')))}|"
            f"receiver_texture_attached={self.safe_metric_value(receiver_snapshot.get('texture_attached', receiver_snapshot.get('receiver_texture_attached', '0')))}|"
            f"receiver_stall={self.safe_metric_value(receiver_snapshot.get('stall', receiver_snapshot.get('receiver_stall', 'n/a')))}|"
            f"receiver_received_fps={self.safe_metric_value(receiver_snapshot.get('received_fps', 'n/a'))}|"
            f"receiver_decoded_fps={self.safe_metric_value(receiver_snapshot.get('decoded_fps', 'n/a'))}|"
            f"receiver_displayed_fps={self.safe_metric_value(receiver_snapshot.get('displayed_fps', 'n/a'))}|"
            f"receiver_bitrate_mbps={self.safe_metric_value(receiver_snapshot.get('bitrate_mbps', 'n/a'))}|"
            f"receiver_packets_received={self.safe_metric_value(receiver_snapshot.get('packets_received', '0'))}|"
            f"receiver_bytes_received={self.safe_metric_value(receiver_snapshot.get('bytes_received', '0'))}|"
            f"receiver_packets_lost={self.safe_metric_value(receiver_snapshot.get('packets_lost', 'n/a'))}|"
            f"receiver_packet_loss_pct={self.safe_metric_value(receiver_snapshot.get('packet_loss_pct', 'n/a'))}|"
            f"receiver_jitter_ms={self.safe_metric_value(receiver_snapshot.get('jitter_ms', 'n/a'))}|"
            f"receiver_rtt_ms={self.safe_metric_value(receiver_snapshot.get('rtt_ms', 'n/a'))}|"
            f"receiver_decoder={self.safe_metric_value(receiver_snapshot.get('decoder', 'n/a'))}|"
            f"receiver_codec={self.safe_metric_value(receiver_snapshot.get('codec', 'n/a'))}|"
            f"receiver_resolution={self.safe_metric_value(receiver_snapshot.get('resolution', 'n/a'))}|"
            f"receiver_queue_backlog={self.safe_metric_value(receiver_snapshot.get('queue_backlog', '0'))}|"
            f"receiver_dropped_frames={self.safe_metric_value(receiver_snapshot.get('dropped_frames', '0'))}|"
            f"receiver_last_frame_age_ms={self.safe_metric_value(receiver_snapshot.get('last_frame_age_ms', 'n/a'))}|"
            f"receiver_metrics_age_ms={self.safe_metric_value(receiver_snapshot.get('metrics_age_ms', 'n/a'))}|"
            f"receiver_last_error={self.safe_metric_value(receiver_snapshot.get('last_error', 'None'))}",
            flush=True,
        )

        return True

    def update_pipeline_state_string(self) -> None:
        if self.pipeline is None:
            self.metrics_pipeline_state = "NULL"
            return

        try:
            _ret, state, _pending = self.pipeline.get_state(0)
            self.metrics_pipeline_state = str(state.value_nick).upper()
        except Exception:
            self.metrics_pipeline_state = "unknown"

    def update_webrtc_state_strings(self) -> None:
        if self.webrtc is None:
            return

        for property_name, attr_name in (
            ("connection-state", "metrics_webrtc_state"),
            ("ice-connection-state", "metrics_ice_state"),
        ):
            try:
                value = self.webrtc.get_property(property_name)
                setattr(self, attr_name, self.enum_to_string(value))
            except Exception:
                pass

    def update_selected_route_string(self) -> None:
        self.metrics_selected_route = f"sender -> {self.config.host}:{self.config.port}"

    def encoder_name(self) -> str:
        if self.video_encoder is not None:
            try:
                factory = self.video_encoder.get_factory()
                if factory is not None:
                    return str(factory.get_name())
            except Exception:
                pass

        codec = str(self.config.codec).lower()
        if codec in ("h264", "avc"):
            return "h264 encoder"
        if codec in ("h265", "hevc"):
            return "h265 encoder"
        if codec == "av1":
            return "av1 encoder"
        return "unknown"

    def on_ice_candidate(self, webrtc: Gst.Element, mline_index: int, candidate: str) -> None:
        if not candidate:
            return

        line = f"ICE|{mline_index}|{base64_encode(candidate)}"
        self.signaling.send_line(line)
        print("[sender] ICE candidate sent to Unity receiver.", flush=True)

    def on_offer_created(self, promise: Gst.Promise, *args) -> None:
        reply = promise.get_reply()

        if reply is None:
            print("[sender] Failed to create SDP offer: empty promise reply.", flush=True)
            return

        try:
            offer = reply.get_value("offer")
        except Exception:
            offer = None

        if offer is None:
            print("[sender] Failed to create SDP offer.", flush=True)
            return

        print("[sender] Created SDP offer.", flush=True)

        if self.webrtc is None:
            print("[sender] webrtc element is missing.", flush=True)
            return

        try:
            sdp_text = offer.sdp.as_text()
        except Exception:
            sdp_text = None

        # AV1: keep NACK, NACK PLI and RTX, but suppress CCM FIR.
        #
        # The Quest receiver was generating an RTCP FIR storm, causing
        # repeated forced AV1 keyframes and visible playback flickering.
        # The SDP must be modified before set-local-description so that
        # the sender's local WebRTC state matches the signaling offer.
        if sdp_text and "AV1/90000" in sdp_text.upper():
            original_lines = sdp_text.replace("\r\n", "\n").splitlines()
            filtered_lines = []
            removed_fir_lines = []

            for line in original_lines:
                normalized = line.strip().lower()

                if (
                    normalized.startswith("a=rtcp-fb:")
                    and normalized.endswith("ccm fir")
                ):
                    removed_fir_lines.append(line)
                    continue

                filtered_lines.append(line)

            if removed_fir_lines:
                filtered_text = "\r\n".join(filtered_lines) + "\r\n"

                result, filtered_sdp = GstSdp.SDPMessage.new()

                if result != GstSdp.SDPResult.OK:
                    print(
                        "[sender] Failed to allocate filtered AV1 SDP offer.",
                        flush=True,
                    )
                    return

                parse_result = GstSdp.sdp_message_parse_buffer(
                    filtered_text.encode("utf-8"),
                    filtered_sdp,
                )

                if parse_result != GstSdp.SDPResult.OK:
                    print(
                        "[sender] Failed to parse filtered AV1 SDP offer.",
                        flush=True,
                    )
                    return

                filtered_offer = GstWebRTC.WebRTCSessionDescription.new(
                    GstWebRTC.WebRTCSDPType.OFFER,
                    filtered_sdp,
                )

                if filtered_offer is None:
                    print(
                        "[sender] Failed to create filtered AV1 SDP offer.",
                        flush=True,
                    )
                    return

                offer = filtered_offer
                sdp_text = filtered_text

                print(
                    "[sender] AV1 SDP munging: removed CCM FIR; "
                    "preserved NACK, NACK PLI and RTX. "
                    f"removed={len(removed_fir_lines)}",
                    flush=True,
                )

        self.webrtc.emit("set-local-description", offer, None)

        if sdp_text:
            print("[sender] SDP OFFER RECOVERY:", flush=True)

            recovery_lines = [
                line
                for line in sdp_text.splitlines()
                if (
                    "rtcp-fb:" in line
                    or "rtx/90000" in line
                    or "apt=" in line
                )
            ]

            if recovery_lines:
                for line in recovery_lines:
                    print(f"[sender]   {line}", flush=True)
            else:
                print("[sender]   NONE", flush=True)

            encoded_offer = base64_encode(sdp_text)
            self.signaling.send_line("OFFER|" + encoded_offer)
            print("[sender] SDP offer sent to Unity receiver.", flush=True)
        else:
            print("[sender] Could not convert SDP offer to text.", flush=True)

    def on_negotiation_needed(self, webrtc: Gst.Element) -> None:
        if self.initial_offer_requested:
            print(
                "[sender] Negotiation request ignored because the "
                "initial SDP offer was already requested.",
                flush=True,
            )
            return

        if self.webrtc is None:
            print("[sender] webrtc element is missing.", flush=True)
            return

        self.initial_offer_requested = True

        print(
            "[sender] Negotiation needed. Creating SDP offer...",
            flush=True,
        )

        promise = Gst.Promise.new_with_change_func(
            self.on_offer_created,
            None,
            None,
        )
        self.webrtc.emit("create-offer", None, promise)

    def request_initial_external_h264_offer(self) -> bool:
        if not self.running.is_set():
            return False

        if self.webrtc is None:
            print(
                "[sender] Cannot request initial offer: "
                "webrtc element is missing.",
                flush=True,
            )
            return False

        try:
            transceiver = self.webrtc.emit(
                "get-transceiver",
                0,
            )
        except Exception as exc:
            print(
                "[sender] Could not inspect WebRTC transceiver 0: "
                f"{exc}",
                flush=True,
            )
            return False

        if transceiver is None:
            self.metrics_last_error = (
                "External H.264 WebRTC transceiver 0 missing"
            )
            print(
                "[sender] External H.264 transceiver 0 was not found.",
                flush=True,
            )
            return False

        print(
            "[sender] External H.264 transceiver 0 is ready. "
            "Requesting initial SDP offer explicitly.",
            flush=True,
        )

        self.on_negotiation_needed(self.webrtc)
        return False

    def handle_answer_message(self, encoded_answer: str) -> bool:
        sdp_text = base64_decode_to_string(encoded_answer)

        if sdp_text is None:
            print("[signaling] Could not decode SDP answer.", flush=True)
            return False

        print("[sender] SDP ANSWER RECOVERY:", flush=True)

        recovery_lines = [
            line
            for line in sdp_text.splitlines()
            if (
                "rtcp-fb:" in line
                or "rtx/90000" in line
                or "apt=" in line
            )
        ]

        if recovery_lines:
            for line in recovery_lines:
                print(f"[sender]   {line}", flush=True)
        else:
            print("[sender]   NONE", flush=True)

        result, sdp = GstSdp.SDPMessage.new()

        if result != GstSdp.SDPResult.OK:
            print("[signaling] Could not allocate SDP answer message.", flush=True)
            return False

        parse_result = GstSdp.sdp_message_parse_buffer(sdp_text.encode("utf-8"), sdp)

        if parse_result != GstSdp.SDPResult.OK:
            print("[signaling] Could not parse SDP answer.", flush=True)
            return False

        answer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.ANSWER,
            sdp,
        )

        if answer is None:
            print("[signaling] Could not create WebRTC answer description.", flush=True)
            return False

        if self.webrtc is None:
            print("[signaling] webrtc element is missing.", flush=True)
            return False

        self.webrtc.emit("set-remote-description", answer, None)

        print("[signaling] SDP answer received and applied.", flush=True)
        return False

    def handle_ice_message(self, mline_text: str, encoded_candidate: str) -> bool:
        candidate = base64_decode_to_string(encoded_candidate)

        if candidate is None:
            print("[signaling] Could not decode ICE candidate.", flush=True)
            return False

        try:
            mline_index = int(mline_text)
        except ValueError:
            print("[signaling] Invalid ICE mline index.", flush=True)
            return False

        if self.webrtc is None:
            print("[signaling] webrtc element is missing.", flush=True)
            return False

        self.webrtc.emit("add-ice-candidate", mline_index, candidate)
        print("[signaling] ICE candidate received from Unity receiver.", flush=True)
        return False

    def handle_signaling_line(self, line: str) -> None:
        if line.startswith("ANSWER|"):
            encoded_answer = line[7:]
            GLib.idle_add(self.handle_answer_message, encoded_answer)
            return

        if line.startswith("ICE|"):
            parts = line.split("|", 2)

            if len(parts) != 3:
                print("[signaling] Malformed ICE message.", flush=True)
                return

            _, mline, candidate = parts
            GLib.idle_add(self.handle_ice_message, mline, candidate)
            return

        print(f"[signaling] Unknown message from Unity: {line}", flush=True)

    def signaling_read_thread(self) -> None:
        while self.running.is_set():
            line = self.signaling.recv_line()

            if line is None:
                print("[signaling] Signaling connection closed.", flush=True)
                self.metrics_signaling_state = "closed"

                if self.main_loop is not None:
                    GLib.idle_add(self.main_loop.quit)
                break

            if line:
                self.handle_signaling_line(line)


    def _ensure_receiver_feedback_fields(self) -> None:
        if not hasattr(self, "receiver_feedback_port"):
            self.receiver_feedback_port = int(getattr(self.config, "receiver_feedback_port", 0) or (int(self.config.port) + 100))

        if not hasattr(self, "receiver_feedback_server_socket"):
            self.receiver_feedback_server_socket = None

        if not hasattr(self, "receiver_feedback_client_socket"):
            self.receiver_feedback_client_socket = None

        if not hasattr(self, "receiver_feedback_thread"):
            self.receiver_feedback_thread = None

        if not hasattr(self, "receiver_feedback_running"):
            self.receiver_feedback_running = threading.Event()

        defaults = {
            "receiver_alive": "no",
            "receiver_display_mode": "n/a",
            "receiver_unity_fps": "n/a",
            "receiver_copied_fps": "n/a",
            "receiver_frame_id": "0",
            "receiver_texture_attached": "no",
            "receiver_stall": "n/a",
        }

        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def start_receiver_feedback_server(self) -> None:
        self._ensure_receiver_feedback_fields()

        if self.receiver_feedback_thread is not None and self.receiver_feedback_thread.is_alive():
            return

        self.receiver_feedback_running.set()

        self.receiver_feedback_thread = threading.Thread(
            target=self.receiver_feedback_server_loop,
            name="ReceiverFeedbackServerThread",
            daemon=True,
        )
        self.receiver_feedback_thread.start()

        print(f"[receiver-feedback] Listening on 0.0.0.0:{self.receiver_feedback_port}", flush=True)

    def stop_receiver_feedback_server(self) -> None:
        self._ensure_receiver_feedback_fields()

        try:
            self.receiver_feedback_running.clear()
        except Exception:
            pass

        for sock_name in ("receiver_feedback_client_socket", "receiver_feedback_server_socket"):
            sock = getattr(self, sock_name, None)

            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass

                try:
                    sock.close()
                except Exception:
                    pass

                setattr(self, sock_name, None)

    def receiver_feedback_server_loop(self) -> None:
        self._ensure_receiver_feedback_fields()

        server_socket = None

        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.settimeout(1.0)
            server_socket.bind(("0.0.0.0", int(self.receiver_feedback_port)))
            server_socket.listen(1)

            self.receiver_feedback_server_socket = server_socket

            while self.receiver_feedback_running.is_set():
                try:
                    client_socket, address = server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                self.receiver_feedback_client_socket = client_socket
                self.receiver_alive = "yes"

                print(f"[receiver-feedback] Unity receiver feedback connected from {address[0]}:{address[1]}", flush=True)

                try:
                    self.receiver_feedback_client_loop(client_socket)
                finally:
                    self.receiver_alive = "no"

                    try:
                        client_socket.close()
                    except Exception:
                        pass

                    self.receiver_feedback_client_socket = None

        except Exception as exc:
            self.metrics_last_error = f"Receiver feedback server error: {exc}"
            print(f"[receiver-feedback] Server error: {exc}", flush=True)

        finally:
            if server_socket is not None:
                try:
                    server_socket.close()
                except Exception:
                    pass

            self.receiver_feedback_server_socket = None

    def receiver_feedback_client_loop(self, client_socket) -> None:
        client_socket.settimeout(1.0)
        pending = ""

        while self.receiver_feedback_running.is_set():
            try:
                data = client_socket.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                break

            try:
                pending += data.decode("utf-8", errors="replace")
            except Exception:
                continue

            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                self.handle_receiver_feedback_line(line.strip())

    def handle_receiver_feedback_line(self, line: str) -> None:
        if not line:
            return

        if not line.startswith("RECEIVER_METRICS|"):
            return

        fields: dict[str, str] = {}

        for part in line.split("|")[1:]:
            if "=" not in part:
                continue

            key, value = part.split("=", 1)
            key = key.strip()
            if key:
                fields[key] = value.strip()

        def get_any(*names: str, default: str = "n/a") -> str:
            for name in names:
                if name in fields:
                    return fields[name]
            return default

        alive = get_any("receiver_alive", "alive", default="1")
        texture = get_any("texture_attached", "texture", default="0")
        stall = get_any("stall", "receiver_stall", default="0")

        normalized = dict(fields)
        normalized.update(
            {
                "receiver_alive": "yes" if alive in ("1", "true", "True", "yes", "Yes") else "no",
                "receiver_display_mode": get_any("display_mode", "display", "receiver_display", default="n/a"),
                "receiver_unity_fps": get_any("unity_fps", "displayed_fps", "receiver_fps", default="n/a"),
                "receiver_copied_fps": get_any("copied_fps", "decoded_fps", "receiver_copied_fps", default="n/a"),
                "receiver_frame_id": get_any("last_frame_id", "frame_id", "receiver_frame_id", default="0"),
                "receiver_texture_attached": "yes" if texture in ("1", "true", "True", "yes", "Yes") else "no",
                "receiver_stall": "yes" if stall in ("1", "true", "True", "yes", "Yes") else "no",
            }
        )

        now_ns = GLib.get_monotonic_time() * 1000
        with self.receiver_metrics_lock:
            self.receiver_metrics.update(normalized)
            self.receiver_metrics_last_seen_ns = now_ns

        # Preserve the original public attributes used by older launchers.
        self.receiver_alive = normalized["receiver_alive"]
        self.receiver_display_mode = normalized["receiver_display_mode"]
        self.receiver_unity_fps = normalized["receiver_unity_fps"]
        self.receiver_copied_fps = normalized["receiver_copied_fps"]
        self.receiver_frame_id = normalized["receiver_frame_id"]
        self.receiver_texture_attached = normalized["receiver_texture_attached"]
        self.receiver_stall = normalized["receiver_stall"]

    def get_receiver_metrics_snapshot(self) -> dict[str, str]:
        now_ns = GLib.get_monotonic_time() * 1000

        with self.receiver_metrics_lock:
            snapshot = dict(self.receiver_metrics)
            last_seen_ns = self.receiver_metrics_last_seen_ns

        if last_seen_ns > 0:
            age_ms = max(0.0, (now_ns - last_seen_ns) / 1_000_000.0)
            snapshot["metrics_age_ms"] = f"{age_ms:.0f}"
            if age_ms > 2500.0:
                snapshot["receiver_alive"] = "no"
                snapshot["receiver_stall"] = "yes"
        else:
            snapshot["metrics_age_ms"] = "n/a"

        # Compatibility aliases expected by the existing METRICS line.
        snapshot.setdefault("display_mode", snapshot.get("receiver_display_mode", "n/a"))
        snapshot.setdefault("unity_fps", snapshot.get("receiver_unity_fps", "n/a"))
        snapshot.setdefault("copied_fps", snapshot.get("receiver_copied_fps", "n/a"))
        snapshot.setdefault("last_frame_id", snapshot.get("receiver_frame_id", "0"))
        snapshot.setdefault("texture_attached", snapshot.get("receiver_texture_attached", "no"))
        snapshot.setdefault("stall", snapshot.get("receiver_stall", "n/a"))

        return snapshot


    def request_webrtc_stats(self) -> None:
        if self.webrtc is None:
            return

        try:
            promise = Gst.Promise.new_with_change_func(
                self.on_webrtc_stats_ready,
                None,
                None,
            )

            self.webrtc.emit("get-stats", None, promise)
        except Exception as exc:
            self.metrics_last_error = f"WebRTC stats request error: {exc}"

    def on_webrtc_stats_ready(self, promise: Gst.Promise, *args) -> None:
        try:
            reply = promise.get_reply()
        except Exception as exc:
            self.metrics_last_error = f"WebRTC stats reply error: {exc}"
            return

        if reply is None:
            return

        try:
            flat = {}
            self.flatten_gst_stats_value(reply, "", flat)
            self.apply_webrtc_stats(flat)
        except Exception as exc:
            self.metrics_last_error = f"WebRTC stats parse error: {exc}"

    def flatten_gst_stats_value(self, value, prefix: str, output: dict) -> None:
        if value is None:
            return

        if isinstance(value, Gst.Structure):
            try:
                count = value.n_fields()
            except Exception:
                count = 0

            for i in range(count):
                try:
                    name = value.nth_field_name(i)
                    child = value.get_value(name)
                except Exception:
                    continue

                key = f"{prefix}.{name}" if prefix else str(name)
                self.flatten_gst_stats_value(child, key, output)

            return

        try:
            # Gst.ValueArray / Gst.ValueList-like values.
            length = len(value)
            if length >= 0 and not isinstance(value, (str, bytes, bytearray)):
                for i in range(length):
                    try:
                        child = value[i]
                    except Exception:
                        continue

                    key = f"{prefix}[{i}]"
                    self.flatten_gst_stats_value(child, key, output)

                return
        except Exception:
            pass

        output[prefix] = value

    def normalize_stats_key(self, key: str) -> str:
        result = []

        for ch in str(key):
            if ch.isalnum():
                result.append(ch.lower())

        return "".join(result)

    def stats_to_float(self, value) -> Optional[float]:
        if value is None:
            return None

        try:
            return float(value)
        except Exception:
            pass

        try:
            text = str(value).strip()
            if not text or text.lower() in ("nan", "none", "n/a"):
                return None
            return float(text)
        except Exception:
            return None

    def stats_to_int(self, value) -> Optional[int]:
        number = self.stats_to_float(value)

        if number is None:
            return None

        try:
            return int(number)
        except Exception:
            return None

    def find_stats_number(self, flat: dict, names: list[str]) -> Optional[float]:
        normalized_targets = [self.normalize_stats_key(name) for name in names]

        for key, value in flat.items():
            normalized_key = self.normalize_stats_key(key)

            for target in normalized_targets:
                if normalized_key.endswith(target) or target in normalized_key:
                    number = self.stats_to_float(value)
                    if number is not None:
                        return number

        return None

    def apply_webrtc_stats(self, flat: dict) -> None:
        # WebRTC outbound counters. These are used as a secondary source/fallback.
        bytes_sent = self.find_stats_number(
            flat,
            [
                "bytesSent",
                "bytes-sent",
                "outbound-rtp.bytesSent",
                "outboundrtp.bytesSent",
            ],
        )

        packets_sent = self.find_stats_number(
            flat,
            [
                "packetsSent",
                "packets-sent",
                "outbound-rtp.packetsSent",
                "outboundrtp.packetsSent",
            ],
        )

        if bytes_sent is not None and bytes_sent > self.metrics_bytes_sent:
            self.metrics_bytes_sent = int(bytes_sent)

        if packets_sent is not None and packets_sent > self.metrics_packets_sent:
            self.metrics_packets_sent = int(packets_sent)

        # GCC / candidate-pair bandwidth estimate, if exposed.
        estimated_bw = self.find_stats_number(
            flat,
            [
                "availableOutgoingBitrate",
                "available-outgoing-bitrate",
                "googAvailableSendBandwidth",
                "googAvailableSendBandwidthBitrate",
                "targetBitrate",
                "target-bitrate",
            ],
        )

        if estimated_bw is not None and estimated_bw > 0:
            # In automatic mode rtpgccbwe is the source of truth. WebRTC
            # candidate-pair stats remain a fallback for manual mode.
            with self.gcc_lock:
                have_gcc_estimate = self.gcc_latest_estimate_bps > 0
            if not self.gcc_enabled or not have_gcc_estimate:
                if estimated_bw > 100000.0:
                    self.metrics_estimated_bandwidth_mbps = f"{estimated_bw / 1000000.0:.2f}"
                else:
                    self.metrics_estimated_bandwidth_mbps = f"{estimated_bw:.2f}"

        # RTT can appear in seconds or milliseconds depending on implementation.
        rtt = self.find_stats_number(
            flat,
            [
                "currentRoundTripTime",
                "current-round-trip-time",
                "roundTripTime",
                "round-trip-time",
                "googRtt",
                "rtt",
            ],
        )

        if rtt is not None and rtt >= 0:
            if rtt <= 10.0:
                self.metrics_rtt_ms = f"{rtt * 1000.0:.1f}"
            else:
                self.metrics_rtt_ms = f"{rtt:.1f}"

        # Packet loss may be exposed as fractionLost or packetsLost.
        fraction_lost = self.find_stats_number(
            flat,
            [
                "fractionLost",
                "fraction-lost",
                "packetsLostFraction",
            ],
        )

        if fraction_lost is not None:
            if fraction_lost <= 1.0:
                self.metrics_packet_loss_pct = f"{fraction_lost * 100.0:.2f}"
            else:
                self.metrics_packet_loss_pct = f"{fraction_lost:.2f}"
        else:
            packets_lost = self.find_stats_number(
                flat,
                [
                    "packetsLost",
                    "packets-lost",
                ],
            )

            packets_received = self.find_stats_number(
                flat,
                [
                    "packetsReceived",
                    "packets-received",
                ],
            )

            if packets_lost is not None and packets_received is not None:
                denominator = packets_lost + packets_received
                if denominator > 0:
                    self.metrics_packet_loss_pct = f"{(packets_lost / denominator) * 100.0:.2f}"
            elif packets_lost is not None:
                self.metrics_packet_loss_pct = f"{packets_lost:.0f} pkts"

        # Jitter is normally seconds in WebRTC stats.
        jitter = self.find_stats_number(
            flat,
            [
                "jitter",
                "jitterBufferDelay",
                "jitter-buffer-delay",
                "googJitterReceived",
            ],
        )

        if jitter is not None and jitter >= 0:
            if jitter <= 10.0:
                self.metrics_jitter_ms = f"{jitter * 1000.0:.1f}"
            else:
                self.metrics_jitter_ms = f"{jitter:.1f}"

    def on_external_h264_access_unit(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
            event = info.get_event()

            if (
                event is not None
                and event.type == Gst.EventType.EOS
            ):
                print(
                    "[external-h264] Parser EOS after "
                    f"{self.external_h264_au_number} access units.",
                    flush=True,
                )

            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()

        if buffer is None:
            return Gst.PadProbeReturn.OK

        duration = self.external_h264_frame_duration_ns

        if duration <= 0:
            duration = int(
                Gst.SECOND // max(1, int(self.config.fps))
            )
            self.external_h264_frame_duration_ns = duration

        if self.external_h264_base_pts_ns is None:
            running_time = 0

            if self.pipeline is not None:
                clock = self.pipeline.get_clock()

                if clock is not None:
                    clock_time = int(clock.get_time())
                    base_time = int(self.pipeline.get_base_time())

                    if clock_time >= base_time:
                        running_time = clock_time - base_time

            # Give the first AU one frame of scheduling headroom.
            self.external_h264_base_pts_ns = (
                running_time + duration
            )

            print(
                "[external-h264] AU timestamp base: "
                f"{self.external_h264_base_pts_ns} ns",
                flush=True,
            )

        frame_number = self.external_h264_au_number
        pts = (
            self.external_h264_base_pts_ns
            + frame_number * duration
        )

        buffer.pts = pts
        buffer.dts = pts
        buffer.duration = duration
        buffer.offset = frame_number
        buffer.offset_end = frame_number + 1

        self.external_h264_au_number += 1

        if (
            self.external_h264_au_number == 1
            or self.external_h264_au_number % 120 == 0
        ):
            print(
                "[external-h264] Timestamped access units: "
                f"{self.external_h264_au_number}",
                flush=True,
            )

        return Gst.PadProbeReturn.OK

    def setup_external_h264_source(self) -> bool:
        if self.pipeline is None:
            print("[external-h264] Pipeline is missing.", flush=True)
            return False

        fifo_path = external_h264_fifo_path()

        if not fifo_path:
            return True

        source = self.pipeline.get_by_name("external_h264_source")

        if source is None:
            print(
                "[external-h264] Could not get appsrc element "
                "named external_h264_source.",
                flush=True,
            )
            return False

        parser = self.pipeline.get_by_name(
            "external_h264_parse"
        )

        if parser is None:
            print(
                "[external-h264] Could not get parser element "
                "named external_h264_parse.",
                flush=True,
            )
            return False

        parser_src = parser.get_static_pad("src")

        if parser_src is None:
            print(
                "[external-h264] Parser src pad is missing.",
                flush=True,
            )
            return False

        self.external_h264_source = source
        self.external_h264_parse = parser
        self.external_h264_fifo = fifo_path
        self.external_h264_au_number = 0
        self.external_h264_base_pts_ns = None
        self.external_h264_frame_duration_ns = int(
            Gst.SECOND // max(1, int(self.config.fps))
        )

        self.external_h264_au_probe_id = parser_src.add_probe(
            (
                Gst.PadProbeType.BUFFER
                | Gst.PadProbeType.EVENT_DOWNSTREAM
            ),
            self.on_external_h264_access_unit,
        )

        self.external_h264_reader_stop.clear()

        self.external_h264_reader_thread = threading.Thread(
            target=self.external_h264_fifo_reader,
            name="ExternalH264FifoReader",
            daemon=True,
        )
        self.external_h264_reader_thread.start()

        print(
            "[external-h264] Live appsrc FIFO reader started: "
            f"{fifo_path}",
            flush=True,
        )

        return True

    def external_h264_fifo_reader(self) -> None:
        source = self.external_h264_source
        fifo_path = self.external_h264_fifo
        fd: Optional[int] = None
        eof_seen = False

        if source is None or not fifo_path:
            print(
                "[external-h264] FIFO reader started without a source or path.",
                flush=True,
            )
            return

        try:
            fd = os.open(
                fifo_path,
                os.O_RDONLY | os.O_NONBLOCK,
            )
            self.external_h264_reader_fd = fd

            print(
                "[external-h264] H.264 FIFO opened for reading.",
                flush=True,
            )

            while (
                self.running.is_set()
                and not self.external_h264_reader_stop.is_set()
            ):
                try:
                    readable, _, _ = select.select(
                        [fd],
                        [],
                        [],
                        0.25,
                    )
                except (OSError, ValueError) as exc:
                    if not self.external_h264_reader_stop.is_set():
                        print(
                            "[external-h264] FIFO select failed: "
                            f"{exc}",
                            flush=True,
                        )
                    break

                if not readable:
                    continue

                try:
                    chunk = os.read(fd, 262144)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    if not self.external_h264_reader_stop.is_set():
                        print(
                            "[external-h264] FIFO read failed: "
                            f"{exc}",
                            flush=True,
                        )
                    break

                if not chunk:
                    eof_seen = True
                    print(
                        "[external-h264] H.264 FIFO reached EOF.",
                        flush=True,
                    )
                    break

                buffer = Gst.Buffer.new_allocate(
                    None,
                    len(chunk),
                    None,
                )
                buffer.fill(0, chunk)

                flow = source.emit("push-buffer", buffer)

                if flow == Gst.FlowReturn.OK:
                    continue

                if flow == Gst.FlowReturn.FLUSHING:
                    break

                print(
                    "[external-h264] appsrc push-buffer returned: "
                    f"{flow}",
                    flush=True,
                )
                break

            if (
                eof_seen
                and self.running.is_set()
                and not self.external_h264_reader_stop.is_set()
            ):
                result = source.emit("end-of-stream")
                print(
                    "[external-h264] appsrc end-of-stream returned: "
                    f"{result}",
                    flush=True,
                )

        except OSError as exc:
            self.metrics_last_error = (
                "External H.264 FIFO error: "
                f"{exc}"
            )
            print(
                "[external-h264] Could not open/read FIFO: "
                f"{exc}",
                flush=True,
            )

            if self.main_loop is not None:
                GLib.idle_add(self.main_loop.quit)

        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

            self.external_h264_reader_fd = None

            print(
                "[external-h264] FIFO reader stopped.",
                flush=True,
            )

    def setup_image_folder_source(self) -> bool:
        if self.pipeline is None:
            print("[image-folder] Pipeline is missing.", flush=True)
            return False

        self.frame_source = self.pipeline.get_by_name("frame_source")

        if self.frame_source is None:
            print("[image-folder] Could not get appsrc element named frame_source.", flush=True)
            return False

        self.image_files = list_supported_images(
            self.config.input_path,
            self.config.image_format,
        )

        if not self.image_files:
            print("[image-folder] No supported image files found.", flush=True)
            print(f"[image-folder] Folder: {self.config.input_path}", flush=True)
            print("[image-folder] Supported: .jpg, .jpeg, .png, .webp, .avif", flush=True)
            return False

        try:
            try:
                import pillow_avif  # type: ignore  # noqa: F401
            except ImportError:
                pass

            from PIL import Image
            self.image_loader = Image
        except ImportError:
            print("[image-folder] Pillow is not installed in the selected Python environment.", flush=True)
            print("[image-folder] Install it with:", flush=True)
            print("               conda install -n gstwebrtc -c conda-forge pillow pillow-avif-plugin -y", flush=True)
            return False

        self.image_index = 0
        self.frame_number = 0
        self.frame_duration_ns = int(Gst.SECOND // max(1, self.config.fps))
        self.loop_image_folder = bool(getattr(self.config, "loop", True))

        caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGB,"
            f"width={self.config.width},"
            f"height={self.config.height},"
            f"framerate={self.config.fps}/1"
        )

        self.frame_source.set_property("caps", caps)
        self.frame_source.set_property("is-live", True)
        self.frame_source.set_property("format", Gst.Format.TIME)
        self.frame_source.set_property("do-timestamp", False)
        self.frame_source.set_property("block", True)

        self.frame_source.connect("need-data", self.on_appsrc_need_data)

        print(
            f"[image-folder] Prepared {len(self.image_files)} image frame(s) "
            f"from {self.config.input_path}",
            flush=True,
        )
        print(
            f"[image-folder] Streaming as RGB "
            f"{self.config.width}x{self.config.height} @ {self.config.fps} FPS",
            flush=True,
        )
        print(f"[image-folder] Looping is {'enabled' if self.loop_image_folder else 'disabled'}.", flush=True)

        return True

    def on_appsrc_need_data(self, appsrc: Gst.Element, length: int) -> None:
        if not self.running.is_set():
            return

        if not self.image_files:
            print("[image-folder] No image files available for appsrc.", flush=True)
            appsrc.emit("end-of-stream")
            return

        if self.image_index >= len(self.image_files):
            if self.loop_image_folder:
                self.image_index = 0
            else:
                print("[image-folder] End of image sequence.", flush=True)
                appsrc.emit("end-of-stream")
                return

        image_path = self.image_files[self.image_index]
        self.image_index += 1

        try:
            frame_bytes = self.load_image_as_rgb_bytes(image_path)
        except Exception as exc:
            print(f"[image-folder] Failed to load image: {image_path}", flush=True)
            print(f"[image-folder] Error: {exc}", flush=True)

            if self.main_loop is not None:
                self.main_loop.quit()
            return

        buffer = Gst.Buffer.new_allocate(None, len(frame_bytes), None)
        buffer.fill(0, frame_bytes)

        pts = self.frame_number * self.frame_duration_ns
        buffer.pts = pts
        buffer.dts = Gst.CLOCK_TIME_NONE
        buffer.duration = self.frame_duration_ns
        buffer.offset = self.frame_number
        buffer.offset_end = self.frame_number + 1

        self.frame_number += 1

        ret = appsrc.emit("push-buffer", buffer)

        if ret != Gst.FlowReturn.OK:
            print(f"[image-folder] appsrc push-buffer returned: {ret}", flush=True)
            if self.main_loop is not None:
                self.main_loop.quit()

    def load_image_as_rgb_bytes(self, image_path: Path) -> bytes:
        if self.image_loader is None:
            raise RuntimeError("Pillow image loader is not initialized.")

        Image = self.image_loader

        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            target_size = (self.config.width, self.config.height)

            if img.size != target_size:
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS

                img = img.resize(target_size, resample=resample)

            return img.tobytes()

    def start_sender_pipeline(self) -> bool:
        desc = build_sender_pipeline_description(self.config)

        print(
            f"[sender] Starting GStreamer WebRTC "
            f"{codec_label(self.config.codec)} sender pipeline:\n{desc}",
            flush=True,
        )

        try:
            self.pipeline = Gst.parse_launch(desc)
        except GLib.Error as exc:
            self.metrics_last_error = f"Pipeline parse error: {exc.message}"
            print(f"[sender] Pipeline parse error: {exc.message}", flush=True)
            return False

        self.webrtc = self.pipeline.get_by_name("sender_webrtc") if self.pipeline else None

        if not self.webrtc:
            self.metrics_last_error = "sender_webrtc element missing"
            print("[sender] Could not get sender_webrtc element.", flush=True)
            return False

        external_fifo = external_h264_fifo_path()

        if external_fifo:
            if not self.setup_external_h264_source():
                return False
        elif is_image_folder_mode(self.config):
            if not self.setup_image_folder_source():
                return False

        self.install_video_loop_guard()
        self.install_metrics_probes()

        if not self.configure_gcc_adaptation():
            return False

        # QGXS AV1 NACK recovery
        if str(self.config.codec).lower() == "av1":
            transceiver = self.webrtc.emit("get-transceiver", 0)

            if transceiver is None:
                raise RuntimeError(
                    "AV1 WebRTC transceiver 0 was not available"
                )

            transceiver.set_property("do-nack", True)

            print(
                "[sender] AV1 loss recovery: "
                f"do-nack={transceiver.get_property('do-nack')}",
                flush=True,
            )

        self.webrtc.connect("on-negotiation-needed", self.on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self.on_ice_candidate)

        self.add_bus_watch(self.pipeline)

        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            self.metrics_last_error = "Failed to set sender pipeline to PLAYING"
            print("[sender] Failed to set sender pipeline to PLAYING.", flush=True)
            return False

        if external_fifo:
            GLib.idle_add(
                self.request_initial_external_h264_offer
            )

        self.install_video_loop_watchdog()
        self.install_metrics_timer()

        return True

    def cleanup(self) -> None:
        self.running.clear()
        self.external_h264_reader_stop.set()
        self.stop_receiver_feedback_server()
        self.metrics_signaling_state = "closed"

        if self.loop_watchdog_id is not None:
            try:
                GLib.source_remove(self.loop_watchdog_id)
            except Exception:
                pass
            self.loop_watchdog_id = None

        if self.metrics_timer_id is not None:
            try:
                GLib.source_remove(self.metrics_timer_id)
            except Exception:
                pass
            self.metrics_timer_id = None

        if self.receiver_feedback_client_socket is not None:
            try:
                self.receiver_feedback_client_socket.close()
            except Exception:
                pass
            self.receiver_feedback_client_socket = None

        if self.receiver_feedback_server_socket is not None:
            try:
                self.receiver_feedback_server_socket.close()
            except Exception:
                pass
            self.receiver_feedback_server_socket = None

        if self.external_h264_source is not None:
            print(
                "[external-h264] Final parsed/timestamped AU count: "
                f"{self.external_h264_au_number}",
                flush=True,
            )

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        if self.external_h264_reader_thread is not None:
            if self.external_h264_reader_thread.is_alive():
                self.external_h264_reader_thread.join(timeout=2.0)

            self.external_h264_reader_thread = None

        self.external_h264_source = None
        self.external_h264_parse = None
        self.external_h264_au_probe_id = None
        self.external_h264_au_number = 0
        self.external_h264_frame_duration_ns = 0
        self.external_h264_base_pts_ns = None
        self.external_h264_fifo = ""
        self.external_h264_reader_fd = None

        self.frame_source = None
        self.video_encoder = None
        self.gcc_estimators.clear()
        self.webrtc = None
        self.pipeline = None

        self.signaling.close()
        self.main_loop = None

    def run(self) -> int:
        self.metrics_signaling_state = "connecting"

        if not self.signaling.connect(self.config.host, self.config.port):
            self.metrics_signaling_state = "failed"
            self.cleanup()
            return 1

        self.metrics_signaling_state = "connected"
        self.start_receiver_feedback_server()

        self.main_loop = GLib.MainLoop()

        self.signaling_thread = threading.Thread(
            target=self.signaling_read_thread,
            name="SignalingReadThread",
            daemon=False,
        )
        self.signaling_thread.start()

        if not self.start_sender_pipeline():
            self.running.clear()
            self.signaling.shutdown_to_unblock()

            if self.signaling_thread.is_alive():
                self.signaling_thread.join()

            self.cleanup()
            return 1

        print("[sender] Running. Unity should receive and display the stream.", flush=True)
        print(f"[sender] Looping is {'enabled' if self.loop_enabled else 'disabled'}.", flush=True)

        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            print("\n[sender] Keyboard interrupt received.", flush=True)

        print("[sender] Stopping...", flush=True)

        self.running.clear()
        self.signaling.shutdown_to_unblock()

        if self.signaling_thread.is_alive():
            self.signaling_thread.join()

        self.cleanup()
        return 0




