#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")

from gi.repository import GLib, Gst, GstSdp, GstWebRTC  # noqa: E402


def b64encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def b64decode(text: str) -> Optional[str]:
    try:
        return base64.b64decode(text.encode("ascii")).decode("utf-8", errors="replace")
    except Exception:
        return None


def element_exists(name: str) -> bool:
    return Gst.ElementFactory.find(name) is not None


def safe_value(value: Any) -> str:
    text = str(value).replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    return text[:180]


@dataclass(frozen=True)
class ReceiverConfig:
    listen_host: str
    port: int
    feedback_port: int
    feedback_interval_ms: int
    latency_ms: int
    sink: str
    decoder: str
    sync: bool


class FeedbackPublisher:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.running = threading.Event()
        self.running.set()
        self.wakeup = threading.Event()
        self.lock = threading.Lock()
        self.latest_line: Optional[str] = None
        self.sock: Optional[socket.socket] = None
        self.thread = threading.Thread(
            target=self._run,
            name="ReceiverFeedbackPublisher",
            daemon=True,
        )
        self.thread.start()

    def publish(self, line: str) -> None:
        with self.lock:
            self.latest_line = line
        self.wakeup.set()

    def _connect(self) -> bool:
        while self.running.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            try:
                sock.connect((self.host, self.port))
                sock.settimeout(2.0)
                self.sock = sock
                print(
                    f"[receiver-feedback] Connected to sender dashboard at "
                    f"{self.host}:{self.port}",
                    flush=True,
                )
                return True
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
                self.wakeup.wait(0.5)
                self.wakeup.clear()
        return False

    def _close_socket(self) -> None:
        sock = self.sock
        self.sock = None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _run(self) -> None:
        while self.running.is_set():
            if self.sock is None and not self._connect():
                break

            self.wakeup.wait(1.0)
            self.wakeup.clear()

            with self.lock:
                line = self.latest_line

            if not line or self.sock is None:
                continue

            try:
                self.sock.sendall((line + "\n").encode("utf-8"))
            except OSError:
                print("[receiver-feedback] Connection lost; retrying.", flush=True)
                self._close_socket()

        self._close_socket()

    def stop(self) -> None:
        self.running.clear()
        self.wakeup.set()
        self._close_socket()
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)


class LinuxWebRTCReceiver:
    DECODER_CANDIDATES = {
        "h264": ["nvh264dec", "vah264dec", "qsvh264dec", "avdec_h264"],
        "h265": ["nvh265dec", "vah265dec", "qsvh265dec", "avdec_h265"],
        "av1": ["nvav1dec", "vaav1dec", "qsvav1dec", "dav1ddec", "av1dec"],
    }

    RTP_ELEMENTS = {
        "h264": ("rtph264depay", "h264parse"),
        "h265": ("rtph265depay", "h265parse"),
        "av1": ("rtpav1depay", "av1parse"),
    }

    def __init__(self, config: ReceiverConfig):
        self.config = config
        self.main_loop = GLib.MainLoop()
        self.running = threading.Event()
        self.running.set()

        self.server_socket: Optional[socket.socket] = None
        self.server_thread: Optional[threading.Thread] = None
        self.connection_socket: Optional[socket.socket] = None
        self.connection_lock = threading.Lock()
        self.send_lock = threading.Lock()

        self.pipeline: Optional[Gst.Pipeline] = None
        self.webrtc: Optional[Gst.Element] = None
        self.rtp_queue: Optional[Gst.Element] = None
        self.display_queue: Optional[Gst.Element] = None
        self.video_chain_created = False
        self.remote_description_set = False
        self.pending_ice: list[tuple[int, str]] = []

        self.feedback: Optional[FeedbackPublisher] = None
        self.sender_host = "n/a"

        self.metrics_lock = threading.Lock()
        self.session_started_ns = 0
        self.last_metrics_ns = 0
        self.last_received_frames = 0
        self.last_decoded_frames = 0
        self.last_displayed_frames = 0
        self.last_rtp_bytes = 0

        self.received_frames = 0
        self.decoded_frames = 0
        self.displayed_frames = 0
        self.rtp_packets = 0
        self.rtp_bytes = 0
        self.last_displayed_ns = 0

        self.received_fps = 0.0
        self.decoded_fps = 0.0
        self.displayed_fps = 0.0
        self.incoming_bitrate_mbps = 0.0

        self.codec = "n/a"
        self.resolution = "n/a"
        self.decoder_name = "n/a"
        self.packet_loss_pct = "n/a"
        self.packets_lost = "n/a"
        self.jitter_ms = "n/a"
        self.rtt_ms = "n/a"
        self.last_error = "None"

        self.metrics_timer_id: Optional[int] = None
        self.stats_request_counter = 0

    def _new_element(self, factory: str, name: str) -> Gst.Element:
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Required GStreamer element is missing: {factory}")
        return element

    def _set_if_present(self, element: Gst.Element, name: str, value: Any) -> None:
        if element.find_property(name) is not None:
            element.set_property(name, value)

    def _choose_decoder(self, codec: str) -> str:
        requested = self.config.decoder.strip().lower()
        if requested != "auto":
            if not element_exists(requested):
                raise RuntimeError(f"Requested decoder is not installed: {requested}")
            return requested

        for candidate in self.DECODER_CANDIDATES[codec]:
            if element_exists(candidate):
                return candidate

        raise RuntimeError(
            f"No usable {codec.upper()} decoder found. Tried: "
            + ", ".join(self.DECODER_CANDIDATES[codec])
        )

    def _choose_sink(self) -> str:
        requested = self.config.sink.strip().lower()
        if requested != "auto":
            if not element_exists(requested):
                raise RuntimeError(f"Requested video sink is not installed: {requested}")
            return requested

        for candidate in ("glimagesink", "waylandsink", "ximagesink", "autovideosink"):
            if element_exists(candidate):
                return candidate

        if element_exists("fakesink"):
            return "fakesink"

        raise RuntimeError("No usable video sink found")

    def _send_signaling_line(self, line: str) -> bool:
        with self.send_lock:
            sock = self.connection_socket
            if sock is None:
                return False
            try:
                sock.sendall((line + "\n").encode("utf-8"))
                return True
            except OSError:
                return False

    def _recv_line(self, sock: socket.socket) -> Optional[str]:
        chunks: list[bytes] = []
        while self.running.is_set():
            try:
                chunk = sock.recv(1)
            except OSError:
                return None
            if not chunk:
                return None
            if chunk == b"\n":
                return b"".join(chunks).decode("utf-8", errors="replace")
            if chunk != b"\r":
                chunks.append(chunk)
        return None

    def _server_loop(self) -> None:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(1.0)
            server.bind((self.config.listen_host, self.config.port))
            server.listen(1)
            self.server_socket = server
        except OSError as exc:
            self.last_error = f"Signaling server error: {exc}"
            print(f"[signaling] Could not listen on port {self.config.port}: {exc}", flush=True)
            GLib.idle_add(self._quit_main_loop)
            return

        print(
            f"[signaling] Ubuntu receiver listening on "
            f"{self.config.listen_host}:{self.config.port}",
            flush=True,
        )

        try:
            while self.running.is_set():
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                client.settimeout(None)
                print(
                    f"[signaling] Sender connected from {address[0]}:{address[1]}",
                    flush=True,
                )

                ready = threading.Event()
                GLib.idle_add(self._begin_session, client, address[0], ready)
                if not ready.wait(timeout=10.0):
                    print("[receiver] Session setup timed out.", flush=True)
                    try:
                        client.close()
                    except OSError:
                        pass
                    continue

                try:
                    while self.running.is_set():
                        line = self._recv_line(client)
                        if line is None:
                            break
                        if line:
                            GLib.idle_add(self._handle_signaling_line, line)
                finally:
                    print("[signaling] Sender disconnected.", flush=True)
                    done = threading.Event()
                    GLib.idle_add(self._end_session, done)
                    done.wait(timeout=10.0)
        finally:
            try:
                server.close()
            except OSError:
                pass
            self.server_socket = None

    def _begin_session(self, sock: socket.socket, sender_host: str, ready: threading.Event) -> bool:
        try:
            self._destroy_pipeline()
            with self.connection_lock:
                self.connection_socket = sock
            self.sender_host = sender_host
            self.remote_description_set = False
            self.pending_ice.clear()
            self.video_chain_created = False
            self._reset_metrics()

            self.feedback = FeedbackPublisher(sender_host, self.config.feedback_port)
            self._create_pipeline()
            ready.set()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[receiver] Session setup failed: {exc}", flush=True)
            ready.set()
        return False

    def _end_session(self, done: threading.Event) -> bool:
        try:
            self._publish_metrics(force_alive=False)
            if self.feedback is not None:
                time.sleep(0.05)
                self.feedback.stop()
                self.feedback = None
            with self.connection_lock:
                sock = self.connection_socket
                self.connection_socket = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            self._destroy_pipeline()
        finally:
            done.set()
        return False

    def _create_pipeline(self) -> None:
        pipeline = Gst.Pipeline.new("qgxs_linux_receiver")
        if pipeline is None:
            raise RuntimeError("Could not create receiver pipeline")

        webrtc = self._new_element("webrtcbin", "receiver_webrtc")
        webrtc.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self._set_if_present(webrtc, "latency", self.config.latency_ms)
        webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        webrtc.connect("pad-added", self._on_webrtc_pad_added)

        pipeline.add(webrtc)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.pipeline = pipeline
        self.webrtc = webrtc

        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Could not set receiver pipeline to PLAYING")

        self.metrics_timer_id = GLib.timeout_add(
            self.config.feedback_interval_ms,
            self._on_metrics_tick,
        )
        print("[receiver] WebRTC pipeline is ready.", flush=True)

    def _destroy_pipeline(self) -> None:
        if self.metrics_timer_id is not None:
            try:
                GLib.source_remove(self.metrics_timer_id)
            except Exception:
                pass
            self.metrics_timer_id = None

        if self.pipeline is not None:
            try:
                bus = self.pipeline.get_bus()
                bus.remove_signal_watch()
            except Exception:
                pass
            self.pipeline.set_state(Gst.State.NULL)

        self.pipeline = None
        self.webrtc = None
        self.rtp_queue = None
        self.display_queue = None
        self.video_chain_created = False
        self.remote_description_set = False
        self.pending_ice.clear()

    def _reset_metrics(self) -> None:
        now = time.monotonic_ns()
        with self.metrics_lock:
            self.session_started_ns = now
            self.last_metrics_ns = now
            self.last_received_frames = 0
            self.last_decoded_frames = 0
            self.last_displayed_frames = 0
            self.last_rtp_bytes = 0
            self.received_frames = 0
            self.decoded_frames = 0
            self.displayed_frames = 0
            self.rtp_packets = 0
            self.rtp_bytes = 0
            self.last_displayed_ns = 0
            self.received_fps = 0.0
            self.decoded_fps = 0.0
            self.displayed_fps = 0.0
            self.incoming_bitrate_mbps = 0.0
            self.codec = "n/a"
            self.resolution = "n/a"
            self.decoder_name = "n/a"
            self.packet_loss_pct = "n/a"
            self.packets_lost = "n/a"
            self.jitter_ms = "n/a"
            self.rtt_ms = "n/a"
            self.last_error = "None"

    def _handle_signaling_line(self, line: str) -> bool:
        if line.startswith("OFFER|"):
            self._handle_offer(line[6:])
            return False

        if line.startswith("ICE|"):
            parts = line.split("|", 2)
            if len(parts) != 3:
                print("[signaling] Malformed ICE message.", flush=True)
                return False
            candidate = b64decode(parts[2])
            if candidate is None:
                print("[signaling] Could not decode ICE candidate.", flush=True)
                return False
            try:
                mline = int(parts[1])
            except ValueError:
                print("[signaling] Invalid ICE mline index.", flush=True)
                return False

            if self.webrtc is None:
                return False
            if self.remote_description_set:
                self.webrtc.emit("add-ice-candidate", mline, candidate)
            else:
                self.pending_ice.append((mline, candidate))
            return False

        print(f"[signaling] Unknown sender message: {line[:120]}", flush=True)
        return False

    def _handle_offer(self, encoded_offer: str) -> None:
        if self.webrtc is None:
            return

        sdp_text = b64decode(encoded_offer)
        if sdp_text is None:
            print("[signaling] Could not decode SDP offer.", flush=True)
            return

        result, sdp = GstSdp.SDPMessage.new()
        if result != GstSdp.SDPResult.OK:
            print("[signaling] Could not allocate SDP offer.", flush=True)
            return

        parse_result = GstSdp.sdp_message_parse_buffer(sdp_text.encode("utf-8"), sdp)
        if parse_result != GstSdp.SDPResult.OK:
            print("[signaling] Could not parse SDP offer.", flush=True)
            return

        offer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.OFFER,
            sdp,
        )
        if offer is None:
            print("[signaling] Could not create SDP offer description.", flush=True)
            return

        print("[signaling] SDP offer received.", flush=True)
        promise = Gst.Promise.new_with_change_func(
            self._on_remote_description_set,
            None,
            None,
        )
        self.webrtc.emit("set-remote-description", offer, promise)

    def _on_remote_description_set(self, promise: Gst.Promise, *args: Any) -> None:
        GLib.idle_add(self._finish_remote_description)

    def _finish_remote_description(self) -> bool:
        if self.webrtc is None:
            return False

        self.remote_description_set = True
        for mline, candidate in self.pending_ice:
            self.webrtc.emit("add-ice-candidate", mline, candidate)
        self.pending_ice.clear()

        promise = Gst.Promise.new_with_change_func(
            self._on_answer_created,
            None,
            None,
        )
        self.webrtc.emit("create-answer", None, promise)
        return False

    def _on_answer_created(self, promise: Gst.Promise, *args: Any) -> None:
        reply = promise.get_reply()
        if reply is None:
            print("[signaling] Empty create-answer reply.", flush=True)
            return

        try:
            answer = reply.get_value("answer")
        except Exception:
            answer = None

        if answer is None or self.webrtc is None:
            print("[signaling] Could not create SDP answer.", flush=True)
            return

        self.webrtc.emit("set-local-description", answer, None)
        try:
            sdp_text = answer.sdp.as_text()
        except Exception:
            sdp_text = None

        if not sdp_text:
            print("[signaling] Could not serialize SDP answer.", flush=True)
            return

        if self._send_signaling_line("ANSWER|" + b64encode(sdp_text)):
            print("[signaling] SDP answer sent.", flush=True)

    def _on_ice_candidate(
        self,
        webrtc: Gst.Element,
        mline_index: int,
        candidate: str,
    ) -> None:
        if not candidate:
            return
        self._send_signaling_line(
            f"ICE|{mline_index}|{b64encode(candidate)}"
        )

    def _caps_text(self, pad: Gst.Pad) -> str:
        caps = pad.get_current_caps()
        if caps is None or caps.is_empty() or caps.is_any():
            caps = pad.query_caps(None)
        return caps.to_string() if caps is not None else "unknown"

    def _on_webrtc_pad_added(self, webrtc: Gst.Element, pad: Gst.Pad) -> None:
        if self.video_chain_created or self.pipeline is None:
            return

        caps = pad.get_current_caps()
        if caps is None or caps.is_empty() or caps.is_any():
            caps = pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            print("[receiver] Incoming pad has no usable caps.", flush=True)
            return

        structure = caps.get_structure(0)
        media = structure.get_string("media") or ""
        encoding = (structure.get_string("encoding-name") or "").upper()

        if media.lower() != "video":
            print(f"[receiver] Ignoring non-video pad: {caps.to_string()}", flush=True)
            return

        codec_map = {"H264": "h264", "H265": "h265", "HEVC": "h265", "AV1": "av1"}
        codec = codec_map.get(encoding)
        if codec is None:
            self.last_error = f"Unsupported RTP encoding: {encoding or caps.to_string()}"
            print(f"[receiver] {self.last_error}", flush=True)
            return

        try:
            self._build_video_chain(pad, codec)
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[receiver] Could not build video chain: {exc}", flush=True)

    def _build_video_chain(self, incoming_pad: Gst.Pad, codec: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("Receiver pipeline is missing")

        depay_name, parser_name = self.RTP_ELEMENTS[codec]
        decoder_name = self._choose_decoder(codec)
        sink_name = self._choose_sink()

        rtp_queue = self._new_element("queue", "receiver_rtp_queue")
        depay = self._new_element(depay_name, f"receiver_{codec}_depay")
        parser = self._new_element(parser_name, f"receiver_{codec}_parse")
        decoder = self._new_element(decoder_name, "receiver_video_decoder")
        convert = self._new_element("videoconvert", "receiver_video_convert")
        display_queue = self._new_element("queue", "receiver_display_queue")
        sink = self._new_element(sink_name, "receiver_video_sink")

        self._set_if_present(rtp_queue, "max-size-buffers", 512)
        self._set_if_present(rtp_queue, "max-size-bytes", 0)
        self._set_if_present(rtp_queue, "max-size-time", 0)
        self._set_if_present(rtp_queue, "leaky", 0)

        self._set_if_present(display_queue, "max-size-buffers", 6)
        self._set_if_present(display_queue, "max-size-bytes", 0)
        self._set_if_present(display_queue, "max-size-time", 0)
        self._set_if_present(display_queue, "leaky", 2)

        self._set_if_present(sink, "sync", self.config.sync)
        self._set_if_present(sink, "qos", True)
        self._set_if_present(sink, "force-aspect-ratio", True)

        if codec == "av1":
            self._set_if_present(depay, "request-keyframe", True)
            self._set_if_present(depay, "wait-for-keyframe", True)

        elements = [rtp_queue, depay, parser, decoder, convert, display_queue, sink]
        for element in elements:
            self.pipeline.add(element)

        for upstream, downstream in zip(elements, elements[1:]):
            if not upstream.link(downstream):
                raise RuntimeError(
                    f"Could not link {upstream.get_name()} -> {downstream.get_name()}"
                )

        queue_sink = rtp_queue.get_static_pad("sink")
        if queue_sink is None:
            raise RuntimeError("RTP queue sink pad is missing")
        if incoming_pad.link(queue_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Could not link webrtcbin video pad to RTP queue")

        incoming_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_rtp_buffer)

        parser_src = parser.get_static_pad("src")
        decoder_src = decoder.get_static_pad("src")
        sink_pad = sink.get_static_pad("sink")

        if parser_src is None or decoder_src is None or sink_pad is None:
            raise RuntimeError("A required telemetry pad is missing")

        parser_src.add_probe(Gst.PadProbeType.BUFFER, self._on_received_frame)
        decoder_src.add_probe(Gst.PadProbeType.BUFFER, self._on_decoded_frame)
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_displayed_frame)

        for element in elements:
            element.sync_state_with_parent()

        self.rtp_queue = rtp_queue
        self.display_queue = display_queue
        self.video_chain_created = True
        with self.metrics_lock:
            self.codec = codec
            self.decoder_name = decoder_name

        print(
            "[receiver] Video chain active: "
            f"codec={codec} decoder={decoder_name} sink={sink_name} "
            f"caps={self._caps_text(incoming_pad)}",
            flush=True,
        )

    def _on_rtp_buffer(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()
        if buffer is not None:
            with self.metrics_lock:
                self.rtp_packets += 1
                self.rtp_bytes += buffer.get_size()
        return Gst.PadProbeReturn.OK

    def _on_received_frame(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        if info.get_buffer() is not None:
            with self.metrics_lock:
                self.received_frames += 1
        return Gst.PadProbeReturn.OK

    def _on_decoded_frame(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        if info.get_buffer() is not None:
            caps = pad.get_current_caps()
            resolution = None
            if caps is not None and caps.get_size() > 0:
                structure = caps.get_structure(0)
                try:
                    width = structure.get_value("width")
                    height = structure.get_value("height")
                    if width and height:
                        resolution = f"{int(width)}x{int(height)}"
                except Exception:
                    pass
            with self.metrics_lock:
                self.decoded_frames += 1
                if resolution:
                    self.resolution = resolution
        return Gst.PadProbeReturn.OK

    def _on_displayed_frame(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        if info.get_buffer() is not None:
            now = time.monotonic_ns()
            with self.metrics_lock:
                self.displayed_frames += 1
                self.last_displayed_ns = now
        return Gst.PadProbeReturn.OK

    def _on_metrics_tick(self) -> bool:
        if self.pipeline is None:
            return False

        now = time.monotonic_ns()
        with self.metrics_lock:
            elapsed = max(1, now - self.last_metrics_ns) / 1_000_000_000.0
            received_delta = max(0, self.received_frames - self.last_received_frames)
            decoded_delta = max(0, self.decoded_frames - self.last_decoded_frames)
            displayed_delta = max(0, self.displayed_frames - self.last_displayed_frames)
            bytes_delta = max(0, self.rtp_bytes - self.last_rtp_bytes)

            self.received_fps = received_delta / elapsed
            self.decoded_fps = decoded_delta / elapsed
            self.displayed_fps = displayed_delta / elapsed
            self.incoming_bitrate_mbps = bytes_delta * 8.0 / elapsed / 1_000_000.0

            self.last_received_frames = self.received_frames
            self.last_decoded_frames = self.decoded_frames
            self.last_displayed_frames = self.displayed_frames
            self.last_rtp_bytes = self.rtp_bytes
            self.last_metrics_ns = now

        self.stats_request_counter += 1
        if self.stats_request_counter >= max(1, 1000 // self.config.feedback_interval_ms):
            self.stats_request_counter = 0
            self._request_webrtc_stats()

        self._publish_metrics(force_alive=True)
        return True

    def _publish_metrics(self, force_alive: bool) -> None:
        now = time.monotonic_ns()
        with self.metrics_lock:
            backlog = 0
            if self.display_queue is not None:
                try:
                    backlog = int(self.display_queue.get_property("current-level-buffers"))
                except Exception:
                    backlog = 0

            if self.last_displayed_ns > 0:
                age_ms = max(0.0, (now - self.last_displayed_ns) / 1_000_000.0)
            else:
                age_ms = max(0.0, (now - self.session_started_ns) / 1_000_000.0)

            stall = int(
                force_alive
                and self.received_frames > 0
                and age_ms > 1500.0
            )
            dropped = max(0, self.decoded_frames - self.displayed_frames - backlog)

            fields = {
                "receiver_alive": int(force_alive),
                "display_mode": "linux-gstreamer",
                "unity_fps": f"{self.displayed_fps:.1f}",
                "copied_fps": f"{self.decoded_fps:.1f}",
                "last_frame_id": self.displayed_frames,
                "received_frames_total": self.received_frames,
                "decoded_frames_total": self.decoded_frames,
                "displayed_frames_total": self.displayed_frames,
                "texture_attached": int(self.video_chain_created),
                "stall": stall,
                "received_fps": f"{self.received_fps:.1f}",
                "decoded_fps": f"{self.decoded_fps:.1f}",
                "displayed_fps": f"{self.displayed_fps:.1f}",
                "bitrate_mbps": f"{self.incoming_bitrate_mbps:.2f}",
                "packets_received": self.rtp_packets,
                "bytes_received": self.rtp_bytes,
                "packets_lost": self.packets_lost,
                "packet_loss_pct": self.packet_loss_pct,
                "jitter_ms": self.jitter_ms,
                "rtt_ms": self.rtt_ms,
                "decoder": self.decoder_name,
                "codec": self.codec,
                "resolution": self.resolution,
                "queue_backlog": backlog,
                "dropped_frames": dropped,
                "last_frame_age_ms": f"{age_ms:.1f}",
                "last_error": self.last_error,
            }

        line = "RECEIVER_METRICS|" + "|".join(
            f"{key}={safe_value(value)}" for key, value in fields.items()
        )

        if self.feedback is not None:
            self.feedback.publish(line)

        print(line, flush=True)

    def _request_webrtc_stats(self) -> None:
        if self.webrtc is None:
            return
        try:
            promise = Gst.Promise.new_with_change_func(
                self._on_webrtc_stats_ready,
                None,
                None,
            )
            self.webrtc.emit("get-stats", None, promise)
        except Exception as exc:
            self.last_error = f"WebRTC stats request error: {exc}"

    def _on_webrtc_stats_ready(self, promise: Gst.Promise, *args: Any) -> None:
        try:
            reply = promise.get_reply()
        except Exception as exc:
            self.last_error = f"WebRTC stats reply error: {exc}"
            return
        if reply is None:
            return
        try:
            flat: dict[str, Any] = {}
            self._flatten_stats(reply, "", flat)
            self._apply_webrtc_stats(flat)
        except Exception as exc:
            self.last_error = f"WebRTC stats parse error: {exc}"

    def _flatten_stats(self, value: Any, prefix: str, output: dict[str, Any]) -> None:
        if value is None:
            return
        if isinstance(value, Gst.Structure):
            for index in range(value.n_fields()):
                name = value.nth_field_name(index)
                child = value.get_value(name)
                key = f"{prefix}.{name}" if prefix else str(name)
                self._flatten_stats(child, key, output)
            return
        try:
            if not isinstance(value, (str, bytes, bytearray)):
                length = len(value)
                for index in range(length):
                    self._flatten_stats(value[index], f"{prefix}[{index}]", output)
                return
        except Exception:
            pass
        output[prefix] = value

    def _normalize_key(self, key: str) -> str:
        return "".join(ch.lower() for ch in str(key) if ch.isalnum())

    def _find_number(self, flat: dict[str, Any], names: list[str]) -> Optional[float]:
        targets = [self._normalize_key(name) for name in names]
        for key, value in flat.items():
            normalized = self._normalize_key(key)
            if not any(normalized.endswith(target) or target in normalized for target in targets):
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    def _apply_webrtc_stats(self, flat: dict[str, Any]) -> None:
        packets_received = self._find_number(flat, ["packetsReceived", "packets-received"])
        packets_lost = self._find_number(flat, ["packetsLost", "packets-lost"])
        jitter = self._find_number(flat, ["jitter", "jitterBufferDelay", "jitter-buffer-delay"])
        rtt = self._find_number(flat, ["currentRoundTripTime", "roundTripTime", "rtt"])

        with self.metrics_lock:
            if packets_lost is not None:
                self.packets_lost = str(max(0, int(packets_lost)))

            if packets_received is not None and packets_lost is not None:
                denominator = max(0.0, packets_received) + max(0.0, packets_lost)
                if denominator > 0:
                    self.packet_loss_pct = f"{max(0.0, packets_lost) * 100.0 / denominator:.2f}"

            if jitter is not None and jitter >= 0:
                self.jitter_ms = f"{jitter * 1000.0:.2f}" if jitter <= 10.0 else f"{jitter:.2f}"

            if rtt is not None and rtt >= 0:
                self.rtt_ms = f"{rtt * 1000.0:.1f}" if rtt <= 10.0 else f"{rtt:.1f}"

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        source = message.src.get_name() if message.src else "unknown"
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            text = error.message if error else "unknown"
            self.last_error = f"ERROR from {source}: {text}"
            print(f"[ERROR from {source}] {text}", flush=True)
            if debug:
                print(f"[DEBUG] {debug}", flush=True)
        elif message.type == Gst.MessageType.WARNING:
            error, debug = message.parse_warning()
            text = error.message if error else "unknown"
            print(f"[WARNING from {source}] {text}", flush=True)
            if debug:
                print(f"[DEBUG] {debug}", flush=True)
        elif message.type == Gst.MessageType.EOS:
            print("[receiver] EOS received.", flush=True)

    def _quit_main_loop(self) -> bool:
        if self.main_loop.is_running():
            self.main_loop.quit()
        return False

    def start(self) -> int:
        self.server_thread = threading.Thread(
            target=self._server_loop,
            name="ReceiverSignalingServer",
            daemon=False,
        )
        self.server_thread.start()

        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            print("\n[receiver] Keyboard interrupt received.", flush=True)
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        if not self.running.is_set():
            return
        self.running.clear()

        server = self.server_socket
        self.server_socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass

        with self.connection_lock:
            sock = self.connection_socket
            self.connection_socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        if self.feedback is not None:
            self.feedback.stop()
            self.feedback = None

        self._destroy_pipeline()

        if self.main_loop.is_running():
            self.main_loop.quit()

        if self.server_thread is not None and self.server_thread.is_alive():
            self.server_thread.join(timeout=3.0)


def parse_args() -> ReceiverConfig:
    parser = argparse.ArgumentParser(
        description="QGXS Ubuntu WebRTC receiver with receiver-to-sender telemetry"
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument(
        "--feedback-port",
        type=int,
        default=0,
        help="Sender feedback port; default is signaling port + 100",
    )
    parser.add_argument("--feedback-interval-ms", type=int, default=500)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument(
        "--sink",
        default="auto",
        help="auto, glimagesink, waylandsink, ximagesink, autovideosink, or fakesink",
    )
    parser.add_argument(
        "--decoder",
        default="auto",
        help="auto or an explicit GStreamer decoder element such as nvh264dec",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Disable sink clock synchronization for diagnostic testing",
    )
    args = parser.parse_args()

    if not (1 <= args.port <= 65535):
        parser.error("--port must be between 1 and 65535")
    feedback_port = args.feedback_port or (args.port + 100)
    if not (1 <= feedback_port <= 65535):
        parser.error("feedback port must be between 1 and 65535")
    if not (100 <= args.feedback_interval_ms <= 5000):
        parser.error("--feedback-interval-ms must be between 100 and 5000")
    if not (0 <= args.latency_ms <= 5000):
        parser.error("--latency-ms must be between 0 and 5000")

    return ReceiverConfig(
        listen_host=args.listen_host,
        port=args.port,
        feedback_port=feedback_port,
        feedback_interval_ms=args.feedback_interval_ms,
        latency_ms=args.latency_ms,
        sink=args.sink,
        decoder=args.decoder,
        sync=not args.no_sync,
    )


def main() -> int:
    Gst.init(None)
    config = parse_args()

    print("============================================================", flush=True)
    print("QGXS Ubuntu WebRTC Receiver", flush=True)
    print("============================================================", flush=True)
    print(f"Signaling: {config.listen_host}:{config.port}", flush=True)
    print(f"Feedback port: sender:{config.feedback_port}", flush=True)
    print(f"Sink: {config.sink}", flush=True)
    print(f"Decoder: {config.decoder}", flush=True)
    print(f"Latency: {config.latency_ms} ms", flush=True)
    print("============================================================", flush=True)

    receiver = LinuxWebRTCReceiver(config)
    return receiver.start()


if __name__ == "__main__":
    raise SystemExit(main())
