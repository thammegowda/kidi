import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import secrets
import socket
import struct
import sys
import threading
import time
import webbrowser

import wifi

ROOT = Path(__file__).resolve().parent
HEADER = struct.Struct(">B3xIQI")
MAX_PAYLOAD = 1024 * 1024
RESOLUTIONS = {"1280x720": (1280, 720), "96x96": (96, 96)}


class LiveSource:
    def __init__(self, stream, connection=None, transport="wifi"):
        self.stream = stream
        self.connection = connection
        self.transport = transport
        self.ended = False

    def close(self):
        if self.connection is not None:
            if self.connection.sock is not None:
                try:
                    self.connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # The peer may already have closed the stream.
            self.stream.close()
            self.connection.close()
        else:
            if self.transport == "usb" and not self.ended:
                try:
                    self.stream.write(b"stream-stop\n")
                    self.stream.flush()
                    deadline = time.monotonic() + 5
                    tail = b""
                    while time.monotonic() < deadline:
                        chunk = self.stream.read(min(4096, getattr(self.stream, "in_waiting", 1) or 1))
                        tail = (tail + chunk)[-64:]
                        if b"KIDI_STREAM_STOPPED\n" in tail.replace(b"\r\n", b"\n"):
                            self.ended = True
                            break
                    if not self.ended:
                        raise wifi.SetupError("USB stop acknowledgement timed out")
                except (wifi.SetupError, OSError, ValueError) as error:
                    print(f"USB stream stop was not acknowledged: {error}", file=sys.stderr)
            self.stream.close()


def open_wifi_source(profile, mode, seconds, resolution="1280x720"):
    connection = wifi.DeviceConnection(profile)
    try:
        connection.connect()
        connection.request(
            "POST", "/v1/stream", json.dumps({"mode": mode, "seconds": seconds, "resolution": resolution}),
            {"Authorization": f"Bearer {profile['token']}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status in (401, 403):
            raise wifi.SecurityError("Stream authentication failed; refusing USB fallback")
        if response.status != 200:
            raise wifi.SetupError(f"Live stream failed with HTTP {response.status}; update firmware if needed")
        if (
            response.getheader("Content-Type") != "application/vnd.kidi.live"
            or response.getheader("X-Kidi-Transport") != "wifi"
            or response.getheader("X-Kidi-Stream-Protocol") != "1"
        ):
            raise wifi.SetupError("Unsupported live stream format")
        if response.readline(1024) != b"KIDI_LIVE_BEGIN\n":
            raise wifi.SetupError("Missing live stream start marker")
        return LiveSource(response, connection)
    except BaseException:
        connection.close()
        raise


def open_usb_source(port, mode, seconds, resolution="1280x720"):
    stream = wifi.open_serial(wifi.choose_port(port))
    try:
        stream.write(f"stream {mode} {seconds} {resolution}\n".encode())
        stream.flush()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            line = stream.readline()
            if line.strip() == b"KIDI_LIVE_BEGIN":
                return LiveSource(stream, transport="usb")
            if line.startswith(b"KIDI_ERROR"):
                raise wifi.SetupError(line.decode(errors="replace").strip())
        raise wifi.SetupError("USB live stream did not start; update the firmware")
    except BaseException:
        stream.close()
        raise


def open_source(args):
    if args.transport != "usb":
        try:
            profile = wifi.cached_profile(args.device)
        except wifi.SecurityError:
            raise
        except wifi.SetupError:
            if args.transport == "wifi":
                raise
        else:
            try:
                return open_wifi_source(profile, args.mode, args.seconds, args.resolution)
            except wifi.WifiUnavailable as error:
                if args.transport == "wifi":
                    raise
                print(f"Wi-Fi unavailable: {error}. Using USB live development fallback.", file=sys.stderr)
    if args.transport != "usb":
        print("Starting USB live development transport.", file=sys.stderr)
    return open_usb_source(args.port, args.mode, args.seconds, args.resolution)


def read_exact(source, size, deadline=None):
    result = bytearray()
    if deadline is None:
        deadline = time.monotonic() + 15
    while len(result) < size:
        if time.monotonic() > deadline:
            raise TimeoutError("Live packet timed out")
        chunk = source.stream.read(size - len(result))
        if not chunk:
            if source.transport == "usb":
                continue
            raise wifi.SetupError("Live stream disconnected before its end marker")
        result.extend(chunk)
    return bytes(result)


def jpeg_dimensions(payload):
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise wifi.SetupError("Incomplete live JPEG")
    position = 2
    while position < len(payload):
        if payload[position] != 0xff:
            raise wifi.SetupError("Invalid JPEG marker")
        while position < len(payload) and payload[position] == 0xff:
            position += 1
        if position >= len(payload):
            break
        marker = payload[position]
        position += 1
        if marker in (0xd9, 0xda):
            break
        if marker == 0x01 or 0xd0 <= marker <= 0xd8:
            continue
        if position + 2 > len(payload):
            break
        length = int.from_bytes(payload[position:position + 2], "big")
        if length < 2 or position + length > len(payload):
            raise wifi.SetupError("Truncated JPEG segment")
        if marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
            if length < 8:
                raise wifi.SetupError("Invalid JPEG frame header")
            height, width = struct.unpack(">HH", payload[position + 3:position + 7])
            return width, height
        position += length
    raise wifi.SetupError("JPEG contains no image dimensions")


def packets(source, resolution="1280x720"):
    started = False
    audio_next = 0
    video_next = 0
    audio_bytes = None
    with_audio = False
    while True:
        header = read_exact(source, HEADER.size)
        kind, sequence, timestamp, length = HEADER.unpack(header)
        if kind not in (1, 2, 3, 4) or header[1:4] != b"\0\0\0" or not 0 < length <= MAX_PAYLOAD:
            raise wifi.SetupError("Invalid live packet header")
        payload = read_exact(source, length)
        if not started:
            if kind != 3:
                raise wifi.SetupError("Stream is missing format metadata")
            info = json.loads(payload)
            width, height = RESOLUTIONS[resolution]
            if (info["protocol"], info["width"], info["height"]) != (1, width, height) or info["sample_rate"] not in (16000, 48000):
                raise wifi.SetupError("Unsupported live media format")
            audio_bytes = info["sample_rate"] // 50 * 2
            if not isinstance(info["audio"], bool):
                raise wifi.SetupError("Invalid audio-enabled metadata")
            with_audio = info["audio"]
            started = True
        if kind == 1:
            if sequence != video_next or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
                raise wifi.SetupError("Missing or malformed live camera frame")
            if jpeg_dimensions(payload) != RESOLUTIONS[resolution]:
                raise wifi.SetupError("Actual JPEG dimensions differ from the requested live resolution")
            video_next += 1
        elif kind == 2:
            if not with_audio:
                raise wifi.SetupError("Unexpected audio in a video-only stream")
            if sequence != audio_next or length % audio_bytes != 0 or timestamp != sequence * 20000:
                raise wifi.SetupError("Live audio discontinuity")
            audio_next += length // audio_bytes
        elif kind == 3:
            info = json.loads(payload)
            if "sdk_diagnostics" in info:
                print("Accessory SDK diagnostic:", info["sdk_diagnostics"], file=sys.stderr)
                if info.get("sdk_diagnostics_truncated"):
                    print("Accessory SDK diagnostics exceeded their bounded buffer.", file=sys.stderr)
            if info.get("audio_dropped", 0) or info.get("audio_overruns", 0):
                raise wifi.SetupError(
                    f"Live audio queue/DMA overrun: dropped={info.get('audio_dropped', 0)}, "
                    f"DMA={info.get('audio_overruns', 0)}, video_acquired={info.get('video_acquired', 0)}, "
                    f"video_sent={info.get('video_sent', 0)}"
                )
        if kind == 4:
            source.ended = True
        yield header, payload
        if kind == 4:
            return


def check_stream(args):
    source = open_source(args)
    frames = 0
    samples = 0
    first = None
    last = 0
    video_bytes = 0
    gaps = []
    received_previous = None
    summary = {}
    started = time.monotonic()
    try:
        for header, payload in packets(source, args.resolution):
            kind, _, timestamp, _ = HEADER.unpack(header)
            if kind == 1:
                received = time.monotonic()
                if received_previous is not None:
                    gaps.append(received - received_previous)
                received_previous = received
                frames += 1
                video_bytes += len(payload)
                first = timestamp if first is None else first
                last = timestamp
            elif kind == 2:
                samples += len(payload) // 2
            elif kind == 3:
                print("Live status:", payload.decode())
            elif kind == 4:
                summary = json.loads(payload)
    finally:
        source.close()
    elapsed = time.monotonic() - started
    if frames < 2 or (args.mode == "av" and samples == 0):
        raise wifi.SetupError("Live test did not receive usable media")
    if args.mode == "video" and samples != 0:
        raise wifi.SetupError("Video-only check received unexpected audio")
    measured_seconds = max(args.seconds, elapsed)
    delivered_fps = frames / measured_seconds
    report = {
        "transport": source.transport, "resolution": args.resolution, "mode": args.mode,
        "frames": frames, "audio_samples": samples, "requested_seconds": args.seconds,
        "received_seconds": elapsed, "delivered_fps": delivered_fps,
        "video_bytes": video_bytes, "payload_mbps": video_bytes * 8 / measured_seconds / 1e6,
        "max_frame_gap_seconds": max(gaps, default=0),
        "camera_first_seconds": first / 1e6, "camera_last_seconds": last / 1e6,
        "device": summary,
    }
    print("Live report:", json.dumps(report, sort_keys=True))
    print(
        f"Live {source.transport.upper()} {args.resolution}: {frames} frames, {samples} audio samples; "
        f"{delivered_fps:.2f} delivered fps, "
        f"camera timestamps {first / 1e6:.3f}..{last / 1e6:.3f}s, received in {elapsed:.3f}s; no files saved"
    )
    if args.min_fps is not None and delivered_fps < args.min_fps:
        raise wifi.SetupError(f"Delivered {delivered_fps:.2f} fps, below required {args.min_fps:.2f}")
    return report


def serve_viewer(args):
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *arguments):
            pass

        def do_GET(self):
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected_host:
                self.send_error(403)
                return
            if self.path == "/":
                html = (
                    (ROOT / "tools/live.html").read_text()
                    .replace("VIEWER_TOKEN", json.dumps(token))
                    .replace("VIEWER_AUDIO", "true" if args.mode == "av" else "false")
                )
                data = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src blob:; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path != "/stream" or not secrets.compare_digest(self.headers.get("X-Kidi-Viewer", ""), token):
                self.send_error(403)
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{expected_host}":
                self.send_error(403)
                return
            if not lock.acquire(blocking=False):
                self.send_error(409, "A preview is already active")
                return
            source = None
            response_started = False
            try:
                source = open_source(args)
                print(f"Live preview connected over {source.transport.upper()}.", flush=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                response_started = True
                for header, payload in packets(source, args.resolution):
                    self.wfile.write(header + payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("Preview stopped; closing accessory stream.", flush=True)
            except (wifi.SetupError, OSError, ValueError, KeyError, http.client.HTTPException) as error:
                print(f"Preview error: {error}", file=sys.stderr, flush=True)
                if not response_started:
                    self.send_error(502, "Accessory connection failed; see terminal")
            finally:
                if source is not None:
                    source.close()
                lock.release()
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Local preview: {url}\nPress Start in the browser. No recordings are saved. Ctrl-C stops the viewer.", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("video", "av"), default="video")
    parser.add_argument("--transport", choices=("auto", "wifi", "usb"), default="auto")
    parser.add_argument("--seconds", type=int, default=300, help="Safety limit, 1-300 seconds; Start can be pressed again")
    parser.add_argument("--port", default="auto")
    parser.add_argument("--device")
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="1280x720")
    parser.add_argument("--check", action="store_true", help="Receive and validate media without opening a viewer")
    parser.add_argument("--min-fps", type=float, help="Fail a headless check below this delivered FPS")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error("--seconds must be 1 to 300")
    if args.min_fps is not None and (not args.check or not 0 < args.min_fps <= 60):
        parser.error("--min-fps requires --check and a value above 0 and at most 60")
    if args.check:
        check_stream(args)
    else:
        serve_viewer(args)


if __name__ == "__main__":
    try:
        main()
    except (wifi.SetupError, OSError, ValueError, KeyError, http.client.HTTPException) as error:
        print(f"Live client error: {error}", file=sys.stderr)
        sys.exit(1)
