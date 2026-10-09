import contextlib
import subprocess
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import struct
import wave
import os
from pathlib import Path
import ssl
import tempfile
import threading
import unittest
from unittest.mock import patch

from cryptography import x509

import wifi
import live
import speed
from urllib.parse import parse_qs, urlsplit


JPEG = (
    b"\xff\xd8\xff\xc0\x00\x11\x08" + struct.pack(">HH", 720, 1280)
    + b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xd9"
)


class FakeDevice:
    def close(self):
        pass


class WifiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kidi-wifi-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(wifi, "PRIVATE", self.root / "private").start()
        patch.object(wifi, "WIFI_CACHE", self.root / ".kidi.wifi.txt").start()
        self.profile, self.key = wifi.generate_identity()

    def test_identity_has_verified_hostname_and_no_saved_server_key(self):
        certificate = x509.load_pem_x509_certificate(self.profile["certificate"].encode())
        names = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertEqual(names.get_values_for_type(x509.DNSName), [self.profile["hostname"]])
        self.assertEqual(len(self.profile["token"]), 64)
        self.assertTrue(self.key.startswith("-----BEGIN PRIVATE KEY-----"))
        self.assertNotIn("private_key", self.profile)

    def test_private_profile_permissions(self):
        wifi.save_profile(self.profile)
        restored = wifi.load_profile(self.profile["device_id"])
        self.assertEqual(restored, self.profile)
        if os.name != "nt":
            self.assertEqual(wifi.profile_path(self.profile["device_id"]).stat().st_mode & 0o777, 0o600)

    def test_rejects_unknown_owner_and_identity_path(self):
        with self.assertRaises(wifi.SecurityError):
            wifi.load_profile(self.profile["device_id"])
        with self.assertRaises(wifi.SecurityError):
            wifi.profile_path("../other")

    def test_windows_encrypted_key_is_not_a_password(self):
        template = (
            '<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">'
            '<MSM><security><sharedKey><protected>{}</protected>'
            '<keyMaterial>test-password</keyMaterial></sharedKey></security></MSM></WLANProfile>'
        )
        self.assertEqual(wifi.password_from_windows_xml(template.format("false")), "test-password")
        with self.assertRaises(wifi.CredentialUnavailable):
            wifi.password_from_windows_xml(template.format("true"))

    def test_macos_credential_request_is_scoped_and_not_logged(self):
        response = subprocess.CompletedProcess([], 0, json.dumps({"password": "test-password"}).encode(), b"")
        output = io.StringIO()
        with (
            patch.object(wifi, "macos_helper", return_value=(self.root, self.root / "helper")),
            patch.object(wifi.subprocess, "run", return_value=response) as run,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(wifi.macos_password("selected-network"), "test-password")
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), {"ssid": "selected-network"})
        self.assertNotIn("test-password", str(run.call_args.args))
        self.assertEqual(output.getvalue(), "")

    def test_cancelled_credential_access_never_prompts_for_password(self):
        with (
            patch.object(wifi.platform, "system", return_value="Darwin"),
            patch.object(wifi, "macos_password", side_effect=wifi.CredentialCancelled("cancelled")),
            patch.object(wifi.getpass, "getpass") as prompt,
            self.assertRaises(wifi.CredentialCancelled),
        ):
            wifi.network_credentials("selected-network", authorize_credentials=True)
        prompt.assert_not_called()
        self.assertFalse(wifi.WIFI_CACHE.exists())

    def test_selected_network_cache_avoids_repeated_keychain_access(self):
        output = io.StringIO()
        with (
            patch.object(wifi.platform, "system", return_value="Darwin"),
            patch.object(wifi, "macos_password", return_value="test-password") as keychain,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(wifi.network_credentials("selected-network", authorize_credentials=True), ("selected-network", "test-password"))
            self.assertEqual(wifi.network_credentials("selected-network"), ("selected-network", "test-password"))
            keychain.assert_called_once_with("selected-network")
            wifi.network_credentials("another-network", authorize_credentials=True)
            self.assertEqual(keychain.call_count, 2)
        self.assertNotIn("test-password", output.getvalue())
        self.assertEqual(set(wifi.read_wifi_cache()), {"selected-network", "another-network"})

    def test_wifi_cache_is_private_without_changing_project_permissions(self):
        if os.name == "nt":
            self.skipTest("POSIX file mode check")
        self.root.chmod(0o755)
        wifi.cache_wifi_credentials("selected-network", "test-password")
        self.assertEqual(wifi.WIFI_CACHE.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o755)
        wifi.WIFI_CACHE.chmod(0o644)
        with self.assertRaises(wifi.SecurityError):
            wifi.read_wifi_cache()

    def test_corrupt_cache_is_not_an_implicit_keychain_request(self):
        wifi.WIFI_CACHE.write_text("not-json")
        if os.name != "nt":
            wifi.WIFI_CACHE.chmod(0o600)
        with patch.object(wifi, "macos_password") as keychain, self.assertRaises(wifi.SecurityError):
            wifi.network_credentials("selected-network")
        keychain.assert_not_called()

    def test_reboot_reconnect_waits_without_requesting_credentials(self):
        wifi.save_profile(self.profile)
        info = {
            "ok": True, "protocol": 1, "configured": True, "device_id": self.profile["device_id"],
            "certificate": self.profile["certificate"],
            "ssid_hash": hashlib.sha256(b"selected-network").hexdigest(),
            "ready": False, "address": "0.0.0.0", "port": 8443,
        }
        with (
            patch.object(wifi, "current_network_ssid", return_value="selected-network"),
            patch.object(wifi, "usb_request", side_effect=[info, {**info, "ready": True, "address": "127.0.0.1"}]),
            patch.object(wifi.time, "sleep"),
            patch.object(wifi, "network_credentials") as credentials,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            profile = wifi.setup_usb(FakeDevice(), check_network=True)
        self.assertEqual(profile["address"], "127.0.0.1")
        credentials.assert_not_called()

    def test_normal_setup_never_opens_keychain_or_asks_for_password(self):
        with (
            patch.object(wifi, "macos_password") as keychain,
            patch.object(wifi, "windows_profile") as windows,
            patch.object(wifi, "current_network_ssid") as discovery,
            patch.object(wifi.getpass, "getpass") as prompt,
        ):
            with self.assertRaises(wifi.CredentialUnavailable):
                wifi.network_credentials("uncached-network")
            with self.assertRaises(wifi.CredentialUnavailable):
                wifi.network_credentials()
            wifi.cache_wifi_credentials("selected-network", "test-password")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(wifi.network_credentials(), ("selected-network", "test-password"))
            keychain.assert_not_called()
            windows.assert_not_called()
            discovery.assert_not_called()
            prompt.assert_not_called()

    def test_normal_configured_setup_does_not_use_os_credential_or_discovery_apis(self):
        wifi.save_profile(self.profile)
        info = {
            "ok": True, "protocol": 1, "configured": True,
            "device_id": self.profile["device_id"], "certificate": self.profile["certificate"],
            "ready": True, "address": "127.0.0.1", "port": 8443,
        }
        with (
            patch.object(wifi, "usb_request", return_value=info),
            patch.object(wifi, "current_network_ssid") as discovery,
            patch.object(wifi, "macos_password") as keychain,
        ):
            self.assertEqual(wifi.setup_usb(FakeDevice(), check_network=True)["address"], "127.0.0.1")
        discovery.assert_not_called()
        keychain.assert_not_called()

    def test_owned_device_certificate_mismatch_fails_closed(self):
        wifi.save_profile(self.profile)
        info = {
            "ok": True, "protocol": 1, "configured": True,
            "device_id": self.profile["device_id"], "certificate": "different certificate",
        }
        with patch.object(wifi, "usb_request", return_value=info), self.assertRaises(wifi.SecurityError):
            wifi.setup_usb(FakeDevice())

    def run_photo(self, transport, error=None, usb_available=True):
        output = self.root / "photo.jpg"
        arguments = ["wifi.py", "photo", "--transport", transport, "--output", str(output)]
        serial_capture = patch.object(wifi.capture, "capture", return_value=("jpeg", JPEG)).start()
        patch.object(wifi.sys, "argv", arguments).start()
        patch.object(wifi, "choose_port", return_value="test-port").start()
        if usb_available:
            patch.object(wifi, "open_serial", return_value=FakeDevice()).start()
        else:
            patch.object(wifi, "open_serial", side_effect=OSError("no USB")).start()
        patch.object(wifi, "setup_usb", return_value=self.profile).start()
        patch.object(wifi, "cached_profile", return_value=self.profile).start()
        network_photo = patch.object(wifi, "wifi_photo", side_effect=error, return_value=JPEG).start()
        logs = io.StringIO()
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            wifi.main()
        return output, logs.getvalue(), serial_capture, network_photo

    def test_auto_prefers_wifi(self):
        output, logs, serial_capture, network_photo = self.run_photo("auto")
        self.assertEqual(output.read_bytes(), JPEG)
        self.assertIn("WIFI photo", logs)
        serial_capture.assert_not_called()
        network_photo.assert_called_once()

    def test_unavailable_wifi_has_explicit_usb_fallback(self):
        output, logs, serial_capture, _ = self.run_photo("auto", wifi.WifiUnavailable("no route"))
        self.assertEqual(output.read_bytes(), JPEG)
        self.assertIn("USB development fallback", logs)
        self.assertIn("USB photo", logs)
        serial_capture.assert_called_once()

    def test_forced_usb_does_not_use_network(self):
        output, logs, serial_capture, network_photo = self.run_photo("usb")
        self.assertEqual(output.read_bytes(), JPEG)
        self.assertIn("USB photo", logs)
        serial_capture.assert_called_once()
        network_photo.assert_not_called()

    def test_wifi_can_capture_without_usb(self):
        output, logs, serial_capture, network_photo = self.run_photo("wifi", usb_available=False)
        self.assertEqual(output.read_bytes(), JPEG)
        self.assertIn("WIFI photo", logs)
        serial_capture.assert_not_called()
        network_photo.assert_called_once()

    def test_authentication_failure_never_falls_back(self):
        with self.assertRaises(wifi.SecurityError):
            self.run_photo("auto", wifi.SecurityError("bad identity"))
        wifi.capture.capture.assert_not_called()
        self.assertFalse((self.root / "photo.jpg").exists())

    def test_interrupted_transfer_never_starts_another_usb_capture(self):
        with self.assertRaises(OSError):
            self.run_photo("auto", OSError("transfer interrupted"))
        wifi.capture.capture.assert_not_called()

    def test_pending_setup_is_not_a_wifi_endpoint(self):
        wifi.save_profile(self.profile)
        with self.assertRaises(wifi.SetupError):
            wifi.cached_profile()

    def test_clip_route_selection_and_fail_closed_behavior(self):
        for kind in ("audio",):
            for transport in ("auto", "wifi", "usb"):
                with self.subTest(kind=kind, transport=transport):
                    output = self.root / f"{kind}-{transport}"
                    arguments = ["wifi.py", kind, "--transport", transport, "--seconds", "3", "--output", str(output)]
                    with (
                        patch.object(wifi.sys, "argv", arguments),
                        patch.object(wifi, "cached_profile", return_value=self.profile),
                        patch.object(wifi, "choose_port", return_value="test-port"),
                        patch.object(wifi, "open_serial", return_value=FakeDevice()) as serial_open,
                        patch.object(wifi, "usb_capture") as usb,
                        patch.object(wifi, "wifi_clip") as wireless,
                    ):
                        wifi.main()
                    if transport == "usb":
                        usb.assert_called_once()
                        wireless.assert_not_called()
                    else:
                        wireless.assert_called_once_with(self.profile, kind, 3, output)
                        usb.assert_not_called()
                        serial_open.assert_not_called()
            for error in (wifi.SecurityError("untrusted"), OSError("transfer interrupted")):
                with self.subTest(kind=kind, error=type(error).__name__):
                    with (
                        patch.object(wifi.sys, "argv", ["wifi.py", kind, "--output", str(self.root / "failed")]),
                        patch.object(wifi, "cached_profile", return_value=self.profile),
                        patch.object(wifi, "wifi_clip", side_effect=error),
                        patch.object(wifi, "usb_capture") as usb,
                        self.assertRaises(type(error)),
                    ):
                        wifi.main()
                    usb.assert_not_called()

    def test_clip_outage_fallback_is_reported(self):
        for kind in ("audio",):
            with self.subTest(kind=kind):
                logs = io.StringIO()
                with (
                    patch.object(wifi.sys, "argv", ["wifi.py", kind, "--output", str(self.root / kind)]),
                    patch.object(wifi, "cached_profile", return_value=self.profile),
                    patch.object(wifi, "choose_port", return_value="test-port"),
                    patch.object(wifi, "open_serial", return_value=FakeDevice()),
                    patch.object(wifi, "setup_usb", return_value=self.profile),
                    patch.object(wifi, "wifi_clip", side_effect=wifi.WifiUnavailable("unreachable")),
                    patch.object(wifi, "usb_capture") as usb,
                    contextlib.redirect_stderr(logs),
                ):
                    wifi.main()
                usb.assert_called_once()
                self.assertIn("USB development fallback", logs.getvalue())

    def test_invalid_clip_duration_does_not_connect(self):
        for seconds in ("0", "11"):
            with (
                patch.object(wifi.sys, "argv", ["wifi.py", "audio", "--seconds", seconds, "--output", str(self.root / "invalid")]),
                patch.object(wifi, "cached_profile") as load,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                wifi.main()
            load.assert_not_called()

    def test_save_refuses_existing_file_and_invalid_jpeg(self):
        output = self.root / "photo.jpg"
        with self.assertRaises(wifi.SetupError):
            wifi.save_photo(output, b"not a JPEG", "wifi")
        output.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            wifi.save_photo(output, JPEG, "wifi")
        self.assertEqual(output.read_bytes(), b"original")


class TlsPhotoTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kidi-tls-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile, key = wifi.generate_identity()
        certificate_path = self.root / "server.pem"
        key_path = self.root / "server-key.pem"
        certificate_path.write_text(self.profile["certificate"])
        key_path.write_text(key)
        token = self.profile["token"]
        self.requests = []
        requests = self.requests
        self.clip_suffix = b"KIDI_CLIP_END\n"
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                authorised = self.headers.get("Authorization") == f"Bearer {token}"
                requests.append(authorised)
                if not authorised:
                    self.send_response(401)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                parameters = parse_qs(urlsplit(self.path).query)
                size = int(parameters["bytes"][0])
                data = speed.pattern(0, size)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("X-Kidi-Speed-Protocol", "1")
                self.send_header("X-Kidi-RSSI", "-50")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                requests.append(self.headers.get("Authorization") == f"Bearer {token}")
                if not requests[-1]:
                    self.send_response(401)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                length = int(self.headers["Content-Length"])
                if self.path.startswith("/v1/speed?"):
                    parameters = parse_qs(urlsplit(self.path).query)
                    size = int(parameters["bytes"][0])
                    data = self.rfile.read(length)
                    if data != speed.pattern(0, size):
                        self.send_response(400)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    response = json.dumps({
                        "protocol": 1, "bytes": size, "elapsed_us": 10000,
                        "rssi_dbm": -50, "cpu_mhz": 240,
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                parameters = json.loads(self.rfile.read(length))
                if self.path == "/v1/stream":
                    width, height = map(int, parameters.get("resolution", "1280x720").split("x"))
                    format_data = json.dumps({
                        "protocol": 1, "width": width, "height": height, "sample_rate": wifi.capture.SAMPLE_RATE,
                        "audio": parameters["mode"] == "av", "max_seconds": parameters["seconds"],
                    }).encode()
                    payload = b"KIDI_LIVE_BEGIN\n"
                    payload += live.HEADER.pack(3, 0, 0, len(format_data)) + format_data
                    image = (
                        b"\xff\xd8\xff\xc0\x00\x11\x08" + struct.pack(">HH", height, width)
                        + b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xd9"
                    )
                    payload += live.HEADER.pack(1, 0, 10000, len(image)) + image
                    if parameters["mode"] == "av":
                        pcm = b"\0\0" * (wifi.capture.SAMPLE_RATE // 50)
                        payload += live.HEADER.pack(2, 0, 0, len(pcm)) + pcm
                    end = b'{"reason":"duration_limit"}'
                    payload += live.HEADER.pack(4, 1, 1000000, len(end)) + end
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.kidi.live")
                    self.send_header("X-Kidi-Transport", "wifi")
                    self.send_header("X-Kidi-Stream-Protocol", "1")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n")
                    return
                if self.path in ("/v1/audio", "/v1/video", "/v1/av"):
                    kind = self.path.rsplit("/", 1)[1]
                    seconds = parameters["seconds"]
                    samples = seconds * wifi.capture.SAMPLE_RATE
                    pcm = struct.pack(f"<{samples}h", *(index % 200 - 100 for index in range(samples)))
                    if kind == "audio":
                        payload = (
                            f"KIDI_AUDIO_FORMAT sample_rate={wifi.capture.SAMPLE_RATE} channels=1 bits=16\n"
                            f"KIDI_BEGIN pcm16 {len(pcm)}\n"
                        ).encode() + pcm + b"\nKIDI_END\n"
                    else:
                        payload = b"".join(
                            f"KIDI_FRAME {timestamp} {len(JPEG)}\n".encode() + JPEG + b"\nKIDI_END\n"
                            for timestamp in (0, 500000)
                        )
                        payload += f"KIDI_VIDEO_END duration_us={seconds * 1000000} frames=2\n".encode()
                        if kind == "av":
                            payload += (
                                f"KIDI_AUDIO_META start_us=0 duration_us={seconds * 1000000} "
                                f"samples={samples} sample_rate={wifi.capture.SAMPLE_RATE} overruns=0 max_read_us=64000\n"
                                f"KIDI_BEGIN pcm16 {len(pcm)}\n"
                            ).encode() + pcm + b"\nKIDI_END\nKIDI_AV_END\n"
                    payload += fixture.clip_suffix
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.kidi.capture")
                    self.send_header("X-Kidi-Transport", "wifi")
                    self.send_header("X-Kidi-Capture-Protocol", "1")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for offset in range(0, len(payload), 4096):
                        chunk = payload[offset:offset + 4096]
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                    return
                width, height = parameters["resolution"].split("x")
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(JPEG)))
                self.send_header("X-Kidi-Width", width)
                self.send_header("X-Kidi-Height", height)
                self.send_header("X-Kidi-Transport", "wifi")
                self.end_headers()
                self.wfile.write(JPEG)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate_path, key_path)
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        self.profile.update(address="127.0.0.1", port=self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_authenticated_photo_over_verified_tls(self):
        self.assertEqual(wifi.wifi_photo(self.profile, "2048x1536"), JPEG)
        self.assertEqual(self.requests, [True])

    def test_connection_retains_the_response_timeout(self):
        connection = wifi.DeviceConnection(self.profile)
        try:
            connection.connect()
            self.assertEqual(connection.sock.gettimeout(), 25)
        finally:
            connection.close()

    def test_wrong_owner_is_rejected(self):
        profile = {**self.profile, "token": "0" * 64}
        with self.assertRaises(wifi.SecurityError):
            wifi.wifi_photo(profile, "2048x1536")
        self.assertEqual(self.requests, [False])

    def test_wrong_device_certificate_fails_before_http(self):
        different, _ = wifi.generate_identity()
        profile = {**self.profile, "certificate": different["certificate"]}
        with self.assertRaises(wifi.SecurityError):
            wifi.wifi_photo(profile, "2048x1536")
        self.assertEqual(self.requests, [])

    def test_audio_over_verified_tls(self):
        for kind in ("audio",):
            with self.subTest(kind=kind), contextlib.redirect_stdout(io.StringIO()):
                output = self.root / (f"{kind}.wav" if kind == "audio" else kind)
                wifi.wifi_clip(self.profile, kind, 1, output)
                if kind != "audio":
                    manifest = json.loads((output / "capture.json").read_text())
                    self.assertEqual(len(manifest["frames"]), 2)
                audio_path = output if kind == "audio" else output / "audio.wav"
                if kind != "video":
                    with wave.open(str(audio_path), "rb") as audio:
                        self.assertEqual(audio.getparams()[:4], (1, 2, wifi.capture.SAMPLE_RATE, wifi.capture.SAMPLE_RATE))
        self.assertEqual(self.requests, [True])

    def test_missing_completion_and_trailing_errors_fail(self):
        for index, suffix in enumerate((b"", b"KIDI_ERROR microphone shutdown failed\n", b"KIDI_CLIP_END\nextra")):
            for kind in ("audio",):
                with self.subTest(kind=kind, suffix=suffix):
                    self.clip_suffix = suffix
                    output = self.root / f"{kind}-invalid-{index}"
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises((ValueError, wifi.SetupError)):
                        wifi.wifi_clip(self.profile, kind, 1, output)
                    self.assertFalse((output / "capture.json").exists())
                    if kind == "audio":
                        self.assertFalse(output.exists())

    def test_wrong_owner_cannot_capture_clips(self):
        for kind in ("audio",):
            with self.subTest(kind=kind), self.assertRaises(wifi.SecurityError):
                wifi.wifi_clip({**self.profile, "token": "0" * 64}, kind, 1, self.root / kind)
        self.assertEqual(self.requests, [False])

    def test_live_protocol_over_verified_tls(self):
        source = live.open_wifi_source(self.profile, "av", 1)
        try:
            packets = list(live.packets(source))
            self.assertEqual([live.HEADER.unpack(header)[0] for header, _ in packets], [3, 1, 2, 4])
        finally:
            source.close()

    def test_live_wrong_owner_fails_closed(self):
        with self.assertRaises(wifi.SecurityError):
            live.open_wifi_source({**self.profile, "token": "0" * 64}, "av", 1)

    def test_low_resolution_live_protocol_over_verified_tls(self):
        source = live.open_wifi_source(self.profile, "av", 1, "96x96")
        try:
            packets = list(live.packets(source, "96x96"))
            self.assertEqual(live.jpeg_dimensions(packets[1][1]), (96, 96))
        finally:
            source.close()

    def test_download_and_upload_speed_protocol_over_verified_tls(self):
        for direction in ("download", "upload"):
            with self.subTest(direction=direction):
                result = speed.transfer(self.profile, direction, 8192, 640)
                self.assertEqual(result["bytes"], 8192)
                self.assertTrue(result["integrity_verified"])
                self.assertGreater(result["mbit_per_second"], 0)
                self.assertGreaterEqual(result["tls_setup_seconds"], 0)

    def test_speed_test_requires_owner_and_valid_limits(self):
        for direction in ("download", "upload"):
            with self.subTest(direction=direction), self.assertRaises(wifi.SecurityError):
                speed.transfer({**self.profile, "token": "0" * 64}, direction, 8192, 4096)
        with self.assertRaises(ValueError):
            speed.transfer(self.profile, "download", 8192, 0)


if __name__ == "__main__":
    unittest.main()
