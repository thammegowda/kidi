from argparse import Namespace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import live
import wifi


def packet(kind, sequence, timestamp, payload):
    return live.HEADER.pack(kind, sequence, timestamp, len(payload)) + payload


FORMAT = json.dumps({
    "protocol": 1, "width": 1280, "height": 720, "sample_rate": 48000,
    "audio": True, "max_seconds": 5,
}).encode()
def jpeg(width, height):
    return (
        b"\xff\xd8\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xd9"
    )


JPEG = jpeg(1280, 720)
END = b'{"reason":"duration_limit"}'


class LiveTest(unittest.TestCase):
    def source(self, payload):
        return live.LiveSource(io.BytesIO(payload))

    def test_browser_rejects_unknown_protocol_before_media(self):
        viewer = (Path(__file__).resolve().parents[1] / "tools/live.html").read_text()
        rejection = 'if (info.protocol !== 1) throw new Error(`Unsupported accessory protocol version: ${info.protocol}.`);'
        self.assertIn(rejection, viewer)
        self.assertIn('if (!formatReceived && kind !== 3)', viewer)
        self.assertLess(viewer.index(rejection), viewer.index("canvas.width = info.width"))

    def test_camera_and_audio_packets(self):
        data = (
            packet(3, 0, 0, FORMAT)
            + packet(1, 0, 10000, JPEG)
            + packet(2, 0, 0, b"\0\0" * 960)
            + packet(2, 1, 20000, b"\0\0" * 1920)
            + packet(1, 1, 120000, JPEG)
            + packet(4, 2, 5000000, END)
        )
        packets = list(live.packets(self.source(data)))
        self.assertEqual(len(packets), 6)
        self.assertEqual(live.HEADER.unpack(packets[-1][0])[0], 4)

    def test_missing_format_and_truncated_payload(self):
        for data in (packet(1, 0, 0, JPEG), packet(3, 0, 0, FORMAT)[:-1]):
            with self.subTest(data=data), self.assertRaises(wifi.SetupError):
                list(live.packets(self.source(data)))

    def test_rejects_missing_end_marker(self):
        with self.assertRaises(wifi.SetupError):
            list(live.packets(self.source(packet(3, 0, 0, FORMAT))))

    def test_rejects_gaps_and_overruns(self):
        for bad in (
            packet(2, 1, 20000, b"\0\0" * 960),
            packet(2, 0, 0, b"\0"),
            packet(1, 1, 0, JPEG),
            packet(3, 0, 1000000, b'{"audio_dropped":1,"audio_overruns":0}'),
        ):
            with self.subTest(packet=bad), self.assertRaises(wifi.SetupError):
                list(live.packets(self.source(packet(3, 0, 0, FORMAT) + bad)))

    def test_packet_size_is_bounded(self):
        data = live.HEADER.pack(1, 0, 0, live.MAX_PAYLOAD + 1)
        with self.assertRaises(wifi.SetupError):
            list(live.packets(self.source(data)))

    def test_live_wifi_prefers_network_and_never_falls_back_on_security_error(self):
        args = Namespace(transport="auto", device=None, port="auto", mode="av", seconds=5, resolution="1280x720")
        source = self.source(b"")
        with (
            patch.object(wifi, "cached_profile", return_value={}),
            patch.object(live, "open_wifi_source", return_value=source),
            patch.object(live, "open_usb_source") as usb,
        ):
            self.assertIs(live.open_source(args), source)
        usb.assert_not_called()
        with (
            patch.object(wifi, "cached_profile", return_value={}),
            patch.object(live, "open_wifi_source", side_effect=wifi.SecurityError("bad certificate")),
            patch.object(live, "open_usb_source") as usb,
            self.assertRaises(wifi.SecurityError),
        ):
            live.open_source(args)
        usb.assert_not_called()

    def test_forced_usb_does_not_open_wifi(self):
        args = Namespace(transport="usb", device=None, port="test-port", mode="video", seconds=5, resolution="1280x720")
        with (
            patch.object(live, "open_usb_source", return_value=self.source(b"")) as usb,
            patch.object(live, "open_wifi_source") as wireless,
        ):
            live.open_source(args)
        usb.assert_called_once_with("test-port", "video", 5, "1280x720")
        wireless.assert_not_called()

    def test_unconfigured_media_backend_never_falls_back_to_usb(self):
        args = Namespace(transport="auto", device=None, port="auto", mode="video",
                         seconds=5, resolution="1280x720")
        with (
            patch.object(wifi, "cached_profile", return_value={}),
            patch.object(live, "open_wifi_source", side_effect=wifi.SetupError("HTTP 501: camera backend not configured")),
            patch.object(live, "open_usb_source") as usb,
            self.assertRaisesRegex(wifi.SetupError, "camera backend not configured"),
        ):
            live.open_source(args)
        usb.assert_not_called()

    def test_usb_stop_waits_for_acknowledgement_before_closing(self):
        class Usb(io.BytesIO):
            def __init__(self, payload):
                super().__init__(payload)
                self.commands = []

            def write(self, payload):
                self.commands.append(payload)
                return len(payload)

            def flush(self):
                pass

        stream = Usb(packet(4, 0, 1000000, b'{"reason":"stopped"}') + b"KIDI_STREAM_STOPPED\r\n")
        source = live.LiveSource(stream, transport="usb")
        source.close()
        self.assertEqual(stream.commands, [b"stream-stop\n"])
        self.assertTrue(source.ended)
        self.assertTrue(stream.closed)

    def test_lowest_resolution_has_matching_metadata_and_actual_jpeg(self):
        metadata = json.loads(FORMAT)
        metadata.update(width=96, height=96)
        image = jpeg(96, 96)
        data = (
            packet(3, 0, 0, json.dumps(metadata).encode())
            + packet(1, 0, 10000, image)
            + packet(2, 0, 0, b"\0\0" * 960)
            + packet(4, 1, 1000000, END)
        )
        result = list(live.packets(self.source(data), "96x96"))
        self.assertEqual(live.jpeg_dimensions(result[1][1]), (96, 96))

    def test_rejects_mismatched_metadata_or_jpeg_dimensions(self):
        metadata = json.loads(FORMAT)
        metadata.update(width=96, height=96)
        for initial, image in ((FORMAT, jpeg(96, 96)), (json.dumps(metadata).encode(), JPEG)):
            with self.subTest(metadata=initial), self.assertRaises(wifi.SetupError):
                list(live.packets(self.source(packet(3, 0, 0, initial) + packet(1, 0, 0, image)), "96x96"))

    def test_low_resolution_is_forwarded_to_wifi(self):
        args = Namespace(transport="wifi", device=None, port="auto", mode="av", seconds=5, resolution="96x96")
        with (
            patch.object(wifi, "cached_profile", return_value={}),
            patch.object(live, "open_wifi_source", return_value=self.source(b"")) as wireless,
        ):
            live.open_source(args)
        wireless.assert_called_once_with({}, "av", 5, "96x96")

    def test_reduced_audio_rate_keeps_packet_continuity(self):
        metadata = json.loads(FORMAT)
        metadata["sample_rate"] = 16000
        data = (
            packet(3, 0, 0, json.dumps(metadata).encode())
            + packet(2, 0, 0, b"\0\0" * 320)
            + packet(2, 1, 20000, b"\0\0" * 640)
            + packet(4, 0, 1000000, END)
        )
        self.assertEqual(len(list(live.packets(self.source(data)))), 4)

    def test_video_only_accepts_split_packets_and_rejects_audio(self):
        metadata = json.loads(FORMAT)
        metadata.update(audio=False, sample_rate=16000, target_fps=10, frame_buffers=1)
        data = (
            packet(3, 0, 0, json.dumps(metadata).encode())
            + packet(1, 0, 10000, JPEG)
            + packet(4, 1, 1000000, END)
        )

        class Fragmented(io.BytesIO):
            def read(self, size):
                return super().read(min(size, 7))

        source = live.LiveSource(Fragmented(data))
        self.assertEqual(len(list(live.packets(source))), 3)
        with self.assertRaisesRegex(wifi.SetupError, "Unexpected audio"):
            list(live.packets(self.source(
                packet(3, 0, 0, json.dumps(metadata).encode()) + packet(2, 0, 0, b"\0\0" * 320)
            )))

    def test_sdk_cleanup_diagnostic_is_framed_and_visible(self):
        diagnostic = b'{"sdk_diagnostics":"gdma_disconnect: no peripheral","sdk_diagnostics_truncated":true}'
        errors = io.StringIO()
        with patch("sys.stderr", errors):
            result = list(live.packets(self.source(
                packet(3, 0, 0, FORMAT) + packet(3, 0, 1000000, diagnostic)
                + packet(4, 0, 1000000, END)
            )))
        self.assertEqual(len(result), 3)
        self.assertIn("gdma_disconnect: no peripheral", errors.getvalue())
        self.assertIn("bounded buffer", errors.getvalue())

    def check_packets(self):
        return [
            (live.HEADER.pack(1, 0, 10000, len(JPEG)), JPEG),
            (live.HEADER.pack(1, 1, 110000, len(JPEG)), JPEG),
            (live.HEADER.pack(4, 2, 1000000, len(END)), END),
        ]

    def test_check_reports_delivered_rate_and_video_bytes(self):
        args = Namespace(mode="video", resolution="1280x720", seconds=1, min_fps=2)
        with (
            patch.object(live, "open_source", return_value=self.source(b"")),
            patch.object(live, "packets", return_value=iter(self.check_packets())),
            patch.object(live.time, "monotonic", side_effect=[0, .01, .11, 1]),
            patch("builtins.print"),
        ):
            report = live.check_stream(args)
        self.assertEqual(report["delivered_fps"], 2)
        self.assertEqual(report["video_bytes"], len(JPEG) * 2)
        self.assertEqual(report["audio_samples"], 0)
        self.assertAlmostEqual(report["max_frame_gap_seconds"], .1)
        self.assertEqual(report["device"]["reason"], "duration_limit")

    def test_check_does_not_hide_network_stalls_in_requested_duration(self):
        args = Namespace(mode="video", resolution="1280x720", seconds=1, min_fps=1)
        with (
            patch.object(live, "open_source", return_value=self.source(b"")),
            patch.object(live, "packets", return_value=iter(self.check_packets())),
            patch.object(live.time, "monotonic", side_effect=[0, .01, 5, 10]),
            patch("builtins.print"),
            self.assertRaisesRegex(wifi.SetupError, "below required"),
        ):
            live.check_stream(args)

    def test_cli_defaults_to_video_and_validates_fps_requirement(self):
        with patch("sys.argv", ["live.py", "--check"]), patch.object(live, "check_stream") as check:
            live.main()
        self.assertEqual(check.call_args.args[0].mode, "video")
        for extra in (["--min-fps", "10"], ["--check", "--min-fps", "nan"], ["--check", "--min-fps", "0"]):
            with (
                self.subTest(arguments=extra),
                patch("sys.argv", ["live.py", *extra]),
                patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                live.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
