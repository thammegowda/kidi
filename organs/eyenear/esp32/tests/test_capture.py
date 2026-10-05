import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import wave


spec = importlib.util.spec_from_file_location("w11_capture", Path(__file__).resolve().parents[1] / "capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class FakePort(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.commands = []

    def write(self, data):
        self.commands.append(data)
        return len(data)

    def flush(self):
        pass

    def open(self):
        pass


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="w11-capture-test-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "capture"
        self.pcm = struct.pack(
            f"<{capture.SAMPLE_RATE}h",
            *(index % 200 - 100 for index in range(capture.SAMPLE_RATE)),
        )
        self.jpeg = b"\xff\xd8test\xff\xd9"

    def av_data(self, samples=capture.SAMPLE_RATE, sample_rate=capture.SAMPLE_RATE, overruns=0, duration=1000000, start=0):
        frames = b"".join(
            f"KIDI_FRAME {timestamp} {len(self.jpeg)}\n".encode() + self.jpeg + b"\nKIDI_END\n"
            for timestamp in (0, 500000)
        )
        return (
            frames
            + b"KIDI_VIDEO_END duration_us=1000000 frames=2\n"
            + (
                f"KIDI_AUDIO_META start_us={start} duration_us={duration} samples={samples} sample_rate={sample_rate} "
                f"overruns={overruns} max_read_us=64000\n"
            ).encode()
            + f"KIDI_BEGIN pcm16 {len(self.pcm)}\n".encode()
            + self.pcm
            + b"\nKIDI_END\nKIDI_AV_END\n"
        )

    def record_av(self, data):
        port = FakePort(data)
        with contextlib.redirect_stdout(io.StringIO()):
            capture.capture_video(port, 1, self.output, with_audio=True)
        return port

    def test_simultaneous_capture(self):
        port = self.record_av(self.av_data())
        self.assertEqual(port.commands, [b"av 1\n"])
        manifest = json.loads((self.output / "capture.json").read_text())
        self.assertEqual(len(manifest["frames"]), 2)
        self.assertEqual(manifest["audio"]["samples"], capture.SAMPLE_RATE)
        self.assertEqual(manifest["audio"]["sample_rate"], capture.SAMPLE_RATE)
        self.assertEqual(manifest["audio"]["overruns"], 0)
        self.assertEqual(manifest["requested_duration_us"], 1000000)
        with wave.open(str(self.output / "audio.wav"), "rb") as audio:
            self.assertEqual(audio.getparams()[:4], (1, 2, capture.SAMPLE_RATE, capture.SAMPLE_RATE))
            self.assertEqual(audio.readframes(capture.SAMPLE_RATE), self.pcm)

    def test_rejects_audio_loss_and_timing_errors(self):
        cases = (
            {"samples": capture.SAMPLE_RATE - 1},
            {"sample_rate": 48000},
            {"overruns": 1},
            {"duration": 1300000},
            {"start": 1000},
        )
        for index, parameters in enumerate(cases):
            with self.subTest(parameters=parameters):
                self.output = Path(self.temporary.name) / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    self.record_av(self.av_data(**parameters))
                self.assertFalse((self.output / "capture.json").exists())

    def test_silent_video_capture(self):
        data = self.av_data().split(b"KIDI_AUDIO_META", 1)[0]
        port = FakePort(data)
        with contextlib.redirect_stdout(io.StringIO()):
            capture.capture_video(port, 1, self.output)
        self.assertEqual(port.commands, [b"video 1\n"])
        manifest = json.loads((self.output / "capture.json").read_text())
        self.assertEqual(len(manifest["frames"]), 2)
        self.assertNotIn("audio", manifest)

    def test_photo_payload(self):
        port = FakePort(
            f"KIDI_BEGIN jpeg {len(self.jpeg)}\n".encode() + self.jpeg + b"\nKIDI_END\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = capture.capture(port, "photo")
        self.assertEqual(result, ("jpeg", self.jpeg))
        self.assertEqual(port.commands, [b"photo\n"])

    def test_audio_sample_format(self):
        data = (
            f"KIDI_AUDIO_FORMAT sample_rate={capture.SAMPLE_RATE} channels=1 bits=16\n"
            f"KIDI_BEGIN pcm16 {len(self.pcm)}\n"
        ).encode() + self.pcm + b"\nKIDI_END\n"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(capture.capture(FakePort(data), "audio 1"), ("pcm16", self.pcm))
        for invalid in (b"", b"KIDI_AUDIO_FORMAT sample_rate=48000 channels=1 bits=16\n"):
            with self.subTest(format=invalid):
                payload = invalid + f"KIDI_BEGIN pcm16 {len(self.pcm)}\n".encode() + self.pcm + b"\nKIDI_END\n"
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    capture.capture(FakePort(payload), "audio 1")

    def photo_data(self, resolution):
        width, height = resolution.split("x")
        return (
            f"KIDI_PHOTO width={width} height={height}\n"
            f"KIDI_BEGIN jpeg {len(self.jpeg)}\n"
        ).encode() + self.jpeg + b"\nKIDI_END\n"

    def test_full_resolution_photo(self):
        port = FakePort(self.photo_data("2048x1536"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = capture.capture(port, "photo 2048x1536", resolution="2048x1536")
        self.assertEqual(result, ("jpeg", self.jpeg))
        self.assertEqual(port.commands, [b"photo 2048x1536\n"])

    def test_rejects_photo_resolution_mismatch(self):
        port = FakePort(self.photo_data("640x480"))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Requested 2048x1536"):
            capture.capture(port, "photo 2048x1536", resolution="2048x1536")

    def test_photo_resolution_cli(self):
        for resolution in (None, "640x480", "2048x1536"):
            with self.subTest(resolution=resolution):
                output = Path(self.temporary.name) / f"photo-{resolution}.jpg"
                port = FakePort(self.photo_data(resolution or "2048x1536"))
                arguments = ["capture.py", "photo", "--port", "test-port", "--output", str(output)]
                if resolution is not None:
                    arguments.extend(["--resolution", resolution])
                with (
                    patch.object(capture.sys, "argv", arguments),
                    patch.object(capture.serial, "Serial", return_value=port),
                    patch.object(capture.time, "sleep"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    capture.main()
                expected = f"photo {resolution or '2048x1536'}\n".encode()
                self.assertEqual(port.commands, [expected])
                self.assertEqual(output.read_bytes(), self.jpeg)

    def test_rejects_invalid_resolution_options_before_opening_serial(self):
        for kind, resolution in (("photo", "800x600"), ("video", "2048x1536")):
            with self.subTest(kind=kind, resolution=resolution):
                arguments = [
                    "capture.py", kind, "--resolution", resolution,
                    "--output", str(self.output),
                ]
                with (
                    patch.object(capture.sys, "argv", arguments),
                    patch.object(capture.serial, "Serial") as serial_port,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    capture.main()
                self.assertEqual(error.exception.code, 2)
                serial_port.assert_not_called()

    def test_temperature_response(self):
        port = FakePort(b"KIDI_TEMPERATURE celsius=42.5 cpu_mhz=240\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(capture.capture(port, "temperature"))
        self.assertEqual(port.commands, [b"temperature\n"])

    def test_sensor_snapshot_reports_unavailable_devices(self):
        snapshot = {
            "schema_version": 1, "timestamp_us": 123, "cpu_mhz": 240,
            "chip_temperature": {"available": True, "celsius": 42.5},
            "imu": {"available": True, "acceleration_g": [0, 0, 1], "gyroscope_dps": [0, 0, 0]},
            "board_temperature": {"available": False, "error": "STS35 not detected"},
            "battery_rail": {"available": True, "voltage_v": 4.1, "battery_presence": "unknown"},
        }
        port = FakePort(("KIDI_SENSORS " + json.dumps(snapshot) + "\n").encode())
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            self.assertIsNone(capture.capture(port, "sensors"))
        self.assertIn("STS35 not detected", errors.getvalue())
        self.assertEqual(port.commands, [b"sensors\n"])

    def test_rejects_payload_size_and_boundary(self):
        for length in (0, 1024 * 1024 + 1):
            with self.subTest(length=length), self.assertRaises(ValueError):
                capture.read_payload(FakePort(b""), length, time.monotonic() + 1)
        with self.assertRaises(ValueError):
            capture.read_payload(FakePort(b"abc\nBAD_END\n"), 3, time.monotonic() + 1)

    def test_maximum_duration_audio_payload(self):
        payload = self.pcm * 10
        port = FakePort(payload + b"\nKIDI_END\n")
        received = capture.read_payload(port, len(payload), time.monotonic() + 1)
        self.assertEqual(len(received), capture.SAMPLE_RATE * 10 * 2)
        self.assertEqual(received, payload)

    def test_rejects_non_increasing_frame_timestamps(self):
        data = self.av_data().replace(b"KIDI_FRAME 500000", b"KIDI_FRAME 0")
        with self.assertRaises(ValueError):
            self.record_av(data)

    def test_firmware_error_is_explicit(self):
        port = FakePort(b"KIDI_ERROR camera initialization failed\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            capture.capture(port, "photo")


if __name__ == "__main__":
    unittest.main()
