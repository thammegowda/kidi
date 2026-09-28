"""Run with python -m unittest discover -s tests -p android_benchmark_test.py."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/android"))
import run


class AndroidBenchmarkTest(unittest.TestCase):
    def test_summary_excludes_warmup_and_preserves_counters(self):
        records = [{"stage": "load", "ms": 8, "peak_rss_kib": 10}]
        for warmup, duration in ((True, 100), (False, 1), (False, 3)):
            records.append({"stage": "transcription", "workload": "final", "warmup": warmup,
                            "wall_ms": duration, "real_time_factor": duration / 10, "peak_rss_kib": 12})
        self.assertEqual(run.summarize(records), {"final": {"wall_ms": 2, "real_time_factor": 0.2}, "load_ms": 8, "peak_rss_kib": 12})
        record = {"token_ids": [2**53 + 1, 123], "passed": False, "metric": 1.23456789}
        self.assertEqual(run.rounded(record), {"token_ids": [2**53 + 1, 123], "passed": False, "metric": 1.23457})
        self.assertEqual(record["metric"], 1.23456789)

    def test_adb_binding_and_failed_execution(self):
        success = subprocess.CompletedProcess([], 0, 'native output\n', "")
        with patch.dict("os.environ", {"ANDROID_HOME": "/sdk with spaces"}), patch.object(run.subprocess, "run", return_value=success) as execute:
            adb = run.Adb("device-serial")
            self.assertEqual(adb.shell("native command"), "native output")
            self.assertEqual(execute.call_args.args[0], ["/sdk with spaces/platform-tools/adb", "-s", "device-serial", "shell", "native command"])
            self.assertEqual(execute.call_args.kwargs["timeout"], 360)
            execute.return_value = subprocess.CompletedProcess([], 1, "", "device failure")
            with self.assertRaisesRegex(RuntimeError, "device failure"):
                adb.shell("native command")
            execute.side_effect = subprocess.TimeoutExpired("adb", 360)
            with self.assertRaises(subprocess.TimeoutExpired):
                adb.shell("native command")

    def test_stale_device_audio_is_rejected_before_benchmarking(self):
        with patch.object(run, "digest", return_value="generated-hash"), patch.object(run, "Adb") as adb:
            adb.return_value.shell.return_value = "old-hash  speech.wav"
            with self.assertRaisesRegex(RuntimeError, "Device speech.wav differs"):
                run.benchmark("device-serial", "unused-report.json")
            adb.return_value.shell.assert_called_once_with(f"sha256sum {run.ROOT}/speech.wav")


if __name__ == "__main__":
    unittest.main()