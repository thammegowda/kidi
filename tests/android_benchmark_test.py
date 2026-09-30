"""Run with python -m unittest discover -s tests -p android_benchmark_test.py."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/android"))
import lowbit
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

    def test_lowbit_npu_measures_without_prepare_library(self):
        case = ("gemma-w4-down-prefill", 4, 128, 6144, 1536)
        script = lowbit.invocation(case, "npu", runs=20, threads=4)
        prepare, measure = script.split(f" && cd {lowbit.ROOT}/runtime ")
        self.assertIn(f"{lowbit.ROOT}/prepare", prepare)
        self.assertIn("--qnn-prepare", prepare)
        self.assertIn(f"LD_LIBRARY_PATH={lowbit.ROOT}/runtime", measure)
        self.assertNotIn("prepare", measure)
        self.assertIn("--bits 4 --m 128 --k 6144 --n 1536 --runs 20", measure)
        self.assertIn("--threads 4", lowbit.invocation(case, "cpu", runs=20, threads=4))
        self.assertNotIn("--threads", lowbit.invocation(case, "gpu", runs=20, threads=4))

    def test_ffn_npu_replay_is_chunked_and_keeps_gate_failures(self):
        script = lowbit.ffn_invocation("npu", "replay", 35, 20, 4, graph_layers=18)
        prepare, measure = script.split(f" && cd {lowbit.ROOT}/runtime ")
        self.assertIn("--qnn-prepare", prepare)
        self.assertIn("--qnn-graph-layers 18", prepare)
        self.assertIn("--qnn-graph-layers 18", measure)
        self.assertIn("ffn-35-replay-g18.bin", measure)
        self.assertNotIn("KIDI_FFN_DIAGNOSE", measure)
        self.assertTrue(measure.endswith("exit $status"))
        eager = lowbit.ffn_invocation("npu", "eager", 35, 20, 4, graph_layers=18)
        self.assertNotIn("--qnn-graph-layers", eager)
        self.assertIn("KIDI_FFN_DIAGNOSE=1", eager)
        self.assertIn("--threads 4", lowbit.ffn_invocation("cpu", "eager", 35, 20, 4))

    def test_ffn_table_reports_speedup_and_replay_equivalence(self):
        base = {"backend": "gpu", "launches_per_step": 105, "steps_per_second": 40.0, "passed": True,
                "output_fingerprints": ["a", "b"]}
        results = [{**base, "mode": "eager", "median_step_ms": 20.0},
                   {**base, "mode": "replay", "launches_per_step": 1, "median_step_ms": 10.0},
                   {**base, "mode": "encode", "median_step_ms": 12.5, "output_fingerprints": ["a", "c"]}]
        repeat = {**base, "mode": "replay", "launches_per_step": 1, "median_step_ms": 12.0}
        rows = {line.split(" | ")[1]: line for line in lowbit.ffn_table(results + [repeat]).splitlines()[2:]}
        self.assertIn("| 1 | 11.000 | 10.00-12.00 | 90.9 | 1.82x | yes | pass |", rows["replay"])
        self.assertIn("| 1.60x | no | pass |", rows["encode"])
        collapsed = [{**result, "outputs_live": [False, False]} for result in results]
        rows = {line.split(" | ")[1]: line for line in lowbit.ffn_table(collapsed).splitlines()[2:]}
        self.assertIn("| n/a (zero output) |", rows["replay"])
        # A live first output is still compared when the second collapsed to zero.
        partial = [{**result, "outputs_live": [True, False]} for result in results]
        rows = {line.split(" | ")[1]: line for line in lowbit.ffn_table(partial).splitlines()[2:]}
        self.assertIn("| yes | pass |", rows["replay"])

    def test_live_check_requires_live_equal_distinct_outputs(self):
        base = {"backend": "npu", "passed": True, "inputs_distinguished": True, "outputs_live": [True, True],
                "output_fingerprints": ["a", "b"]}
        self.assertIn("| yes | yes | yes | pass |",
                      lowbit.in_place_table([{**base, "mode": "eager"}, {**base, "mode": "replay"}]))
        stale = {**base, "mode": "replay", "output_fingerprints": ["a", "a"], "inputs_distinguished": False}
        self.assertIn("FAIL", lowbit.in_place_table([{**base, "mode": "eager"}, stale]))

    def test_lowbit_result_parsing_and_table(self):
        self.assertEqual(lowbit.parse('native log\n{"median_ms": 1}\n'), {"median_ms": 1})
        with self.assertRaisesRegex(RuntimeError, "no result"):
            lowbit.parse("error: no device")
        label = lowbit.CASES[0][0]
        results = [{"label": label, "backend": "cpu", "precision": "W8A8", "m": 1500, "median_ms": 10.0},
                   {"label": label, "backend": "gpu", "precision": "W8A8", "m": 1500, "median_ms": 2.5},
                   {"label": label, "backend": "npu", "error": "unsupported"}]
        row = lowbit.table(results).splitlines()[2]
        self.assertEqual(row, f"| {label} | W8A8 | 1500 | 10.000 | 2.500 | failed | 4.00x | - |")


if __name__ == "__main__":
    unittest.main()