"""Validate CPU benchmark measurements without loading a model."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_benchmark", ROOT / "benchmarks/cpu/run.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class CpuBenchmarkTest(unittest.TestCase):
    def records(self):
        records = [{"stage": "runtime", "device": "cpu:0"}, {"stage": "load", "ms": 100}]
        for iteration, speed in enumerate((1, 10, 20, 30)):
            row = {name: 100 for name in BENCHMARK.METRICS}
            row.update(stage="generation", workload="short", iteration=iteration, warmup=iteration == 0,
                       decode_tps=speed, input_token_ids=[1, 2], token_ids=[3, 4, 5, 6], output_tokens=4,
                       observed_output_tokens=4, decode_tokens=3, first_token_ms=1000 if iteration == 0 else 10)
            records.append(row)
        return records

    def settings(self):
        return {"workload": "short", "warmups": 1, "output_tokens": 4}

    def test_summary_excludes_cold_run(self):
        summary = BENCHMARK.summarize(self.records(), self.settings(), 3)["short"]
        self.assertEqual(summary["decode_tps"]["median"], 20)
        self.assertEqual(summary["decode_tps"]["min"], 10)
        self.assertEqual(summary["decode_tps"]["max"], 30)
        self.assertEqual(summary["cold_first_token_ms"], 1000)
        self.assertEqual(summary["first_token_ms"]["median"], 10)

    def test_rejects_missing_or_changed_tokens(self):
        for change in ("missing", "tokens", "prompt", "count", "warmup", "nan", "device", "reuse"):
            with self.subTest(change=change):
                records = self.records()
                if change == "missing":
                    records.pop()
                elif change == "tokens":
                    records[-1]["token_ids"][0] = 99
                elif change == "prompt":
                    records[-1]["input_token_ids"] = [99]
                elif change == "count":
                    records[-1]["observed_output_tokens"] = 3
                elif change == "warmup":
                    records[-1]["warmup"] = True
                elif change == "nan":
                    records[-1]["decode_tps"] = float("nan")
                elif change == "reuse":
                    records[-1]["reused_prompt_tokens"] = 1
                else:
                    records[0]["device"] = "cuda:0"
                with self.assertRaises(ValueError):
                    BENCHMARK.summarize(records, self.settings(), 3)

    def test_comparison_checks_tokens_before_speed(self):
        baseline = BENCHMARK.summarize(self.records(), self.settings(), 3)
        current = copy.deepcopy(baseline)
        current["short"]["decode_tps"]["median"] = 40
        BENCHMARK.compare(current, baseline)
        self.assertEqual(current["short"]["baseline_decode_speedup"], 2)
        current["short"]["token_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            BENCHMARK.compare(current, baseline)

    def test_environment_forces_cpu_and_explicit_profiling(self):
        with patch.dict(os.environ, {"KIDI_ACCELERATOR": "cuda:0", "KIDI_REPLAY": "0",
                                     "KIDI_PROFILE_OPS": "sync", "KIDI_CUDA_PROFILE": "1",
                                     "KIDI_HNSW_EF": "10"}):
            environment = BENCHMARK.benchmark_environment("default", False, 1)
            self.assertEqual(environment["KIDI_ACCELERATOR"], "cpu")
            self.assertNotIn("KIDI_REPLAY", environment)
            self.assertNotIn("KIDI_PROFILE_OPS", environment)
            self.assertNotIn("KIDI_CUDA_PROFILE", environment)
            self.assertNotIn("KIDI_HNSW_EF", environment)
            environment = BENCHMARK.benchmark_environment("on", True, 2)
            self.assertEqual(environment["KIDI_REPLAY"], "1")
            self.assertEqual(environment["KIDI_PROFILE_OPS_SKIP"], "2")
            environment = BENCHMARK.benchmark_environment("default", False, 1, 256)
            self.assertEqual(environment["KIDI_HNSW_EF"], "256")

    def test_rejects_invalid_hnsw_options(self):
        for options in (["--hnsw-ef", "0"], ["--hnsw-ef", "-1"],
                        ["--dense-lm-head", "--hnsw-ef", "256"]):
            with self.subTest(options=options), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                BENCHMARK.main(options)
            self.assertEqual(error.exception.code, 2)

    def test_hnsw_settings_reject_a_missing_index(self):
        with tempfile.TemporaryDirectory() as temporary, patch("sys.stderr"), patch("sys.stdout"):
            output = Path(temporary) / "report.json"
            status = BENCHMARK.main(["--model", temporary, "--hnsw-ef", "256", "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertIn("refusing to measure a dense fallback", json.loads(output.read_text())["error"])

    def test_operator_summary_groups_shapes_without_counting_other_output(self):
        text = "\n".join((
            "native diagnostic output",
            "kidi_op|phase=decode_body|op=packed_linear|calls=2|run_ns=100|prepare_ns=20|dispatch_ns=10",
            "kidi_op|phase=decode_body|op=packed_linear|calls=3|run_ns=200|prepare_ns=0|dispatch_ns=5",
            "kidi_op|phase=prefill_body|op=attention|calls=1|run_ns=50",
        ))
        summary = BENCHMARK.operator_summary(text)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0], {"phase": "decode_body", "op": "packed_linear", "calls": 5,
                                      "run_ns": 300, "prepare_ns": 20, "dispatch_ns": 15})

    def test_rejects_overwriting_baseline(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            BENCHMARK.main(["--baseline", "same.json", "--output", "same.json"])
        self.assertEqual(error.exception.code, 2)

    def test_native_failures_save_report_and_partial_logs(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                binary = root / "benchmark"
                binary.write_bytes(b"benchmark")
                output = root / "report.json"
                result = subprocess.CompletedProcess([], 1, "partial stdout", "native failure")
                error = subprocess.TimeoutExpired("benchmark", 1, output=b"partial stdout", stderr=b"native timeout")
                with (
                    patch.object(BENCHMARK.subprocess, "run", return_value=result,
                                 side_effect=error if timeout else None),
                    patch.object(BENCHMARK.subprocess, "check_output",
                                 side_effect=lambda *args, **kwargs: "commit" if kwargs.get("text") else b"diff"),
                    patch("sys.stderr"), patch("sys.stdout"),
                ):
                    status = BENCHMARK.main(["--model", str(root), "--binary", str(binary), "--threads", "1",
                                             "--output", str(output)])
                self.assertEqual(status, 1)
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "failed")
                run = report["runs"]["1"]
                self.assertEqual(run["status"], "failed")
                self.assertEqual(Path(run["stdout"]).read_text(), "partial stdout")
                self.assertEqual(Path(run["stderr"]).read_text(), "native timeout" if timeout else "native failure")


if __name__ == "__main__":
    unittest.main()