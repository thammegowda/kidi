"""Check CPU/CUDA benchmark reporting without model or GPU dependencies."""

import argparse
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cuda_benchmark", ROOT / "benchmarks/cuda/run.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)
QUICK_SPEC = importlib.util.spec_from_file_location("quick_e4b", ROOT / "benchmarks/cuda/quick_e4b.py")
QUICK = importlib.util.module_from_spec(QUICK_SPEC)
QUICK_SPEC.loader.exec_module(QUICK)


class BenchmarkInventoryTest(unittest.TestCase):
    def test_inventory_matches_device(self):
        for device in ("cpu", "cuda", "cuda:2"):
            with self.subTest(device=device):
                arguments = argparse.Namespace(
                    device=device, repeats=1, asr_quality=False, models=[],
                    cpu_reference=Path("reference.json"), output=Path("results.json"),
                )
                reference = json.dumps({"inputs": {}, "host": {}, "protocol": {}})

                def check_output(command, **kwargs):
                    if command[0] == "nvidia-smi":
                        if device == "cpu":
                            raise FileNotFoundError("nvidia-smi")
                        return "GPU inventory\n"
                    self.assertEqual(command[0], "git")
                    return "commit\n"

                with (
                    patch.object(argparse.ArgumentParser, "parse_args", return_value=arguments),
                    patch.object(Path, "read_text", return_value=reference),
                    patch.object(Path, "mkdir"),
                    patch.object(Path, "rglob", return_value=[]),
                    patch.object(Path, "iterdir", return_value=[]),
                    patch.object(Path, "write_text") as write,
                    patch.object(BENCHMARK, "digest", return_value="sha256"),
                    patch.object(BENCHMARK.subprocess, "check_output", side_effect=check_output) as command,
                ):
                    BENCHMARK.main()

                report = json.loads(write.call_args.args[0])
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["gpu_inventory"], None if device == "cpu" else "GPU inventory\n")
                inventory_calls = [call for call in command.call_args_list if call.args[0][0] == "nvidia-smi"]
                self.assertEqual(len(inventory_calls), 0 if device == "cpu" else 1)


class QuickE4BBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.case = {"name": "p2", "input_token_ids": [1, 2]}
        self.protocol = {"warmups": 1, "repeats": 2, "output_tokens": 2}
        self.rows = []
        for iteration, (prefill, decode) in enumerate(((20.0, 2.0), (100.0, 5.0), (200.0, 10.0))):
            self.rows.append({
                "stage": "generation", "iteration": iteration, "warmup": iteration == 0,
                "input_token_ids": [1, 2], "prompt_tokens": 2, "output_tokens": 2,
                "token_ids": [7, 8], "text": "test", "first_token_seconds": 2 / prefill,
                "decode_seconds": 1 / decode, "prefill_through_first_token_tps": prefill,
                "decode_interval_tps": decode,
            })

    def test_summary_excludes_warmup_and_detects_output_drift(self):
        result = QUICK.summarize_rows(self.rows, self.case, self.protocol, "kidi")
        self.assertEqual((result["prefill_tps"], result["ttft_ms"], result["decode_tps"]), (150.0, 15.0, 7.5))
        self.assertTrue(result["deterministic"])
        self.rows[-1]["token_ids"] = [7, 9]
        self.assertFalse(QUICK.summarize_rows(self.rows, self.case, self.protocol, "kidi")["deterministic"])

    def test_rejects_mismatched_prompts_and_output_lengths(self):
        for key, value, message in (
            ("input_token_ids", [1, 3], "prompt token mismatch"),
            ("token_ids", [7], "output token count mismatch"),
        ):
            with self.subTest(key=key):
                row = self.rows[1]
                original = row[key]
                row[key] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    QUICK.summarize_rows(self.rows, self.case, self.protocol, "kidi")
                row[key] = original

    def test_protocol_fingerprint_ignores_repeat_counts_but_not_workload(self):
        protocol = {**self.protocol, "workloads": [self.case]}
        changed_runs = {**protocol, "warmups": 2, "repeats": 5}
        changed_workload = {**protocol, "workloads": [{"name": "p2", "input_token_ids": [1, 3]}]}
        self.assertEqual(QUICK.protocol_fingerprint(protocol), QUICK.protocol_fingerprint(changed_runs))
        self.assertNotEqual(QUICK.protocol_fingerprint(protocol), QUICK.protocol_fingerprint(changed_workload))

    def test_reference_quality_stops_at_eos_and_marks_truncation(self):
        protocol = {
            "warmups": 0, "repeats": 1, "output_tokens": 4,
            "workloads": [{"name": "wmt0", "input_token_ids": [1, 2]},
                          {"name": "wmt1", "input_token_ids": [3, 4]},
                          {"name": "wmt2", "input_token_ids": [5, 6]}],
        }
        records = []
        for case, tokens in zip(protocol["workloads"], ([7, 106, 9, 1], [7, 8, 9, 10], [11, 1, 13, 14])):
            records.append({
                "workload": case["name"], "iteration": 0, "warmup": False,
                "input_token_ids": case["input_token_ids"], "prompt_tokens": 2, "output_tokens": 4,
                "token_ids": tokens, "device": "CUDA0", "first_token_seconds": 0.02,
                "decode_seconds": 0.3, "prefill_through_first_token_tps": 100.0,
                "decode_interval_tps": 10.0,
            })
        answers, truncated = QUICK.reference_quality_tokens(records, protocol, "llamacpp")
        self.assertEqual(answers, [[7], [7, 8, 9, 10], [11]])
        self.assertEqual(truncated, [False, True, False])

    def test_vllm_quality_validates_prompt_and_stop_reason(self):
        protocol = {"output_tokens": 4, "workloads": [{"name": "wmt0", "input_token_ids": [1, 2]},
                                                       {"name": "wmt1", "input_token_ids": [3, 4]}]}
        records = [
            {"stage": "eval_item", "workload": "wmt0", "input_token_ids": [1, 2],
             "token_ids": [7, 8], "truncated": False, "finish_reason": "stop", "stop_reason": 106},
            {"stage": "eval_item", "workload": "wmt1", "input_token_ids": [3, 4],
             "token_ids": [7, 8, 9, 10], "truncated": True, "finish_reason": "length", "stop_reason": None},
        ]
        self.assertEqual(QUICK.stopped_vllm_quality_tokens(records, protocol),
                         ([[7, 8], [7, 8, 9, 10]], [False, True]))
        records[0]["input_token_ids"] = [1, 3]
        with self.assertRaisesRegex(RuntimeError, "invalid quality output"):
            QUICK.stopped_vllm_quality_tokens(records, protocol)
        records[0]["input_token_ids"] = [1, 2]
        records[0]["stop_reason"] = 50
        with self.assertRaisesRegex(RuntimeError, "invalid quality stopping reason"):
            QUICK.stopped_vllm_quality_tokens(records, protocol)

    def test_quality_rejects_outdated_stopping_method(self):
        fixture = {"items": 1}
        fixture_hash = QUICK.fingerprint(fixture)
        quality = {
            "vllm": {"fixture_fingerprint": fixture_hash, "stopping": "explicit_stop_tokens"},
            "llamacpp": {"fixture_fingerprint": fixture_hash, "stopping": "trim_after_forced_generation"},
        }
        QUICK.validate_reference_quality(quality, fixture)
        del quality["vllm"]["stopping"]
        with self.assertRaisesRegex(RuntimeError, "vllm.*--refresh-quality"):
            QUICK.validate_reference_quality(quality, fixture)

    def test_quality_rejects_same_length_different_prompt_tokens(self):
        fixture = {"items": 1, "prompts": ["prompt"], "references": ["reference"],
                   "input_token_ids_list": [[1, 2]], "context_tokens": 1024, "max_output_tokens": 192}
        records = [
            {"stage": "runtime", "device": "cuda:0"}, {"stage": "load", "execution": "cuda"},
            {"stage": "eval_item", "id": 0, "input_token_ids": [1, 3], "prompt_tokens": 2,
             "token_ids": [7], "truncated": False},
            {"stage": "eval", "items": 1, "outputs": ["answer"], "token_ids": [[7]]},
        ]
        args = argparse.Namespace(binary="runner", warmups=1, cache="bf16", label="test")
        with patch.object(QUICK, "execute", return_value=records):
            with self.assertRaisesRegex(RuntimeError, "prompt or output mismatch"):
                QUICK.run_quality({"chunk": 1024, "threads": 8}, fixture, args, {})


if __name__ == "__main__":
    unittest.main()