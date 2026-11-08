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


if __name__ == "__main__":
    unittest.main()