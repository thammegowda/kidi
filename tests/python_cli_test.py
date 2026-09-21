"""Run against an installed wheel: python -m unittest discover -s tests -p python_cli_test.py."""

import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import kidi


class PythonCliTest(unittest.TestCase):
    def test_repeatable_entrypoint(self):
        self.assertEqual(kidi.main(["--version"]), 0)
        self.assertEqual(kidi.main(["--version"]), 0)
        with self.assertRaises(ValueError):
            kidi.main(["--model", "truncated\0argument"])

    def test_launchers(self):
        console = Path(sys.executable).parent / ("kidi.exe" if os.name == "nt" else "kidi")
        for arguments, expected in [(["--version"], 0), (["generate", "--help"], 0), (["unknown-command"], 2)]:
            with self.subTest(arguments=arguments):
                module = subprocess.run([sys.executable, "-m", "kidi", *arguments], capture_output=True, check=False)
                script = subprocess.run([str(console), *arguments], capture_output=True, check=False)
                self.assertEqual(module.returncode, expected)
                self.assertEqual((script.returncode, script.stdout, script.stderr),
                                 (module.returncode, module.stdout, module.stderr))

    @unittest.skipUnless(os.environ.get("KIDI_CHAT_MODEL"), "set KIDI_CHAT_MODEL for real chat CLI checks")
    def test_ordered_chat_lines(self):
        command = ([os.environ["KIDI_BIN"]] if "KIDI_BIN" in os.environ else [sys.executable, "-m", "kidi"])
        command += ["generate", "-m", os.environ["KIDI_CHAT_MODEL"],
                    "--backend", os.environ.get("KIDI_TEST_BACKEND", "ynnpack"),
                    "--context-size", "256", "--cache-tokens", "512", "--ignore-eos"]
        invalid_limits = subprocess.run(command + ["--max-active", "0"], input="", text=True,
                        capture_output=True, timeout=120)
        self.assertEqual(invalid_limits.returncode, 2, invalid_limits.stderr)
        requests = [
            {"id": "last-alphabetically", "max_tokens": 24, "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Name some colors."}]},
            {"id": "first-alphabetically", "max_tokens": 1, "messages": [
                {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]}]},
            {"id": 77, "max_tokens": 2, "messages": [{"role": "user", "content": "Hello\nworld"}]},
            {"max_tokens": 3, "messages": [{"role": "developer", "content": "Be concise."},
                                          {"role": "user", "content": "Name a country."}]},
        ]
        payload = "\r\n".join(json.dumps(record, ensure_ascii=False) for record in requests)
        serial = subprocess.run(command + ["--max-active", "1", "--queue-size", "1"],
                                input=payload, text=True, capture_output=True, timeout=120)
        self.assertEqual(serial.returncode, 0, serial.stderr)
        expected = [json.loads(line) for line in serial.stdout.splitlines()]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chat input.jsonl"
            destination = Path(directory) / "chat output.jsonl"
            source.write_text(payload, encoding="utf-8")
            concurrent = subprocess.run(command + ["--max-active", "2", "--queue-size", "2",
                                                   "-i", str(source), "-o", str(destination)],
                                        text=True, capture_output=True, timeout=120)
            self.assertEqual(concurrent.returncode, 0, concurrent.stderr)
            self.assertEqual(concurrent.stdout, "")
            actual = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(actual, expected)
        self.assertEqual([record["request_id"] for record in actual], [1, 2, 3, 4])
        self.assertEqual([record["id"] for record in actual], ["last-alphabetically", "first-alphabetically", 77, 4])
        self.assertTrue(all(record["message"]["role"] == "assistant" for record in actual))
        invalid = ["{bad", "", json.dumps({"messages": []}),
                   json.dumps({"messages": [{"role": "tool", "content": "x"},
                                            {"role": "user", "content": "Hi"}]}),
                   json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
                   json.dumps({"messages": [{"role": "user", "content": "Hi"}], "temperature": 0.5})]
        for line in invalid:
            with self.subTest(line=line):
                result = subprocess.run(command + ["--max-active", "1", "--queue-size", "1"],
                                        input=json.dumps(requests[1]) + "\n" + line + "\n",
                                        text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("input line 2", result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()