"""Run against an installed wheel: python -m unittest discover -s tests -p python_cli_test.py."""

import os
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import kidi


class HubCliTest(unittest.TestCase):
    def test_resolution_only_after_valid_arguments(self):
        with patch("kidi.hub.resolve") as resolve:
            self.assertEqual(kidi.main(["chat", "--help", "-m", "@google/example"]), 0)
            self.assertEqual(kidi.main(["chat", "-m", "@google/example", "--not-an-option"]), 2)
            self.assertEqual(kidi.main(["inspect", "-m", "/nonexistent-kidi-test"]), 1)
            resolve.assert_not_called()
        for args, cache in [(["-m", "@google/example"], "~/.cache/kidi/model-hub"),
                            (["--model=@google/example", "-c", "custom cache"], "custom cache"),
                            (["-m=@google/example", "--cache=other"], "other")]:
            with self.subTest(args=args), patch("kidi.hub.resolve", return_value=Path("/nonexistent-kidi-test")) as resolve:
                self.assertEqual(kidi.main(["inspect", *args]), 1)
                resolve.assert_called_once_with("google/example", cache)

    def test_missing_extra(self):
        from kidi.hub import resolve

        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with self.assertRaisesRegex(RuntimeError, r"kidi\[hf\]"):
                resolve("google/example")

    @unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "requires kidi[hf]")
    def test_cached_setup_and_validation(self):
        import yaml
        from kidi.hub import resolve
        from kidi.converters.gemma4 import configure

        with tempfile.TemporaryDirectory() as root:
            cache = Path(root).resolve()
            snapshot = cache / "models--google--example" / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            original = {"model_type": "gemma4", "text_config": {"hidden_size": 4},
                        "quantization_config": {"quant_method": "gemma"}}
            (snapshot / "config.json").write_text(json.dumps(original))
            with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
                with self.assertRaisesRegex(ValueError, "Missing model.safetensors"):
                    resolve("google/example", cache)
                self.assertFalse((snapshot / "model.yaml").exists())
                (snapshot / "model.safetensors").write_bytes(b"test weights")
                (snapshot / "tokenizer.json").write_text("{}")
                (snapshot / "tokenizer_config.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "chat_template"):
                    resolve("google/example", cache)
                self.assertFalse((snapshot / "model.yaml").exists())
                (snapshot / "chat_template.jinja").write_text("{{ messages }}")
                self.assertEqual(resolve("google/example@revision", cache), snapshot)
                self.assertEqual(download.call_args.kwargs["revision"], snapshot.name)
                self.assertEqual(download.call_args.kwargs["cache_dir"], cache)
                config = snapshot / "model.yaml"
                self.assertEqual(yaml.safe_load(config.read_text())["model"]["quantization_config"],
                                 original["quantization_config"])
                self.assertEqual(yaml.safe_load(config.read_text())["decode"],
                                 {"maximum_new_tokens": 8192, "context_size": 16384})
                saved = config.read_bytes()
                self.assertEqual(resolve("google/example", cache), snapshot)
                self.assertEqual(config.read_bytes(), saved)
                with self.assertRaises(FileExistsError):
                    configure(snapshot)
                self.assertEqual(config.read_bytes(), saved)
                self.assertFalse(list(snapshot.glob(".model.yaml.*")))
                self.assertEqual((snapshot / "model.safetensors").read_bytes(), b"test weights")
                legacy = yaml.safe_load(saved)
                legacy["decode"] = {"maximum_new_tokens": 256, "context_size": 2048}
                config.write_text(yaml.safe_dump(legacy))
                resolve("google/example", cache)
                self.assertEqual(config.read_bytes(), saved)
                customized = yaml.safe_load(saved)
                customized["decode"]["maximum_new_tokens"] = 512
                config.write_text(yaml.safe_dump(customized))
                custom_bytes = config.read_bytes()
                resolve("google/example", cache)
                self.assertEqual(config.read_bytes(), custom_bytes)
                (snapshot / "model.safetensors").unlink()
                with self.assertRaisesRegex(ValueError, "Missing model.safetensors"):
                    resolve("google/example", cache)
                config.unlink()
                original["model_type"] = "unsupported"
                (snapshot / "config.json").write_text(json.dumps(original))
                download.reset_mock()
                with self.assertRaisesRegex(ValueError, "Automatic setup supports"):
                    resolve("google/example", cache)
                self.assertEqual(download.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "requires kidi[hf]")
    def test_preconfigured_package_paths(self):
        import yaml
        from kidi.hub import resolve

        with tempfile.TemporaryDirectory() as root:
            snapshot = Path(root).resolve() / ("b" * 40)
            snapshot.mkdir()
            document = {"format_version": 1, "model": {"type": "rtg_transformer_nmt"}, "decode": {},
                        "weights_file": "weights.safetensors",
                        "tokenizers": {"source": "src.json.gz", "target": "tgt.json.gz"}}
            config = snapshot / "model.yaml"
            config.write_text(yaml.safe_dump(document))
            for name in ("weights.safetensors", "src.json.gz", "tgt.json.gz"):
                (snapshot / name).write_bytes(b"fixture")
            saved = config.read_bytes()
            with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
                self.assertEqual(resolve("owner/rtg", root), snapshot)
                self.assertEqual(config.read_bytes(), saved)
                self.assertIn("src.json.gz", download.call_args.kwargs["allow_patterns"])
                for unsafe in ("../outside", "/absolute", "*.safetensors", "C:\\weights"):
                    document["weights_file"] = unsafe
                    config.write_text(yaml.safe_dump(document))
                    with self.assertRaisesRegex(ValueError, "Invalid package file path"):
                        resolve("owner/rtg", root)


class PythonCliTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("KIDI_CHAT_MODEL"), "set KIDI_CHAT_MODEL for real chat CLI checks")
    def test_interactive_chat(self):
        command = ([os.environ["KIDI_BIN"]] if "KIDI_BIN" in os.environ else [sys.executable, "-m", "kidi"])
        options = ["-m", os.environ["KIDI_CHAT_MODEL"],
                   "--backend", os.environ.get("KIDI_TEST_BACKEND", "ynnpack"),
                   "--context-size", "256", "--max-new-tokens", "16"]
        system = "Be concise."
        prompts = ["My name is Alex. Say hello.", "What is my name?"]
        result = subprocess.run(command + ["chat", *options, "--system", system, "--color", "never"],
                                input="\n".join(["/help", *prompts, "/clear", "x " * 300,
                                                 "/system Be concise.", "/multiline", "Hello", "world",
                                                 "/send", "/exit", "Not read."]) + "\n",
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("\x1b", result.stdout)
        self.assertEqual(result.stdout.count("Kidi chat"), 1)
        self.assertEqual(result.stdout.count("[load "), 1)
        self.assertRegex(result.stdout.split("You> ", 1)[0], r"\[load [0-9.]+s \| RAM ")
        self.assertEqual(result.stdout.count("Assistant> "), 3)
        self.assertIn("Error: ", result.stdout)
        self.assertIn("Conversation cleared.", result.stdout)
        summary_pattern = (r"\[(\d+)tok @ ([0-9.]+|n/a) ?tok/s \| 1st ([0-9.]+)s last ([0-9.]+)s"
                   r" \| RAM (?:[0-9.]+GiB|n/a) (?:[0-9]+%|n/a) (?:headroom|free)"
                   r" (?:[0-9.]+GiB|n/a) total\]")
        summaries = re.findall(summary_pattern, result.stdout)
        self.assertEqual(len(summaries), 3)
        for tokens, speed, first_token, total in summaries:
            self.assertGreaterEqual(int(tokens), 0)
            if speed != "n/a":
                self.assertGreater(float(speed.split()[0]), 0)
            self.assertGreaterEqual(float(total), float(first_token))
        if sys.platform in {"darwin", "linux", "win32"}:
            memory = re.findall(r"RAM ([0-9.]+)GiB ([0-9]+)% (?:headroom|free) ([0-9.]+)GiB total",
                                result.stdout)
            self.assertEqual(len(memory), 4)
            for used, percent, total in memory:
                self.assertGreater(float(used), 0)
                self.assertGreater(float(total), 0)
                self.assertGreaterEqual(float(percent), 0)
                self.assertLessEqual(float(percent), 100)
        transcript = re.sub(r"\n" + summary_pattern, "", result.stdout)
        replies = [part.split("\nYou> ", 1)[0] for part in transcript.split("Assistant> ")[1:]]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompts[0]},
                    {"role": "assistant", "content": replies[0]}, {"role": "user", "content": prompts[1]}]
        reference = subprocess.run(command + ["generate", *options, "--max-active", "1", "--queue-size", "1"],
                                   input=json.dumps({"messages": messages}) + "\n", text=True,
                                   capture_output=True, timeout=120)
        self.assertEqual(reference.returncode, 0, reference.stderr)
        self.assertEqual(replies[1], json.loads(reference.stdout)["message"]["content"])
        self.assertEqual(int(summaries[1][0]), len(json.loads(reference.stdout)["token_ids"]))
        single = subprocess.run(command + ["chat", *options[:-1], "1", "--color", "never", "--profile"],
                    input="Hi\n/exit\n", text=True, capture_output=True, timeout=120)
        self.assertEqual(single.returncode, 0, single.stderr)
        self.assertIn("n/a tok/s", single.stdout)
        self.assertIn("|decode_tokens=0|decode_ns=0", single.stderr)
        for color, expected in [("auto", False), ("always", True)]:
            with self.subTest(color=color):
                session = subprocess.run(command + ["chat", *options, "--color", color],
                                         input="", text=True, capture_output=True, timeout=120,
                                         env={**os.environ, "NO_COLOR": "1"})
                self.assertEqual(session.returncode, 0, session.stderr)
                self.assertEqual("\x1b[" in session.stdout, expected)

    @unittest.skipUnless(os.environ.get("KIDI_CHAT_MODEL") and os.name == "posix", "requires a POSIX chat terminal")
    def test_interactive_stream_and_cancel(self):
        import pty
        import select
        import signal

        command = ([os.environ["KIDI_BIN"]] if "KIDI_BIN" in os.environ else [sys.executable, "-m", "kidi"])
        command += ["chat", "-m", os.environ["KIDI_CHAT_MODEL"],
                    "--backend", os.environ.get("KIDI_TEST_BACKEND", "ynnpack"), "--context-size", "1024",
                    "--max-new-tokens", "256", "--ignore-eos", "--color", "never"]
        master, slave = pty.openpty()
        process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        os.close(slave)
        pending = b""

        def read_until(marker):
            nonlocal pending
            deadline = time.monotonic() + 120
            while marker not in pending:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, pending[-500:])
                if select.select([master], [], [], remaining)[0]:
                    pending += os.read(master, 65536)
            end = pending.index(marker) + len(marker)
            consumed, pending = pending[:end], pending[end:]
            return consumed

        try:
            startup = read_until(b"You> ")
            self.assertRegex(startup, rb"\[load [0-9.]+s \| RAM ")
            self.assertRegex(startup, rb"(?:headroom|free) (?:[0-9.]+GiB|n/a) total")
            self.assertNotIn(b"tok/s", startup)
            os.write(master, b"Count from one to a hundred.\n")
            read_until(b"Assistant> ")
            if not pending:
                self.assertTrue(select.select([master], [], [], 120)[0], "no streamed output")
                pending += os.read(master, 65536)
            self.assertTrue(pending)
            self.assertNotIn(b"You> ", pending, "reply was buffered until completion")
            os.kill(process.pid, signal.SIGINT)
            cancelled = read_until(b"You> ")
            self.assertIn(b"Cancelled", cancelled)
            self.assertNotIn(b"tok/s", cancelled)
            os.kill(process.pid, signal.SIGINT)
            read_until(b"You> ")
            os.write(master, b"/clear\n")
            self.assertIn(b"Conversation cleared.", read_until(b"You> "))
            if sys.platform == "darwin":
                import ctypes

                class Usage(ctypes.Structure):
                    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
                        (name, ctypes.c_uint64) for name in
                        ("user_time", "system_time", "package_wakeups", "interrupt_wakeups", "pageins",
                         "wired", "resident", "footprint", "start", "exit")]

                os.write(master, b"Say hello.\n")
                response = read_until(b"You> ")
                match = re.search(rb"RAM ([0-9.]+)GiB", response)
                self.assertIsNotNone(match, response)
                libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
                libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                usage = Usage()
                self.assertEqual(libproc.proc_pid_rusage(process.pid, 0, ctypes.byref(usage)), 0)
                self.assertAlmostEqual(float(match[1]), usage.footprint / 2**30, delta=0.10)
            os.write(master, b"/exit\n")
            self.assertEqual(process.wait(timeout=10), 0)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            os.close(master)

    def test_repeatable_entrypoint(self):
        self.assertEqual(kidi.main(["--version"]), 0)
        self.assertEqual(kidi.main(["--version"]), 0)
        with self.assertRaises(ValueError):
            kidi.main(["--model", "truncated\0argument"])

    def test_launchers(self):
        console = Path(sys.executable).parent / ("kidi.exe" if os.name == "nt" else "kidi")
        for arguments, expected in [(["--version"], 0), (["generate", "--help"], 0), (["chat", "--help"], 0),
                        (["generate", "--interactive"], 2), (["generate", "--system", "Hi"], 2),
                        (["chat", "--in", "missing"], 2), (["chat", "--max-active", "2"], 2),
                        (["unknown-command"], 2)]:
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