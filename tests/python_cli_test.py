"""Run against an installed wheel: python -m unittest discover -s tests -p python_cli_test.py."""

import os
import json
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import kidi


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
        self.assertEqual(result.stdout.count("[loaded in "), 1)
        self.assertRegex(result.stdout.split("You> ", 1)[0], r"\[loaded in [0-9.]+ s \| (?:RSS|footprint) ")
        self.assertEqual(result.stdout.count("Assistant> "), 3)
        self.assertIn("Error: ", result.stdout)
        self.assertIn("Conversation cleared.", result.stdout)
        summary_pattern = (r"\[(\d+) tokens \| decode ([0-9.]+ tok/s|n/a) \| first token ([0-9.]+) s"
                   r" \| total ([0-9.]+) s \| (?:RSS|footprint) (?:[0-9.]+ GiB|n/a)"
                   r" \| RAM (?:headroom (?:[0-9]+%|n/a) \((?:[0-9.]+ GiB total|total n/a)\)"
                   r"|free (?:[0-9.]+/[0-9.]+ GiB \([0-9.]+%\)|n/a))\]")
        summaries = re.findall(summary_pattern, result.stdout)
        self.assertEqual(len(summaries), 3)
        for tokens, speed, first_token, total in summaries:
            self.assertGreaterEqual(int(tokens), 0)
            if speed != "n/a":
                self.assertGreater(float(speed.split()[0]), 0)
            self.assertGreaterEqual(float(total), float(first_token))
        if sys.platform == "darwin":
            memory = re.findall(r"footprint ([0-9.]+) GiB \| RAM headroom ([0-9]+)% \(([0-9.]+) GiB total\)",
                                result.stdout)
            self.assertEqual(len(memory), 4)
            for footprint, percent, total in memory:
                self.assertGreater(float(footprint), 0)
                self.assertGreater(float(total), 0)
                self.assertGreaterEqual(int(percent), 0)
                self.assertLessEqual(int(percent), 100)
        elif sys.platform in {"linux", "win32"}:
            memory = re.findall(r"RSS ([0-9.]+) GiB \| RAM free ([0-9.]+)/([0-9.]+) GiB \(([0-9.]+)%\)",
                                result.stdout)
            self.assertEqual(len(memory), 4)
            for resident, free, total, percent in memory:
                self.assertGreater(float(resident), 0)
                self.assertGreater(float(total), 0)
                self.assertGreaterEqual(float(free), 0)
                self.assertLessEqual(float(free), float(total))
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
        self.assertIn("decode n/a", single.stdout)
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
            self.assertRegex(startup, rb"\[loaded in [0-9.]+ s \| (?:RSS|footprint) ")
            self.assertIn(b"RAM ", startup)
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
            self.assertNotIn(b"tokens | decode", cancelled)
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
                match = re.search(rb"footprint ([0-9.]+) GiB", response)
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