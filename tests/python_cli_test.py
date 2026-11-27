"""Run against an installed wheel: python -m unittest discover -s tests -p python_cli_test.py."""

import os
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kidi


class HubCliTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_omnivoice_activation_calibration(self):
        from tools.converters.omnivoice import _activation_quantization

        edges = (
            "attention_norm",
            "qkv",
            "attention_output",
            "attention_residual",
            "ffn_norm",
            "gate_up",
            "ffn_hidden",
            "ffn_residual",
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "calibration.json"
            layers = {}
            for layer in range(2):
                layers[str(layer)] = {
                    edge: {
                        bucket: {
                            "symmetric_int8_scale": 1.0,
                            "symmetric_int8_scale_p99": 0.9,
                            "symmetric_int8_scale_p99_9": 0.8,
                            "symmetric_int8_scale_p99_99": 0.7 + layer / 10,
                        }
                        for bucket in ("early", "middle", "late")
                    }
                    for edge in edges
                }
            path.write_text(json.dumps({"format_version": 1, "layers": layers}))
            result = _activation_quantization(path, 2, "p99.99")
            self.assertEqual(result["bucket_order"], ["early", "middle", "late"])
            self.assertEqual(result["edge_order"], list(edges))
            self.assertEqual(result["scales"][0][0][0], 0.7)
            self.assertAlmostEqual(result["scales"][1][2][-1], 0.8)
            with self.assertRaisesRegex(ValueError, "expected 3"):
                _activation_quantization(path, 3, "p99.99")

    def _install_mock_hf(self, root: Path, repo_root: Path, log_path: Path) -> str:
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        script = bin_dir / "hf"
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, shutil, sys\n"
            "from pathlib import Path\n"
            f"repo_root = Path({str(repo_root)!r})\n"
            f"log_path = Path({str(log_path)!r})\n"
            "args = sys.argv[1:]\n"
            "with log_path.open('a', encoding='utf-8') as f:\n"
            "    f.write(json.dumps(args) + '\\n')\n"
            "assert args[0] == 'download', args\n"
            "repo_id = args[1]\n"
            "local_dir = Path(args[args.index('--local-dir') + 1])\n"
            "local_dir.mkdir(parents=True, exist_ok=True)\n"
            "src = repo_root / repo_id\n"
            "if not src.is_dir():\n"
            "    sys.exit(1)\n"
            "if 'config.json' in args and '--include' not in args:\n"
            "    cfg = src / 'config.json'\n"
            "    if not cfg.is_file():\n"
            "        sys.exit(1)\n"
            "    shutil.copy2(cfg, local_dir / 'config.json')\n"
            "else:\n"
            "    includes = [args[i + 1] for i, a in enumerate(args) if a == '--include']\n"
            "    for rel in includes:\n"
            "        candidate = src / rel\n"
            "        if candidate.is_file():\n"
            "            dst = local_dir / rel\n"
            "            dst.parent.mkdir(parents=True, exist_ok=True)\n"
            "            shutil.copy2(candidate, dst)\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return str(bin_dir)

    def test_resolution_only_after_valid_arguments_and_missing_hf(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            repos = base / "repos"
            log_path = base / "hf.log"
            bin_dir = self._install_mock_hf(base, repos, log_path)
            with patch.dict(os.environ, {"PATH": bin_dir}):
                self.assertEqual(kidi.main(["generate", "--help", "-m", "@google/example"]), 0)
                self.assertEqual(kidi.main(["generate", "-m", "@google/example", "--not-an-option"]), 2)
                self.assertEqual(kidi.main(["inspect", "-m", "/nonexistent-kidi-test"]), 1)
                self.assertFalse(log_path.exists())
                for args, cache_name in [
                    (["--model=@google/example", "-c", str(base / "custom cache")], base / "custom cache"),
                    (["-m=@google/example", f"--cache={base / 'other'}"], base / "other"),
                ]:
                    log_path.unlink(missing_ok=True)
                    self.assertEqual(kidi.main(["inspect", *args]), 1)
                    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(calls), 1)
                    self.assertIn(str(cache_name / "google" / "example"), calls[0])
            with patch.dict(os.environ, {"PATH": ""}):
                self.assertEqual(kidi.main(["inspect", "-m", "@google/example", "-c", str(base / "empty")]), 1)

    def test_whisper_checkpoint_stays_unchanged(self):
        for model, hidden, layers, heads, ffn, bins, vocabulary in [
            ("whisper-tiny", 384, 4, 6, 1536, 80, 51865),
            ("whisper-large-v3", 1280, 32, 20, 5120, 128, 51866),
            ("whisper-large-v3-turbo", 1280, 32, 20, 5120, 128, 51866),
        ]:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as root:
                base = Path(root).resolve()
                repos = base / "repos"
                cache = base / "cache"
                log_path = base / "hf.log"
                bin_dir = self._install_mock_hf(base, repos, log_path)
                repo_dir = repos / "openai" / model
                repo_dir.mkdir(parents=True)
                config = {
                    "model_type": "whisper",
                    "architectures": ["WhisperForConditionalGeneration"],
                    "activation_function": "gelu",
                    "scale_embedding": False,
                    "is_encoder_decoder": True,
                    "use_cache": True,
                    "activation_dropout": 0.0,
                    "attention_dropout": 0.0,
                    "decoder_layerdrop": 0.0,
                    "dropout": 0.0,
                    "encoder_layerdrop": 0.0,
                    "max_source_positions": 1500,
                    "max_target_positions": 448,
                    "d_model": hidden,
                    "encoder_layers": layers,
                    "decoder_layers": 4 if model.endswith("-turbo") else layers,
                    "encoder_attention_heads": heads,
                    "decoder_attention_heads": heads,
                    "encoder_ffn_dim": ffn,
                    "decoder_ffn_dim": ffn,
                    "num_mel_bins": bins,
                    "vocab_size": vocabulary,
                }
                (repo_dir / "config.json").write_text(json.dumps(config))
                for name in ("model.safetensors", "tokenizer.json", "preprocessor_config.json", "generation_config.json"):
                    (repo_dir / name).write_text("{}")
                with patch.dict(os.environ, {"PATH": bin_dir}):
                    # inspect downloads into <cache>/openai/<model> cleanly (then fails reading empty safetensors)
                    self.assertEqual(kidi.main(["inspect", "-m", f"@openai/{model}", "-c", str(cache)]), 1)
                    local_dir = cache / "openai" / model
                    self.assertTrue((local_dir / "config.json").is_file())
                    self.assertTrue((local_dir / "preprocessor_config.json").is_file())
                    self.assertTrue((local_dir / "generation_config.json").is_file())
                    self.assertFalse((local_dir / "model.yaml").exists())
                if model == "whisper-large-v3":
                    shutil.rmtree(cache)
                    config["num_mel_bins"] = 80
                    (repo_dir / "config.json").write_text(json.dumps(config))
                    with patch.dict(os.environ, {"PATH": bin_dir}):
                        self.assertEqual(kidi.main(["inspect", "-m", f"@openai/{model}", "-c", str(cache)]), 1)

    def test_whisper_window_validation(self):
        import wave

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "speech.wav"
            for rate in (8000, 16000):
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\0\0" * 100)
                original = path.read_bytes()
                base = ["transcribe", "-m", "/nonexistent-kidi-test", "--in", str(path)]
                for option, value in [("--chunk-seconds", "0"), ("--chunk-seconds", "31"),
                                      ("--max-new-tokens", "0"), ("--max-new-tokens", "445")]:
                    self.assertEqual(kidi.main([*base, option, value]), 2)
                if rate == 8000:
                    self.assertEqual(kidi.main(base), 2)
                else:
                    self.assertEqual(kidi.main([*base, "--out", str(path)]), 2)
                    alias = Path(root) / "alias.wav"
                    alias.hardlink_to(path)
                    self.assertEqual(kidi.main([*base, "--out", str(alias)]), 2)
                self.assertEqual(path.read_bytes(), original)

    def test_cached_setup_and_package_validation(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            repos = base / "repos"
            cache = base / "cache"
            log_path = base / "hf.log"
            bin_dir = self._install_mock_hf(base, repos, log_path)

            gemma_repo = repos / "google" / "example"
            gemma_repo.mkdir(parents=True)
            original = {
                "model_type": "gemma4",
                "text_config": {"hidden_size": 4},
                "quantization_config": {"quant_method": "gemma"},
            }
            (gemma_repo / "config.json").write_text(json.dumps(original))
            with patch.dict(os.environ, {"PATH": bin_dir}):
                # Missing model.safetensors
                self.assertEqual(kidi.main(["inspect", "-m", "@google/example", "-c", str(cache)]), 1)
                (gemma_repo / "model.safetensors").write_bytes(b"invalid safetensors")
                (gemma_repo / "tokenizer.json").write_text("{}")
                (gemma_repo / "tokenizer_config.json").write_text("{}")
                # Missing chat_template
                self.assertEqual(kidi.main(["inspect", "-m", "@google/example", "-c", str(cache)]), 1)
                (gemma_repo / "chat_template.jinja").write_text("{{ messages }}")
                log_path.unlink(missing_ok=True)
                self.assertEqual(kidi.main(["inspect", "-m", "@google/example@rev1", "-c", str(cache)]), 1)
                calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(calls), 2)
                self.assertIn("rev1", calls[0])
                self.assertFalse((cache / "google" / "example" / "model.yaml").exists())

                # Cached <cache>/google/example resolves without invoking hf at all (even with empty PATH).
                log_path.unlink(missing_ok=True)
            with patch.dict(os.environ, {"PATH": ""}):
                self.assertEqual(kidi.main(["inspect", "-m", "@google/example", "-c", str(cache)]), 1)
                self.assertFalse(log_path.exists())

            # RTG, OmniVoice, and Kokoro config.json path validation
            rtg_repo = repos / "owner" / "rtg"
            rtg_repo.mkdir(parents=True)
            rtg_doc = {
                "model_type": "rtg_transformer_nmt",
                "decode": {"beam_size": 4},
                "io": {"input_format": "moses_tokenized", "output_format": "moses_tokenized"},
                "weights_file": "weights.safetensors",
                "tokenizers": {"source": "src.json.gz", "target": "tgt.json.gz"},
            }
            for name in ("weights.safetensors", "src.json.gz", "tgt.json.gz"):
                (rtg_repo / name).write_bytes(b"fixture")
            with patch.dict(os.environ, {"PATH": bin_dir}):
                for unsafe in ("../outside", "/absolute", "*.safetensors", "C:\\weights"):
                    shutil.rmtree(cache, ignore_errors=True)
                    log_path.unlink(missing_ok=True)
                    rtg_doc["weights_file"] = unsafe
                    (rtg_repo / "config.json").write_text(json.dumps(rtg_doc))
                    self.assertEqual(kidi.main(["inspect", "-m", "@owner/rtg", "-c", str(cache)]), 1)
                    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(calls), 1)

    def test_gemma4_assistant_resolution_and_cli_speculative(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            repos = base / "repos"
            cache = base / "cache"
            log_path = base / "hf.log"
            bin_dir = self._install_mock_hf(base, repos, log_path)

            assistant_repo = repos / "google" / "gemma-4-E2B-it-assistant"
            assistant_repo.mkdir(parents=True)
            shutil.copy2(repo_root / "tests/data/gemma4-assistant/config.json", assistant_repo / "config.json")
            shutil.copy2(repo_root / "tests/data/gemma4-assistant/model.safetensors",
                         assistant_repo / "model.safetensors")

            target_dir = base / "target"
            target_dir.mkdir()
            shutil.copy2(repo_root / "tests/data/gemma4/model.safetensors", target_dir / "model.safetensors")
            doc = json.loads((repo_root / "tests/data/gemma4/config.json").read_text())
            doc["decode"] = {"context_size": 16, "maximum_new_tokens": 6}
            (target_dir / "config.json").write_text(json.dumps(doc))
            (target_dir / "tokenizer.json").write_text(json.dumps({
                "version": "1.0", "pre_tokenizer": {"type": "WhitespaceSplit"},
                "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
                "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": {
                    "<pad>": 0, "<eos>": 1, "<bos>": 2, "<turn|>": 3, "<|turn>": 4, "<unk>": 5,
                    "alpha": 6, "beta": 7, "gamma": 8, "delta": 9, "theta": 10, "zeta": 11,
                    "eta": 12, "iota": 13, "kappa": 14, "lambda": 15}},
            }))
            (target_dir / "tokenizer_config.json").write_text("{}")
            (target_dir / "chat_template.jinja").write_text(
                "<bos>{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}<|turn>model\n"
            )
            req_file = base / "req.jsonl"
            out_file = base / "out.jsonl"
            req_file.write_text('{"messages":[{"role":"user","content":"alpha beta"}],"max_tokens":4}\n')

            out_file.unlink(missing_ok=True)
            shutil.rmtree(cache, ignore_errors=True)
            log_path.unlink(missing_ok=True)
            with patch.dict(os.environ, {"PATH": bin_dir}):
                # Missing relative path without @ is treated as a local directory, not a Hub ID.
                code = kidi.main([
                    "generate", "-m", str(target_dir), "-c", str(cache),
                    "--assistant-model", "google/gemma-4-E2B-it-assistant", "--draft-tokens", "2",
                    "--ignore-eos", "-i", str(req_file), "-o", str(out_file), "--profile",
                ])
                self.assertEqual(code, 1)
                self.assertFalse(log_path.exists())

                # Existing relative directory without @ loads locally without invoking hf.
                previous_cwd = Path.cwd()
                try:
                    os.chdir(repos)
                    code = kidi.main([
                        "generate", "-m", str(target_dir), "-c", str(cache),
                        "--assistant-model", "google/gemma-4-E2B-it-assistant", "--draft-tokens", "2",
                        "--ignore-eos", "-i", str(req_file), "-o", str(out_file), "--profile",
                    ])
                finally:
                    os.chdir(previous_cwd)
                self.assertEqual(code, 0)
                self.assertFalse(log_path.exists())

                # @org/id resolves through the Hub.
                out_file.unlink(missing_ok=True)
                code = kidi.main([
                    "generate", "-m", str(target_dir), "-c", str(cache),
                    "--assistant-model", "@google/gemma-4-E2B-it-assistant", "--draft-tokens", "2",
                    "--ignore-eos", "-i", str(req_file), "-o", str(out_file), "--profile",
                ])
                self.assertEqual(code, 0)
                self.assertTrue((cache / "google" / "gemma-4-E2B-it-assistant" / "config.json").is_file())
                self.assertFalse((cache / "google" / "gemma-4-E2B-it-assistant" / "model.yaml").exists())
                row = json.loads(out_file.read_text().strip())
                self.assertTrue(row["speculative"])
                self.assertGreater(row["verification_steps"], 0)


class PythonCliTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("KIDI_WHISPER_MODEL") and os.environ.get("KIDI_WHISPER_WAV"),
                         "set KIDI_WHISPER_MODEL and KIDI_WHISPER_WAV to an English PCM16 mono 16 kHz fixture")
    def test_whisper_long_file(self):
        import wave

        command = ([os.environ["KIDI_BIN"]] if "KIDI_BIN" in os.environ else [sys.executable, "-m", "kidi"])
        command += ["transcribe", "-m", os.environ["KIDI_WHISPER_MODEL"], "--language", "en",
                    "--backend", os.environ.get("KIDI_TEST_BACKEND", "ynnpack")]
        source = os.environ["KIDI_WHISPER_WAV"]
        with wave.open(source, "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            frames = audio.readframes(audio.getnframes())
        self.assertGreater(len(frames), 0)
        self.assertLessEqual(len(frames), 30 * 16000 * 2)
        baseline = subprocess.run(command + ["--in", source, "--max-new-tokens", "444"],
                                  text=True, capture_output=True, timeout=180)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertTrue(baseline.stdout.strip())
        with tempfile.TemporaryDirectory() as root:
            recording = Path(root) / "long.wav"
            window = frames + bytes(30 * 16000 * 2 - len(frames))
            with wave.open(str(recording), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(window + window + frames)
            transcript = Path(root) / "transcript.txt"
            result = subprocess.run(command + ["--in", str(recording), "--out", str(transcript),
                                                "--max-new-tokens", "444", "--profile"],
                                    text=True, capture_output=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([line.strip() for line in transcript.read_text().splitlines()],
                             [baseline.stdout.strip()] * 3)
            self.assertIn("|chunks=3", result.stderr)
            limited = Path(root) / "limited.txt"
            result = subprocess.run(command + ["--in", str(recording), "--out", str(limited),
                                                "--max-new-tokens", "1"],
                                    text=True, capture_output=True, timeout=180)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("reached --max-new-tokens", result.stderr)
            self.assertEqual(limited.read_text(), "")

    def test_repeatable_entrypoint(self):
        self.assertEqual(kidi.main(["--version"]), 0)
        self.assertEqual(kidi.main(["--version"]), 0)
        with self.assertRaises(ValueError):
            kidi.main(["--model", "truncated\0argument"])

    def test_launchers(self):
        console = Path(sys.executable).parent / ("kidi.exe" if os.name == "nt" else "kidi")
        for arguments, expected in [(["--version"], 0), (["generate", "--help"], 0), (["serve", "--help"], 0),
                        (["translate", "--help"], 0),
                        (["generate", "--interactive"], 2), (["generate", "--system", "Hi"], 2),
                        (["generate", "--beam-size", "2"], 2), (["translate", "--max-new-tokens", "8"], 2),
                        (["serve", "--in", "missing"], 2),
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