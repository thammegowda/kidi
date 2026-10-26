"""Run pinned native CPU/CUDA model baselines against a recorded CPU reference."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import wave


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records_from(path):
    records = []
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))
        elif line.startswith("kidi_tts_bench|"):
            fields = dict(field.split("=", 1) for field in line.split("|")[1:])
            record = {key: value if key == "backend" else float(value) if key == "rtf" else int(value)
                      for key, value in fields.items()}
            record["warmup"] = record["repetition"] == 0
            records.append(record)
    return records


def summarize(label, records, reference, repeats):
    summary = {}
    workloads = ("short", "long") if label.startswith("gemma") else ("draft_2s", "final") if label.startswith("whisper") else ("synthesis",)
    for workload in workloads:
        rows = [row for row in records if not row.get("warmup", True)
                and (row.get("workload") == workload or workload == "synthesis")]
        if len(rows) != repeats:
            raise RuntimeError(f"{label}/{workload}: expected {repeats} measured runs, got {len(rows)}")
        expected_rows = [row for row in reference["records"] if not row.get("warmup", True)
                         and (row.get("workload") == workload or workload == "synthesis")]
        identity = "token_hash" if workload == "synthesis" else "token_ids"
        if any(row[identity] != rows[0][identity] for row in rows):
            raise RuntimeError(f"{label}/{workload}: repeated benchmark outputs changed")
        metrics = ("total_ns", "generation_ns", "decode_ns", "preparation_ns", "rtf") if workload == "synthesis" else (
            ("prefill_tps", "decode_tps", "first_token_ms", "generation_ms") if label.startswith("gemma")
            else ("wall_ms", "real_time_factor", "encode_ms", "decode_ms"))
        result = {key: median(row[key] for row in rows) for key in metrics}
        result.update({"warm_repeat_count": len(rows), "deterministic_tokens": True,
                       "exact_cpu_tokens": rows[0][identity] == expected_rows[0][identity]})
        if workload == "synthesis":
            result["audio_seconds"] = rows[0]["samples"] / rows[0]["sample_rate"]
            result["cpu_speedup"] = reference["summary"][workload]["total_ns"] / result["total_ns"]
        else:
            result["text"] = rows[0]["text"]
            if label.startswith("gemma"):
                text = result["text"].lower()
                if "binary" not in text or "search" not in text or len(set(rows[0]["token_ids"])) < 12:
                    raise RuntimeError("Gemma output failed the task/repetition sanity check")
                result["prefill_speedup"] = result["prefill_tps"] / reference["summary"][workload]["prefill_tps"]
                result["decode_speedup"] = result["decode_tps"] / reference["summary"][workload]["decode_tps"]
            else:
                if not result["exact_cpu_tokens"]:
                    raise RuntimeError("Whisper transcription differs from the pinned CPU fixture")
                result["cpu_speedup"] = reference["summary"][workload]["wall_ms"] / result["wall_ms"]
        summary[workload] = result
    return summary


def waveform_quality(path, reference_text, whisper, work, asr):
    import numpy as np

    with wave.open(str(path)) as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise RuntimeError("TTS benchmark did not produce mono PCM16 audio")
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    quality = {"sample_rate": rate, "samples": len(samples), "rms": float(np.sqrt(np.mean(samples * samples))),
               "finite": bool(np.isfinite(samples).all()), "waveform_sha256": digest(path)}
    if not quality["finite"] or quality["rms"] < 0.001:
        raise RuntimeError("Synthesis produced non-finite or silent audio")
    if asr:
        from scipy.signal import resample_poly

        audio_16k = resample_poly(samples, 2, 3) if rate == 24000 else samples
        if rate not in (16000, 24000):
            raise RuntimeError("unsupported TTS sample rate for the quality proxy")
        converted = work / "omnivoice-asr-16k.wav"
        with wave.open(str(converted), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(np.clip(np.rint(audio_16k * 32768), -32768, 32767).astype("<i2").tobytes())
        result = subprocess.run([str(ROOT / "build-release/kidi"), "transcribe", "--model", whisper,
                                 "--backend", "ynnpack", "--threads", "8", "--language", "en", "--in", str(converted)],
                                capture_output=True, text=True, check=True)
        quality["asr_transcript"] = result.stdout.strip()
        source = re.findall(r"[a-z0-9]+", reference_text.lower())
        target = re.findall(r"[a-z0-9]+", result.stdout.lower())
        previous = list(range(len(target) + 1))
        for row, word in enumerate(source, 1):
            current = [row]
            for column, candidate in enumerate(target, 1):
                current.append(min(current[-1] + 1, previous[column] + 1,
                                   previous[column - 1] + (word != candidate)))
            previous = current
        quality["asr_word_error_rate"] = previous[-1] / max(len(source), 1)
        quality["asr_proxy_pass"] = quality["asr_word_error_rate"] <= 0.5
        if not quality["asr_proxy_pass"]:
            raise RuntimeError(f"TTS ASR proxy failed: {quality}")
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference", type=Path, default=ROOT / "benchmarks/cpu/reference.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/benchmarks/native-results.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", choices=("gemma", "whisper", "omnivoice"), default=["gemma", "whisper", "omnivoice"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--asr-quality", action="store_true")
    arguments = parser.parse_args()
    cpu = arguments.device == "cpu"
    if (not cpu and not re.fullmatch(r"cuda(?::[0-9]+)?", arguments.device)) or not 1 <= arguments.repeats <= 20:
        parser.error("expected cpu or cuda[:INDEX] and 1-20 measured repeats")
    if arguments.asr_quality:
        from scipy.signal import resample_poly
        if not callable(resample_poly):
            parser.error("SciPy resample_poly is required for the ASR proxy")

    reference = json.loads(arguments.cpu_reference.read_text())
    inputs = reference["inputs"]
    work = ROOT / ".cache" / ("cpu-baseline" if cpu else "cuda-baseline") / arguments.output.stem
    work.mkdir(parents=True, exist_ok=True)
    report = {"version": 1, "started": datetime.now(UTC).isoformat(), "status": "running",
              "device": arguments.device, "cpu_reference": str(arguments.cpu_reference),
              "inputs": inputs, "host": reference["host"], "protocol": dict(reference["protocol"]), "runs": {}}
    report["protocol"].update({"backend": arguments.device, "measured_repeats": arguments.repeats,
                                "kv_cache_note": "CPU auto policy" if cpu else "CUDA FP32; CPU reference retains auto policy"})
    report["gpu_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap", "--format=csv"], text=True)
    report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    report["source"] = {"kidi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                         "ynnpack_commit": subprocess.check_output(["git", "-C", "third_party/ynnpack-dev", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    paths = [ROOT / "CMakeLists.txt", Path(__file__).resolve(), ROOT / "benchmarks/android/baseline.cpp", ROOT / "benchmarks/tts/synthesis.cpp"]
    paths.extend((ROOT / "src/kidi").rglob("*.cpp"))
    paths.extend((ROOT / "src/kidi").rglob("*.h"))
    paths.extend(path for path in (ROOT / "third_party/ynnpack-dev/ynnpack/cuda").iterdir() if path.is_file())
    paths.extend([ROOT / "third_party/ynnpack-dev/ynnpack/include/ynnpack_cuda.h", ROOT / "third_party/ynnpack-dev/ynnpack/subgraph/dot.cc"])
    report["source"]["file_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))}
    report["source"]["binary_sha256"] = {name: digest(ROOT / "build-release" / name) for name in ("kidi", "kidi_android_baseline", "kidi_tts_bench")}
    report["memory_note"] = "GPU peaks include runtime-owned allocations only, excluding driver/context and cuBLAS-private allocations."
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    for model in arguments.models:
        label = {"gemma": "gemma4-e4b-qat", "whisper": "whisper-small-w8a8", "omnivoice": "omnivoice-q8"}[model]
        directory = inputs["models"][label]["directory"]
        environment = dict(os.environ, KIDI_ACCELERATOR=arguments.device)
        if model == "omnivoice":
            settings = reference["protocol"]["omnivoice"]
            command = [str(ROOT / "build-release/kidi_tts_bench"), directory, arguments.device, str(arguments.repeats + 1), "8", settings["text"], str(settings["steps"]),
                       json.dumps({key: settings[key] for key in ("language", "pace", "seed", "voice")})]
            environment["KIDI_TTS_WAV_OUT"] = str(work / "omnivoice.wav")
        else:
            command = [str(ROOT / "build-release/kidi_android_baseline"), model, directory, "8", str(arguments.repeats)]
            if model == "whisper":
                command.append(inputs["audio"]["path"])
        stdout, stderr, rss = (work / f"{model}.{suffix}" for suffix in ("stdout", "stderr", "rss"))
        run = {"command": command, "environment": {key: environment[key] for key in ("KIDI_ACCELERATOR", "KIDI_TTS_WAV_OUT") if key in environment}, "status": "running"}
        report["runs"][label] = run
        save()
        print(f"Starting {label} on {arguments.device}", flush=True)
        try:
            with stdout.open("w") as output, stderr.open("w") as errors:
                with subprocess.Popen(["/usr/bin/time", "-f", "%M", "-o", str(rss), *command], stdout=subprocess.PIPE, stderr=errors, env=environment, cwd=ROOT, text=True) as process:
                    for line in process.stdout:
                        output.write(line)
                        output.flush()
                        if line.startswith("{"):
                            record = json.loads(line)
                            print(f"  {model}: {record.get('stage')} {record.get('workload', '')} iteration={record.get('iteration', '')}", flush=True)
                        elif line.startswith("kidi_tts_bench|"):
                            print(f"  {model}: synthesis repetition completed", flush=True)
                    exit_code = process.wait()
            run.update({"exit_code": exit_code, "records": records_from(stdout), "stdout": str(stdout), "stderr": str(stderr), "peak_host_rss_kib": int(rss.read_text().splitlines()[-1])})
            if exit_code:
                raise RuntimeError(stderr.read_text()[-4000:])
            run["summary"] = summarize(label, run["records"], reference["runs"][label], arguments.repeats)
            run["peak_gpu_allocated_bytes"] = max(row.get("gpu_peak_bytes", 0) for row in run["records"])
            if model == "omnivoice":
                run["quality"] = waveform_quality(work / "omnivoice.wav", settings["text"], inputs["models"]["whisper-small-w8a8"]["directory"], work, arguments.asr_quality)
            run["status"] = "completed"
            print(json.dumps({"model": label, "summary": run["summary"], "quality": run.get("quality"), "gpu_peak_gib": run["peak_gpu_allocated_bytes"] / 2**30}, indent=2), flush=True)
        except Exception as error:
            run.update({"status": "failed", "error": str(error)})
            report["status"] = "failed"
            save()
            raise
        save()
    report.update({"status": "completed", "completed": datetime.now(UTC).isoformat()})
    save()


if __name__ == "__main__":
    main()