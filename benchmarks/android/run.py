"""Run native Android CPU baselines through ADB (Python 3.12+)."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess


ROOT = "/data/local/tmp/kidi-baseline"
GEMMA_MODEL = "google/gemma-4-E2B-it-qat-mobile-transformers"
WHISPER_MODEL = "openai/whisper-tiny"
GEMMA = f"{ROOT}/models/gemma4"
WHISPER = f"{ROOT}/models/whisper-tiny"


def timestamp():
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def digest(file):
    with Path(file).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 5) if math.isfinite(value) else None
    return value


def save_report(file, report):
    Path(file).write_text(json.dumps(rounded(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def emit(record):
    print(json.dumps(rounded(record), ensure_ascii=False, allow_nan=False), flush=True)


def command(arguments, timeout=360):
    result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed ({result.returncode}): {result.stderr or result.stdout}")
    return result


class Adb:
    def __init__(self, serial):
        sdk = Path(os.environ.get("ANDROID_HOME") or Path.home() / "Library/Android/sdk")
        self.arguments = [str(sdk / "platform-tools/adb"), "-s", serial, "shell"]

    def shell(self, script, timeout=360):
        result = command([*self.arguments, script], timeout=timeout)
        return result.stdout.strip()


def snapshot(adb):
    thermal = adb.shell("dumpsys thermalservice")
    temperatures = {}
    for match in re.finditer(r"Temperature\{mValue=([\d.-]+), mType=(\d+), mName=([^,]+), mStatus=(\d+)\}", thermal):
        temperatures[match[3]] = {"celsius": float(match[1]), "type": int(match[2]), "status": int(match[4])}
    battery = adb.shell("dumpsys battery")

    def number(pattern, text, divisor=1):
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            return None
        value = float(match[1]) if "." in match[1] else int(match[1])
        return value / divisor if divisor != 1 else value

    cpus = [item["celsius"] for item in temperatures.values() if item["type"] == 0]
    return {
        "at": timestamp(),
        "thermal_status": number(r"Thermal Status: (\d+)", thermal),
        "cpu_max_celsius": max(cpus) if cpus else None,
        "skin_celsius": next((item["celsius"] for item in temperatures.values() if item["type"] == 3), None),
        "battery_celsius": number(r"^\s*temperature: ([-\d.]+)$", battery, 10),
        "battery_percent": number(r"^\s*level: ([-\d.]+)$", battery),
        "powered": bool(re.search(r"^\s*(?:AC|USB|Wireless) powered: true$", battery, re.MULTILINE)),
        "available_memory_kib": number(r"MemAvailable:\s+(\d+)", adb.shell("cat /proc/meminfo")),
    }


def summarize(records):
    summary = {}
    workloads = dict.fromkeys(item["workload"] for item in records if item.get("workload"))
    for workload in workloads:
        rows = [item for item in records if item.get("workload") == workload and not item.get("warmup")]
        metrics = ("prefill_tps", "decode_tps", "first_token_ms", "generation_ms") if workload in ("short", "long") else ("wall_ms", "real_time_factor")
        summary[workload] = {key: median(row[key] for row in rows) for key in metrics}
    summary["load_ms"] = next(item["ms"] for item in records if item["stage"] == "load")
    summary["peak_rss_kib"] = max(item["peak_rss_kib"] for item in records)
    return summary


def benchmark(serial, output):
    speech_sha256 = digest(Path(__file__).resolve().parent / ".cache/speech.wav")
    adb = Adb(serial)
    if adb.shell(f"sha256sum {ROOT}/speech.wav").split()[0] != speech_sha256:
        raise RuntimeError("Device speech.wav differs from the generated fixture; push .cache/speech.wav again")
    features = re.search(r"Features\s*:\s*(.*)", adb.shell("cat /proc/cpuinfo"))
    report = {
        "version": 1, "started": timestamp(),
        "device": {
            "model": adb.shell("getprop ro.product.model"), "soc": adb.shell("getprop ro.soc.model"),
            "android": adb.shell("getprop ro.build.version.release"), "abi": adb.shell("getprop ro.product.cpu.abilist"),
            "runner_cgroup": adb.shell("cat /proc/self/cgroup"),
        },
        "source": {
            "commit": command(["git", "rev-parse", "HEAD"]).stdout.strip(),
            "dirty_files": command(["git", "diff", "--name-only", "HEAD"]).stdout.strip().split("\n"),
            "harness_sha256": digest("benchmarks/android/baseline.cpp"),
            "binary_sha256": digest("build-android-baseline/kidi_android_baseline"),
            "ynnpack_commit": command(["git", "-C", "third_party/ynnpack-dev", "rev-parse", "HEAD"]).stdout.strip(),
        },
        "inputs": {
            "gemma_model": GEMMA_MODEL, "whisper_model": WHISPER_MODEL, "requested_revision": "main",
            "gemma_directory": GEMMA, "whisper_directory": WHISPER,
            "speech_sha256": speech_sha256,
            "weight_sha256": adb.shell(f"sha256sum {GEMMA}/model.safetensors {WHISPER}/model.safetensors").splitlines(),
        },
        "protocol": {
            "build": "NDK 28 Release (-O3 -DNDEBUG), ARM64 CPU", "threads": [4, 1, 8, 2, 6, 4],
            "warm_repeats": 3, "warmup_repeats": 1, "context_tokens": 9216, "output_tokens": 64,
            "ignore_eos": True, "prefill_chunk": 32, "packed_prefill": True, "stream_text": True,
            "note": "Shell process, not app UI. Models run separately; peak RSS is per-process cumulative. No governor/affinity changes.",
        },
        "runs": [],
    }
    if features:
        report["device"]["cpu_features"] = features[1]
    save_report(output, report)
    expected = {}
    for index, threads in enumerate(report["protocol"]["threads"]):
        for model, directory in (("gemma", GEMMA), ("whisper", WHISPER)):
            run = {"index": index, "threads": threads, "model": model, "before": snapshot(adb)}
            arguments = f"{model} {directory} {threads} 3" + (f" {ROOT}/speech.wav" if model == "whisper" else "")
            print(f"Run {index + 1}/{len(report['protocol']['threads'])}: {model}, {threads} threads", flush=True)
            try:
                run["records"] = [json.loads(line) for line in adb.shell(f"timeout 300 {ROOT}/runner {arguments}").splitlines()]
                run["after"] = snapshot(adb)
                run["summary"] = summarize(run["records"])
                run["exact_tokens"] = True
                for item in run["records"]:
                    if "token_ids" in item:
                        key = f"{model}/{item['workload']}"
                        expected.setdefault(key, item["token_ids"])
                        run["exact_tokens"] &= item["token_ids"] == expected[key]
                report["runs"].append(run)
                save_report(output, report)
                emit({"threads": threads, "model": model, **run["summary"], "exact_tokens": run["exact_tokens"]})
            except Exception as error:
                run["error"] = str(error)
                if not report["runs"] or report["runs"][-1] is not run:
                    report["runs"].append(run)
                save_report(output, report)
                raise
    report["completed"] = timestamp()
    save_report(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial")
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    benchmark(arguments.serial, arguments.output)


if __name__ == "__main__":
    main()