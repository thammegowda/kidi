#!/usr/bin/env python3
"""Measure warmed single-shape decoding and optional device-wide macOS GPU counters."""

import argparse
import json
import os
from pathlib import Path
import plistlib
import selectors
import statistics
import subprocess
import tempfile
import threading
import time


def gpu_counters():
    result = subprocess.run(
        ["ioreg", "-r", "-c", "AGXAccelerator", "-a"],
        capture_output=True, check=True, timeout=5,
    )
    devices = plistlib.loads(result.stdout)
    for device in devices:
        counters = device.get("PerformanceStatistics", {})
        if "Device Utilization %" in counters:
            return {
                key: counters[key]
                for key in ("Device Utilization %", "Renderer Utilization %", "Tiler Utilization %")
                if key in counters
            }
    raise RuntimeError("IORegistry does not expose GPU utilization counters")


def record_fields(stderr, prefix):
    lines = [line for line in stderr.splitlines() if line.startswith(prefix + "|")]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one {prefix} record")
    return dict(field.split("=", 1) for field in lines[0].split("|")[1:])


def trial(args, sentences):
    command = [
        str(args.binary.resolve()), "translate", "--model", str(args.model.resolve()),
        "--backend", args.backend, "--beam-size", "1", "--max-extra-tokens", "50",
        "--threads", "8", "--stats", "--profile",
        "--batch-size", str(args.batch_size),
        "--batch-window", str(args.batch_window),
    ]
    samples = []
    sampling_errors = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append(gpu_counters())
            except Exception as error:
                sampling_errors.append(str(error))
                return
            stop.wait(args.sample_ms / 1000)

    sampler = None
    with tempfile.TemporaryFile(mode="w+") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, bufsize=0)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + args.timeout

        def request():
            process.stdin.write(("\n".join(sentences) + "\n").encode())
            process.stdin.flush()
            response = bytearray()
            while response.count(b"\n") < len(sentences):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("translation timed out")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    stderr.seek(0)
                    raise RuntimeError("decoder exited without output: " + stderr.read())
                response.extend(chunk)
            return response.decode().splitlines()

        try:
            expected = request()
            for _ in range(args.warmups - 1):
                if request() != expected:
                    raise RuntimeError("warmup output changed for the same input")
            if args.sample_ms:
                gpu_counters()
                sampler = threading.Thread(target=sample)
                sampler.start()
            started = time.perf_counter()
            for _ in range(args.sentences // args.batch_window):
                if request() != expected:
                    raise RuntimeError("measured output changed for the same input")
            elapsed = time.perf_counter() - started
            stop.set()
            if sampler:
                sampler.join()
            process.stdin.close()
            process.wait(timeout=max(1, deadline - time.monotonic()))
            if process.returncode:
                raise RuntimeError(f"decoder exited with {process.returncode}")
            stderr.seek(0)
            log = stderr.read()
        finally:
            stop.set()
            if sampler:
                sampler.join()
            if process.poll() is None:
                process.kill()
                process.wait()
            selector.close()
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
    if sampling_errors:
        raise RuntimeError("GPU sampling failed: " + "; ".join(sampling_errors))
    metrics = record_fields(log, "kidi_metrics")
    profile = record_fields(log, "kidi_profile")
    total_batches = args.warmups + args.sentences // args.batch_window
    total_sentences = total_batches * args.batch_window
    if int(metrics["translated_items"]) != total_sentences:
        raise RuntimeError("decoder sentence count mismatch")
    total_tokens = int(metrics["target_tokens"])
    if total_tokens % total_batches:
        raise RuntimeError("repeated sentence target-token count is inconsistent")
    tokens = total_tokens // total_batches * (args.sentences // args.batch_window)
    utilization = {}
    if samples:
        for key in samples[0]:
            values = sorted(item[key] for item in samples)
            utilization[key] = {
                "mean": statistics.mean(values), "median": statistics.median(values),
                "p95": values[min(len(values) - 1, int(len(values) * 0.95))], "max": max(values),
            }
    return {
        "seconds": elapsed, "target_tokens": tokens, "target_tok_s": tokens / elapsed,
        "sentences": sentences, "translations": expected, "batch_size": args.batch_size, "samples": len(samples),
        "device_wide_gpu": utilization, "graph_compile_s": int(profile["graph_compile_ns"]) / 1e9,
        "command": command,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build-release/kidi"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("ynnpack", "mps"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="use the first batch-window nonempty lines")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-window", type=int)
    parser.add_argument("--sentences", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--sample-ms", type=int, default=0, help="0 disables device-wide sampling")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.batch_window is None:
        args.batch_window = args.batch_size
    if min(args.sentences, args.warmups, args.repetitions, args.timeout) <= 0 or args.sample_ms < 0:
        parser.error("counts and timeout must be positive; sample-ms must be nonnegative")
    if not 1 <= args.batch_size <= 256 or not args.batch_size <= args.batch_window <= 4096 or args.sentences % args.batch_window:
        parser.error("batch-size must be 1..256; window must be batch-size..4096 and divide sentences")
    with args.input.open() as stream:
        sentences = []
        for line in stream:
            if line.strip():
                sentences.append(line.strip())
            if len(sentences) == args.batch_window:
                break
    if len(sentences) != args.batch_window:
        parser.error("input has fewer nonempty lines than batch-window")
    trials = []
    for _ in range(args.repetitions):
        result = trial(args, sentences)
        trials.append(result)
        print(json.dumps({"trial": result}), flush=True)
    print(json.dumps({
        "backend": args.backend, "model": str(args.model), "sample_ms": args.sample_ms,
        "batch_size": args.batch_size,
        "batch_window": args.batch_window,
        "median_target_tok_s": statistics.median(result["target_tok_s"] for result in trials),
        "protocol": "fixed source batch repeated, warmups excluded; timed IPC/tokenization/detokenization included",
    }))


if __name__ == "__main__":
    main()