"""Repeat warm CPU Gemma workloads through the existing native benchmark."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median, pstdev
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "@google/gemma-4-E4B-it-qat-mobile-transformers"
METRICS = ("decode_tps", "decode_interval_tps", "prefill_tps", "first_token_ms",
           "generation_ms", "preparation_ms", "peak_rss_kib")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records_from(text):
    records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("native benchmark records must be JSON objects")
    return records


def summarize(records, settings, repeats):
    runtime = [row for row in records if row.get("stage") == "runtime"]
    loads = [row for row in records if row.get("stage") == "load"]
    if len(runtime) != 1 or runtime[0].get("device") != "cpu:0" or len(loads) != 1:
        raise ValueError("expected one CPU runtime record and one model load")
    summary = {}
    workloads = ("short", "long") if settings["workload"] == "both" else (settings["workload"],)
    for workload in workloads:
        rows = [row for row in records if row.get("stage") == "generation" and row.get("workload") == workload]
        if len(rows) != settings["warmups"] + repeats:
            raise ValueError(f"{workload}: missing warmup or measured generations")
        first = rows[0]
        for iteration, row in enumerate(rows):
            if row["iteration"] != iteration or row["warmup"] != (iteration < settings["warmups"]):
                raise ValueError(f"{workload}: invalid warmup/iteration sequence")
            if (row["output_tokens"] != settings["output_tokens"] or
                    row["observed_output_tokens"] != settings["output_tokens"] or
                    len(row["token_ids"]) != settings["output_tokens"] or
                    row["decode_tokens"] != settings["output_tokens"] - 1):
                raise ValueError(f"{workload}: incomplete generation")
            if row["token_ids"] != first["token_ids"] or row["input_token_ids"] != first["input_token_ids"]:
                raise ValueError(f"{workload}: token IDs changed between repetitions")
            if row.get("reused_prompt_tokens", 0):
                raise ValueError(f"{workload}: prompt KV cache was reused; expected independent requests")
            for name in METRICS:
                if not math.isfinite(row[name]) or row[name] < 0 or (name != "preparation_ms" and row[name] == 0):
                    raise ValueError(f"{workload}: invalid {name}")
        measured = rows[settings["warmups"]:]
        result = {"input_token_ids": first["input_token_ids"], "token_ids": first["token_ids"],
                  "cold_first_token_ms": first["first_token_ms"], "measured_repeats": repeats}
        for name in METRICS:
            values = [row[name] for row in measured]
            result[name] = {"median": median(values), "min": min(values), "max": max(values), "stdev": pstdev(values)}
        summary[workload] = result
    return summary


def compare(summary, reference):
    for workload, current in summary.items():
        previous = reference[workload]
        if (current["token_ids"] != previous["token_ids"] or
                current["input_token_ids"] != previous["input_token_ids"]):
            raise ValueError(f"{workload}: token IDs differ from baseline")
        current["baseline_decode_speedup"] = current["decode_tps"]["median"] / previous["decode_tps"]["median"]


def operator_summary(text):
    groups = {}
    for line in text.splitlines():
        fields = dict(part.split("=", 1) for part in line.split("|") if "=" in part)
        if "run_ns" not in fields or "op" not in fields:
            continue
        key = (fields.get("phase", ""), fields["op"])
        group = groups.setdefault(key, {"phase": key[0], "op": key[1], "calls": 0,
                                        "run_ns": 0, "prepare_ns": 0, "dispatch_ns": 0})
        for name in ("calls", "run_ns", "prepare_ns", "dispatch_ns"):
            group[name] += int(fields.get(name, 0))
    return sorted(groups.values(), key=lambda row: row["run_ns"], reverse=True)


def benchmark_environment(replay, profile_ops, warmups, hnsw_ef=None):
    environment = dict(os.environ, KIDI_ACCELERATOR="cpu")
    for name in ("KIDI_REPLAY", "KIDI_PROFILE_OPS", "KIDI_PROFILE_OPS_SKIP", "KIDI_CUDA_PROFILE",
                 "KIDI_HNSW_EF"):
        environment.pop(name, None)
    if replay != "default":
        environment["KIDI_REPLAY"] = "1" if replay == "on" else "0"
    if profile_ops:
        environment.update(KIDI_PROFILE_OPS="host", KIDI_PROFILE_OPS_SKIP=str(warmups))
    if hnsw_ef is not None:
        environment["KIDI_HNSW_EF"] = str(hnsw_ef)
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--packed-prefill", action="store_true")
    parser.add_argument("--workload", choices=("short", "long", "both"), default="short")
    parser.add_argument("--prompt", default="Explain how binary search works in detail.")
    parser.add_argument("--replay", choices=("default", "on", "off"), default="default")
    parser.add_argument("--profile-ops", action="store_true")
    parser.add_argument("--dense-lm-head", action="store_true", help="bypass the checkpoint HNSW index")
    parser.add_argument("--hnsw-ef", type=int, help="override HNSW search effort (otherwise use the index default)")
    parser.add_argument("--build", action="store_true", help="incrementally build the native benchmark first")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-release")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/benchmarks/cpu/latest.json")
    arguments = parser.parse_args(argv)
    if (not 1 <= arguments.repeats <= 20 or not 1 <= arguments.warmups <= 20 or
            not 2 <= arguments.tokens < arguments.context_tokens or arguments.prefill_chunk < 1 or
            arguments.jobs < 1 or not math.isfinite(arguments.timeout) or arguments.timeout <= 0 or
            any(threads < 1 or threads > (os.cpu_count() or 1) for threads in arguments.threads) or
            len(set(arguments.threads)) != len(arguments.threads) or not arguments.prompt.strip()):
        parser.error("expected positive thread/job counts, 1-20 repeats/warmups, and 2 <= tokens < context")
    if arguments.hnsw_ef is not None and arguments.hnsw_ef < 1:
        parser.error("expected positive HNSW search effort")
    if arguments.dense_lm_head and arguments.hnsw_ef is not None:
        parser.error("--dense-lm-head cannot be combined with --hnsw-ef")
    output = arguments.output.expanduser().resolve()
    if arguments.baseline and arguments.baseline.expanduser().resolve() == output:
        parser.error("baseline and output must be different files")
    settings = {"output_tokens": arguments.tokens, "context_tokens": arguments.context_tokens,
                "prefill_chunk_tokens": arguments.prefill_chunk, "packed_prefill": arguments.packed_prefill,
                "warmups": arguments.warmups, "workload": arguments.workload, "prompt": arguments.prompt}
    if arguments.dense_lm_head:
        settings["dense_lm_head"] = True
    report = {"version": 1, "started": datetime.now(timezone.utc).isoformat(), "status": "running",
              "protocol": {"settings": settings, "replay": arguments.replay, "operator_profiling": arguments.profile_ops},
              "runs": {}}
    if arguments.hnsw_ef is not None:
        report["protocol"]["hnsw"] = {"ef": arguments.hnsw_ef}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        if arguments.model.startswith("@"):
            from kidi.hub import resolve
            model = resolve(arguments.model[1:])
        else:
            model = Path(arguments.model).expanduser().resolve()
        if not model.is_dir():
            raise ValueError(f"not a model directory: {model}")
        index_path = model / "lm_head.hnsw"
        if "hnsw" in report["protocol"] and not index_path.is_file():
            raise ValueError("HNSW settings require MODEL/lm_head.hnsw; refusing to measure a dense fallback")
        report["model"] = str(model)
        reference = json.loads(arguments.baseline.expanduser().read_text()) if arguments.baseline else None
        if reference and (reference.get("status") != "completed" or reference["protocol"] != report["protocol"] or
                          reference["model"] != report["model"]):
            raise ValueError("baseline model/protocol differs; compare identical workloads and profiling modes")
        build_dir = arguments.build_dir.expanduser().resolve()
        binary = (arguments.binary or build_dir / "kidi_android_baseline").expanduser().resolve()
        if arguments.build:
            if arguments.binary:
                raise ValueError("--build and --binary cannot be combined")
            subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
                            "-DKIDI_BUILD_BENCHMARKS=ON"], check=True)
            subprocess.run(["cmake", "--build", str(build_dir), "--target", "kidi_android_baseline",
                            "-j", str(arguments.jobs)], check=True)
        if not binary.is_file():
            raise ValueError(f"missing benchmark binary: {binary}; run with --build")
        report["host"] = {"platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count(),
                          "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                          "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None}
        report["source"] = {
            "kidi_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "ynnpack_commit": subprocess.check_output(["git", "-C", "third_party/ynnpack-dev", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
            "binary": str(binary), "binary_sha256": digest(binary), "harness_sha256": digest(Path(__file__)),
        }
        if index_path.is_file() and not arguments.dense_lm_head:
            report["source"]["hnsw_index_sha256"] = digest(index_path)
            report["source"]["hnsw_index_bytes"] = index_path.stat().st_size
        if (reference and reference.get("source", {}).get("hnsw_index_sha256") !=
                report["source"].get("hnsw_index_sha256")):
            raise ValueError("baseline HNSW index differs; compare identical topology and search settings")
        cache = binary.parent / "CMakeCache.txt"
        if cache.is_file():
            names = {"CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
                     "KIDI_ENABLE_CUDA", "KIDI_ENABLE_VULKAN", "KIDI_BUILD_BENCHMARKS"}
            report["source"]["cmake"] = {}
            for line in cache.read_text().splitlines():
                declaration, separator, value = line.partition("=")
                name = declaration.partition(":")[0]
                if separator and name in names:
                    report["source"]["cmake"][name] = value
        environment = benchmark_environment(arguments.replay, arguments.profile_ops, arguments.warmups,
                                            arguments.hnsw_ef)
        report["environment"] = {key: value for key, value in environment.items() if key.startswith("KIDI_")}
        save()
        print("threads workload   decode tok/s (min..max)    warm TTFT ms   peak GiB   baseline", flush=True)
        for threads in arguments.threads:
            command = [str(binary), "gemma", str(model), str(threads), str(arguments.repeats), json.dumps(settings)]
            stdout = output.parent / f"{output.stem}-j{threads}.stdout.jsonl"
            stderr = output.parent / f"{output.stem}-j{threads}.stderr.log"
            run = {"command": command, "status": "running", "stdout": str(stdout), "stderr": str(stderr)}
            report["runs"][str(threads)] = run
            save()
            try:
                completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True,
                                           timeout=arguments.timeout)
            except subprocess.TimeoutExpired as error:
                stdout.write_text(error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or "")
                stderr.write_text(error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or "")
                raise RuntimeError(f"{threads} threads: native benchmark timed out; partial logs: {stdout}, {stderr}") from error
            stdout.write_text(completed.stdout)
            stderr.write_text(completed.stderr)
            run["exit_code"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"{threads} threads: native benchmark failed; see {stderr}\n{completed.stderr[-2000:]}")
            run["records"] = records_from(completed.stdout)
            run["summary"] = summarize(run["records"], settings, arguments.repeats)
            if arguments.profile_ops:
                run["operators"] = operator_summary(completed.stderr)
            if reference:
                if str(threads) not in reference["runs"]:
                    raise ValueError(f"baseline has no {threads}-thread run")
                compare(run["summary"], reference["runs"][str(threads)]["summary"])
            for workload, summary in run["summary"].items():
                speed = summary["decode_tps"]
                relative = f'{summary["baseline_decode_speedup"]:.3f}x' if reference else "-"
                print(f'{threads:7} {workload:8} {speed["median"]:8.2f} ({speed["min"]:.2f}..{speed["max"]:.2f})'
                      f' {summary["first_token_ms"]["median"]:13.2f}'
                      f' {summary["peak_rss_kib"]["max"] / 2**20:10.2f} {relative:>10}', flush=True)
            for operation in run.get("operators", [])[:5]:
                print(f'  {operation["phase"]}/{operation["op"]}: {operation["run_ns"] / 1e6:.2f} ms'
                      f' in {operation["calls"]} calls', flush=True)
            run["status"] = "completed"
            save()
        if digest(binary) != report["source"]["binary_sha256"]:
            raise RuntimeError("benchmark binary changed during measurement")
        if ("hnsw_index_sha256" in report["source"] and
                digest(index_path) != report["source"]["hnsw_index_sha256"]):
            raise RuntimeError("HNSW index changed during measurement")
        report.update(status="completed", completed=datetime.now(timezone.utc).isoformat())
        save()
        print(f"Report: {output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        for run in report["runs"].values():
            if run["status"] == "running":
                run["status"] = "failed"
        report.update(status="failed", error=str(error))
        save()
        print(f"CPU benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())