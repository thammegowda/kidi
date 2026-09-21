#!/usr/bin/env python3
"""Benchmark kidi RTG FP32, BF16, and INT8 packages on identical text."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


HERE = Path(__file__).resolve().parent
KIDI_ROOT = HERE.parents[1]
WORKSPACE_ROOT = KIDI_ROOT.parent
DEFAULT_MODEL_PREFIX = WORKSPACE_ROOT / "models/rtg500eng-tfm9L6L768d-bsz720k-ens05-kidi-canonical"
SACREBLEU_VERSION = "2.6.0"
PORTALOCKER_VERSION = "3.2.0"
PINNED_PACKAGES = {
    "sacrebleu": SACREBLEU_VERSION,
    "portalocker": PORTALOCKER_VERSION,
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
METRICS_PATTERN = re.compile(
    r"^kidi_metrics\|input_lines=(?P<input_lines>\d+)\|"
    r"translated_items=(?P<translated_items>\d+)\|target_tokens=(?P<target_tokens>\d+)$",
    re.MULTILINE,
)
PROFILE_PATTERN = re.compile(r"^kidi_profile\|(?P<fields>[^\r\n]+)$", re.MULTILINE)


class RunResult(NamedTuple):
    wall_elapsed: float
    model_load_elapsed: float
    inference_elapsed: float
    translated_items: int
    target_tokens: int


def csv_values(text: str) -> list[str]:
    values = [value.strip().lower() for value in text.split(",") if value.strip()]
    valid = {"fp32", "bf16", "int8"}
    if not values or len(values) != len(set(values)) or any(value not in valid for value in values):
        raise argparse.ArgumentTypeError("expected unique comma-separated values from: fp32,bf16,int8")
    return values


def percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def prefix_file(source: Path, destination: Path, count: int) -> int:
    lines = source.read_text(encoding="utf-8").splitlines()
    if count > 0:
        lines = lines[:count]
    if not lines:
        raise ValueError(f"input has no lines: {source}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def ensure_sacrebleu(cache_dir: Path):
    package_dir = cache_dir / "python"
    dataset_dir = cache_dir / "datasets"
    os.environ["SACREBLEU"] = str(dataset_dir)
    installed_versions = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(package_dir)])
        if distribution.metadata["Name"]
    }
    if any(installed_versions.get(name) != version for name, version in PINNED_PACKAGES.items()):
        shutil.rmtree(package_dir, ignore_errors=True)
        package_dir.mkdir(parents=True, exist_ok=True)
        requirements = [f"{name}=={version}" for name, version in PINNED_PACKAGES.items()]
        print(f"installing {' '.join(requirements)} into {package_dir}", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--target",
                str(package_dir),
                *requirements,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("failed to install the pinned SacreBLEU benchmark dependency")
    sys.path.insert(0, str(package_dir))
    importlib.invalidate_caches()
    module = importlib.import_module("sacrebleu")
    if module.__version__ != SACREBLEU_VERSION:
        raise RuntimeError(f"expected SacreBLEU {SACREBLEU_VERSION}, found {module.__version__}")
    return module


def prepare_corpus(args: argparse.Namespace, sacrebleu, input_path: Path, reference_path: Path) -> tuple[int, str]:
    if args.input:
        count = prefix_file(args.input, input_path, args.num)
        if args.reference:
            reference_count = prefix_file(args.reference, reference_path, args.num)
            if reference_count != count:
                raise RuntimeError(f"input/reference line mismatch: {count}/{reference_count}")
        return count, str(args.input)

    dataset = sacrebleu.DATASETS.get(args.test_set)
    if dataset is None:
        raise RuntimeError(f"unknown SacreBLEU test set: {args.test_set}")
    if args.lang_pair not in dataset.langpairs:
        raise RuntimeError(f"{args.test_set} has no language pair {args.lang_pair}")
    print(f"preparing SacreBLEU {args.test_set} {args.lang_pair}", flush=True)
    source_file = Path(dataset.get_source_file(args.lang_pair))
    reference_files = [Path(path) for path in dataset.get_reference_files(args.lang_pair)]
    if not reference_files:
        raise RuntimeError(f"{args.test_set} {args.lang_pair} has no reference")

    tokenizer_module = importlib.import_module("sacrebleu.tokenizers.tokenizer_13a")
    tokenizer = tokenizer_module.Tokenizer13a()
    source_lines = source_file.read_text(encoding="utf-8").splitlines()
    reference_lines = reference_files[0].read_text(encoding="utf-8").splitlines()
    if args.num > 0:
        source_lines = source_lines[: args.num]
        reference_lines = reference_lines[: args.num]
    if not source_lines or len(source_lines) != len(reference_lines):
        raise RuntimeError(f"downloaded input/reference line mismatch: {len(source_lines)}/{len(reference_lines)}")
    input_path.write_text("\n".join(tokenizer(line) for line in source_lines) + "\n", encoding="utf-8")
    reference_path.write_text("\n".join(reference_lines) + "\n", encoding="utf-8")
    return len(source_lines), f"SacreBLEU {args.test_set} {args.lang_pair}"


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def translated_item_count(path: Path) -> int:
    return sum(bool(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines())


def mismatch_count(reference: Path, candidate: Path) -> int:
    reference_lines = reference.read_text(encoding="utf-8", errors="replace").splitlines()
    candidate_lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(reference_lines) != len(candidate_lines):
        raise RuntimeError("cannot compare outputs with different line counts")
    return sum(left != right for left, right in zip(reference_lines, candidate_lines))


def chrf(scorer, reference: Path | None, hypothesis: Path) -> str:
    if reference is None:
        return ""
    hypotheses = hypothesis.read_text(encoding="utf-8", errors="replace").splitlines()
    references = reference.read_text(encoding="utf-8", errors="replace").splitlines()
    return f"{scorer.corpus_score(hypotheses, [references], n_bootstrap=0).score:.4f}"


def model_encoding(model: Path) -> str:
    with (model / "model.safetensors").open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    dtypes = {
        tensor["dtype"]
        for name, tensor in header.items()
        if name.endswith(".weight") and len(tensor["shape"]) == 2
    }
    if len(dtypes) != 1:
        return "mixed:" + ",".join(sorted(dtypes))
    dtype = dtypes.pop()
    return "INT8_PER_CHANNEL" if dtype == "I8" else dtype


def machine_description() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def make_command(args: argparse.Namespace, model: Path, input_path: Path, output_path: Path) -> list[str]:
    command = [
        str(args.binary),
        "generate",
        "--model",
        str(model),
        "--in",
        str(input_path),
        "--out",
        str(output_path),
        "--beam-size",
        str(args.beam_size),
        "--max-extra-tokens",
        str(args.max_extra_tokens),
        "--length-penalty",
        str(args.length_penalty),
        "--stats",
        "--profile",
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    return command


def run_once(
    name: str,
    command: list[str],
    expected_lines: int,
    expected_translated_items: int,
    output: Path,
    log: Path,
    timeout_sec: int,
    environment: dict[str, str],
) -> RunResult:
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as log_stream:
        result = subprocess.run(
            command,
            text=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            env=environment,
            check=False,
        )
    wall_elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log}")
    actual_lines = line_count(output)
    if actual_lines != expected_lines:
        raise RuntimeError(f"{name} produced {actual_lines}/{expected_lines} lines; see {log}")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    matches = list(METRICS_PATTERN.finditer(log_text))
    if len(matches) != 1:
        raise RuntimeError(f"{name} emitted {len(matches)} valid metrics records; expected exactly one; see {log}")
    metrics = {key: int(matches[0].group(key)) for key in ("input_lines", "translated_items", "target_tokens")}
    if metrics["input_lines"] != expected_lines:
        raise RuntimeError(f"{name} reported {metrics['input_lines']}/{expected_lines} input lines; see {log}")
    if metrics["translated_items"] > metrics["input_lines"]:
        raise RuntimeError(f"{name} reported more translated items than input lines; see {log}")
    if metrics["translated_items"] != expected_translated_items:
        raise RuntimeError(
            f"{name} reported {metrics['translated_items']}/{expected_translated_items} translated items; see {log}"
        )
    profile_matches = list(PROFILE_PATTERN.finditer(log_text))
    if len(profile_matches) != 1:
        raise RuntimeError(f"{name} emitted {len(profile_matches)} profile records; expected exactly one; see {log}")
    profile: dict[str, str] = {}
    for field in profile_matches[0].group("fields").split("|"):
        if "=" not in field:
            raise RuntimeError(f"{name} emitted a malformed profile field; see {log}")
        key, value = field.split("=", 1)
        if key in profile:
            raise RuntimeError(f"{name} emitted duplicate profile field {key}; see {log}")
        profile[key] = value
    try:
        package_load_ns = int(profile["package_load_ns"])
        graph_compile_ns = int(profile["graph_compile_ns"])
        inference_ns = int(profile["translate_ns"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"{name} emitted invalid timing fields; see {log}") from error
    if min(package_load_ns, graph_compile_ns) < 0 or inference_ns <= 0:
        raise RuntimeError(f"{name} emitted invalid timing values; see {log}")
    return RunResult(
        wall_elapsed,
        (package_load_ns + graph_compile_ns) / 1.0e9,
        inference_ns / 1.0e9,
        metrics["translated_items"],
        metrics["target_tokens"],
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    source_origin: str,
    input_path: Path,
    reference_path: Path | None,
    rows: list[dict[str, str]],
    commands: dict[str, list[str]],
    workdir: Path,
) -> None:
    by_precision = {row["precision"]: row for row in rows}
    bf16 = by_precision.get("bf16")
    int8 = by_precision.get("int8")
    comparison = ""
    if bf16 and int8:
        comparison = f"{float(int8['median_inference_sec']) / float(bf16['median_inference_sec']):.2f}x"

    lines = [
        "# Kidi RTG precision benchmark",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Protocol",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| CPU | {machine_description()} |",
        f"| OS | {platform.platform()} |",
        f"| Binary | `{args.binary}` |",
        f"| Input source | `{source_origin}` |",
        f"| SacreBLEU version | {SACREBLEU_VERSION} |",
        f"| SacreBLEU cache | `{args.cache_dir}` |",
        f"| Input copy | `{input_path}` |",
        f"| Reference | `{reference_path or '<none>'}` |",
        f"| Sentences | {line_count(input_path)} |",
        f"| Precisions | `{','.join(args.precisions)}` |",
        f"| Beam size | {args.beam_size} |",
        f"| Maximum extra tokens | {args.max_extra_tokens} |",
        f"| Length penalty | {args.length_penalty} |",
        f"| Warmups per precision | {args.warmup_runs} |",
        f"| Measured repetitions | {args.repetitions} |",
        "| Execution | warm-file-cache, fresh process per trial, round-robin repetitions |",
        "| Throughput timing | aggregate `Translator::translate` time; excludes package load and graph compilation |",
        f"| Runtime threads | {args.threads} total YNNPACK threads; BLAS thread env fixed to 1 |",
        "",
        "## Results",
        "",
        "| Precision | Encoding | Model GiB | Median inference s | p95 inference s | Median load s | Median wall s | Lines/s | Target tokens | Target tok/s | Mismatch vs FP32 | chrF/FP32 | chrF/reference | Inference timings s | Wall timings s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['precision']} | {row['encoding']} | {row['model_gib']} | {row['median_inference_sec']} | "
            f"{row['p95_inference_sec']} | {row['median_model_load_sec']} | {row['median_wall_sec']} | "
            f"{row['lines_per_sec']} | {row['target_tokens']} | "
            f"{row['target_tokens_per_sec']} | {row['mismatches_fp32']} | {row['chrf_fp32']} | "
            f"{row['chrf_reference']} | `{row['inference_timings_sec']}` | `{row['wall_timings_sec']}` |"
        )
    if comparison:
        lines.extend(["", f"BF16 throughput relative to INT8 by median inference time: **{comparison}**."])

    lines.extend(["", "## Commands", ""])
    for precision, command in commands.items():
        lines.extend([f"### {precision}", "", "```bash", shlex.join(command), "```", ""])
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Workdir: `{workdir}`",
            f"- Outputs: `{workdir / 'outputs'}`",
            f"- Logs: `{workdir / 'logs'}`",
            f"- Command plan: `{workdir / 'command-plan.txt'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=KIDI_ROOT / "build-release/kidi")
    parser.add_argument("--test-set", default="wmt14/full", help="SacreBLEU test-set registry key")
    parser.add_argument("--lang-pair", default="fr-en", help="SacreBLEU source-target language pair")
    parser.add_argument("--input", type=Path, help="already tokenized input override")
    parser.add_argument("--reference", type=Path, help="reference for --input")
    parser.add_argument("--num", type=int, default=32, help="input prefix length; 0 uses every line")
    parser.add_argument("--precisions", type=csv_values, default=["fp32", "bf16", "int8"])
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--bf16-model", type=Path, default=Path(str(DEFAULT_MODEL_PREFIX) + "-bf16"))
    parser.add_argument("--int8-model", type=Path, default=Path(str(DEFAULT_MODEL_PREFIX) + "-int8"))
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-extra-tokens", type=int, default=50)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--threads", type=int, default=8, help="total YNNPACK threads including the caller")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache")
    parser.add_argument("--outdir", type=Path, default=HERE)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()

    if args.num < 0:
        parser.error("--num cannot be negative")
    if args.beam_size <= 0 or args.max_extra_tokens <= 0 or args.length_penalty < 0:
        parser.error("decode settings must be positive, except length penalty may be zero")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    if args.warmup_runs < 0 or args.repetitions <= 0:
        parser.error("--warmup-runs must be non-negative and --repetitions must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error(f"not an executable: {args.binary}")
    if args.input and not args.input.is_file():
        parser.error(f"input is not a file: {args.input}")
    if args.reference and not args.reference.is_file():
        parser.error(f"reference is not a file: {args.reference}")
    if args.reference and not args.input:
        parser.error("--reference requires --input")
    models = {"fp32": args.fp32_model, "bf16": args.bf16_model, "int8": args.int8_model}
    for precision in args.precisions:
        if not models[precision].is_dir():
            parser.error(f"{precision} model is not a directory: {models[precision]}")
    return args


def main() -> int:
    args = parse_args()
    sacrebleu = ensure_sacrebleu(args.cache_dir)
    scorer = sacrebleu.metrics.CHRF()
    models = {"fp32": args.fp32_model, "bf16": args.bf16_model, "int8": args.int8_model}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = args.workdir or args.outdir / f"work-{timestamp}"
    outputs = workdir / "outputs"
    logs = workdir / "logs"
    for directory in (args.outdir, workdir, outputs, logs):
        directory.mkdir(parents=True, exist_ok=True)

    input_copy = workdir / "input.fr.tok"
    reference_copy = workdir / "reference.en"
    sentence_count, source_origin = prepare_corpus(args, sacrebleu, input_copy, reference_copy)
    expected_lines = line_count(input_copy)
    if expected_lines != sentence_count:
        raise RuntimeError(f"prepared input line mismatch: {expected_lines}/{sentence_count}")
    expected_translated_items = translated_item_count(input_copy)
    if args.input and not args.reference:
        reference_copy = None

    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    commands: dict[str, list[str]] = {}
    for precision in args.precisions:
        commands[precision] = make_command(args, models[precision], input_copy, Path("<OUTPUT>"))
    (workdir / "command-plan.txt").write_text(
        "\n".join(f"{precision}: {shlex.join(command)}" for precision, command in commands.items()) + "\n",
        encoding="utf-8",
    )

    final_outputs: dict[str, Path] = {}
    measured_results: dict[str, list[RunResult]] = {precision: [] for precision in args.precisions}
    for precision in args.precisions:
        for warmup in range(1, args.warmup_runs + 1):
            print(f"warming {precision}, run={warmup}/{args.warmup_runs}", flush=True)
            output = outputs / f"{precision}-warmup{warmup}.txt"
            command = make_command(args, models[precision], input_copy, output)
            run_once(
                precision,
                command,
                expected_lines,
                expected_translated_items,
                output,
                logs / f"{precision}-warmup{warmup}.log",
                args.timeout_sec,
                environment,
            )

    for repetition in range(1, args.repetitions + 1):
        offset = (repetition - 1) % len(args.precisions)
        order = args.precisions[offset:] + args.precisions[:offset]
        for precision in order:
            print(f"running {precision}, run={repetition}/{args.repetitions}", flush=True)
            output = outputs / f"{precision}-run{repetition}.txt"
            command = make_command(args, models[precision], input_copy, output)
            result = run_once(
                precision,
                command,
                expected_lines,
                expected_translated_items,
                output,
                logs / f"{precision}-run{repetition}.log",
                args.timeout_sec,
                environment,
            )
            measured_results[precision].append(result)
            final_outputs[precision] = output

    baseline_precision = "fp32" if "fp32" in final_outputs else args.precisions[0]
    baseline = final_outputs[baseline_precision]
    rows: list[dict[str, str]] = []
    for precision in args.precisions:
        results = measured_results[precision]
        inference_values = [result.inference_elapsed for result in results]
        wall_values = [result.wall_elapsed for result in results]
        load_values = [result.model_load_elapsed for result in results]
        median_inference = statistics.median(inference_values)
        target_token_counts = {result.target_tokens for result in results}
        if len(target_token_counts) != 1:
            raise RuntimeError(f"{precision} target-token count changed across measured repetitions")
        target_tokens = target_token_counts.pop()
        output = final_outputs[precision]
        rows.append(
            {
                "precision": precision,
                "encoding": model_encoding(models[precision]),
                "model_bytes": str((models[precision] / "model.safetensors").stat().st_size),
                "model_gib": f"{(models[precision] / 'model.safetensors').stat().st_size / 2**30:.3f}",
                "median_inference_sec": f"{median_inference:.4f}",
                "p95_inference_sec": f"{percentile(inference_values, 0.95):.4f}",
                "median_model_load_sec": f"{statistics.median(load_values):.4f}",
                "median_wall_sec": f"{statistics.median(wall_values):.4f}",
                "lines_per_sec": f"{sentence_count / median_inference:.2f}",
                "target_tokens": str(target_tokens),
                "target_tokens_per_sec": f"{statistics.median(result.target_tokens / result.inference_elapsed for result in results):.2f}",
                "mismatches_fp32": str(mismatch_count(baseline, output)),
                "exact_fp32": "yes" if baseline.read_bytes() == output.read_bytes() else "no",
                "chrf_fp32": chrf(scorer, baseline, output),
                "chrf_reference": chrf(scorer, reference_copy, output),
                "inference_timings_sec": ",".join(f"{value:.4f}" for value in inference_values),
                "wall_timings_sec": ",".join(f"{value:.4f}" for value in wall_values),
                "output": str(output),
            }
        )

    csv_path = args.outdir / f"results-{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.outdir / f"report-{timestamp}.md"
    write_report(report_path, args, source_origin, input_copy, reference_copy, rows, commands, workdir)
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Workdir: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())