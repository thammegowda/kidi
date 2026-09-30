"""Benchmark W8/W4/W2 (INT8 activation) projections and a captured decode FFN stack on CPU, Adreno GPU, and Hexagon NPU
through ADB (Python 3.12+)."""

import argparse
import json
from pathlib import Path
from statistics import median

from run import Adb, command, save_report, snapshot, timestamp


ROOT = "/data/local/tmp/kidi-lowbit"
BACKENDS = ("cpu", "gpu", "npu")
# Real projection shapes: Whisper Small INT8 FFN, and Gemma 4 E2B QAT MLPs (W4 in layers 0-14, W2 in 15-34).
# Gate and up projections share an input, so they run as one projection with concatenated outputs.
CASES = (
    ("whisper-w8-fc1-encoder", 8, 1500, 768, 3072),
    ("whisper-w8-fc2-encoder", 8, 1500, 3072, 768),
    ("whisper-w8-fc1-decode", 8, 1, 768, 3072),
    ("whisper-w8-fc2-decode", 8, 1, 3072, 768),
    ("gemma-w4-gate-up-decode", 4, 1, 1536, 12288),
    ("gemma-w4-down-decode", 4, 1, 6144, 1536),
    ("gemma-w4-gate-up-prefill", 4, 128, 1536, 12288),
    ("gemma-w4-down-prefill", 4, 128, 6144, 1536),
    ("gemma-w2-gate-up-decode", 2, 1, 1536, 24576),
    ("gemma-w2-down-decode", 2, 1, 12288, 1536),
    ("gemma-w2-gate-up-prefill", 2, 128, 1536, 24576),
    ("gemma-w2-down-prefill", 2, 128, 12288, 1536),
)
# Launch modes for the decode FFN stack: eager launches every operator; replay launches one captured step.
FFN_MODES = {
    "cpu": ("eager", "replay"),
    "gpu": ("eager", "eager-async", "encode", "replay", "replay-queued"),
    "npu": ("eager", "replay"),
}


def qnn_files(sdk, arch):
    runtime = [sdk / "lib/aarch64-android/libQnnHtp.so", sdk / f"lib/aarch64-android/libQnnHtpV{arch}Stub.so",
               sdk / f"lib/hexagon-v{arch}/unsigned/libQnnHtpV{arch}Skel.so"]
    return runtime, sdk / "lib/aarch64-android/libQnnHtpPrepare.so"


def push(adb, sources, destination):
    command([*adb.arguments[:-1], "push", *map(str, sources), destination], timeout=600)


def parse(output):
    lines = [line for line in output.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"benchmark produced no result: {output}")
    return json.loads(lines[-1])


def invocation(case, backend, runs, threads):
    label, bits, rows, width, outputs = case
    arguments = f"{ROOT}/bench --backend {backend} --label {label} --bits {bits} --m {rows} --k {width} --n {outputs}"
    if backend == "cpu":
        return f"{arguments} --runs {runs} --threads {threads}"
    if backend == "gpu":
        return f"{arguments} --runs {runs}"
    context = f"{ROOT}/contexts/{label}.bin"
    # Contexts are prepared with the 81 MB prepare library, then measured from a runtime-only directory.
    prepare = (f"cd {ROOT}/prepare && LD_LIBRARY_PATH={ROOT}/prepare ADSP_LIBRARY_PATH={ROOT}/prepare "
               f"{arguments} --runs 1 --warmups 0 --qnn-lib libQnnHtp.so --qnn-prepare --qnn-context {context} >/dev/null")
    measure = (f"cd {ROOT}/runtime && LD_LIBRARY_PATH={ROOT}/runtime ADSP_LIBRARY_PATH={ROOT}/runtime "
               f"{arguments} --runs {runs} --qnn-lib libQnnHtp.so --qnn-context {context}")
    return f"{prepare} && {measure}"


# Exit status 3 means the benchmark ran but failed a correctness gate; keep its JSON result for the report.
def keep_gate_failures(script):
    return f"{script}; status=$?; if [ $status -eq 3 ]; then exit 0; fi; exit $status"


def ffn_invocation(backend, mode, layers, runs, threads, graph_layers=0):
    arguments = f"{ROOT}/bench --workload ffn --backend {backend} --mode {mode} --layers {layers} --runs {runs}"
    if backend == "cpu":
        return keep_gate_failures(f"{arguments} --threads {threads}")
    if backend == "gpu":
        return keep_gate_failures(arguments)
    # The HTP prepare library runs out of memory finalizing all 35 MLPs (about 495 MB of weights) as one graph, so
    # replay captures the step as consecutive graphs of `graph_layers` layers.
    if mode == "replay" and graph_layers:
        arguments += f" --qnn-graph-layers {graph_layers}"
    context = f"{ROOT}/contexts/ffn-{layers}-{mode}-g{graph_layers if mode == 'replay' else 0}.bin"
    # Prepared once with the prepare library, then reused by later repeats.
    prepare = (f"if [ ! -f {context}.json ]; then cd {ROOT}/prepare && LD_LIBRARY_PATH={ROOT}/prepare "
               f"ADSP_LIBRARY_PATH={ROOT}/prepare {arguments.replace(f'--runs {runs}', '--runs 1 --warmups 0')} "
               f"--qnn-lib libQnnHtp.so --qnn-prepare --qnn-context {context} >/dev/null; fi")
    # Eager NPU runs also compare every graph with the reference applied to its own inputs (per-stage error).
    diagnose = "KIDI_FFN_DIAGNOSE=1 " if mode == "eager" else ""
    measure = (f"cd {ROOT}/runtime && {diagnose}LD_LIBRARY_PATH={ROOT}/runtime ADSP_LIBRARY_PATH={ROOT}/runtime "
               f"{arguments} --qnn-lib libQnnHtp.so --qnn-context {context}")
    return f"{prepare} && {keep_gate_failures(measure)}"


def equivalence(item, eager):
    """Compares replay and eager fingerprints for every input whose output is live (not all zero)."""
    pairs = zip(item["output_fingerprints"], eager["output_fingerprints"],
                item.get("outputs_live", [True, True]), eager.get("outputs_live", [True, True]))
    live = [(ours, theirs) for ours, theirs, alive, eager_alive in pairs if alive or eager_alive]
    if not live:
        return "n/a (zero output)"
    return "yes" if all(ours == theirs for ours, theirs in live) else "no"


def ffn_table(results):
    rows = ["| Backend | Mode | Launches/step | Step ms (median) | Range | Steps/s | vs eager | Replay = eager | Gate |",
            "|---|---|---:|---:|---|---:|---:|---|---|"]
    for backend, modes in FFN_MODES.items():
        groups = {mode: [item for item in results if item.get("backend") == backend and item.get("mode") == mode]
                  for mode in modes}
        steps = {mode: [item["median_step_ms"] for item in group if "median_step_ms" in item]
                 for mode, group in groups.items()}
        eager = next((item for item in groups["eager"] if "median_step_ms" in item), None)
        for mode in modes:
            measured = [item for item in groups[mode] if "median_step_ms" in item]
            if not measured:
                if groups[mode]:
                    rows.append(f"| {backend} | {mode} | - | failed | - | - | - | - | "
                                f"{groups[mode][0].get('error', 'failed')[:60]} |")
                continue
            step = median(steps[mode])
            speedup = f"{median(steps['eager']) / step:.2f}x" if steps["eager"] else "-"
            same = "-" if not eager or mode == "eager" else equivalence(measured[0], eager)
            passed = all(item["passed"] for item in measured)
            rows.append(f"| {backend} | {mode} | {measured[0]['launches_per_step']} | {step:.3f} | "
                        f"{min(steps[mode]):.2f}-{max(steps[mode]):.2f} | {1000 / step:.1f} | {speedup} | {same} | "
                        f"{'pass' if passed else 'FAIL'} |")
    return "\n".join(rows)


def in_place_table(results):
    rows = ["| Backend | Live outputs | Inputs distinguished | Replay = eager | Gate |", "|---|---|---|---|---|"]
    for backend in FFN_MODES:
        found = {item.get("mode"): item for item in results if item.get("backend") == backend}
        eager, replay = found.get("eager", {}), found.get("replay", {})
        if "output_fingerprints" not in eager or "output_fingerprints" not in replay:
            if found:
                rows.append(f"| {backend} | - | - | - | failed |")
            continue
        live = all(eager["outputs_live"]) and all(replay["outputs_live"])
        same = eager["output_fingerprints"] == replay["output_fingerprints"]
        passed = live and same and replay["inputs_distinguished"] and replay["passed"] and eager["passed"]
        rows.append(f"| {backend} | {'yes' if live else 'no'} | {'yes' if replay['inputs_distinguished'] else 'no'} | "
                    f"{'yes' if same else 'no'} | {'pass' if passed else 'FAIL'} |")
    return "\n".join(rows)


def table(results):
    rows = ["| Case | Precision | Rows | CPU ms | GPU ms | NPU ms | GPU/CPU | NPU/CPU |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, *_ in CASES:
        found = {item["backend"]: item for item in results if item.get("label") == label}
        if not found:
            continue
        sample = next(iter(found.values()))
        timing = {backend: found.get(backend, {}).get("median_ms") for backend in BACKENDS}
        cells = [f"{timing[backend]:.3f}" if timing[backend] else "failed" for backend in BACKENDS]
        speedups = [f"{timing['cpu'] / timing[backend]:.2f}x" if timing["cpu"] and timing[backend] else "-"
                    for backend in ("gpu", "npu")]
        rows.append(f"| {label} | {sample.get('precision', '-')} | {sample.get('m', '-')} | " + " | ".join(cells + speedups) + " |")
    return "\n".join(rows)


def benchmark(arguments):
    adb = Adb(arguments.serial)
    backends = [backend for backend in BACKENDS if backend != "npu" or arguments.qnn_sdk]
    adb.shell(f"rm -rf {ROOT} && mkdir -p {ROOT}/runtime {ROOT}/prepare {ROOT}/contexts")
    push(adb, [arguments.binary], f"{ROOT}/bench")
    adb.shell(f"chmod 755 {ROOT}/bench")
    runtime_bytes = None
    if arguments.qnn_sdk:
        runtime, prepare = qnn_files(Path(arguments.qnn_sdk), arguments.htp_arch)
        push(adb, runtime, f"{ROOT}/runtime/")
        push(adb, [*runtime, prepare], f"{ROOT}/prepare/")
        runtime_bytes = sum(file.stat().st_size for file in runtime)
    report = {
        "started": timestamp(),
        "device": {key: adb.shell(f"getprop {key}") for key in ("ro.product.model", "ro.soc.model", "ro.build.fingerprint")},
        "cpu_threads": arguments.threads,
        "qnn_runtime_bytes": runtime_bytes,
        "before": snapshot(adb),
        "results": [],
    }
    if arguments.workload == "ffn":
        report["layers"] = arguments.layers
        report["npu_graph_layers"] = arguments.npu_graph_layers
        # Deep unnormalized stacks decay outputs to zero, so replay/eager equivalence and in-place input updates are
        # checked on a live 2-layer stack.
        report["in_place"] = []
        for backend in backends:
            for mode in ("eager", "replay"):
                script = ffn_invocation(backend, mode, 2, 3, arguments.threads, arguments.npu_graph_layers)
                try:
                    result = parse(adb.shell(script, timeout=900))
                except RuntimeError as error:
                    result = {"backend": backend, "mode": mode, "error": str(error)}
                report["in_place"].append(result)
        live_check = in_place_table(report["in_place"])
        print(live_check, flush=True)
        # Round-robin repeats spread thermal and clock drift across all modes instead of favoring early ones.
        for repeat in range(arguments.repeats):
            for backend in backends:
                for mode in FFN_MODES[backend]:
                    script = ffn_invocation(backend, mode, arguments.layers, arguments.runs, arguments.threads,
                                            arguments.npu_graph_layers)
                    try:
                        result = parse(adb.shell(script, timeout=3600))
                    except RuntimeError as error:
                        result = {"backend": backend, "mode": mode, "error": str(error)}
                    result["repeat"] = repeat
                    report["results"].append(result)
                    status = (f"{result['median_step_ms']:.3f} ms/step, {result['launches_per_step']} launches"
                              if "median_step_ms" in result else result["error"])
                    print(f"[{repeat}] {backend} {mode:13}: {status}", flush=True)
        summary = "2-layer live check:\n" + live_check + "\n\n" + ffn_table(report["results"])
    else:
        selected = [case for case in CASES if not arguments.cases or any(part in case[0] for part in arguments.cases)]
        for case in selected:
            for backend in backends:
                try:
                    result = parse(adb.shell(invocation(case, backend, arguments.runs, arguments.threads), timeout=900))
                except RuntimeError as error:
                    result = {"label": case[0], "backend": backend, "error": str(error)}
                report["results"].append(result)
                status = (f"{result['median_ms']:.3f} ms, max diff {result['check']['max_difference']}"
                          if "median_ms" in result else result["error"])
                print(f"{case[0]:28} {backend}: {status}", flush=True)
        summary = table(report["results"])
    report["after"] = snapshot(adb)
    report["finished"] = timestamp()
    save_report(arguments.output, report)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial")
    parser.add_argument("output", help="JSON report path; keep it under the ignored .cache directory")
    parser.add_argument("--binary", default="build-android-baseline/kidi_lowbit_bench")
    parser.add_argument("--qnn-sdk", help="local QAIRT SDK root; omit to skip the NPU backend")
    parser.add_argument("--htp-arch", default="79", help="Hexagon architecture of the device (V79 for SM8750)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads; the Android app defaults to 4")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cases", nargs="*", help="only run projection cases whose label contains one of these strings")
    parser.add_argument("--workload", choices=("projection", "ffn"), default="projection",
                        help="single projections, or the captured Gemma 4 decode FFN stack")
    parser.add_argument("--layers", type=int, default=35, help="FFN stack depth (Gemma 4 E2B has 35 layers)")
    parser.add_argument("--repeats", type=int, default=3, help="round-robin repeats of the FFN mode matrix")
    parser.add_argument("--npu-graph-layers", type=int, default=18,
                        help="layers per captured NPU graph (0 captures the whole step as one graph)")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
