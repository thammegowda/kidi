#!/usr/bin/env python3
"""Compare CPU Gemma speculative decoding off/on with identical greedy outputs and matched settings."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess


ROOT = Path(__file__).resolve().parents[3]
PROMPT = "Explain how binary search works in at least 300 words, including an example and its time complexity."


def measure(arguments, drafts, reference):
    settings = {
        "prompt": arguments.prompt,
        "workload": "short",
        "output_tokens": arguments.output_tokens,
        "context_tokens": arguments.context_tokens,
        "prefill_chunk_tokens": arguments.context_tokens,
        "packed_prefill": True,
        "warmups": arguments.warmups,
        "dense_lm_head": not arguments.hnsw,
        "kv_cache_precision": arguments.kv_cache_precision,
    }
    if drafts:
        settings.update(assistant_model=str(arguments.assistant), draft_tokens=drafts)
    command = [str(arguments.runner), "gemma", str(arguments.model), str(arguments.threads),
               str(arguments.repeats), json.dumps(settings)]
    environment = {**os.environ, "KIDI_ACCELERATOR": "cpu"}
    if arguments.hnsw_ef is not None:
        environment["KIDI_HNSW_EF"] = str(arguments.hnsw_ef)
    process = subprocess.run(command, cwd=ROOT, env=environment,
                             capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Speculative draft length {drafts} failed:\n{process.stderr}\n{process.stdout}")
    records = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    generations = [row for row in records if row["stage"] == "generation"]
    runs = [row for row in generations if not row["warmup"]]
    if len(runs) != arguments.repeats or len(generations) != arguments.repeats + arguments.warmups:
        raise ValueError(f"Expected {arguments.repeats} measurements, found {len(runs)}")
    for row in generations:
        if (row["output_tokens"] != arguments.output_tokens or row["observed_output_tokens"] != arguments.output_tokens
                or row["threads"] != arguments.threads or row["speculative"] != bool(drafts)):
            raise ValueError("Measured settings or output counts differ from the requested comparison")
        if drafts and (not row["draft_tokens"] or not row["verification_steps"]):
            raise ValueError("Speculative decoding was requested but no drafting or verification was measured")
        if reference is None:
            reference = {"input": row["input_token_ids"], "tokens": row["token_ids"], "text": row["text"]}
        if row["input_token_ids"] != reference["input"]:
            raise ValueError("Speculative decoding modes received different prompt token IDs")
        if row["token_ids"] != reference["tokens"]:
            mismatch = next((index for index, pair in enumerate(zip(row["token_ids"], reference["tokens"]))
                             if pair[0] != pair[1]), min(len(reference["tokens"]), len(row["token_ids"])))
            raise ValueError(f"Speculative draft length {drafts} changes greedy output at token {mismatch}")
        if row["text"] != reference["text"]:
            raise ValueError("Speculative decoding modes decoded different text from identical token IDs")
    proposed = sum(row["draft_tokens"] for row in runs)
    accepted = sum(row["accepted_draft_tokens"] for row in runs)
    summary = {
        "draft_length": drafts,
        "prompt_tokens": runs[0]["prompt_tokens"],
        "output_tokens": arguments.output_tokens,
        "decode_tps": median(row["decode_interval_tps"] for row in runs),
        "decode_tps_min": min(row["decode_interval_tps"] for row in runs),
        "decode_tps_max": max(row["decode_interval_tps"] for row in runs),
        "prefill_tps": median(row["prefill_through_first_token_tps"] for row in runs),
        "ttft_ms": median(row["first_token_seconds"] * 1000 for row in runs),
        "peak_rss_gib": max(row["peak_rss_kib"] for row in runs) / 1048576,
        "draft_acceptance": accepted / proposed if proposed else None,
        "verification_steps": median(row["verification_steps"] for row in runs),
        "draft_ms": median(row["draft_ms"] for row in runs),
        "verify_ms": median(row["verify_ms"] for row in runs),
        "greedy_output_matches_baseline": True,
    }
    return {"settings": settings, "summary": summary, "records": records}, reference


def render_speculative_svg(tsv_path: Path = Path(__file__).resolve().with_name("speculative_results.tsv")) -> str:
    rows = []
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append({
                "label": row["label"],
                "runtime": row["runtime"],
                "tier": row["tier"],
                "base": float(row["base_decode"]),
                "hnsw": float(row["hnsw_decode"]) if row["hnsw_decode"] else None,
                "specdec": float(row["specdec_decode"]) if row["specdec_decode"] else None,
                "hnsw_specdec": float(row["hnsw_specdec_decode"]) if row["hnsw_specdec_decode"] else None,
                "hnsw_spd": float(row["hnsw_speedup"]) if row["hnsw_speedup"] else None,
                "specdec_spd": float(row["specdec_speedup"]) if row["specdec_speedup"] else None,
                "hnsw_specdec_spd": float(row["hnsw_specdec_speedup"]) if row["hnsw_specdec_speedup"] else None,
                "base_chrf": float(row["base_chrf"]),
                "hnsw_chrf": float(row["hnsw_chrf"]) if row["hnsw_chrf"] else None,
                "notes": row["notes"],
            })

    bar_h = 9.5
    gap = 2.4
    row_pad = 14.0
    for row in rows:
        n_bars = 1 + int(row["hnsw"] is not None) + int(row["specdec"] is not None) + int(row["hnsw_specdec"] is not None)
        row["n_bars"] = n_bars
        row["height"] = n_bars * bar_h + max(0, n_bars - 1) * gap + row_pad

    width = 1200
    px0, px1 = 185.0, 795.0
    qx0, qx1 = 835.0, 1150.0
    frame_x1 = 1165.0
    y0 = 110.0
    y1 = y0 + sum(row["height"] for row in rows)
    height = int(round(y1 + 114))

    def x_tps(val: float) -> float:
        return px0 + (val / 118.0) * (px1 - px0)

    def x_chrf(val: float) -> float:
        return qx0 + ((val - 54.9) / (57.7 - 54.9)) * (qx1 - qx0)

    bf16_ref_tps = next(
        (row["base"] for row in rows if row["label"] == "Kidi BF16"),
        rows[0]["base"],
    )

    col_base = "#94a3b8"
    col_hnsw = "#0284c7"
    col_mtp = "#D55E00"
    col_hnsw_mtp = "#7c2d12"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
        'style="font-family: \'Inter\', -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        '  <defs>',
        '    <linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1">',
        '      <stop offset="0%" stop-color="#f8fafc"/><stop offset="100%" stop-color="#f1f5f9"/>',
        '    </linearGradient>',
        '    <filter id="card-shadow" x="-2%" y="-2%" width="104%" height="106%">',
        '      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.05"/>',
        '    </filter>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" rx="14" fill="url(#bg-grad)" stroke="#cbd5e1" stroke-width="1.2"/>',
        f'  <text x="{width / 2}" y="31" text-anchor="middle" font-size="17" font-weight="800" letter-spacing="-0.3" fill="#0f172a">'
        'Gemma 4 E2B on Arm CPU (Apple M5, 4P Threads) — Speculative Decoding (K=2) &amp; HNSW</text>',
        f'  <text x="{width / 2}" y="51" text-anchor="middle" font-size="12" font-weight="500" fill="#475569">'
        f'Speed (31-token prompt, 64 output tokens; brackets vs. Kidi BF16 {bf16_ref_tps:.2f} tok/s) vs. Generation Quality (150 WMT23 chrF)</text>',
        f'  <rect x="{px0}" y="{y0}" width="{frame_x1 - px0}" height="{y1 - y0}" rx="6" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2" filter="url(#card-shadow)"/>',
    ]

    for xt in (0, 20, 40, 60, 80, 100):
        px = x_tps(xt)
        if xt > 0:
            svg.append(f'  <line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y1}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'  <line x1="{px:.1f}" y1="{y1}" x2="{px:.1f}" y2="{y1 + 5}" stroke="#64748b" stroke-width="1.2"/>')
        svg.append(f'  <text x="{px:.1f}" y="{y1 + 18}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">{xt}</text>')
    svg.append(f'  <text x="{(px0 + px1) / 2:.1f}" y="{y1 + 37}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e293b">Decode Speed (tokens / sec) →</text>')

    div_x = (px1 + qx0) / 2
    svg.append(f'  <rect x="{div_x:.1f}" y="{y0 + 1:.1f}" width="{frame_x1 - div_x - 1:.1f}" height="{y1 - y0 - 2:.1f}" fill="#f8fafc" opacity="0.65"/>')
    svg.append(f'  <line x1="{div_x:.1f}" y1="{y0}" x2="{div_x:.1f}" y2="{y1}" stroke="#cbd5e1" stroke-width="1.2"/>')

    svg.append(f'  <text x="{(qx0 + qx1) / 2:.1f}" y="{y0 - 30}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e293b">Generation Quality (WMT23 chrF ↑) →</text>')
    for qt in (55.0, 55.5, 56.0, 56.5, 57.0, 57.5):
        qx = x_chrf(qt)
        svg.append(f'  <line x1="{qx:.1f}" y1="{y0}" x2="{qx:.1f}" y2="{y1}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'  <line x1="{qx:.1f}" y1="{y0 - 5}" x2="{qx:.1f}" y2="{y0}" stroke="#64748b" stroke-width="1.2"/>')
        svg.append(f'  <text x="{qx:.1f}" y="{y0 - 9}" text-anchor="middle" font-size="10.5" font-weight="600" fill="#475569">{qt:.1f}</text>')

    qx_bf16 = x_chrf(57.27)
    svg.append(f'  <line x1="{qx_bf16:.1f}" y1="{y0}" x2="{qx_bf16:.1f}" y2="{y1}" stroke="#64748b" stroke-width="1.4" stroke-dasharray="4,3"/>')
    svg.append(f'  <text x="{qx_bf16:.1f}" y="{y1 + 17}" text-anchor="middle" font-size="10" font-weight="600" font-style="italic" fill="#475569">BF16 Ref (57.27)</text>')

    ry_cur = y0
    prev_tier = None
    for idx, row in enumerate(rows):
        ry_top = ry_cur
        ry_mid = ry_top + row["height"] / 2
        ry_cur += row["height"]
        tier_group = "Q4_g32" if row["tier"] == "Q4_g32" else row["tier"]
        if prev_tier is not None and tier_group != prev_tier:
            svg.append(f'  <line x1="28" y1="{ry_top:.1f}" x2="{frame_x1}" y2="{ry_top:.1f}" stroke="#cbd5e1" stroke-width="1"/>')
        elif idx > 0:
            svg.append(f'  <line x1="{px0}" y1="{ry_top:.1f}" x2="{frame_x1}" y2="{ry_top:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        prev_tier = tier_group

        is_kidi = row["runtime"] == "Kidi"
        svg.append(
            f'  <text x="{px0 - 10:.1f}" y="{ry_mid + 4:.1f}" text-anchor="end" font-size="11.5" '
            f'font-weight="{"700" if is_kidi else "600"}" fill="{"#0f172a" if is_kidi else "#334155"}">{row["label"]}</text>'
        )

        base_spd = row["base"] / bf16_ref_tps
        base_lbl = f'{row["base"]:.2f} ({base_spd:.2f}x)' if row["specdec"] is not None else f'{row["base"]:.2f} ({base_spd:.2f}x · no SpecDec)'
        bars = [("Base", row["base"], col_base, "#475569", base_lbl, row["base_chrf"])]
        if row["hnsw"] is not None:
            bars.append(("HNSW", row["hnsw"], col_hnsw, "#0369a1", f'{row["hnsw"]:.2f} ({row["hnsw"] / bf16_ref_tps:.2f}x)', row["hnsw_chrf"]))
        if row["specdec"] is not None:
            bars.append(("SpecDec", row["specdec"], col_mtp, "#c2410c", f'{row["specdec"]:.2f} ({row["specdec"] / bf16_ref_tps:.2f}x)', row["base_chrf"]))
        if row["hnsw_specdec"] is not None:
            bars.append(("HNSW+SpecDec", row["hnsw_specdec"], col_hnsw_mtp, "#7c2d12", f'{row["hnsw_specdec"]:.2f} ({row["hnsw_specdec"] / bf16_ref_tps:.2f}x)', row["hnsw_chrf"]))

        block_h = len(bars) * bar_h + (len(bars) - 1) * gap
        start_y = ry_mid - block_h / 2
        for b_idx, (_, val, fill_col, txt_col, label_txt, chrf_val) in enumerate(bars):
            by = start_y + b_idx * (bar_h + gap)
            bw = max(1.0, x_tps(val) - px0)
            svg.append(f'  <rect x="{px0:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" rx="2.5" fill="{fill_col}"/>')
            fw = "700" if b_idx == len(bars) - 1 and is_kidi else "600"
            svg.append(f'  <text x="{px0 + bw + 5:.1f}" y="{by + bar_h - 1.5:.1f}" font-size="9.5" font-weight="{fw}" fill="{txt_col}">{label_txt}</text>')

            cy = by + bar_h / 2
            cx = x_chrf(chrf_val)
            svg.append(f'  <line x1="{div_x + 6:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>')
            svg.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.1" fill="{fill_col}" stroke="#ffffff" stroke-width="1.1"/>')
            if b_idx == 0:
                side_x = cx - 7.0 if chrf_val >= 56.9 else cx + 7.0
                anchor = "end" if chrf_val >= 56.9 else "start"
                svg.append(f'  <text x="{side_x:.1f}" y="{cy + 3.2:.1f}" text-anchor="{anchor}" font-size="9.0" font-weight="600" fill="#475569">{chrf_val:.2f}</text>')
            elif b_idx == 1 and row["hnsw_chrf"] is not None and abs(row["hnsw_chrf"] - row["base_chrf"]) >= 0.04:
                side_x = cx - 7.0 if chrf_val >= 56.9 else cx + 7.0
                anchor = "end" if chrf_val >= 56.9 else "start"
                svg.append(f'  <text x="{side_x:.1f}" y="{cy + 3.2:.1f}" text-anchor="{anchor}" font-size="9.0" font-weight="600" fill="#0369a1">{chrf_val:.2f}</text>')

    svg.append(f'  <line x1="{px0}" y1="{y0}" x2="{px0}" y2="{y1}" stroke="#64748b" stroke-width="1.3"/>')
    svg.append(f'  <line x1="{px0}" y1="{y1}" x2="{frame_x1}" y2="{y1}" stroke="#64748b" stroke-width="1.3"/>')
    svg.append(f'  <line x1="{div_x:.1f}" y1="{y0}" x2="{frame_x1}" y2="{y0}" stroke="#64748b" stroke-width="1.3"/>')

    leg_y = y1 + 50
    svg.append(f'  <rect x="40" y="{leg_y:.0f}" width="1125" height="48" rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2"/>')
    legend = [
        (col_base, "Baseline (SpecDec Off)", "600"),
        (col_hnsw, "+ HNSW (ef=800)", "600"),
        (col_mtp, "+ SpecDec (K=2)", "700"),
        (col_hnsw_mtp, "+ HNSW + SpecDec (K=2)", "700"),
    ]
    lx = 56.0
    for col, text, fw in legend:
        svg.append(f'  <rect x="{lx:.1f}" y="{leg_y + 18.5:.1f}" width="11" height="11" rx="2.5" fill="{col}"/>')
        svg.append(f'  <text x="{lx + 16:.1f}" y="{leg_y + 28.0:.1f}" font-size="11" font-weight="{fw}" fill="#0f172a">{text}</text>')
        lx += 24.0 + len(text) * 6.4 + 12.0
    svg.append(f'  <text x="1145" y="{leg_y + 28.0:.1f}" text-anchor="end" font-size="9.8" font-weight="500" fill="#64748b">† LiteRT-LM uses bundled TFLite drafter (MixedQ) · Dots share mode color &amp; chrF scale</text>')
    svg.append('</svg>')
    return "\n".join(svg) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "build-mtp/kidi_android_baseline")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--assistant", type=Path)
    parser.add_argument("--render-svg", type=Path, help="render speculative comparison SVG from --tsv")
    parser.add_argument("--tsv", type=Path, default=Path(__file__).resolve().with_name("speculative_results.tsv"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--draft-tokens", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--kv-cache-precision", choices=["auto", "fp32", "int8"], default="int8")
    parser.add_argument("--hnsw", action="store_true", help="use MODEL/lm_head.hnsw instead of the dense target head")
    parser.add_argument("--hnsw-ef", type=int, help="override HNSW search effort")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/benchmarks/mtp-e2b-int8.json")
    arguments = parser.parse_args()
    if arguments.render_svg:
        svg_path = arguments.render_svg.expanduser().resolve()
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(render_speculative_svg(arguments.tsv.expanduser().resolve()), encoding="utf-8")
        print(f"Wrote {svg_path}")
        return
    if not arguments.model or not arguments.assistant:
        parser.error("--model and --assistant are required unless --render-svg is given")
    if (arguments.threads < 1 or arguments.repeats < 1 or arguments.warmups < 1
            or arguments.output_tokens < 3 or arguments.context_tokens <= arguments.output_tokens
            or not arguments.draft_tokens or any(value < 1 or value > 16 for value in arguments.draft_tokens)
            or (arguments.hnsw_ef is not None and arguments.hnsw_ef < 1)):
        parser.error("Counts must be positive, the context must fit the output, and draft lengths must be 1 to 16")
    for name in ("runner", "model", "assistant", "output"):
        setattr(arguments, name, getattr(arguments, name).expanduser().absolute())
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                if platform.system() == "Darwin" else platform.processor()),
        "threads": arguments.threads,
        "warmups": arguments.warmups,
        "repeats": arguments.repeats,
        "target_model": str(arguments.model),
        "assistant_model": str(arguments.assistant),
        "assistant_compute_precision": "fp32",
        "kv_cache_precision": arguments.kv_cache_precision,
        "lm_head": "hnsw" if arguments.hnsw else "dense",
        "runner_sha256": hashlib.sha256(arguments.runner.read_bytes()).hexdigest(),
        "runs": [],
    }
    target_config = arguments.model / "config.json"
    if target_config.is_file():
        document = target_config.read_bytes()
        report["target_config_sha256"] = hashlib.sha256(document).hexdigest()
        report["target_weight_bits"] = json.loads(document).get("quantization_config", {}).get("num_bits")
    if arguments.hnsw:
        index = arguments.model / "lm_head.hnsw"
        if not index.is_file():
            parser.error("--hnsw requires MODEL/lm_head.hnsw")
        report["hnsw_index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
        report["hnsw_ef"] = arguments.hnsw_ef
    reference = None
    for drafts in [0, *dict.fromkeys(arguments.draft_tokens)]:
        result, reference = measure(arguments, drafts, reference)
        speed = result["summary"]["decode_tps"]
        baseline = report["runs"][0]["summary"]["decode_tps"] if report["runs"] else speed
        result["summary"]["speedup"] = speed / baseline
        report["runs"].append(result)
        label = f"SpecDec on, draft={drafts}" if drafts else "SpecDec off"
        print(f"{label}: {speed:.2f} tokens/s, {speed / baseline:.3f}x baseline", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Identical greedy outputs verified. Results: {arguments.output}")


if __name__ == "__main__":
    main()
