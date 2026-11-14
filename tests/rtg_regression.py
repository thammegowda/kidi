#!/usr/bin/env python3
"""Download the RTG 500 model and check multilingual translation regressions."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path


MODEL_ID = "thammegowda/rtg-500eng-v1"
CHECKSUMS = {
    "model.safetensors": "3a5a876292e8877004afc40588603ea09c1d9d85aa6726dbf90aa89ef0c3b5b7",
    "tokenizer.src.json.gz": "6ca4f081c20b5b1dbc6aac0b7a1a64c09a644f8d578b3494427bce0bbdb96bcf",
    "tokenizer.tgt.json.gz": "72722e53c66e440ad98a13d6a503d6cd76dd9f6c3d7f12efbfc880cb54b8b16b",
}
PAYLOAD = ("config.json", *CHECKSUMS)
DATA = Path(__file__).resolve().parent / "data"
INPUT = DATA / "sample.input.txt"
LANGUAGES = DATA / "sample.input.lang.txt"
EXPECTED = DATA / "sample.eng.rtg-500eng-v1.expect.txt"
OUTPUT = DATA / "sample.eng.rtg-500eng-v1.out.txt"


def model_directory() -> Path:
    hub = Path(os.environ.get("KIDI_HUB", "~/.cache/kidi/model-hub")).expanduser()
    return hub.resolve() / MODEL_ID


def verify_model(directory: Path) -> None:
    if not (directory / "config.json").is_file() or not (directory / "config.json").stat().st_size:
        raise ValueError(f"Model config.json not installed at {directory}; run make setup-test first")
    for filename, expected in CHECKSUMS.items():
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"Model file {filename} missing at {directory}; run make setup-test first")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Model checksum mismatch: {filename}; remove {directory} and rerun make setup-test")


def setup() -> None:
    directory = model_directory()
    flag = directory / "._OK"
    if flag.is_file() and all((directory / name).is_file() for name in PAYLOAD):
        print(f"Skipping RTG model setup: {flag}")
        return
    flag.unlink(missing_ok=True)
    hf = Path(sys.executable).with_name("hf")
    cmd = [str(hf) if hf.is_file() else "hf", "download", MODEL_ID, "--local-dir", str(directory)]
    for name in PAYLOAD:
        cmd.extend(["--include", name])
    subprocess.run(cmd, check=True)
    verify_model(directory)
    flag.write_text("ok\n", encoding="utf-8")
    print(f"Verified {MODEL_ID} at {directory}")


def read_lines(filename: Path) -> list[str]:
    lines = filename.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"Empty input or blank translation in {filename}")
    return lines


def score_output(expected: list[str], output: Path, minimum: float) -> float:
    from sacrebleu.metrics import CHRF

    hypotheses = read_lines(output)
    if len(hypotheses) != len(expected):
        raise ValueError(f"Output has {len(hypotheses)} lines; expected {len(expected)}")
    scorer = CHRF(char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False, eps_smoothing=False)
    score = scorer.corpus_score(hypotheses, [expected]).score
    print(f"chrF2={score:.4f} minimum={minimum:.4f} sentences={len(expected)}")
    print(f"signature: {scorer.get_signature()}")
    print(f"output: {output}")
    if not math.isfinite(score) or score + 1e-9 < minimum:
        raise ValueError(f"Translation regression: chrF2 {score:.4f} is below {minimum:.4f}")
    return score


def run(args: argparse.Namespace) -> None:
    if not math.isfinite(args.min_chrf) or not 0 <= args.min_chrf <= 100:
        raise ValueError("--min-chrf must be between 0 and 100")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    source = read_lines(INPUT)
    expected = read_lines(EXPECTED)
    languages = read_lines(LANGUAGES)
    if len(source) != 50 or len(expected) != 50 or len(languages) != 50:
        raise ValueError("Input, expected output, and language labels must each contain 50 lines")
    directory = model_directory()
    verify_model(directory)
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise ValueError(f"Kidi executable not found: {binary}")
    OUTPUT.unlink(missing_ok=True)
    subprocess.run(
        [
            str(binary), "translate", "--model", str(directory),
            "--backend", args.backend, "--threads", str(args.threads),
            "--beam-size", "1", "--batch-size", "1", "--batch-window", "1",
            "--max-extra-tokens", "50", "--length-penalty", "0.6",
            "--in", str(INPUT), "--out", str(OUTPUT),
        ],
        check=True,
        timeout=600,
    )
    score_output(expected, OUTPUT, args.min_chrf)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="download and verify the pinned Hub model")
    regression = commands.add_parser("run", help="translate the fixture and enforce its chrF threshold")
    regression.add_argument("--binary", type=Path, default=Path("build-release/kidi"))
    regression.add_argument("--backend", choices=("ynnpack", "mps"), default="ynnpack")
    regression.add_argument("--threads", type=int, default=8)
    regression.add_argument("--min-chrf", type=float, default=99.0)
    args = parser.parse_args()
    try:
        if args.command == "setup":
            setup()
        else:
            run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"RTG regression failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())