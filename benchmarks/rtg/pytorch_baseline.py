#!/usr/bin/env python3
"""Run one exact-token PyTorch/RTG greedy-decoding benchmark trial."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple


PROCESS_STARTED = time.perf_counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="exported RTG experiment directory")
    parser.add_argument("--input", type=Path, required=True, help="Moses-tokenized input")
    parser.add_argument("--output", type=Path, required=True, help="translation output")
    parser.add_argument("--batch-sentences", type=int, default=1, help="maximum sentences decoded together")
    parser.add_argument("--max-extra-tokens", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_sentences <= 0 or args.max_extra_tokens <= 0 or args.threads <= 0:
        parser.error("batch size, maximum extra tokens, and threads must be positive")
    if not args.model.is_dir() or not (args.model / "rtg.zip").is_file():
        parser.error(f"not an exported RTG model with rtg.zip: {args.model}")
    if not args.input.is_file():
        parser.error(f"input is not a file: {args.input}")
    return args


def cleaned_ids(token_ids: Sequence[int], end_id: int, pad_id: int) -> List[int]:
    result = []
    for token_id in token_ids:
        if token_id == end_id:
            break
        if token_id != pad_id:
            result.append(token_id)
    return result


def main() -> int:
    args = parse_args()
    os.environ["RTG_CPUS"] = str(args.threads)
    sys.path.insert(0, str(args.model / "rtg.zip"))

    model_load_started = time.perf_counter()
    import torch
    from rtg import TranslationExperiment, device
    from rtg.module.decoder import Decoder

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    experiment = TranslationExperiment(args.model, read_only=True)
    decoder = Decoder.new(experiment, ensemble=1)
    model_load_elapsed = time.perf_counter() - model_load_started
    lines = args.input.read_text(encoding="utf-8").splitlines()
    translations = [""] * len(lines)
    translated_items = 0
    target_tokens = 0

    records: List[Tuple[int, str]] = [(index, line) for index, line in enumerate(lines) if line]
    inference_started = time.perf_counter()
    for offset in range(0, len(records), args.batch_sentences):
        batch = records[offset : offset + args.batch_sentences]
        source_ids = [decoder.inp_vocab.encode_as_ids(line, add_eos=True, add_bos=False) for _, line in batch]
        maximum_source_length = max(len(ids) for ids in source_ids)
        source = torch.full(
            (len(batch), maximum_source_length), decoder.inp_vocab.pad_idx, dtype=torch.long, device=device
        )
        source_lengths = torch.tensor([len(ids) for ids in source_ids], dtype=torch.long, device=device)
        for row, ids in enumerate(source_ids):
            source[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        hypotheses = decoder.greedy_decode(source, source_lengths, max_len=args.max_extra_tokens)
        for (record_index, _), (_, token_ids) in zip(batch, hypotheses):
            output_ids = cleaned_ids(token_ids, decoder.eos_val, decoder.pad_val)
            translations[record_index] = decoder.out_vocab.decode_ids(output_ids, trunc_eos=True)
            translated_items += 1
            target_tokens += len(output_ids)
    inference_elapsed = time.perf_counter() - inference_started

    args.output.write_text("\n".join(translations) + ("\n" if lines else ""), encoding="utf-8")
    wall_elapsed = time.perf_counter() - PROCESS_STARTED
    target_tokens_per_sec = target_tokens / inference_elapsed if inference_elapsed > 0 else 0.0
    print(
        "rtg_metrics"
        f"|input_lines={len(lines)}"
        f"|translated_items={translated_items}"
        f"|target_tokens={target_tokens}"
        f"|inference_sec={inference_elapsed:.6f}"
        f"|model_load_sec={model_load_elapsed:.6f}"
        f"|wall_sec={wall_elapsed:.6f}"
        f"|target_tokens_per_sec={target_tokens_per_sec:.6f}"
        f"|batch_sentences={args.batch_sentences}"
        f"|threads={args.threads}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())