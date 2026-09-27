#!/usr/bin/env python3
"""Generate independent Whisper feature, layer, logit, and token references."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, attn_implementation="eager"
    ).eval()
    waveform, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    features = inputs.input_features

    with torch.inference_mode():
        conv1 = torch.nn.functional.gelu(model.model.encoder.conv1(features))
        conv2 = torch.nn.functional.gelu(model.model.encoder.conv2(conv1))
        encoder = model.model.encoder(features).last_hidden_state
        language = model.generation_config.lang_to_id[f"<|{args.language}|>"]
        prefix = torch.tensor(
            [[
                model.generation_config.decoder_start_token_id,
                language,
                model.generation_config.task_to_id["transcribe"],
                model.generation_config.no_timestamps_token_id,
            ]]
        )
        logits = model(input_features=features, decoder_input_ids=prefix).logits[0, -1]
        generated = model.generate(
            features,
            language=args.language,
            task="transcribe",
            return_timestamps=False,
            max_new_tokens=args.tokens,
            do_sample=False,
        )[0]

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features": features,
        "conv1": conv1,
        "conv2": conv2,
        "encoder": encoder,
        "logits": logits,
    }
    shapes = {}
    for name, value in arrays.items():
        array = value.detach().cpu().numpy().astype(np.float32, copy=False)
        array.tofile(args.output / f"{name}.f32")
        shapes[name] = list(array.shape)
    generated_ids = generated.tolist()
    metadata = {
        "shapes": shapes,
        "prefix": prefix[0].tolist(),
        "generated_ids": generated_ids,
        "text": processor.decode(generated_ids, skip_special_tokens=True),
        "logits_top": int(logits.argmax()),
    }
    (args.output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()