#!/usr/bin/env python3
"""Create Kidi configuration beside an unchanged Hugging Face Gemma 4 checkpoint."""

import argparse
import json
from pathlib import Path

import yaml


def configure(directory: Path) -> Path:
    original = json.loads((directory / "config.json").read_text())
    model = original["text_config"]
    if original.get("model_type") != "gemma4" or model.get("enable_moe_block"):
        raise ValueError("Expected a dense Gemma 4 checkpoint")
    for filename in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        if not (directory / filename).is_file():
            raise ValueError(f"Missing {filename}")
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
    if not tokenizer_config.get("chat_template") and not (directory / "chat_template.jinja").is_file():
        raise ValueError("Missing chat_template.jinja or chat_template in tokenizer_config.json")
    model["type"] = "gemma4_text"
    if "quantization_config" in original:
        if original["quantization_config"].get("quant_method") != "gemma":
            raise ValueError("Only the native Gemma QAT Safetensors format is supported")
        model["quantization_config"] = original["quantization_config"]
    document = {
        "format_version": 1,
        "weights_file": "model.safetensors",
        "tokenizer_file": "tokenizer.json",
        "model": model,
        "decode": {"maximum_new_tokens": 256, "context_size": 2048},
    }
    destination = directory / "model.yaml"
    with destination.open("x") as output:
        yaml.safe_dump(document, output, sort_keys=False)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(configure(args.directory))


if __name__ == "__main__":
    main()