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
    for filename in ("model.safetensors", "tokenizer.json"):
        if not (directory / filename).is_file():
            raise ValueError(f"Missing {filename}")
    model["type"] = "gemma4_text"
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