#!/usr/bin/env python3
"""Create Kidi configuration beside an unchanged Hugging Face Gemma 4 checkpoint."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml


def configure(directory: Path, *, upgrade_defaults: bool = False) -> Path:
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
        "decode": {"maximum_new_tokens": 8192, "context_size": 16384},
    }
    destination = directory / "model.yaml"
    replace = False
    if destination.exists() and upgrade_defaults:
        legacy = {**document, "decode": {"maximum_new_tokens": 256, "context_size": 2048}}
        if yaml.safe_load(destination.read_text(encoding="utf-8")) != legacy:
            return destination
        replace = True
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".model.yaml.", delete=False) as output:
            temporary = Path(output.name)
            yaml.safe_dump(document, output, sort_keys=False)
        if replace:
            os.replace(temporary, destination)
            print("[kidi] Updated generated Gemma defaults: 8192 output tokens, 16384 context tokens", file=sys.stderr)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(configure(args.directory))


if __name__ == "__main__":
    main()