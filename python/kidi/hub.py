"""Resolve supported Hugging Face model repositories to local Kidi packages."""

import json
import sys
from pathlib import Path, PurePosixPath

DEFAULT_CACHE = "~/.cache/kidi/model-hub"


def _package_files(document: object) -> list[str]:
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise ValueError("Unsupported model.yaml; expected a Kidi format_version 1 package")
    model = document.get("model", {})
    if not isinstance(model, dict) or not isinstance(document.get("decode"), dict):
        raise ValueError("model.yaml requires model and decode mappings")
    names = [document.get("weights_file")]
    if model.get("type") == "gemma4_text":
        names.append(document.get("tokenizer_file"))
    elif model.get("type") == "rtg_transformer_nmt":
        tokenizers = document.get("tokenizers", {})
        if not isinstance(tokenizers, dict):
            raise ValueError("model.yaml requires a tokenizers mapping")
        names.extend([tokenizers.get("source"), tokenizers.get("target")])
    else:
        raise ValueError(f"Unsupported Kidi model type: {model.get('type')!r}")
    for name in names:
        if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts or any(char in name for char in "\\:*?[]\0")):
            raise ValueError(f"Invalid package file path: {name!r}")
    return names


def resolve(reference: str, cache: str | Path = DEFAULT_CACHE) -> Path:
    """Download/reuse a supported repository; optional @revision pins a model commit."""
    try:
        import yaml
        from filelock import FileLock
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import validate_repo_id
    except ImportError as error:
        raise RuntimeError("Hub models require optional dependencies. Install with: pip install 'kidi[hf]'") from error

    repo_id, separator, revision = reference.partition("@")
    validate_repo_id(repo_id)
    if separator and not revision:
        raise ValueError("A model revision after @ must not be empty")
    cache_dir = Path(cache).expanduser().resolve()
    print(f"[kidi] Resolving @{reference} in {cache_dir}", file=sys.stderr)
    metadata = Path(snapshot_download(repo_id, revision=revision or None, cache_dir=cache_dir,
                                      allow_patterns=["model.yaml", "config.json"]))
    manifest = metadata / "model.yaml"
    document = None
    if manifest.is_file():
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        files = _package_files(document)
        gemma = document["model"]["type"] == "gemma4_text"
    else:
        config_path = metadata / "config.json"
        if not config_path.is_file():
            raise ValueError("Repository has neither a Kidi model.yaml nor a supported Gemma 4 config.json")
        original = json.loads(config_path.read_text(encoding="utf-8"))
        text = original.get("text_config", {})
        if original.get("model_type") != "gemma4" or not isinstance(text, dict) or not text or text.get("enable_moe_block"):
            raise ValueError("Automatic setup supports dense Gemma 4 checkpoints or ready-made Kidi packages only")
        quantization = original.get("quantization_config")
        if quantization and quantization.get("quant_method") != "gemma":
            raise ValueError("Only native Gemma mobile QAT or original floating-point checkpoints are supported")
        files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        gemma = True

    optional = []
    if gemma:
        tokenizer_file = document["tokenizer_file"] if document is not None else "tokenizer.json"
        parent = PurePosixPath(tokenizer_file).parent
        optional = [str(parent / "tokenizer_config.json"), str(parent / "chat_template.jinja")]
    directory = Path(snapshot_download(repo_id, revision=metadata.name, cache_dir=cache_dir,
                                       allow_patterns=list(dict.fromkeys([*files, *optional, "model.yaml", "config.json"]))))
    for name in files:
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise ValueError(f"Missing {name} in @{reference}; retry online to complete the download. "
                             "Sharded and non-Safetensors checkpoints are not supported.")
    lock_path = cache_dir / ".locks" / f"kidi-{repo_id.replace('/', '--')}-{directory.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        if not (directory / "model.yaml").is_file():
            from .converters.gemma4 import configure

            configure(directory)
        _package_files(yaml.safe_load((directory / "model.yaml").read_text(encoding="utf-8")))
    print(f"[kidi] Model ready: {directory}", file=sys.stderr)
    return directory