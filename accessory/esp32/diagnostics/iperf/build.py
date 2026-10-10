import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
SDK = ROOT / ".sdk"
TOOLS = ROOT / ".tools"
BUILD = ROOT / ".build"


def main():
    parser = argparse.ArgumentParser(description="Build the pinned official Wi-Fi example for an explicit target")
    parser.add_argument("--target", required=True, help="ESP-IDF chip identifier; no board/flash layout is assumed")
    parser.add_argument("--defaults", action="append", type=Path, default=[],
                        help="Additional board SDKCONFIG defaults file; repeat for multiple files")
    parser.add_argument("--sdk", type=Path, default=Path(os.environ.get("IDF_PATH", SDK)))
    parser.add_argument("--tools", type=Path, default=Path(os.environ.get("IDF_TOOLS_PATH", TOOLS)))
    args = parser.parse_args()
    if not re.fullmatch(r"esp32[a-z0-9]*", args.target):
        parser.error("--target must be an ESP-IDF chip identifier, for example esp32c6")
    for path in args.defaults:
        if not path.is_file():
            parser.error(f"SDKCONFIG defaults file not found: {path}")
    sdk = args.sdk.resolve()
    tools = args.tools.resolve()
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    revision = subprocess.run(
        ["git", "-C", str(sdk), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != manifest["commit"]:
        raise RuntimeError("Installed SDK does not match the official source revision")
    candidates = list((tools / "python_env").glob("idf5.5_py*_env/bin/python"))
    if len(candidates) != 1:
        raise RuntimeError("Install one isolated ESP-IDF 5.5 Python environment first")
    python = candidates[0]
    environment = {
        **os.environ, "IDF_PATH": str(sdk), "IDF_TOOLS_PATH": str(tools),
        "IDF_PYTHON_ENV_PATH": str(python.parent.parent),
    }
    exported = subprocess.run(
        [str(python), str(sdk / "tools/idf_tools.py"), "export", "--format", "key-value"],
        env=environment, capture_output=True, text=True, check=True,
    )
    for line in exported.stdout.splitlines():
        if "=" not in line:
            raise RuntimeError("Unexpected SDK environment export")
        name, value = line.split("=", 1)
        if name not in ("PATH", "IDF_PYTHON_ENV_PATH", "OPENOCD_SCRIPTS", "ESP_ROM_ELF_DIR",
                        "ESP_IDF_VERSION", "IDF_DEACTIVATE_FILE_PATH"):
            raise RuntimeError(f"Unexpected SDK environment key: {name}")
        environment[name] = value.replace("${PATH}", os.environ["PATH"]).replace("$PATH", os.environ["PATH"])
    defaults_files = [ROOT / "upstream/sdkconfig.defaults"]
    chip_defaults = ROOT / "upstream" / f"sdkconfig.defaults.{args.target}"
    if chip_defaults.is_file():
        defaults_files.append(chip_defaults)
    defaults_files.extend(path.resolve() for path in args.defaults)
    defaults = ";".join(str(path) for path in defaults_files)
    build = BUILD / args.target
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(python), str(sdk / "tools/idf.py"), "-C", str(ROOT / "upstream"), "-B", str(build),
            "-D", f"IDF_TARGET={args.target}", "-D", f"SDKCONFIG={build / 'sdkconfig'}",
            "-D", f"SDKCONFIG_DEFAULTS={defaults}", "build",
        ],
        env=environment, check=True,
    )


if __name__ == "__main__":
    main()
