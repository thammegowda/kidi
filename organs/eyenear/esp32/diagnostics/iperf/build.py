import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SDK = ROOT / ".sdk"
TOOLS = ROOT / ".tools"
BUILD = ROOT / ".build"


def main():
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    revision = subprocess.run(
        ["git", "-C", str(SDK), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != manifest["commit"]:
        raise RuntimeError("Installed SDK does not match the official source revision")
    candidates = list((TOOLS / "python_env").glob("idf5.5_py*_env/bin/python"))
    if len(candidates) != 1:
        raise RuntimeError("Install one isolated ESP-IDF 5.5 Python environment first")
    python = candidates[0]
    environment = {
        **os.environ, "IDF_PATH": str(SDK), "IDF_TOOLS_PATH": str(TOOLS),
        "IDF_PYTHON_ENV_PATH": str(python.parent.parent),
    }
    exported = subprocess.run(
        [str(python), str(SDK / "tools/idf_tools.py"), "export", "--format", "key-value"],
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
    defaults = ";".join(str(path) for path in (
        ROOT / "upstream/sdkconfig.defaults",
        ROOT / "upstream/sdkconfig.defaults.esp32s3",
        ROOT / "usb.defaults",
    ))
    BUILD.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(python), str(SDK / "tools/idf.py"), "-C", str(ROOT / "upstream"), "-B", str(BUILD),
            "-D", "IDF_TARGET=esp32s3", "-D", f"SDKCONFIG={BUILD / 'sdkconfig'}",
            "-D", f"SDKCONFIG_DEFAULTS={defaults}", "build",
        ],
        env=environment, check=True,
    )


if __name__ == "__main__":
    main()
