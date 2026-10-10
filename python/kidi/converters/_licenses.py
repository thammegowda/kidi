"""License-bundle helpers shared by model converters."""

import shutil
from pathlib import Path


LICENSE_ROOT = Path(__file__).with_name("licenses")


def write_license_bundle(
    destination: Path,
    static_files: dict[str, str],
    notice: str,
    copied_files: dict[str, Path] | None = None,
) -> list[str]:
    licenses = destination / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    result = ["NOTICE"]
    for output_name, source_name in static_files.items():
        source = LICENSE_ROOT / source_name
        if not source.is_file():
            raise ValueError(f"Missing bundled converter license: {source}")
        shutil.copy2(source, licenses / output_name)
        result.append(f"licenses/{output_name}")
    for output_name, source in (copied_files or {}).items():
        if not source.is_file():
            raise ValueError(f"Missing source model license: {source}")
        shutil.copy2(source, licenses / output_name)
        result.append(f"licenses/{output_name}")
    (destination / "NOTICE").write_text(notice.rstrip() + "\n", encoding="utf-8")
    return result
