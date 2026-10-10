import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iperf_build", ROOT / "diagnostics/iperf/build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class IperfBuildTest(unittest.TestCase):
    def test_target_and_defaults_are_validated_before_sdk_commands(self):
        for arguments in ([], ["--target", "wrong-chip"], [
            "--target", "esp32c6", "--defaults", "/nonexistent/kidi-sdkconfig.defaults",
        ]):
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", ["build.py", *arguments]),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(build.subprocess, "run") as runner,
                self.assertRaises(SystemExit) as error,
            ):
                build.main()
            self.assertEqual(error.exception.code, 2)
            runner.assert_not_called()

    def test_explicit_target_has_isolated_build_and_no_board_flash_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "upstream").mkdir()
            tools = root / "tools"
            python = tools / "python_env/idf5.5_py3.12_env/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            sdk = root / "sdk"
            sdk.mkdir()
            extra = root / "board.defaults"
            extra.write_text("CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y\n")
            revision = "pinned-sdk-revision"
            (root / "source-manifest.json").write_text(json.dumps({"commit": revision}))
            results = [
                subprocess.CompletedProcess([], 0, stdout=revision + "\n"),
                subprocess.CompletedProcess([], 0, stdout="PATH=/sdk/bin:$PATH\n"),
                subprocess.CompletedProcess([], 0),
            ]
            arguments = [
                "build.py", "--target", "esp32c6", "--sdk", str(sdk),
                "--tools", str(tools), "--defaults", str(extra),
            ]
            with (
                patch("sys.argv", arguments),
                patch.object(build, "ROOT", root),
                patch.object(build, "BUILD", root / ".build"),
                patch.object(build.subprocess, "run", side_effect=results) as runner,
            ):
                build.main()
            command = runner.call_args.args[0]
            self.assertIn("IDF_TARGET=esp32c6", command)
            self.assertIn(str(root / ".build/esp32c6"), command)
            self.assertIn(f"SDKCONFIG={root / '.build/esp32c6/sdkconfig'}", command)
            self.assertIn(
                f"SDKCONFIG_DEFAULTS={root / 'upstream/sdkconfig.defaults'};{extra}", command
            )
            self.assertFalse(any("usb.defaults" in argument or "esp32s3" in argument for argument in command))
            self.assertEqual(runner.call_args.kwargs["env"]["IDF_PATH"], str(sdk))

    def test_wrong_sdk_revision_fails_before_build(self):
        with (
            patch("sys.argv", ["build.py", "--target", "esp32c6"]),
            patch.object(build.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout="different-sdk\n",
            )) as runner,
            self.assertRaisesRegex(RuntimeError, "official source revision"),
        ):
            build.main()
        self.assertEqual(runner.call_count, 1)


if __name__ == "__main__":
    unittest.main()
