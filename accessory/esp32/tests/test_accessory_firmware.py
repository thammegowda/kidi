from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def defaults(chip):
    values = {}
    for line in (ROOT / f"firmware/{chip}/sdkconfig.defaults").read_text().splitlines():
        if line.startswith("CONFIG_") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


class AccessoryFirmwareTest(unittest.TestCase):
    def test_product_sources_are_flat_cpp23(self):
        firmware = ROOT / "firmware"
        for chip in ("c6", "p4"):
            cmake = (firmware / chip / "CMakeLists.txt").read_text()
            self.assertIn("set(CMAKE_CXX_STANDARD 23)", cmake)
        owned_roots = [
            firmware / "c6/main",
            firmware / "p4/main",
            *(path for path in (firmware / "components").iterdir() if path.name.startswith("kidi_")),
        ]
        for root in owned_roots:
            self.assertFalse(list(root.rglob("*.c")), root)
            self.assertFalse((root / "include").exists(), root)
            for implementation in root.glob("*.cpp"):
                if implementation.stem.endswith("_main"):
                    continue
                self.assertTrue(
                    implementation.with_suffix(".h").exists(),
                    f"{implementation} must colocate its header",
                )

    def test_c6_owns_control_range_and_host_wake(self):
        config = defaults("c6")
        self.assertEqual(config["CONFIG_IDF_TARGET"], '"esp32c6"')
        self.assertEqual(config["CONFIG_LWIP_TCP_LOCAL_PORT_RANGE_START"], "61440")
        self.assertEqual(config["CONFIG_LWIP_TCP_LOCAL_PORT_RANGE_END"], "65535")
        self.assertEqual(config["CONFIG_ESP_HOSTED_CP_FEAT_NW_SPLIT"], "y")
        self.assertEqual(config["CONFIG_ESP_HOSTED_CP_FEAT_HOST_PS"], "y")
        self.assertEqual(config["CONFIG_ESP_HOSTED_CP_FEAT_HOST_PS_HOST_WAKEUP_GPIO"], "2")
        self.assertEqual(config["CONFIG_BT_NIMBLE_EXT_ADV"], "y")

    def test_p4_owns_media_range_and_uses_board_wiring(self):
        config = defaults("p4")
        self.assertEqual(config["CONFIG_IDF_TARGET"], '"esp32p4"')
        self.assertEqual(config["CONFIG_ESP32P4_SELECTS_REV_LESS_V3"], "y")
        self.assertEqual(config["CONFIG_ESP32P4_REV_MIN_100"], "y")
        self.assertEqual(config["CONFIG_LWIP_TCP_LOCAL_PORT_RANGE_START"], "49152")
        self.assertEqual(config["CONFIG_LWIP_TCP_LOCAL_PORT_RANGE_END"], "61439")
        self.assertEqual(config["CONFIG_ESP_HOSTED_HOST_FEAT_NW_SPLIT"], "y")
        self.assertEqual(config["CONFIG_ESP_HOSTED_HOST_FEAT_POWER_SAVE_WAKEUP_GPIO"], "6")
        self.assertEqual(config["CONFIG_ESP_HOSTED_HOST_RESET_GPIO"], "54")
        self.assertEqual(config["CONFIG_ESP_HOSTED_CP_TARGET_ESP32C6"], "y")

    def test_hosted_versions_are_locked_and_no_flash_target_exists(self):
        for chip in ("c6", "p4"):
            lock = (ROOT / f"firmware/{chip}/dependencies.lock").read_text()
            self.assertIn("version: 3.0.9", lock)
            self.assertIn("version: 5.5.5", lock)
        makefile = (ROOT / "Makefile").read_text()
        self.assertNotIn("\nflash:", makefile)
        self.assertNotIn("\nrestore:", makefile)


if __name__ == "__main__":
    unittest.main()
