import unittest

import speed


class SpeedTest(unittest.TestCase):
    def test_pattern_handles_partial_and_unaligned_blocks(self):
        for offset in (0, 127, 255, 256, 640):
            for size in (1, 255, 640, 4096):
                with self.subTest(offset=offset, size=size):
                    self.assertEqual(
                        speed.pattern(offset, size),
                        bytes((offset + index) & 0xff for index in range(size)),
                    )

    def test_invalid_limits_fail_before_connecting(self):
        for direction, size, block in (
            ("download", 1, 4096), ("upload", speed.MAX_BYTES + 1, 4096),
            ("download", 8192, 16385), ("invalid", 8192, 4096),
        ):
            with self.subTest(direction=direction, size=size, block=block), self.assertRaises(ValueError):
                speed.transfer({}, direction, size, block)


if __name__ == "__main__":
    unittest.main()
