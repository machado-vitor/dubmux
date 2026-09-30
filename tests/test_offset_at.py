"""offset_at must degrade to "no measurement" when the donor window is short."""
import unittest

import numpy as np

import dubmux

SR = dubmux.SR


def fake_windows(target_s, donor_s, lag_s=0.0):
    """read_window stand-in: a shared noise track, donor delayed by lag_s."""
    rng = np.random.default_rng(7)
    track = rng.standard_normal(int(SR * 400)).astype(np.float32)

    def read_window(path, idx, start, dur, speed=1.0):
        if path == "target":
            a = int(start * SR)
            return track[a:a + int(target_s * SR)]
        a = int((start - lag_s) * SR)
        return track[a:a + int(min(dur, donor_s) * SR)]

    return read_window


class OffsetAtTests(unittest.TestCase):
    def setUp(self):
        self._orig = (dubmux.read_window, dubmux.MATCH_MODE["v"])

    def tearDown(self):
        dubmux.read_window, dubmux.MATCH_MODE["v"] = self._orig

    def test_phat_short_donor_window_returns_no_measurement(self):
        dubmux.MATCH_MODE["v"] = "phat"
        dubmux.read_window = fake_windows(target_s=90, donor_s=60)
        off, conf = dubmux.offset_at("target", 0, "donor", 1, 120.0, 90, 30)
        self.assertIsNone(off)
        self.assertEqual(conf, 0.0)

    def test_phat_full_donor_window_finds_the_lag(self):
        dubmux.MATCH_MODE["v"] = "phat"
        dubmux.read_window = fake_windows(target_s=90, donor_s=150, lag_s=2.5)
        off, conf = dubmux.offset_at("target", 0, "donor", 1, 120.0, 90, 30)
        assert off is not None
        # the donor's content runs lag_s late, so it must be pulled earlier
        self.assertAlmostEqual(off, -2.5, delta=0.002)
        self.assertGreater(conf, 1.35)


if __name__ == "__main__":
    unittest.main()
