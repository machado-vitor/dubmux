"""find_speed must only stretch the donor when the stretch is proven."""
import contextlib
import io
import unittest

import dubmux

DUR_T = 1355.4          # a 22.6 min episode


def fake_offset_at(offset_of, conf_of, unmeasurable=()):
    """offset_at stand-in driven by a model of the donor.

    offset_of(t, speed) -> seconds; conf_of(speed) -> lock confidence (None =
    no lock at that speed); unmeasurable: (t, speed) pairs that return nothing.
    """
    def offset_at(target, t_idx, donor, d_idx, start, win, search, guess=0.0,
                  speed=1.0):
        conf = conf_of(speed)
        if conf is None or any(abs(start - t) < 1 and abs(speed - s) < 1e-9
                               for t, s in unmeasurable):
            return None, 0.0
        return offset_of(start, speed), conf
    return offset_at


class FindSpeedTests(unittest.TestCase):
    def setUp(self):
        self._orig = dubmux.offset_at

    def tearDown(self):
        dubmux.offset_at = self._orig

    def run_find_speed(self, dur_d):
        with contextlib.redirect_stdout(io.StringIO()):
            sp, _ = dubmux.find_speed("target", "donor", 0, 1, {"duration": DUR_T},
                                      {"duration": dur_d}, 30)
        return sp

    def test_longer_donor_with_a_step_keeps_nominal_speed(self):
        # Same speed, one edit step at 12 min, 3.4 s of extra runtime. The
        # file-length ratio wins on spread but locks 25x worse than nominal and
        # the middle probe cannot measure it: stretching here was the bug.
        dur_d = DUR_T + 3.4
        ratio = dur_d / DUR_T

        def offset_of(t, speed):
            base = 1.2 if t < 720 else -1.6
            return base + (speed - 1.0) * t

        def conf_of(speed):
            if abs(speed - 1.0) < 1e-9:
                return 186.4
            return 7.3 if abs(speed - ratio) < 1e-9 else None

        dubmux.offset_at = fake_offset_at(offset_of, conf_of,
                                          unmeasurable=[(DUR_T * 0.5, ratio)])
        self.assertEqual(self.run_find_speed(dur_d), 1.0)

    def test_fitted_speed_needs_the_middle_to_confirm_it(self):
        # Comparable locks, but the mid-runtime probe measures nothing at the
        # fitted speed: no proof, so no stretch.
        dur_d = DUR_T * 1.004
        ratio = dur_d / DUR_T
        conf = {1.0: 4.0, ratio: 5.0}
        dubmux.offset_at = fake_offset_at(
            lambda t, speed: 0.8 + (speed - ratio) * t,
            lambda speed: next((c for s, c in conf.items() if abs(speed - s) < 1e-9), None),
            unmeasurable=[(DUR_T * 0.5, ratio)])
        self.assertEqual(self.run_find_speed(dur_d), 1.0)

    def test_genuine_pal_donor_is_stretched(self):
        # A PAL rip runs 4.27% fast: nothing locks at nominal speed, and at the
        # PAL ratio both far probes land on the same offset.
        pal = dubmux.PAL_RATIO
        dubmux.offset_at = fake_offset_at(
            lambda t, speed: 0.5 + (speed - pal) * t,
            lambda speed: 5.0 if abs(speed - pal) < 1e-9 else None)
        self.assertAlmostEqual(self.run_find_speed(DUR_T / pal), pal, places=9)

    def test_genuine_small_drift_is_corrected(self):
        # A real 0.15% speed difference: the offset grows linearly, the middle
        # probe sits on the line, and the fitted speed locks better than nominal.
        true = 1.0015

        def conf_of(speed):
            if abs(speed - 1.0) < 1e-9:
                return 6.0
            return 9.0 if abs(speed - true) < 1e-6 else None

        dubmux.offset_at = fake_offset_at(lambda t, speed: 0.3 + (speed - true) * t, conf_of)
        self.assertAlmostEqual(self.run_find_speed(DUR_T * 0.9985), true, places=6)


if __name__ == "__main__":
    unittest.main()
