"""Lightness of a multi-paint mix follows total pigment, not each paint's share."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paint_manager


def _curve(labs_by_ratio):
    return {
        "samples": [
            {"ratio": ratio, "rgb": [0, 0, 0], "lab": lab}
            for ratio, lab in labs_by_ratio
        ]
    }


RED = _curve([
    (0.0, [100.0, 0.0, 0.0]),
    (0.2, [82.0, 28.0, 18.0]),
    (0.4, [64.0, 42.0, 30.0]),
    (1.0, [42.0, 55.0, 40.0]),
])
BLUE = _curve([
    (0.0, [100.0, 0.0, 0.0]),
    (0.2, [82.0, 8.0, -22.0]),
    (0.4, [64.0, 14.0, -36.0]),
    (1.0, [40.0, 22.0, -50.0]),
])


class MixLightnessTests(unittest.TestCase):
    def test_one_paint_matches_calibration_lightness(self):
        lab = paint_manager._predict_mix_lab_from_components(
            [(RED, None, 0.2)],
            white_ratio=0.8,
            library_group=None,
            paint_ids=["red"],
        )
        self.assertIsNotNone(lab)
        self.assertAlmostEqual(lab[0], 82.0, places=4)
        self.assertAlmostEqual(lab[1], 28.0, places=4)
        self.assertAlmostEqual(lab[2], 18.0, places=4)

    def test_two_paints_use_combined_pigment_load(self):
        # 20% red + 20% blue + 60% white is a 40% pigment load.
        # The old lookup treated each paint as a 20% tint (L* 82) and averaged
        # those pale swatches, so the solver kept adding pigment.
        lab = paint_manager._predict_mix_lab_from_components(
            [(RED, None, 0.2), (BLUE, None, 0.2)],
            white_ratio=0.6,
            library_group=None,
            paint_ids=["red", "blue"],
        )
        self.assertIsNotNone(lab)
        # Each 40% swatch is L* 64. The mix is darker than that, because the
        # two hues do not reflect the same light, and it stays warm of neutral.
        self.assertLess(lab[0], 63.5)
        self.assertGreater(lab[0], 50.0)
        self.assertGreater(lab[1], 10.0)

    def test_same_lightness_different_hues_mix_darker(self):
        lab = paint_manager._combine_labs_subtractive(
            np.array([[50.0, 55.0, 30.0], [50.0, 5.0, -50.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
        self.assertLess(lab[0], 45.0)
        self.assertGreater(lab[0], 30.0)
        self.assertGreater(lab[1], 15.0)

    def test_same_colour_keeps_its_lightness(self):
        lab = paint_manager._combine_labs_subtractive(
            np.array([[50.0, 40.0, 20.0], [50.0, 40.0, 20.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
        self.assertAlmostEqual(lab[0], 50.0, delta=0.5)
        self.assertAlmostEqual(lab[1], 40.0, delta=0.5)
        self.assertAlmostEqual(lab[2], 20.0, delta=0.5)

    def test_similar_hues_barely_darken(self):
        lab = paint_manager._combine_labs_subtractive(
            np.array([[50.0, 55.0, 30.0], [50.0, 48.0, 28.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
        self.assertAlmostEqual(lab[0], 50.0, delta=1.0)

    def test_batch_matches_scalar_for_same_recipe(self):
        scalar = paint_manager._predict_mix_lab_from_components(
            [(RED, None, 0.25), (BLUE, None, 0.15)],
            white_ratio=0.6,
            library_group=None,
            paint_ids=["red", "blue"],
        )
        batch = paint_manager._predict_mix_lab_batch(
            [RED, BLUE],
            [None, None],
            [1.0, 1.0],
            np.array([[0.25, 0.15]]),
            min_white_ratio=0.0,
            max_total_pigment=1.0,
            library_group=None,
            paint_ids=["red", "blue"],
        )
        self.assertEqual(batch.shape, (1, 3))
        np.testing.assert_allclose(batch[0], scalar, atol=0.05)


if __name__ == "__main__":
    unittest.main()
