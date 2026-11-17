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
        # The 40% swatches are L* 64. The old per-paint lookup predicted L* 82.
        self.assertAlmostEqual(lab[0], 64.0, delta=2.0)
        self.assertGreater(lab[1], 10.0)

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
