"""The faster recipe search must visit the same grid and pick the same winner."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paint_manager as pm


def _reference_walk(axes, max_total, min_ratio):
    """The previous depth-first grid, including its 1e-9 slack."""
    found = []

    def walk(depth, chosen, total):
        if depth == len(axes):
            found.append(list(chosen))
            return
        remaining = len(axes) - depth
        min_needed = (remaining - 1) * min_ratio
        max_this = min(float(axes[depth][-1]), max_total - total - min_needed)
        if max_this < min_ratio:
            return
        for ratio in axes[depth]:
            value = float(ratio)
            if value > max_this + 1e-9:
                break
            walk(depth + 1, chosen + [value], total + value)

    walk(0, [], 0.0)
    return found


class RecipeSearchSpeedTests(unittest.TestCase):
    def test_ratio_grid_keeps_the_walk_winner(self):
        axes = [
            np.arange(0.02, 0.42, 0.05),
            np.arange(0.02, 0.37, 0.05),
            np.arange(0.02, 0.5, 0.07),
            np.arange(0.02, 0.3, 0.04),
        ]
        max_total = 0.62
        reference = _reference_walk(axes, max_total, min_ratio=0.02)
        self.assertGreater(len(reference), 10)

        seen = []

        def evaluate(rows):
            seen.extend(rows.tolist())
            # Flat score: the first visited feasible point must win ties.
            return np.ones(rows.shape[0], dtype=np.float64)

        winner, error = pm._best_on_ratio_axes(
            axes, evaluate, max_total, min_ratio=0.02, chunk_size=3
        )
        self.assertEqual(winner, reference[0])
        self.assertEqual(error, 1.0)
        self.assertEqual(np.round(seen, 8).tolist(), np.round(reference, 8).tolist())

        axes3 = axes[:3]
        reference3 = _reference_walk(axes3, max_total, min_ratio=0.02)
        seen3 = []

        def evaluate3(rows):
            seen3.extend(rows.tolist())
            return np.ones(rows.shape[0], dtype=np.float64)

        winner3, _error3 = pm._best_on_ratio_axes(
            axes3, evaluate3, max_total, min_ratio=0.02, chunk_size=5
        )
        self.assertEqual(winner3, reference3[0])
        self.assertEqual(np.round(seen3, 8).tolist(), np.round(reference3, 8).tolist())

    def test_one_pigment_matches_scalar_grid(self):
        target = pm.rgb_to_lab([80, 140, 160])
        recipe = pm.find_best_one_pigment_recipe(target, "cool-blue", "#0831a8", "chromacryl")
        self.assertIsNotNone(recipe)
        calibration = pm._load_calibration_cached("cool-blue", "chromacryl")
        cal_max = max(0.0, min(1.0, pm.get_calibration_max_ratio(calibration)))
        _min_white, max_total = pm.get_white_mix_limits(target, "chromacryl")
        upper = min(max_total, cal_max)
        best_ratio = None
        best_error = float("inf")
        for test_ratio in np.arange(0.0, upper + 0.001, 0.01):
            predicted = pm.predict_mix_lab_for_paint_ratios(
                "chromacryl", ["cool-blue"], [float(test_ratio)], 1.0 - float(test_ratio)
            )
            error = pm.delta_e_lab(target, predicted)
            if error < best_error:
                best_error = float(error)
                best_ratio = float(test_ratio)
        self.assertEqual(recipe["pigment_ratio"], best_ratio)
        self.assertAlmostEqual(recipe["error"], best_error, places=6)

    def test_uncalibrated_tint_batch_matches_scalar(self):
        rgb = [200, 30, 40]
        samples = np.array([0.0, 1e-12, 0.01, 0.37, 0.5, 0.99, 1.0], dtype=np.float64)
        batch = pm._uncalibrated_tint_lab_batch(rgb, samples)
        for index, eff in enumerate(samples):
            scalar = pm._uncalibrated_tint_lab(rgb, float(eff))
            np.testing.assert_allclose(batch[index], scalar, atol=1e-8)

    def test_feedback_bias_cache_reloads_after_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = pm.FEEDBACK_BIAS_DIR
            pm.FEEDBACK_BIAS_DIR = Path(tmp)
            pm._FEEDBACK_BIAS_CACHE.clear()
            try:
                self.assertIsNone(pm.get_paint_bias("cache-group", "cool-blue"))
                pm.save_feedback_bias("cache-group", {"cool-blue": [1.0, -0.5, 0.25]})
                self.assertEqual(pm.get_paint_bias("cache-group", "Cool-Blue"), [1.0, -0.5, 0.25])
                loaded = pm.load_feedback_bias("cache-group")
                loaded["cool-blue"] = [9.0, 9.0, 9.0]
                self.assertEqual(pm.get_paint_bias("cache-group", "cool-blue"), [1.0, -0.5, 0.25])
            finally:
                pm.FEEDBACK_BIAS_DIR = original
                pm._FEEDBACK_BIAS_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
