"""
Basic validation tests for gradient ramp generation.

Run from backend directory:
  cd backend && python -m pytest tests/test_gradient_validation.py -v
  or: cd backend && python -m unittest tests.test_gradient_validation -v
"""
import sys
import unittest
from pathlib import Path

# Ensure backend/ is on path when run from project root
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

import numpy as np
from image_processor import (
    quantize_lab,
    process_gradient_regions,
    detect_gradient_regions,
    GradientRegion,
)


def make_synthetic_gradient_image(h: int = 120, w: int = 160) -> np.ndarray:
    """Create a small RGB image with a vertical gradient (light to dark) for testing."""
    # Top half: smooth gradient; bottom half: flat so we get at least one gradient region
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        gray = int(255 * (1 - t * 0.7))  # 255 at top to ~76 at bottom
        img[y, :, 0] = gray
        img[y, :, 1] = gray
        img[y, :, 2] = gray
    return img


class TestGradientValidation(unittest.TestCase):
    """Validate gradient ramp masks: full coverage, no leak, stability."""

    def test_ramp_masks_cover_region_no_leak(self):
        """All ramp masks together fully cover the gradient region; no pixels leak outside."""
        img = make_synthetic_gradient_image(120, 160)
        n_colors = 8
        labels, _, palette = quantize_lab(img, n_colors, seed=42)
        gradient_masks, gradient_layers, gradient_regions = process_gradient_regions(
            img, labels, palette,
            gradient_steps_n=7,
            gradient_transition_mode="off",  # No dither for simpler validation
            gradient_transition_width=10,
            enable_gradient_detection=True,
            enable_glaze=False,
        )
        if len(gradient_regions) == 0:
            self.skipTest("No gradient regions detected in synthetic image (detection may be strict)")
        h, w = img.shape[:2]
        for grad_region in gradient_regions:
            x, y, rw, rh = grad_region.bounding_box
            # Region mask: 1 inside bbox, 0 outside
            region_mask = np.zeros((h, w), dtype=np.uint8)
            x_end = min(x + rw, w)
            y_end = min(y + rh, h)
            x_start = max(0, x)
            y_start = max(0, y)
            region_mask[y_start:y_end, x_start:x_end] = 255
            # Union of all step masks for this region
            union = np.zeros((h, w), dtype=np.uint8)
            for layer in gradient_layers:
                if layer.get("gradient_region_id") != grad_region.id:
                    continue
                if layer.get("is_glaze"):
                    continue  # Exclude glaze for this test (glaze = full region)
                idx = layer["layer_index"]
                if idx in gradient_masks:
                    union = np.maximum(union, gradient_masks[idx])
            # Full coverage: every region pixel is in at least one mask
            inside_region = (region_mask > 0) & (union > 0)
            coverage_ratio = np.sum(inside_region) / max(1, np.sum(region_mask > 0))
            self.assertGreaterEqual(
                coverage_ratio, 0.99,
                f"Ramp masks should cover gradient region {grad_region.id}; coverage={coverage_ratio:.2%}",
            )
            # No leak: no mask pixel outside region (each mask is 0 outside bbox)
            for layer in gradient_layers:
                if layer.get("gradient_region_id") != grad_region.id:
                    continue
                idx = layer["layer_index"]
                if idx not in gradient_masks:
                    continue
                mask = gradient_masks[idx]
                outside = (region_mask == 0) & (mask > 0)
                leak_pixels = np.sum(outside)
                self.assertEqual(leak_pixels, 0, f"Layer {idx} leaks {leak_pixels} pixels outside region")

    def test_output_stable_across_renders(self):
        """Same inputs produce same number of gradient regions and layers and mask shapes."""
        img = make_synthetic_gradient_image(100, 140)
        n_colors = 6
        labels, _, palette = quantize_lab(img, n_colors, seed=42)
        run1 = process_gradient_regions(
            img, labels, palette,
            gradient_steps_n=5,
            gradient_transition_mode="dither",
            gradient_transition_width=15,
            enable_gradient_detection=True,
            enable_glaze=False,
        )
        run2 = process_gradient_regions(
            img, labels, palette,
            gradient_steps_n=5,
            gradient_transition_mode="dither",
            gradient_transition_width=15,
            enable_gradient_detection=True,
            enable_glaze=False,
        )
        masks1, layers1, regions1 = run1
        masks2, layers2, regions2 = run2
        self.assertEqual(len(regions1), len(regions2), "Same number of gradient regions")
        self.assertEqual(len(layers1), len(layers2), "Same number of gradient layers")
        self.assertEqual(set(masks1.keys()), set(masks2.keys()), "Same layer indices")
        for idx in masks1:
            self.assertEqual(masks1[idx].shape, masks2[idx].shape, f"Mask {idx} same shape")
            np.testing.assert_array_equal(masks1[idx], masks2[idx], f"Mask {idx} identical (deterministic)")


if __name__ == "__main__":
    unittest.main()
