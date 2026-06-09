"""Tests for economical paint mask generation."""
import unittest

import numpy as np

from image_processor import (
    build_paint_masks_economical,
    smart_overpaint_expansion,
    fill_holes_covered_by_later_layers,
    ensure_complete_coverage,
    mask_paint_overlap_stats,
    ensure_base_masks_complete_coverage,
)


def _synthetic_labels(h: int = 120, w: int = 160) -> np.ndarray:
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:, : w // 3] = 0
    labels[:, w // 3 : 2 * w // 3] = 1
    labels[:, 2 * w // 3 :] = 2
    labels[10:14, w // 3 + 2 : w // 3 + 6] = 0
    labels[100:104, 2 * w // 3 + 2 : 2 * w // 3 + 6] = 2
    return labels


def _base_masks_from_labels(labels: np.ndarray) -> dict:
    n = int(labels.max()) + 1
    base = {}
    for idx in range(n):
        base[idx] = ((labels == idx).astype(np.uint8)) * 255
    ensure_base_masks_complete_coverage(base, labels, n)
    return base


class TestPaintMasks(unittest.TestCase):
    def test_economical_covers_all_label_pixels(self):
        labels = _synthetic_labels()
        order = [0, 1, 2]
        base = _base_masks_from_labels(labels)
        masks = build_paint_masks_economical(
            base, labels, order, overpaint_mm=5.0, max_side=1000
        )
        for idx in order:
            owner = (labels == idx).astype(np.uint8) * 255
            self.assertTrue(np.all((masks[idx] > 0) | (owner == 0)))

    def test_economical_less_redundant_paint_than_legacy_pipeline(self):
        labels = _synthetic_labels()
        order = [0, 1, 2]
        base = _base_masks_from_labels(labels)
        h, w = labels.shape
        palette = [
            {"index": 0, "rgb": [200, 0, 0], "coverage": 33.0},
            {"index": 1, "rgb": [0, 200, 0], "coverage": 33.0},
            {"index": 2, "rgb": [0, 0, 200], "coverage": 34.0},
        ]
        quantized = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(3):
            quantized[labels == idx] = palette[idx]["rgb"]

        economical = build_paint_masks_economical(
            base, labels, order, overpaint_mm=5.0, max_side=1000
        )
        legacy = smart_overpaint_expansion(base, order, overpaint_mm=5.0, max_side=1000)
        legacy = ensure_complete_coverage(legacy, order, quantized, labels, palette)
        legacy = fill_holes_covered_by_later_layers(legacy, order)

        eco_stats = mask_paint_overlap_stats(economical, labels, order)
        leg_stats = mask_paint_overlap_stats(legacy, labels, order)

        self.assertLessEqual(
            eco_stats["total_paint_pixels"], leg_stats["total_paint_pixels"]
        )
        self.assertLessEqual(
            eco_stats["redundant_pixels"], leg_stats["redundant_pixels"]
        )

    def test_registration_fringe_only_on_later_colors(self):
        labels = _synthetic_labels()
        order = [0, 1, 2]
        base = _base_masks_from_labels(labels)
        masks = build_paint_masks_economical(
            base, labels, order, overpaint_mm=8.0, max_side=1000
        )
        wrong = (masks[0] > 0) & (labels == 2)
        self.assertEqual(int(np.sum(wrong)), 0)


if __name__ == "__main__":
    unittest.main()
