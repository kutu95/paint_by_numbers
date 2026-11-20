import unittest

import cv2
import numpy as np

from image_processor import _restore_palette_chroma, quantize_lab


def _chroma(rgb) -> float:
    lab = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return float(np.hypot(lab[1] - 128.0, lab[2] - 128.0))


class PaletteChromaTests(unittest.TestCase):
    def test_mixed_region_moves_toward_its_vivid_pixels(self):
        image = np.zeros((40, 100, 3), dtype=np.uint8)
        image[:, :70] = (90, 115, 75)
        image[:, 70:] = (30, 190, 40)
        labels = np.zeros(image.shape[:2], dtype=np.int32)
        mean = image.reshape(-1, 3).mean(axis=0).astype(int)
        palette = [{"index": 0, "rgb": mean.tolist(), "hex": "#000000"}]
        restored = _restore_palette_chroma(image, labels, palette)[0]["rgb"]
        self.assertGreater(_chroma(restored), _chroma(mean) + 2.0)
        self.assertLess(_chroma(restored), _chroma((30, 190, 40)))
        self.assertGreater(restored[1], restored[0])
        self.assertGreater(restored[1], restored[2])

    def test_neutral_region_stays_neutral(self):
        image = np.full((40, 40, 3), (140, 142, 138), dtype=np.uint8)
        labels = np.zeros(image.shape[:2], dtype=np.int32)
        palette = [{"index": 0, "rgb": [140, 142, 138], "hex": "#8c8e8a"}]
        restored = _restore_palette_chroma(image, labels, palette)[0]["rgb"]
        self.assertLess(_chroma(restored), 8.0)

    def test_forced_colour_is_not_shifted(self):
        image = np.zeros((40, 100, 3), dtype=np.uint8)
        image[:, :70] = (90, 115, 75)
        image[:, 70:] = (30, 190, 40)
        labels = np.zeros(image.shape[:2], dtype=np.int32)
        palette = [{"index": 0, "rgb": [10, 20, 30], "hex": "#0a141e", "must_include": True}]
        restored = _restore_palette_chroma(image, labels, palette)[0]
        self.assertEqual(restored["rgb"], [10, 20, 30])
        self.assertEqual(restored["hex"], "#0a141e")

    def test_quantize_palette_is_at_least_as_vivid_as_the_region_average(self):
        rng = np.random.default_rng(0)
        base = np.array([70, 110, 60], dtype=np.int16)
        noise = rng.integers(-25, 50, (80, 120, 3), dtype=np.int16)
        noise[:, :, 1] += 30
        image = np.clip(base + noise, 0, 255).astype(np.uint8)
        _labels, _quantized, palette = quantize_lab(image, 4, seed=1)
        means = []
        labels = _labels.ravel()
        flat = image.reshape(-1, 3)
        for entry in palette:
            sel = flat[labels == entry["index"]]
            if sel.shape[0] < 32:
                continue
            means.append(_chroma(sel.mean(axis=0)))
            self.assertGreaterEqual(_chroma(entry["rgb"]) + 0.5, _chroma(sel.mean(axis=0)))
        self.assertGreater(float(np.mean([_chroma(e["rgb"]) for e in palette])), float(np.mean(means)))


if __name__ == "__main__":
    unittest.main()
