import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_processor import (
    load_rgb_image_oriented,
    refresh_project_color_management,
)

DISPLAY_P3 = Path("/System/Library/ColorSync/Profiles/Display P3.icc")


def _mean_saturation(rgb: np.ndarray) -> float:
    pixels = rgb.astype(np.float32)
    high = pixels.max(axis=2)
    low = pixels.min(axis=2)
    sat = np.where(high > 1, (high - low) / np.maximum(high, 1), 0)
    return float(sat.mean())


class ColorProfileTests(unittest.TestCase):
    def test_untagged_png_keeps_its_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.png"
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[:, :, 0] = 180
            rgb[:, :, 1] = 40
            rgb[:, :, 2] = 20
            Image.fromarray(rgb).save(path)
            loaded = load_rgb_image_oriented(str(path))
        self.assertEqual(loaded.shape, rgb.shape)
        self.assertTrue(np.array_equal(loaded, rgb))

    @unittest.skipUnless(DISPLAY_P3.is_file(), "Display P3 profile is not on this machine")
    def test_display_p3_jpeg_is_converted_to_srgb(self):
        profile = DISPLAY_P3.read_bytes()
        # A red that is inside Display P3. Read as sRGB it is milder than the converted colour.
        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        rgb[:, :] = (210, 40, 50)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p3.jpg"
            Image.fromarray(rgb).save(path, format="JPEG", quality=95, icc_profile=profile)
            loaded = load_rgb_image_oriented(str(path))
            raw = np.asarray(Image.open(path).convert("RGB"))
        self.assertGreater(_mean_saturation(loaded), _mean_saturation(raw) + 0.02)
        self.assertGreater(float(loaded[:, :, 0].mean()), float(raw[:, :, 0].mean()))

    @unittest.skipUnless(DISPLAY_P3.is_file(), "Display P3 profile is not on this machine")
    def test_wide_gamut_project_is_rewritten_once(self):
        profile = DISPLAY_P3.read_bytes()
        rgb = np.zeros((40, 80, 3), dtype=np.uint8)
        rgb[:, :] = (210, 40, 50)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "proj" / "source"
            source.mkdir(parents=True)
            original = source / "original.jpg"
            Image.fromarray(rgb).save(original, format="JPEG", quality=95, icc_profile=profile)
            # The dull copy the old pipeline would have saved: same numbers, no profile.
            Image.fromarray(rgb).save(source / "oriented.jpg", format="JPEG", quality=90)
            with patch("project_store.PROJECTS_DIR", root):
                refresh_project_color_management("proj")
                oriented = source / "oriented.jpg"
                stamp = oriented.stat().st_mtime_ns
                converted = load_rgb_image_oriented(str(oriented))
                refresh_project_color_management("proj")
            self.assertEqual((source / "color-managed-v1").read_text(), "converted")
            self.assertEqual(oriented.stat().st_mtime_ns, stamp)
            self.assertTrue((source / "display.jpg").is_file())
            self.assertGreater(_mean_saturation(converted), _mean_saturation(rgb) + 0.02)


if __name__ == "__main__":
    unittest.main()
