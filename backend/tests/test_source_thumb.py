import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from project_store import (
    DISPLAY_MAX_SIDE,
    THUMB_MAX_SIDE,
    ensure_source_thumb,
    write_source_display_for_oriented,
    write_source_thumb_for_oriented,
)


class SourceThumbTests(unittest.TestCase):
    def test_thumb_is_a_small_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            oriented = Path(tmp) / "oriented.jpg"
            image = np.random.default_rng(0).integers(0, 255, (1200, 2400, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(oriented), image))
            self.assertTrue(write_source_thumb_for_oriented(oriented))
            thumb = Path(tmp) / "thumb.jpg"
            self.assertTrue(thumb.is_file())
            self.assertLess(thumb.stat().st_size, oriented.stat().st_size)
            decoded = cv2.imread(str(thumb))
            self.assertIsNotNone(decoded)
            height, width = decoded.shape[:2]
            self.assertEqual(max(height, width), THUMB_MAX_SIDE)
            self.assertLess(min(height, width), THUMB_MAX_SIDE)

    def test_display_jpeg_is_capped_for_onscreen_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            oriented = Path(tmp) / "oriented.jpg"
            image = np.zeros((2000, 4000, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(oriented), image))
            self.assertTrue(write_source_display_for_oriented(oriented))
            display = Path(tmp) / "display.jpg"
            decoded = cv2.imread(str(display))
            self.assertIsNotNone(decoded)
            height, width = decoded.shape[:2]
            self.assertEqual(max(height, width), DISPLAY_MAX_SIDE)
            self.assertLess(display.stat().st_size, oriented.stat().st_size)

    def test_current_thumb_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "proj" / "source"
            source.mkdir(parents=True)
            cv2.imwrite(str(source / "oriented.jpg"), np.zeros((600, 800, 3), dtype=np.uint8))
            with patch("project_store.PROJECTS_DIR", root):
                first = ensure_source_thumb("proj")
                self.assertIsNotNone(first)
                stamp = first.stat().st_mtime_ns
                again = ensure_source_thumb("proj")
            self.assertEqual(again.stat().st_mtime_ns, stamp)


if __name__ == "__main__":
    unittest.main()
