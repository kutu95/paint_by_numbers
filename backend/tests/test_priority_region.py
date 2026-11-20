import unittest

import numpy as np

from image_processor import quantize_for_pipeline


class PriorityRegionTests(unittest.TestCase):
    def test_lasso_keeps_a_small_saturated_patch_inside_a_dark_figure(self):
        image = np.full((90, 120, 3), (150, 190, 220), dtype=np.uint8)
        image[15:78, 30:70] = (28, 24, 22)
        body = image[15:78, 30:70].astype(np.int16)
        shift = np.array([[0, 0, 0], [8, 5, 4], [3, 2, 1], [12, 7, 5]], dtype=np.int16)
        pick = (np.arange(body.shape[0])[:, None] + np.arange(body.shape[1])[None, :]) % shift.shape[0]
        image[15:78, 30:70] = np.clip(body + shift[pick], 0, 255).astype(np.uint8)
        image[48:66, 40:58] = (190, 42, 48)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[15:78, 30:70] = 1.0

        labels, _quantized, _palette = quantize_for_pipeline(
            image,
            8,
            saturation_boost=1.0,
            style_preset="none",
            favor_skin_tones=False,
            priority_region_mask=mask,
            priority_region_strength=0.7,
            mask_detail_level=0.5,
        )

        shorts = labels[48:66, 40:58]
        body = labels[20:40, 34:66]
        shorts_label = int(np.bincount(shorts.ravel()).argmax())
        body_label = int(np.bincount(body.ravel()).argmax())
        self.assertNotEqual(shorts_label, body_label)
        shorts_fraction = float(np.mean(shorts == shorts_label))
        self.assertGreater(shorts_fraction, 0.8)


if __name__ == "__main__":
    unittest.main()
