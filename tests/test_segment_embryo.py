import unittest

import numpy as np

from qlivecell.embseg.core.utils_ES import segment_embryo


class TestSegmentEmbryo(unittest.TestCase):
    def test_segment_bright_region_in_uint8_image(self):
        y, x = np.ogrid[:48, :48]
        image = np.full((48, 48), 5, dtype=np.uint8)
        image[(y - 24) ** 2 + (x - 24) ** 2 < 13**2] = 220

        embryo, background, level_set, embryo_pixels, background_pixels = (
            segment_embryo(image, 60, (5, 3, 6, 2, 40))
        )

        self.assertEqual(level_set.shape, image.shape)
        self.assertEqual(embryo.dtype, image.dtype)
        np.testing.assert_array_equal(embryo + background, image)
        self.assertEqual(len(embryo_pixels) + len(background_pixels), image.size)
        self.assertGreater(embryo[24, 24], background[24, 24])
