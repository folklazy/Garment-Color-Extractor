"""Unit tests for Color Spaces & Distance calculations."""
import unittest
import numpy as np
from garment_color.color_spaces import (
    delta_e_cie76,
    delta_e_weighted,
    rgb_to_hex,
    srgb_to_cielab,
    srgb_to_hsv,
)

class TestColorSpaces(unittest.TestCase):
    def test_rgb_to_hex(self):
        self.assertEqual(rgb_to_hex(255, 0, 0), "#FF0000")
        self.assertEqual(rgb_to_hex(0, 255, 0), "#00FF00")
        self.assertEqual(rgb_to_hex(0, 0, 255), "#0000FF")
        self.assertEqual(rgb_to_hex(20, 20, 20), "#141414")

    def test_srgb_to_cielab(self):
        pure_white = np.array([[255, 255, 255]], dtype=np.uint8)
        lab_white = srgb_to_cielab(pure_white)
        self.assertAlmostEqual(lab_white[0, 0], 100.0, places=1)
        self.assertAlmostEqual(lab_white[0, 1], 0.0, delta=1.0)
        self.assertAlmostEqual(lab_white[0, 2], 0.0, delta=1.0)

        pure_black = np.array([[0, 0, 0]], dtype=np.uint8)
        lab_black = srgb_to_cielab(pure_black)
        self.assertAlmostEqual(lab_black[0, 0], 0.0, places=1)

    def test_delta_e_weighted(self):
        lab1 = np.array([50.0, 10.0, 20.0], dtype=np.float32)
        lab2 = np.array([50.0, 10.0, 20.0], dtype=np.float32)
        d = delta_e_weighted(lab1, lab2[None, :])
        self.assertAlmostEqual(float(d[0]), 0.0, places=2)

if __name__ == "__main__":
    unittest.main()

