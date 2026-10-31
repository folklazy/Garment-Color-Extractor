"""Unit tests for GarmentColorExtractor engine."""
import unittest
import numpy as np
from garment_color.core import GarmentColorExtractor

class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = GarmentColorExtractor()

    def test_pure_red_garment(self):
        img = np.full((100, 100, 3), (200, 30, 30), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)
        res = self.extractor.predict(img, mask, garment_type="upper")
        self.assertEqual(res.color_id, 5)  # Red
        self.assertEqual(res.color_name, "Red")

    def test_pure_white_garment(self):
        img = np.full((100, 100, 3), (245, 245, 245), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)
        res = self.extractor.predict(img, mask, garment_type="upper")
        self.assertEqual(res.color_id, 3)  # Light / White
        self.assertEqual(res.color_name, "Light")

    def test_pure_black_garment(self):
        img = np.full((100, 100, 3), (20, 20, 20), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)
        res = self.extractor.predict(img, mask, garment_type="lower")
        self.assertEqual(res.color_id, 1)  # Dark / Black
        self.assertEqual(res.color_name, "Dark")

    def test_multicolor_pattern(self):
        # Half bright yellow, half bright blue
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = (230, 200, 30)  # Yellow
        img[50:, :] = (30, 80, 190)   # Blue
        mask = np.ones((100, 100), dtype=bool)
        res = self.extractor.predict(img, mask, garment_type="upper")
        self.assertTrue(res.is_multicolor)
        self.assertEqual(res.color_id, 12)  # Multicolor

    def test_empty_mask(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        res = self.extractor.predict(img, mask, garment_type="upper")
        self.assertEqual(res.color_id, 0)
        self.assertEqual(res.color_name, "Uncharacterized")

if __name__ == "__main__":
    unittest.main()

