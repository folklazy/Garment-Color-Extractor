"""Unit tests for Color Taxonomy."""
import unittest
from garment_color.taxonomy import COLOR_REGISTRY, get_color_by_id, get_color_by_name

class TestTaxonomy(unittest.TestCase):
    def test_total_colors(self):
        self.assertEqual(len(COLOR_REGISTRY), 13)  # 0 to 12

    def test_color_lookup_by_id(self):
        dark = get_color_by_id(1)
        self.assertEqual(dark.name, "Dark")
        self.assertEqual(dark.id, 1)

        blue = get_color_by_id(10)
        self.assertEqual(blue.name, "Blue")
        self.assertEqual(blue.id, 10)

        multi = get_color_by_id(12)
        self.assertEqual(multi.name, "Multicolor")

    def test_color_lookup_by_name(self):
        red = get_color_by_name("red")
        self.assertIsNotNone(red)
        self.assertEqual(red.id, 5)

        yellow = get_color_by_name("YELLOW")
        self.assertIsNotNone(yellow)
        self.assertEqual(yellow.id, 7)

if __name__ == "__main__":
    unittest.main()
