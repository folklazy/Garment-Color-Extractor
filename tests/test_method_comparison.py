"""Independent expected cases for new method comparison risks; CPU only."""
import unittest
import numpy as np
from garment_color.research_methods import ResearchColorSuite, blend_m5, predict_m1_garment


class MethodComparisonContracts(unittest.TestCase):
    def test_m1_known_single_and_multicolor(self):
        red = np.tile([0., 1., 1.], (40, 1))
        blue = np.tile([210., 1., 1.], (30, 1))
        green = np.tile([120., 1., 1.], (30, 1))
        self.assertEqual(predict_m1_garment(red)['pred'], 5)
        self.assertEqual(predict_m1_garment(np.r_[red, blue, green])['pred'], 12)
        with self.assertRaises(ValueError):
            predict_m1_garment(np.zeros((0, 3)))

    def test_real_m5_class_column_alignment_mass_and_non_alias(self):
        # HGB labels: P(12)=.2, P(5)=.8; ET labels: P(10)=1.
        # Mixed base: 12=.12, 5=.48, 10=.40. Prototype all class3.
        prototype = np.zeros(11)
        prototype[2] = 1
        p = blend_m5([.2, .8], [12, 5], [1.], [10], prototype)
        expected = np.zeros(13)
        expected[[12, 5, 10, 3]] = [.12, .408, .34, .132]
        np.testing.assert_allclose(p, expected, atol=1e-14)
        self.assertEqual(p.argmax(), 5)  # not prototype/M4=3, nor ET=10
        self.assertAlmostEqual(p.sum(), 1)
        with self.assertRaises(ValueError):
            blend_m5([1.], [0], [1.], [10], prototype)

    def test_threshold_selection_uses_explicit_calibration_rows(self):
        suite = ResearchColorSuite()
        suite.bcn_to_ccihp[:, 1] = 1
        rows = [{'gt_color': c, 'features': {'bcn_hist': np.full(11, 1/11)}} for c in [1, 12]]
        # Every distribution has secondary mass0: both candidates tie -> smaller .4.
        self.assertEqual(suite._calibrate(rows, 'M2', None, [.4, .8], .72), .4)
        self.assertEqual(suite.calibration_log['M2:GLOBAL']['rows'], 2)

    def test_feature_dimension_finite_and_palette_mass(self):
        suite = ResearchColorSuite()
        pixels = np.full((10, 3), [190, 20, 20], dtype=np.uint8)
        f = suite.extract_kmeans_and_features(pixels, pixels)
        self.assertEqual(f['feat_vec'].shape, (84,))
        self.assertTrue(np.isfinite(f['feat_vec']).all())
        self.assertAlmostEqual(f['weights'].sum(), 1)
        with self.assertRaises(ValueError):
            suite.extract_kmeans_and_features(pixels[:0], pixels[:0])


if __name__ == '__main__':
    unittest.main()
