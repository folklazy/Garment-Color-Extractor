"""Independent labels/denominators and new artifact model contracts, CPU only."""
import unittest,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_t4_pilot import eligible,metrics,order

class PilotContracts(unittest.TestCase):
    def test_eligibility_does_not_restrict_input_to_color_labels(self):
        mask=np.ones((10,10),bool);colors=np.full((10,10),5,np.uint8);colors[:2]=0
        self.assertEqual(eligible(colors,mask),(5,None))
        colors[2]=0;self.assertEqual(eligible(colors,mask),(None,'CHARACTERIZED_FRACTION_LT80'))
        colors[:]=5;colors[:5]=10;self.assertEqual(eligible(colors,mask),(None,'MODAL_FRACTION_LT80'))
    def test_hand_counted_accuracy_and_fixed_macro(self):
        m=metrics([1,1,2],[1,2,2]);self.assertAlmostEqual(m['strict_accuracy'],2/3)
        self.assertAlmostEqual(m['macroF1_fixed12'],(2/3+2/3)/12)
        self.assertIsNone(metrics([],[])['strict_accuracy'])
    def test_deterministic_membership_order_independent_of_input_order(self):
        ids=['a','b','c','d'];self.assertEqual(order(ids,'Training',20261101),order(ids[::-1],'Training',20261101))
    def test_artifact_roundtrip_classes_no_data_and_hash(self):
        try:import torch
        except ImportError as error:raise unittest.SkipTest(str(error))
        from garment_color.learned import build_model,LearnedColorPredictor,ALGORITHM
        model=build_model()
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
            model[-1].bias[4]=10
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'model.pt';torch.save(dict(algorithm_id=ALGORITHM,state_dict=model.state_dict(),mean=[0.]*62,scale=[1.]*62,class_order=list(range(1,13))),f)
            p=LearnedColorPredictor(f);r=p.predict_pixels(np.full((100,3),[200,30,30],np.uint8),5)
            self.assertEqual(r['color_id'],5);self.assertIsNone(r['calibrated_probability'])
            self.assertEqual(p.predict_pixels(np.zeros((0,3),np.uint8),5)['status'],'no_data')
            with self.assertRaises(ValueError):LearnedColorPredictor(f,expected_sha256='0'*64)
