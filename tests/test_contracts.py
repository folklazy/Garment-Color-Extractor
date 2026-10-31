"""Regression cases for observations, input contracts and service serialization."""
import base64
import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2
from garment_color import GarmentColorExtractor,create_color_card
from garment_color.features import extract_features,ClusterInfo
from garment_color.classifier import PrototypeRuleClassifier

class ObservationContracts(unittest.TestCase):
    def setUp(self):
        self.x=GarmentColorExtractor()
        self.rgb=np.full((30,50,3),(200,30,30),np.uint8)
    def test_single_color_has_one_palette_entry_with_mass_one(self):
        f=extract_features(self.rgb.reshape(-1,3))
        self.assertEqual(len(f.clusters),1)
        self.assertEqual(f.primary_cluster.weight,1.)
        self.assertEqual(f.secondary_cluster.weight,0.)
        self.assertEqual(f.feature_vector_58d.shape,(58,))
    def test_empty_mask_is_not_a_gray_observation(self):
        r=self.x.predict(self.rgb,np.zeros((30,50),bool))
        d=json.loads(r.to_json());self.assertEqual(d['status'],'no_data')
        self.assertIsNone(d['hex_code']);self.assertIsNone(d['rgb_swatch'])
        self.assertIsNone(d['calibrated_probability']);self.assertEqual(d['palette'],[])
    def test_multicolor_hex_and_rgb_describe_same_observed_color(self):
        im=self.rgb.copy();im[:15]=[230,200,30];im[15:]=[30,80,190]
        r=self.x.predict(im,apply_erosion=False)
        self.assertTrue(r.is_multicolor)
        self.assertNotEqual(r.hex_code,'#FF00FF')
        self.assertEqual(r.hex_code,'#%02X%02X%02X'%r.rgb_swatch)
        self.assertAlmostEqual(sum(c['weight'] for c in r.palette),1.)
    def test_same_family_clusters_are_combined_before_label_selection(self):
        f=extract_features(self.rgb.reshape(-1,3))
        f.clusters=[ClusterInfo(np.array([44.,58.5,41.2]),np.array([200,30,30]),.4,'#C81E1E'),
                    ClusterInfo(np.array([41.5,14.8,-52.5]),np.array([30,80,190]),.3,'#1E50BE'),
                    ClusterInfo(np.array([46.,14.8,-52.5]),np.array([35,90,200]),.3,'#235AC8')]
        r=PrototypeRuleClassifier(multicolor_sec_ratio_min=.5).classify(f)
        self.assertEqual(r.color_id,10)  # Blue family .6 versus red .4
        self.assertAlmostEqual(r.confidence,.6)
    def test_strict_input_contracts(self):
        bad=[dict(image=self.rgb.astype(float)),dict(image=self.rgb[:,:,:1]),
             dict(image=self.rgb,mask=np.zeros((2,3),bool)),dict(image=self.rgb,mask=np.full((30,50),2,np.uint8)),
             dict(image=self.rgb,bbox=(100,100,120,120)),dict(image=self.rgb,bbox=(5,5,2,10)),
             dict(image=self.rgb,bbox=(0,0,float('nan'),10)),dict(image=self.rgb,garment_type='shirt'),
             dict(image=self.rgb,mask=np.ones((30,50),bool),bbox=(0,0,10,10))]
        for kwargs in bad:
            with self.subTest(kwargs=list(kwargs)):
                with self.assertRaises(ValueError):self.x.predict(**kwargs)
    def test_bbox_clip_path_and_bgr_produce_same_observation(self):
        a=self.x.predict(self.rgb,bbox=(-2.5,-3.2,4.2,5.2),apply_erosion=False)
        self.assertEqual(a.input_pixels,30)
        self.assertEqual(a.region_source,'bbox')
        b=self.x.predict(self.rgb[:,:,::-1].copy(),is_bgr=True,apply_erosion=False)
        self.assertEqual(a.hex_code,b.hex_code)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'red.png';cv2.imwrite(str(p),self.rgb[:,:,::-1])
            self.assertEqual(self.x.predict(p).color_id,5)
    def test_thin_region_retained_and_kernel_validated(self):
        mask=np.zeros((30,50),bool);mask[15,2:48]=True
        r=self.x.predict(self.rgb,mask)
        self.assertEqual(r.total_pixels,46)
        with self.assertRaises(ValueError):GarmentColorExtractor(erode_kernel_size=4)
    def test_card_geometry_and_no_data(self):
        for mask in [np.ones((30,50),bool),np.zeros((30,50),bool)]:
            r=self.x.predict(self.rgb,mask)
            self.assertEqual(create_color_card(self.rgb,mask,r,320,500).shape,(500,320,3))
        with self.assertRaises(ValueError):create_color_card(self.rgb,mask,r,100,100)

class APIContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from api.server import app
        except ImportError as error:raise unittest.SkipTest(str(error))
        cls.client=TestClient(app)
    def encoded(self,a):
        ok,data=cv2.imencode('.png',a);self.assertTrue(ok)
        return base64.b64encode(data).decode()
    def test_success_has_score_metadata_and_no_probability(self):
        res=self.client.post('/api/v1/extract-color',json={'image_base64':self.encoded(np.full((10,10,3),(30,30,200),np.uint8))})
        self.assertEqual(res.status_code,200);d=res.json()
        self.assertEqual(d['algorithm_id'],'prototype_rules_v2');self.assertIsNone(d['calibrated_probability'])
    def test_empty_region_json_null_and_bad_mask_422(self):
        image=self.encoded(np.zeros((10,10,3),np.uint8))
        r=self.client.post('/api/v1/extract-color',json={'image_base64':image,'mask_base64':self.encoded(np.zeros((10,10),np.uint8))})
        self.assertEqual(r.status_code,200);self.assertIsNone(r.json()['hex_code'])
        r=self.client.post('/api/v1/extract-color',json={'image_base64':image,'mask_base64':self.encoded(np.zeros((2,3),np.uint8))})
        self.assertEqual(r.status_code,422)
    def test_invalid_encoding_and_garment_type_are_client_errors(self):
        r=self.client.post('/api/v1/extract-color',json={'image_base64':'not base64!'})
        self.assertEqual(r.status_code,400)
        r=self.client.post('/api/v1/extract-color',json={'image_base64':'anything','garment_type':'invalid'})
        self.assertEqual(r.status_code,422)
