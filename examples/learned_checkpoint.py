"""Use an explicitly supplied local pilot artifact; does not download weights."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from garment_color.learned import LearnedColorPredictor
from garment_color.preprocessor import MaskPreprocessor

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--category',type=int,choices=[5,7,9,12],required=True);args=p.parse_args()
    bgr=cv2.imread(str(args.image));mask=cv2.imread(str(args.mask),cv2.IMREAD_UNCHANGED)
    if bgr is None or mask is None or mask.shape!=bgr.shape[:2] or not np.isin(mask,[0,1,255]).all():
        raise ValueError('Readable image and matching single-channel binary garment mask required')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    pixels=MaskPreprocessor().extract_clean_pixels(rgb,mask>0,True)
    predictor=LearnedColorPredictor(args.checkpoint,device='cpu',expected_sha256=args.sha256)
    print(json.dumps(predictor.predict_pixels(pixels,args.category),indent=2,allow_nan=False))
if __name__=='__main__':main()
