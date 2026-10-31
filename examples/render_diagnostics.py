"""Render deterministic synthetic cards; no dataset, camera, or model required."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from garment_color import GarmentColorExtractor,create_color_card

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).parent/'assets/color_diagnostics.png')
    args=parser.parse_args();extractor=GarmentColorExtractor();cards=[]
    for kind in ['single','multicolor','low_chroma','empty']:
        im=np.full((240,320,3),(225,226,228),np.uint8)
        mask=np.zeros((240,320),np.uint8)
        cv2.rectangle(mask,(80,30),(240,215),1,-1)
        if kind=='single':im[mask>0]=[200,30,30]
        elif kind=='multicolor':
            im[mask>0]=[230,200,30];im[125:216,80:241]=[30,80,190]
        elif kind=='low_chroma':im[mask>0]=[100,100,100]
        else:mask[:]=0
        pred=extractor.predict(im,mask=mask)
        cards.append(create_color_card(im,mask,pred))
    grid=np.vstack([np.hstack(cards[:2]),np.hstack(cards[2:])])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(args.output),grid):raise RuntimeError('Could not write diagnostic image')
    print(args.output)

if __name__=='__main__':main()
