"""Code-rendered diagnostic card. Output is BGR for cv2.imwrite; no model calls."""
from __future__ import annotations
import cv2
import numpy as np
from .core import ColorPrediction


def create_color_card(image_rgb: np.ndarray, mask: np.ndarray, prediction: ColorPrediction,
                      card_width: int = 480, card_height: int = 620) -> np.ndarray:
    """Show the actual input region, observed palette and explicit score semantics.

    The card uses ASCII labels because OpenCV's built-in fonts do not render Thai.
    English/Thai names remain available in structured JSON.
    """
    if not isinstance(card_width,int) or not isinstance(card_height,int) or card_width<320 or card_height<500:
        raise ValueError('card_width>=320 and card_height>=500 are required')
    if image_rgb.dtype!=np.uint8 or image_rgb.ndim!=3 or image_rgb.shape[2]!=3:
        raise ValueError('image_rgb must be uint8 RGB')
    if mask.shape!=image_rgb.shape[:2] or mask.ndim!=2:
        raise ValueError('mask must match the image grid')
    card=np.full((card_height,card_width,3),(247,246,244),np.uint8)
    margin=20; muted=(110,99,90);ink=(43,34,27)
    def text(value,x,y,size=.48,color=ink):
        while value and cv2.getTextSize(value,cv2.FONT_HERSHEY_SIMPLEX,size,1)[0][0]>card_width-x-margin:
            value=value[:-4]+'...' if len(value)>4 else ''
        cv2.putText(card,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
    cv2.rectangle(card,(0,0),(card_width,66),(45,35,24),-1)
    text('OBSERVED COLOR',margin,30,.7,(255,255,255))
    text('Region inspection / prototype rules',margin,51,.40,(205,218,224))
    top=84;bottom=int(card_height*.46);preview_h=bottom-top
    cv2.rectangle(card,(margin,top),(card_width-margin,bottom),(230,230,230),-1)
    yy,xx=np.where(mask>0)
    if len(xx):
        y0,y1=max(0,yy.min()-8),min(image_rgb.shape[0],yy.max()+9)
        x0,x1=max(0,xx.min()-8),min(image_rgb.shape[1],xx.max()+9)
        crop=image_rgb[y0:y1,x0:x1].copy();cm=(mask[y0:y1,x0:x1]>0)
        crop[~cm]=(crop[~cm]*.2+235*.8).astype(np.uint8)
        contours,_=cv2.findContours(cm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop,contours,-1,(40,170,140),1)
        crop=cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)
        scale=min((card_width-2*margin)/crop.shape[1],preview_h/crop.shape[0])
        nw,nh=max(1,int(crop.shape[1]*scale)),max(1,int(crop.shape[0]*scale))
        preview=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
        px=(card_width-nw)//2;py=top+(preview_h-nh)//2
        card[py:py+nh,px:px+nw]=preview
    else:text('No pixels selected',margin+15,top+40,.5,muted)
    y=bottom+28
    text(prediction.color_name if prediction.status=='ok' else 'No color observation',margin,y,.65)
    text('ID '+str(prediction.color_id)+' / '+prediction.status,margin,y+23,.43,muted)
    y+=45
    palette=prediction.palette
    if palette:
        cumulative=0.;bar_w=card_width-2*margin
        for i,entry in enumerate(palette):
            left=margin+int(cumulative*bar_w);cumulative+=entry['weight']
            right=margin+int(cumulative*bar_w) if i<len(palette)-1 else card_width-margin
            r,g,b=entry['rgb'];cv2.rectangle(card,(left,y),(right,y+32),(b,g,r),-1)
        text('Observed palette / full region pixel fractions',margin,y+51,.39,muted)
    else:text('Palette unavailable (empty region)',margin,y+20,.43,muted)
    y+=78
    text('Swatch: '+str(prediction.hex_code or 'N/A'),margin,y,.48)
    text('Support: '+f'{prediction.confidence:.1%}'+' (not probability)',margin,y+23,.43)
    text(f'Input {prediction.input_pixels:,} px / analyzed {prediction.total_pixels:,} px',margin,y+46,.40,muted)
    text(f'Region: {prediction.region_source}',margin,y+68,.40,muted)
    text('Image appearance; material hue is not verified.',margin,card_height-32,.36,muted)
    text('View JSON warnings before interpreting the label.',margin,card_height-15,.36,muted)
    return card
