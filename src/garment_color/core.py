"""CPU-only observed-color extraction with explicit input and evidence contracts."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass,field
from pathlib import Path
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .classifier import PrototypeRuleClassifier
from .color_spaces import rgb_to_hex
from .features import GarmentFeatures, extract_features
from .preprocessor import MaskPreprocessor
from .taxonomy import COLOR_REGISTRY, ColorInfo, get_color_by_id


@dataclass
class ColorPrediction:
    color_id: int
    color_name: str
    color_name_th: str
    confidence: float
    hex_code: Optional[str]
    rgb_swatch: Optional[Tuple[int, int, int]]
    is_multicolor: bool
    secondary_color_id: Optional[int]
    secondary_color_name: Optional[str]
    secondary_hex_code: Optional[str]
    total_pixels: int
    latency_ms: float
    reasoning: str
    algorithm_id: str = 'prototype_rules_v2'
    status: str = 'ok'
    score_kind: str = 'supporting_color_family_pixel_fraction_not_probability'
    calibrated_probability: Optional[float] = None
    region_source: str = 'mask'
    input_pixels: int = 0
    erosion_retained_fraction: Optional[float] = None
    palette: List[Dict[str,Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to a serializable Python dictionary."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert prediction result to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False,allow_nan=False)


class GarmentColorExtractor:
    """Observed RGB-region colors; no fitted M5 model or accuracy guarantee.

    ``confidence`` is a legacy name for heuristic supporting pixel fraction.
    It is not a probability that the returned linguistic label is correct.
    """

    def __init__(
        self,
        erode_kernel_size: int = 5,
        min_pixels_for_erosion: int = 80,
        navy_luminance_cutoff: float = 26.78,
        multicolor_sec_ratio_min: float = 0.27,
    ):
        self.preprocessor = MaskPreprocessor(
            erode_kernel_size=erode_kernel_size,
            min_pixels_for_erosion=min_pixels_for_erosion,
        )
        self.classifier = PrototypeRuleClassifier(
            navy_luminance_cutoff=navy_luminance_cutoff,
            multicolor_sec_ratio_min=multicolor_sec_ratio_min,
        )

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        mask: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        garment_type: str = "upper",
        apply_erosion: bool = True,
        is_bgr: bool = False,
    ) -> ColorPrediction:
        """Extract and classify color from a garment image region.
        
        Args:
          image: RGB numpy array (H, W, 3), or file path string.
          mask: Binary boolean or uint8 mask of the garment (H, W). If None, bbox is used.
          bbox: (x1, y1, x2, y2) bounding box if mask is not provided.
          garment_type: 'upper' (top clothing), 'lower' (pants/skirts), or 'hair'.
          apply_erosion: Whether to apply boundary erosion to reduce possible edge bleed.
          is_bgr: Set to True if image is in OpenCV BGR format instead of RGB.
          
        Returns:
          ColorPrediction object with color_id, color_name, confidence, hex, and swatches.
        """
        t0 = time.perf_counter()

        # Handle file path input
        if garment_type not in ('upper','lower','hair','other'):
            raise ValueError('garment_type must be upper, lower, hair, or other')
        if mask is not None and bbox is not None:
            raise ValueError('Supply a mask or a bbox, not both')
        notes=['uncalibrated_rule_score','observed_image_color_not_intrinsic_material_color']
        if isinstance(image, (str,Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not read image file from path: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or 0 in image.shape[:2]:
                raise ValueError('image must be nonempty uint8 RGB with shape (H,W,3); convert explicitly before calling')
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image

        h, w = img_rgb.shape[:2]

        # Resolve mask from bbox if mask is None
        if mask is None:
            if bbox is not None:
                b=np.asarray(bbox)
                if b.shape!=(4,) or b.dtype.kind not in 'iuf' or not np.isfinite(b).all():
                    raise ValueError('bbox must contain four finite XYXY coordinates')
                x1, y1, x2, y2 = bbox
                if x2<=x1 or y2<=y1:raise ValueError('bbox must have positive width and height')
                mask = np.zeros((h, w), dtype=bool)
                x1_c,y1_c=max(0,math.floor(x1)),max(0,math.floor(y1))
                x2_c,y2_c=min(w,math.ceil(x2)),min(h,math.ceil(y2))
                if x2_c<=x1_c or y2_c<=y1_c:raise ValueError('bbox does not intersect the image')
                mask[y1_c:y2_c, x1_c:x2_c] = True
                source='bbox';notes.append('bbox_may_include_background_or_other_parts')
            else:
                mask = np.ones((h, w), dtype=bool)
                source='whole_image';notes.append('whole_image_region_not_garment_segmentation')
        else:
            if not isinstance(mask,np.ndarray) or mask.shape!=(h,w) or mask.dtype.kind not in 'bui' or not np.isin(mask,[0,1,255]).all():
                raise ValueError('mask must match image (H,W) and contain only binary 0/1 or 0/255 values')
            if mask.dtype != bool:
                mask = mask > 0
            source='mask'

        # Step 1: Preprocess and extract clean pixels
        raw_pixels = self.preprocessor.extract_clean_pixels(
            img_rgb, mask, apply_erosion=apply_erosion
        )

        # Observed color clustering. No illumination-invariant recovery is applied.
        features = extract_features(raw_pixels, n_clusters=4)

        # Step 3: fixed prototype rules (no trained ensemble artifact)
        cls_result = self.classifier.classify(features, target_part=garment_type)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        hex_code=cls_result.hex_code if features.total_pixels else None
        swatch_tuple=tuple(int(hex_code[i:i+2],16) for i in (1,3,5)) if hex_code else None
        input_count=int(mask.sum())
        if input_count<40:notes.append('insufficient_region_pixels' if input_count else 'empty_region')
        if features.total_pixels:
            L,a,b=features.median_lab
            if np.hypot(a,b)<11.5:notes.append('low_chroma_no_material_hue_recovery')
            if L<12:notes.append('very_dark_region')
            if L>98:notes.append('near_clipped_highlights')
        palette=[dict(rgb=[int(x) for x in c.centroid_rgb],hex=c.hex_code,weight=float(c.weight),
                 color_id=self.classifier.classify_cluster(c.centroid_lab,garment_type)) for c in features.clusters]

        return ColorPrediction(
            color_id=cls_result.color_id,
            color_name=cls_result.color_name,
            color_name_th=cls_result.color_name_th,
            confidence=cls_result.confidence,
            hex_code=hex_code,
            rgb_swatch=swatch_tuple,
            is_multicolor=cls_result.is_multicolor,
            secondary_color_id=cls_result.secondary_color_id,
            secondary_color_name=cls_result.secondary_color_name,
            secondary_hex_code=cls_result.secondary_hex_code,
            total_pixels=features.total_pixels,
            latency_ms=latency_ms,
            reasoning=cls_result.reasoning,
            status='ok' if features.total_pixels else 'no_data',region_source=source,input_pixels=input_count,
            erosion_retained_fraction=features.total_pixels/input_count if input_count else None,
            palette=palette,warnings=notes,
        )

