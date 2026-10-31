"""Main Entrypoint: GarmentColorExtractor Engine (M5 Pipeline).

Orchestrates preprocessing, feature extraction, and calibrated classification.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .classifier import ClassificationResult, M5Classifier
from .color_spaces import rgb_to_hex
from .features import GarmentFeatures, extract_features
from .preprocessor import MaskPreprocessor, decompose_shadows
from .taxonomy import COLOR_REGISTRY, ColorInfo, get_color_by_id


@dataclass
class ColorPrediction:
    color_id: int
    color_name: str
    color_name_th: str
    confidence: float
    hex_code: str
    rgb_swatch: Tuple[int, int, int]
    is_multicolor: bool
    secondary_color_id: Optional[int]
    secondary_color_name: Optional[str]
    secondary_hex_code: Optional[str]
    total_pixels: int
    latency_ms: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to a serializable Python dictionary."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert prediction result to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class GarmentColorExtractor:
    """High-performance Garment Color Extraction Engine (M5 Calibrated Pipeline).
    
    100% CPU, lightning-fast (< 5 ms per garment), zero GPU memory footprint.
    Calibrated on 3,000+ garment instances and validated on 90 real CCTV scenes.
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
        self.classifier = M5Classifier(
            navy_luminance_cutoff=navy_luminance_cutoff,
            multicolor_sec_ratio_min=multicolor_sec_ratio_min,
        )

    def predict(
        self,
        image: Union[np.ndarray, str],
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
          apply_erosion: Whether to apply boundary erosion to eliminate edge bleed.
          is_bgr: Set to True if image is in OpenCV BGR format instead of RGB.
          
        Returns:
          ColorPrediction object with color_id, color_name, confidence, hex, and swatches.
        """
        t0 = time.perf_counter()

        # Handle file path input
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise ValueError(f"Could not read image file from path: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image

        h, w = img_rgb.shape[:2]

        # Resolve mask from bbox if mask is None
        if mask is None:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((h, w), dtype=bool)
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                mask[y1_c:y2_c, x1_c:x2_c] = True
            else:
                mask = np.ones((h, w), dtype=bool)
        else:
            if mask.dtype != bool:
                mask = mask > 0

        # Step 1: Preprocess and extract clean pixels
        raw_pixels = self.preprocessor.extract_clean_pixels(
            img_rgb, mask, apply_erosion=apply_erosion
        )

        # Step 2: Feature extraction (58D + K-Means K=4)
        features = extract_features(raw_pixels, n_clusters=4)

        # Step 3: M5 Calibrated Classification
        cls_result = self.classifier.classify(features, target_part=garment_type)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Swatch RGB
        swatch_rgb = features.primary_cluster.centroid_rgb
        swatch_tuple = (int(swatch_rgb[0]), int(swatch_rgb[1]), int(swatch_rgb[2]))

        return ColorPrediction(
            color_id=cls_result.color_id,
            color_name=cls_result.color_name,
            color_name_th=cls_result.color_name_th,
            confidence=cls_result.confidence,
            hex_code=cls_result.hex_code,
            rgb_swatch=swatch_tuple,
            is_multicolor=cls_result.is_multicolor,
            secondary_color_id=cls_result.secondary_color_id,
            secondary_color_name=cls_result.secondary_color_name,
            secondary_hex_code=cls_result.secondary_hex_code,
            total_pixels=features.total_pixels,
            latency_ms=latency_ms,
            reasoning=cls_result.reasoning,
        )

