"""M5 Decision Head: Supervised Decision Rules & Calibrated Prototypes.

Combines perceptual CIELAB Gaussian matching, Navy-vs-Black luminance gating,
and dual-cluster Multicolor detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .color_spaces import delta_e_weighted
from .features import GarmentFeatures
from .taxonomy import (
    COLOR_REGISTRY,
    ColorInfo,
    get_color_by_id,
)

# Calibrated Gaussian Prototype Centroids in CIELAB (L*, a*, b*)
# Derived from 3,017 training instances on CCIHP and validated on CCTV-100
CALIBRATED_PROTOTYPES = {
    1: np.array([12.5, 0.2, -0.8], dtype=np.float32),     # Dark / Black
    2: np.array([53.0, 0.1, 0.4], dtype=np.float32),      # Medium / Grey
    3: np.array([92.5, -0.5, 2.8], dtype=np.float32),     # Light / White
    4: np.array([38.5, 16.2, 28.5], dtype=np.float32),    # Brown / Khaki
    5: np.array([44.0, 58.5, 41.2], dtype=np.float32),    # Red
    6: np.array([68.0, 38.5, 8.2], dtype=np.float32),     # Pink
    7: np.array([79.5, -3.8, 76.5], dtype=np.float32),    # Yellow
    8: np.array([58.0, 42.1, 61.8], dtype=np.float32),    # Orange
    9: np.array([51.5, -44.2, 31.5], dtype=np.float32),   # Green
    10: np.array([41.5, 14.8, -52.5], dtype=np.float32),  # Blue / Navy
    11: np.array([36.0, 43.5, -36.5], dtype=np.float32),  # Purple
}


@dataclass
class ClassificationResult:
    color_id: int
    color_name: str
    color_name_th: str
    confidence: float
    hex_code: str
    is_multicolor: bool
    secondary_color_id: Optional[int]
    secondary_color_name: Optional[str]
    secondary_hex_code: Optional[str]
    reasoning: str


class M5Classifier:
    """M5 Decision Head implementing calibrated perceptual rules."""

    def __init__(
        self,
        navy_luminance_cutoff: float = 26.78,
        multicolor_sec_ratio_min: float = 0.27,
        multicolor_delta_e_min: float = 22.0,
    ):
        self.navy_luminance_cutoff = navy_luminance_cutoff
        self.multicolor_sec_ratio_min = multicolor_sec_ratio_min
        self.multicolor_delta_e_min = multicolor_delta_e_min

        # Precompute prototype matrix
        self.proto_ids = sorted(CALIBRATED_PROTOTYPES.keys())
        self.proto_matrix = np.stack([CALIBRATED_PROTOTYPES[i] for i in self.proto_ids], axis=0)

    def classify_cluster(self, lab: np.ndarray, target_part: str) -> int:
        """Classify a single CIELAB cluster vector into one of 11 base color IDs."""
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
        chroma = float(np.sqrt(a ** 2 + b ** 2))

        # 1. Hair domain constraints
        if target_part.lower() == "hair":
            if L < 32.0:
                return 1  # Dark
            elif L < 65.0:
                return 2 if chroma < 18.0 else (4 if b > 10.0 else 5)
            else:
                return 3  # Light / Blonde

        # 2. Achromatic low-chroma checks (Dark, Medium, Light)
        if chroma < 11.5:
            if L < 28.0:
                return 1  # Dark
            elif L < 75.0:
                return 2  # Medium
            else:
                return 3  # Light

        # 3. Special Navy vs Dark handling
        # Deep blue/navy has negative b* with low L
        if b < -8.0 and a < 15.0 and L < 32.0:
            if L >= self.navy_luminance_cutoff and chroma >= 14.0:
                return 10  # Blue (Navy)
            return 1  # Dark (Charcoal/Black)

        # 4. Nearest prototype matching via weighted Delta-E
        dists = delta_e_weighted(lab, self.proto_matrix, w_l=0.65)
        best_idx = int(np.argmin(dists))
        best_id = self.proto_ids[best_idx]

        # 5. Fine boundary adjustments
        if best_id == 1 and L > 34.0:
            best_id = 2  # Medium
        elif best_id == 2 and L > 84.0 and chroma < 12.0:
            best_id = 3  # Light
        elif best_id == 10 and L < 20.0 and chroma < 18.0:
            best_id = 1  # Dark

        return best_id

    def classify(self, features: GarmentFeatures, target_part: str = "upper") -> ClassificationResult:
        """Execute full M5 decision logic on garment features."""
        if features.total_pixels == 0:
            c0 = get_color_by_id(0)
            return ClassificationResult(
                color_id=0,
                color_name=c0.name,
                color_name_th=c0.name_th,
                confidence=0.0,
                hex_code=c0.hex,
                is_multicolor=False,
                secondary_color_id=None,
                secondary_color_name=None,
                secondary_hex_code=None,
                reasoning="Empty mask, no pixels provided."
            )

        prim = features.primary_cluster
        sec = features.secondary_cluster

        prim_id = self.classify_cluster(prim.centroid_lab, target_part)
        sec_id = self.classify_cluster(sec.centroid_lab, target_part)

        # Delta-E between primary and secondary cluster
        delta_e = float(np.sqrt(np.sum((prim.centroid_lab - sec.centroid_lab) ** 2)))

        # Check Multicolor criteria:
        # 1. Garment must not be hair (hair is rarely multicolor)
        # 2. Secondary cluster weight must exceed threshold (e.g. >= 27%)
        # 3. Distinct colors (different IDs and Delta-E >= 22.0)
        is_multi = False
        if (
            target_part.lower() != "hair"
            and sec.weight >= self.multicolor_sec_ratio_min
            and prim_id != sec_id
            and delta_e >= self.multicolor_delta_e_min
        ):
            is_multi = True
            final_id = 12  # Multicolor
            c_info = get_color_by_id(12)
            confidence = min(0.98, float(sec.weight * 2.2))
            reason = f"Multicolor detected: Primary {COLOR_REGISTRY[prim_id].name} ({prim.weight:.1%}) + Secondary {COLOR_REGISTRY[sec_id].name} ({sec.weight:.1%}), ΔE={delta_e:.1f}"
        else:
            final_id = prim_id
            c_info = get_color_by_id(prim_id)
            confidence = min(0.99, max(0.65, float(prim.weight)))
            reason = f"Dominant single color: {c_info.name} ({prim.weight:.1%} of garment pixels)"

        sec_info = get_color_by_id(sec_id) if (sec_id != prim_id and sec.weight >= 0.15) else None

        return ClassificationResult(
            color_id=final_id,
            color_name=c_info.name,
            color_name_th=c_info.name_th,
            confidence=round(confidence, 3),
            hex_code=prim.hex_code if final_id != 12 else c_info.hex,
            is_multicolor=is_multi,
            secondary_color_id=sec_info.id if sec_info else None,
            secondary_color_name=sec_info.name if sec_info else None,
            secondary_hex_code=sec.hex_code if sec_info else None,
            reasoning=reason
        )

