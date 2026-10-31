"""Heuristic lightness gates and weighted-Euclidean prototype matching.

This is not the supervised M5 ensemble in the historical research scripts.
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

# Fixed legacy rule prototypes, not Gaussian distributions. No fitted-model
# artifact binds these values to the historical CCIHP calibration experiment.
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


class PrototypeRuleClassifier:
    """Rule classifier; output scores are not calibrated probabilities."""

    def __init__(
        self,
        navy_luminance_cutoff: float = 26.78,
        multicolor_sec_ratio_min: float = 0.27,
        multicolor_delta_e_min: float = 22.0,
    ):
        if not np.isfinite(navy_luminance_cutoff) or not 0<=navy_luminance_cutoff<=100:
            raise ValueError('navy_luminance_cutoff must be in [0,100]')
        if not np.isfinite(multicolor_sec_ratio_min) or not 0<=multicolor_sec_ratio_min<=1:
            raise ValueError('multicolor_sec_ratio_min must be in [0,1]')
        if not np.isfinite(multicolor_delta_e_min) or multicolor_delta_e_min<0:
            raise ValueError('multicolor_delta_e_min must be finite and nonnegative')
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
        """Classify aggregated color-family mass, preserving observed palette."""
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

        # Folds can create several clusters of one named color. Aggregate their
        # mass before choosing a label; the single largest cluster is insufficient.
        groups={}
        for cluster in features.clusters:
            cid=self.classify_cluster(cluster.centroid_lab,target_part)
            groups.setdefault(cid,[]).append(cluster)
        ranked=sorted(groups,key=lambda cid:(-sum(c.weight for c in groups[cid]),cid))
        prim_id=ranked[0];sec_id=ranked[1] if len(ranked)>1 else prim_id
        prim=max(groups[prim_id],key=lambda c:c.weight)
        sec=max(groups[sec_id],key=lambda c:c.weight)
        primary_mass=sum(c.weight for c in groups[prim_id])
        secondary_mass=sum(c.weight for c in groups[sec_id]) if len(ranked)>1 else 0.

        # Delta-E between primary and secondary cluster
        delta_e = float(np.sqrt(np.sum((prim.centroid_lab - sec.centroid_lab) ** 2)))

        # Check Multicolor criteria:
        # 1. Garment must not be hair (hair is rarely multicolor)
        # 2. Secondary cluster weight must exceed threshold (e.g. >= 27%)
        # 3. Distinct colors (different IDs and Delta-E >= 22.0)
        is_multi = False
        if (
            target_part.lower() != "hair"
            and secondary_mass >= self.multicolor_sec_ratio_min
            and prim_id != sec_id
            and delta_e >= self.multicolor_delta_e_min
        ):
            is_multi = True
            final_id = 12  # Multicolor
            c_info = get_color_by_id(12)
            confidence = primary_mass+secondary_mass
            reason = f"Multiple color families: {COLOR_REGISTRY[prim_id].name} ({primary_mass:.1%}) + {COLOR_REGISTRY[sec_id].name} ({secondary_mass:.1%}); centroid distance={delta_e:.1f}. Score is supporting pixel mass, not probability."
        else:
            final_id = prim_id
            c_info = get_color_by_id(prim_id)
            confidence = primary_mass
            reason = f"Dominant color family: {c_info.name} ({primary_mass:.1%} of sampled-region pixels); score is not calibrated probability."

        sec_info = get_color_by_id(sec_id) if (sec_id != prim_id and secondary_mass >= 0.15) else None

        return ClassificationResult(
            color_id=final_id,
            color_name=c_info.name,
            color_name_th=c_info.name_th,
            confidence=float(confidence),
            hex_code=prim.hex_code,
            is_multicolor=is_multi,
            secondary_color_id=sec_info.id if sec_info else None,
            secondary_color_name=sec_info.name if sec_info else None,
            secondary_hex_code=sec.hex_code if sec_info else None,
            reasoning=reason
        )


# Legacy import name retained; no trained ensemble is implied by this alias.
M5Classifier = PrototypeRuleClassifier

