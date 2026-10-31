"""Deterministic observed-color palette and a legacy 58-element descriptor.

The descriptor is diagnostic; the shipped rule classifier does not consume it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .color_spaces import rgb_to_hex, srgb_to_cielab, srgb_to_hsv


@dataclass
class ClusterInfo:
    centroid_lab: np.ndarray      # Shape (3,)
    centroid_rgb: np.ndarray      # Shape (3,) uint8
    weight: float                 # Fraction of pixels (0.0 to 1.0)
    hex_code: str                 # e.g. '#2D5FB9'


@dataclass
class GarmentFeatures:
    total_pixels: int
    median_rgb: Tuple[int, int, int]
    median_hex: str
    median_lab: Tuple[float, float, float]
    median_hsv: Tuple[float, float, float]
    clusters: List[ClusterInfo]
    primary_cluster: ClusterInfo
    secondary_cluster: ClusterInfo
    feature_vector_58d: np.ndarray


def extract_features(pixels_rgb: np.ndarray, n_clusters: int = 4) -> GarmentFeatures:
    """Extract observed palette; weights refer to all supplied pixels."""
    if not isinstance(pixels_rgb,np.ndarray) or pixels_rgb.ndim!=2 or pixels_rgb.shape[1]!=3 or pixels_rgb.dtype!=np.uint8:
        raise ValueError('pixels_rgb must be uint8 RGB with shape (N, 3)')
    if not isinstance(n_clusters,int) or n_clusters<1:
        raise ValueError('n_clusters must be a positive integer')
    n_pixels = len(pixels_rgb)
    if n_pixels == 0:
        empty_cluster = ClusterInfo(
            centroid_lab=np.array([50.0, 0.0, 0.0], dtype=np.float32),
            centroid_rgb=np.array([128, 128, 128], dtype=np.uint8),
            weight=0.0,
            hex_code="#808080",
        )
        return GarmentFeatures(
            total_pixels=0,
            median_rgb=(128, 128, 128),
            median_hex="#808080",
            median_lab=(50.0, 0.0, 0.0),
            median_hsv=(0.0, 0.0, 0.5),
            clusters=[],
            primary_cluster=empty_cluster,
            secondary_cluster=empty_cluster,
            feature_vector_58d=np.zeros(58, dtype=np.float32),
        )

    # Convert color spaces
    pixels_lab = srgb_to_cielab(pixels_rgb)
    pixels_hsv = srgb_to_hsv(pixels_rgb)

    # Medians
    med_rgb = np.median(pixels_rgb, axis=0).astype(np.uint8)
    med_hex = rgb_to_hex(med_rgb[0], med_rgb[1], med_rgb[2])
    med_lab = tuple(float(x) for x in np.median(pixels_lab, axis=0))
    med_hsv = tuple(float(x) for x in np.median(pixels_hsv, axis=0))

    # Fast K-Means (K=4) clustering in CIELAB
    # Fit a bounded, deterministic sample; identical colors must not create
    # artificial secondary clusters. Predict weights on every supplied pixel.
    sample_pixels=pixels_lab
    if n_pixels>2000:
        indices=np.random.RandomState(42).choice(n_pixels,2000,replace=False)
        sample_pixels=pixels_lab[indices]
    k = min(n_clusters, max(1, n_pixels // 15),len(np.unique(sample_pixels,axis=0)))
    if k <= 1 or n_pixels < 20:
        c_lab = np.mean(pixels_lab, axis=0).astype(np.float32)
        c_rgb = np.mean(pixels_rgb, axis=0).astype(np.uint8)
        c_hex = rgb_to_hex(c_rgb[0], c_rgb[1], c_rgb[2])
        c_info = ClusterInfo(c_lab, c_rgb, 1.0, c_hex)
        clusters = [c_info]
    else:
        # Sample for speed if large
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=max(256, k * 20),
            n_init=3,
        ).fit(sample_pixels)

        labels = kmeans.predict(pixels_lab)
        bincounts = np.bincount(labels, minlength=k)
        sorted_order = sorted(range(k),key=lambda i:(-int(bincounts[i]),*kmeans.cluster_centers_[i].tolist()))

        clusters = []
        for idx in sorted_order:
            w = float(bincounts[idx]) / float(n_pixels)
            if bincounts[idx]==0:
                continue
            centroid_l = kmeans.cluster_centers_[idx].astype(np.float32)

            # Map centroid back to RGB
            c_rgb_pixels = pixels_rgb[labels == idx]
            if len(c_rgb_pixels) > 0:
                c_rgb = np.median(c_rgb_pixels, axis=0).astype(np.uint8)
            else:
                c_rgb = med_rgb
            c_hex = rgb_to_hex(c_rgb[0], c_rgb[1], c_rgb[2])
            clusters.append(ClusterInfo(centroid_l, c_rgb, w, c_hex))

    prim = clusters[0]
    sec = clusters[1] if len(clusters) > 1 else ClusterInfo(prim.centroid_lab.copy(),prim.centroid_rgb.copy(),0.,prim.hex_code)

    # Build 58D feature vector
    feats: List[float] = []
    # 1. Quantiles for LAB (5 quantiles x 3 channels = 15)
    for ch in range(3):
        q = np.percentile(pixels_lab[:, ch], [10, 25, 50, 75, 90])
        feats.extend(q.tolist())

    # 2. Quantiles for HSV (5 quantiles x 3 channels = 15)
    for ch in range(3):
        q = np.percentile(pixels_hsv[:, ch], [10, 25, 50, 75, 90])
        feats.extend(q.tolist())

    # 3. Standard deviations (LAB 3 + HSV 3 = 6)
    feats.extend(np.std(pixels_lab, axis=0).tolist())
    feats.extend(np.std(pixels_hsv, axis=0).tolist())

    # 4. Top 3 Cluster centroids LAB and weights (3 clusters x 4 = 12)
    for i in range(3):
        if i < len(clusters):
            feats.extend(clusters[i].centroid_lab.tolist())
            feats.append(clusters[i].weight)
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    # 5. Contrast and ratios (10 features)
    delta_e_prim_sec = float(np.sqrt(np.sum((prim.centroid_lab - sec.centroid_lab) ** 2)))
    weight_ratio = float(sec.weight / (prim.weight + 1e-6))
    chroma_prim = float(np.sqrt(prim.centroid_lab[1] ** 2 + prim.centroid_lab[2] ** 2))
    chroma_sec = float(np.sqrt(sec.centroid_lab[1] ** 2 + sec.centroid_lab[2] ** 2))
    feats.extend([
        delta_e_prim_sec,
        weight_ratio,
        prim.weight,
        sec.weight,
        chroma_prim,
        chroma_sec,
        float(n_pixels),
        float(med_lab[0]),
        float(med_hsv[1]),
        float(med_hsv[2]),
    ])

    feat_vec = np.array(feats[:58], dtype=np.float32)
    return GarmentFeatures(
        total_pixels=n_pixels,
        median_rgb=(int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2])),
        median_hex=med_hex,
        median_lab=med_lab,
        median_hsv=med_hsv,
        clusters=clusters,
        primary_cluster=prim,
        secondary_cluster=sec,
        feature_vector_58d=feat_vec,
    )

