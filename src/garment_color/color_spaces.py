"""Vectorized Color Space Conversions and Distance Metrics.

Provides high-performance NumPy/OpenCV conversions between sRGB, CIELAB, and HSV,
as well as perceptual color difference (Delta-E) formulas.
"""
from __future__ import annotations

import cv2
import numpy as np


def srgb_to_cielab(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 array (N, 3) to CIELAB float32 array (N, 3).
    
    Standard OpenCV LAB uses scaled values:
      L_cv = L * 255 / 100
      a_cv = a + 128
      b_cv = b + 128
    This function restores true standard CIELAB ranges:
      L* in [0, 100]
      a* in [-128, 127]
      b* in [-128, 127]
    """
    if len(rgb_uint8) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    arr = np.asarray(rgb_uint8, dtype=np.uint8).reshape(-1, 1, 3)
    lab_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = lab_cv[:, 0] * (100.0 / 255.0)
    a = lab_cv[:, 1] - 128.0
    b = lab_cv[:, 2] - 128.0
    return np.stack([L, a, b], axis=1)


def srgb_to_hsv(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 array (N, 3) to HSV float32 array (N, 3).
    
    Returns:
      H in [0, 360)
      S in [0, 1]
      V in [0, 1]
    """
    if len(rgb_uint8) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    arr = np.asarray(rgb_uint8, dtype=np.uint8).reshape(-1, 1, 3)
    hsv_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    H = hsv_cv[:, 0] * 2.0
    S = hsv_cv[:, 1] / 255.0
    V = hsv_cv[:, 2] / 255.0
    return np.stack([H, S, V], axis=1)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB integers (0-255) to uppercase HEX string '#RRGGBB'."""
    r_c = max(0, min(255, int(r)))
    g_c = max(0, min(255, int(g)))
    b_c = max(0, min(255, int(b)))
    return f"#{r_c:02X}{g_c:02X}{b_c:02X}"


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Calculate CIE76 Euclidean distance between CIELAB vectors."""
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def delta_e_weighted(lab: np.ndarray, prototypes: np.ndarray, w_l: float = 0.65) -> np.ndarray:
    """Calculate luminance-discounted Delta-E matrix.
    
    Discounting L* prevents cast shadows or mild overexposure from overriding
    strong chromatic signatures (a*, b*).
    
    Args:
      lab: Shape (N, 3) or (3,)
      prototypes: Shape (P, 3)
      w_l: Luminance weight (default 0.65 for shadow robustness)
      
    Returns:
      Distance matrix of shape (N, P) or (P,)
    """
    lab_arr = np.asarray(lab, dtype=np.float32)
    proto_arr = np.asarray(prototypes, dtype=np.float32)

    if lab_arr.ndim == 1:
        diff = lab_arr[None, :] - proto_arr
        dist2 = w_l * (diff[:, 0] ** 2) + (diff[:, 1] ** 2) + (diff[:, 2] ** 2)
        return np.sqrt(dist2)
    else:
        diff = lab_arr[:, None, :] - proto_arr[None, :, :]
        dist2 = w_l * (diff[:, :, 0] ** 2) + (diff[:, :, 1] ** 2) + (diff[:, :, 2] ** 2)
        return np.sqrt(dist2)
