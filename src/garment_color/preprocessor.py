"""Optional boundary erosion and a separate, unused luminance-filter helper.

Neither operation guarantees removal of contamination or illumination effects.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


class MaskPreprocessor:
    """Preprocesses segmentation masks to extract high-fidelity garment pixels."""

    def __init__(
        self,
        erode_kernel_size: int = 5,
        min_pixels_for_erosion: int = 80,
        min_pixels_remaining: int = 40,
    ):
        if not isinstance(erode_kernel_size,int) or erode_kernel_size<1 or erode_kernel_size%2==0:
            raise ValueError('Erosion kernel size must be a positive odd integer')
        if min_pixels_for_erosion<1 or min_pixels_remaining<1:
            raise ValueError('Minimum pixel counts must be positive')
        self.erode_kernel_size = erode_kernel_size
        self.min_pixels_for_erosion = min_pixels_for_erosion
        self.min_pixels_remaining = min_pixels_remaining

    def erode_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Erode mask boundary to eliminate tile/wall/skin edge artifacts.
        
        Only applies erosion if the mask is large enough to retain sufficient pixels.
        """
        mask_uint8 = binary_mask.astype(np.uint8)
        pixel_count = np.count_nonzero(mask_uint8)

        if pixel_count < self.min_pixels_for_erosion:
            return binary_mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.erode_kernel_size, self.erode_kernel_size)
        )
        eroded = cv2.erode(mask_uint8, kernel, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)

        if np.count_nonzero(eroded) >= self.min_pixels_remaining:
            return eroded > 0

        # Fallback to 3x3 kernel if 5x5 eroded too much
        small=min(self.erode_kernel_size,3)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small, small))
        eroded_small = cv2.erode(mask_uint8, kernel_small, iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        if np.count_nonzero(eroded_small) >= self.min_pixels_remaining:
            return eroded_small > 0

        return binary_mask

    def extract_clean_pixels(
        self,
        image_rgb: np.ndarray,
        binary_mask: np.ndarray,
        apply_erosion: bool = True
    ) -> np.ndarray:
        """Extract foreground RGB pixels with optional boundary erosion.
        
        Returns:
          Array of shape (N, 3) with uint8 RGB pixel values.
        """
        if binary_mask is None or np.count_nonzero(binary_mask) == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        target_mask = self.erode_mask(binary_mask) if apply_erosion else binary_mask
        pixels = image_rgb[target_mask]

        if len(pixels) == 0:
            # Fallback to uneroded mask if eroded mask became empty
            pixels = image_rgb[binary_mask]

        return pixels


def decompose_shadows(pixels_lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy luminance-range filter, unused by GarmentColorExtractor.

    This is not a physical shadow decomposition or intrinsic-color recovery.
    
    Returns:
      (clean_pixels_lab, is_valid_mask)
    """
    if len(pixels_lab) == 0:
        return pixels_lab, np.zeros((0,), dtype=bool)

    L = pixels_lab[:, 0]
    med_L = float(np.median(L))

    # If the overall garment is naturally dark (black/navy), do not filter dark pixels
    if med_L < 22.0:
        valid = L >= 2.0  # only remove total void/zero pixels
    else:
        # Garment has mid or light tone; filter out extreme shadows and extreme specular highlights
        valid = (L >= 12.0) & (L <= 98.0)

    # If filtering removed too many pixels (>85%), relax filter
    if np.count_nonzero(valid) < 0.15 * len(pixels_lab):
        valid = np.ones(len(pixels_lab), dtype=bool)

    return pixels_lab[valid], valid

