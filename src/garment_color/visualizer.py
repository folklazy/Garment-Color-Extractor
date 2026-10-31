"""Visual Inspection & Swatch Card Generator.

Generates visual color diagnostics, overlays, and HTML/SVG color cards for inspection.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import ColorPrediction


def create_color_card(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    prediction: ColorPrediction,
    card_width: int = 400,
    card_height: int = 500,
) -> np.ndarray:
    """Generate a clean, high-resolution visual inspection card as a BGR image.
    
    Contains:
      - Cropped garment image with mask outline
      - Dominant color swatch square
      - Color name (EN & TH) + CCIHP Color ID badge
      - Confidence score & Latency
    """
    card = np.full((card_height, card_width, 3), 245, dtype=np.uint8)  # Off-white background

    # 1. Header Banner
    header_color = (40, 40, 40)
    cv2.rectangle(card, (0, 0), (card_width, 60), header_color, -1)
    cv2.putText(
        card,
        "GARMENT COLOR DIAGNOSTIC",
        (15, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 2. Crop garment region
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) > 0:
        y1, y2 = max(0, y_indices.min() - 10), min(image_rgb.shape[0], y_indices.max() + 10)
        x1, x2 = max(0, x_indices.min() - 10), min(image_rgb.shape[1], x_indices.max() + 10)
        crop = image_rgb[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]

        # Draw green contour around garment
        contours, _ = cv2.findContours(
            crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)

        # Resize crop to fit preview box (width 360, height 220)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        h_c, w_c = crop_bgr.shape[:2]
        scale = min(360.0 / w_c, 220.0 / h_c)
        new_w, new_h = int(w_c * scale), int(h_c * scale)
        resized_crop = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Place preview inside card
        off_x = (card_width - new_w) // 2
        off_y = 80 + (220 - new_h) // 2
        card[off_y : off_y + new_h, off_x : off_x + new_w] = resized_crop
        cv2.rectangle(card, (off_x, off_y), (off_x + new_w, off_y + new_h), (200, 200, 200), 1)

    # 3. Swatch & Result Box
    box_y = 320
    cv2.rectangle(card, (20, box_y), (card_width - 20, card_height - 20), (255, 255, 255), -1)
    cv2.rectangle(card, (20, box_y), (card_width - 20, card_height - 20), (210, 210, 210), 1)

    # Swatch square (BGR format for OpenCV)
    r, g, b = prediction.rgb_swatch
    swatch_bgr = (int(b), int(g), int(r))
    cv2.rectangle(card, (35, box_y + 20), (105, box_y + 90), swatch_bgr, -1)
    cv2.rectangle(card, (35, box_y + 20), (105, box_y + 90), (120, 120, 120), 1)

    # Text details
    cv2.putText(
        card,
        f"{prediction.color_name} (ID: {prediction.color_id})",
        (120, box_y + 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        card,
        f"HEX: {prediction.hex_code}",
        (120, box_y + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        card,
        f"Confidence: {prediction.confidence * 100:.1f}%",
        (120, box_y + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 140, 40),
        1,
        cv2.LINE_AA,
    )

    # Footer latency & pixel count
    cv2.putText(
        card,
        f"Pixels: {prediction.total_pixels}  |  Latency: {prediction.latency_ms:.1f}ms",
        (35, box_y + 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )

    return card
