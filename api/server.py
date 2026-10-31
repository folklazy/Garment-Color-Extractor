"""FastAPI Microservice for Garment Color Extractor.

Launch with:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from garment_color.core import GarmentColorExtractor
from .schemas import ExtractColorRequest, ExtractColorResponse

app = FastAPI(
    title="Garment Color Extractor API",
    version="1.0.0",
    description="High-performance, CPU-only garment color classification microservice (M5 Engine).",
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = GarmentColorExtractor()


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode base64 string to RGB image array."""
    try:
        # Strip header if present (e.g. data:image/jpeg;base64,...)
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from bytes")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image data: {e}",
        )


def decode_base64_mask(b64_str: str) -> np.ndarray:
    """Decode base64 string to binary boolean mask."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        mask_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise ValueError("Could not decode mask from bytes")
        return mask_gray > 0
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 mask data: {e}",
        )


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "engine": "M5_Shadow_Aware_CIELAB", "version": "1.0.0"}


@app.post(
    "/api/v1/extract-color",
    response_model=ExtractColorResponse,
    tags=["Color Extraction"],
    summary="Extract Garment Color from Base64 Image & Mask",
)
def extract_color(req: ExtractColorRequest) -> ExtractColorResponse:
    """Classify the dominant color of a garment given an image and mask/bbox."""
    img_rgb = decode_base64_image(req.image_base64)
    mask = decode_base64_mask(req.mask_base64) if req.mask_base64 else None

    pred = extractor.predict(
        image=img_rgb,
        mask=mask,
        bbox=req.bbox,
        garment_type=req.garment_type,
        apply_erosion=req.apply_erosion,
    )

    return ExtractColorResponse(
        color_id=pred.color_id,
        color_name=pred.color_name,
        color_name_th=pred.color_name_th,
        confidence=pred.confidence,
        hex_code=pred.hex_code,
        rgb_swatch=pred.rgb_swatch,
        is_multicolor=pred.is_multicolor,
        secondary_color_id=pred.secondary_color_id,
        secondary_color_name=pred.secondary_color_name,
        secondary_hex_code=pred.secondary_hex_code,
        total_pixels=pred.total_pixels,
        latency_ms=pred.latency_ms,
        reasoning=pred.reasoning,
    )

