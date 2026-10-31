"""Pydantic Request & Response Schemas for Color Extraction API."""
from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class ExtractColorRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG or PNG image")
    mask_base64: Optional[str] = Field(None, description="Optional Base64 encoded binary mask (PNG)")
    bbox: Optional[Tuple[int, int, int, int]] = Field(None, description="Optional (x1, y1, x2, y2) bounding box")
    garment_type: str = Field("upper", description="Garment type: 'upper', 'lower', or 'hair'")
    apply_erosion: bool = Field(True, description="Whether to apply edge erosion")


class ExtractColorResponse(BaseModel):
    color_id: int = Field(..., description="CCIHP Color ID (0 to 12)")
    color_name: str = Field(..., description="Standard English color name")
    color_name_th: str = Field(..., description="Thai color name")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0")
    hex_code: str = Field(..., description="Representative HEX color string (e.g. #2D5FB9)")
    rgb_swatch: Tuple[int, int, int] = Field(..., description="Representative RGB color swatch (0-255)")
    is_multicolor: bool = Field(..., description="True if garment is patterned or multicolor")
    secondary_color_id: Optional[int] = Field(None, description="Secondary color ID if multicolor")
    secondary_color_name: Optional[str] = Field(None, description="Secondary color name if multicolor")
    secondary_hex_code: Optional[str] = Field(None, description="Secondary color HEX if multicolor")
    total_pixels: int = Field(..., description="Number of garment pixels analyzed")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    reasoning: str = Field(..., description="Decision logic explanation")
