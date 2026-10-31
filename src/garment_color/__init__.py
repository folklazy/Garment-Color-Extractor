"""Garment Color Extractor (M5 Engine).

A lightweight, CPU-optimized, shadow-aware garment color classifier calibrated on CCIHP.
"""

from .core import ColorPrediction, GarmentColorExtractor
from .taxonomy import (
    COLOR_REGISTRY,
    ColorInfo,
    get_color_by_id,
    get_color_by_name,
)
from .visualizer import create_color_card

__version__ = "1.0.0"
__author__ = "folklazy"
__all__ = [
    "GarmentColorExtractor",
    "ColorPrediction",
    "ColorInfo",
    "COLOR_REGISTRY",
    "get_color_by_id",
    "get_color_by_name",
    "create_color_card",
]
