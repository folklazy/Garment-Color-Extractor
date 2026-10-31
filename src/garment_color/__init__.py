"""Observed-color extraction with heuristic labels and explicit evidence limits."""

from .core import ColorPrediction, GarmentColorExtractor
from .taxonomy import (
    COLOR_REGISTRY,
    ColorInfo,
    get_color_by_id,
    get_color_by_name,
)
from .visualizer import create_color_card

__version__ = "1.1.0"
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
