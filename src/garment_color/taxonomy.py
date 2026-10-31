"""CCIHP 12-Color Taxonomy & Color Palette Definitions.

Color vocabulary and illustrative swatches; not an intrinsic-color calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ColorInfo:
    id: int
    name: str
    name_th: str
    hex: str
    rgb: Tuple[int, int, int]
    cielab: Tuple[float, float, float]
    description: str


# Canonical 12-Color Taxonomy (CCIHP Standard)
COLOR_REGISTRY: Dict[int, ColorInfo] = {
    0: ColorInfo(
        id=0,
        name="Uncharacterized",
        name_th="ไม่ระบุสี",
        hex="#808080",
        rgb=(128, 128, 128),
        cielab=(53.59, 0.0, 0.0),
        description="Unlabeled, skin, or transparent region"
    ),
    1: ColorInfo(
        id=1,
        name="Dark",
        name_th="สีดำ / โทนเข้ม",
        hex="#1A1A1A",
        rgb=(26, 26, 26),
        cielab=(10.0, 0.0, 0.0),
        description="Black, deep charcoal, very dark navy, or dark tone"
    ),
    2: ColorInfo(
        id=2,
        name="Medium",
        name_th="สีเทา / โทนกลาง",
        hex="#808080",
        rgb=(128, 128, 128),
        cielab=(53.59, 0.0, 0.0),
        description="Medium grey, slate grey, heather grey, mid-tone"
    ),
    3: ColorInfo(
        id=3,
        name="Light",
        name_th="สีขาว / โทนสว่าง",
        hex="#F0F0F0",
        rgb=(240, 240, 240),
        cielab=(94.68, 0.0, 0.0),
        description="White, ivory, cream, off-white, very light tone"
    ),
    4: ColorInfo(
        id=4,
        name="Brown",
        name_th="สีน้ำตาล / สีกากี",
        hex="#7B4B27",
        rgb=(123, 75, 39),
        cielab=(37.26, 17.65, 30.12),
        description="Brown, beige, khaki, tan, coffee, camel"
    ),
    5: ColorInfo(
        id=5,
        name="Red",
        name_th="สีแดง",
        hex="#C42021",
        rgb=(196, 32, 33),
        cielab=(43.52, 60.12, 42.18),
        description="Red, crimson, maroon, burgundy, scarlet"
    ),
    6: ColorInfo(
        id=6,
        name="Pink",
        name_th="สีชมพู",
        hex="#E5859E",
        rgb=(229, 133, 158),
        cielab=(66.35, 40.12, 6.25),
        description="Pink, rose, peach, salmon, magenta"
    ),
    7: ColorInfo(
        id=7,
        name="Yellow",
        name_th="สีเหลือง",
        hex="#E6C82B",
        rgb=(230, 200, 43),
        cielab=(80.15, -4.12, 75.82),
        description="Yellow, gold, mustard, lemon"
    ),
    8: ColorInfo(
        id=8,
        name="Orange",
        name_th="สีส้ม",
        hex="#E36F1E",
        rgb=(227, 111, 30),
        cielab=(58.21, 41.52, 62.15),
        description="Orange, amber, rust, terracotta, tangerine"
    ),
    9: ColorInfo(
        id=9,
        name="Green",
        name_th="สีเขียว",
        hex="#2B8C3E",
        rgb=(43, 140, 62),
        cielab=(52.18, -45.12, 32.18),
        description="Green, olive, emerald, lime, forest green"
    ),
    10: ColorInfo(
        id=10,
        name="Blue",
        name_th="สีน้ำเงิน / สีฟ้า",
        hex="#2D5FB9",
        rgb=(45, 95, 185),
        cielab=(42.51, 15.12, -54.18),
        description="Blue, navy, denim, sky blue, cyan, cobalt"
    ),
    11: ColorInfo(
        id=11,
        name="Purple",
        name_th="สีม่วง",
        hex="#7B3798",
        rgb=(123, 55, 152),
        cielab=(35.18, 45.12, -38.12),
        description="Purple, violet, lavender, indigo, lilac"
    ),
    12: ColorInfo(
        id=12,
        name="Multicolor",
        name_th="หลากสี / มีลวดลาย",
        hex="#FF00FF",
        rgb=(255, 0, 255),
        cielab=(60.0, 70.0, -50.0),
        description="Patterned, striped, plaid, floral, multi-colored"
    ),
}

# Reverse lookup by lower-case name
NAME_TO_ID: Dict[str, int] = {
    c.name.lower(): c.id for c in COLOR_REGISTRY.values()
}

# BCN-11 (Basic Color Naming 11 colors) mapping to CCIHP ID
BCN11_TO_CCIHP: Dict[str, int] = {
    "black": 1,
    "grey": 2,
    "white": 3,
    "brown": 4,
    "red": 5,
    "pink": 6,
    "yellow": 7,
    "orange": 8,
    "green": 9,
    "blue": 10,
    "purple": 11,
}


def get_color_by_id(color_id: int) -> ColorInfo:
    """Retrieve color info by CCIHP integer ID (0 to 12)."""
    return COLOR_REGISTRY.get(color_id, COLOR_REGISTRY[0])


def get_color_by_name(name: str) -> Optional[ColorInfo]:
    """Retrieve color info by case-insensitive name."""
    cid = NAME_TO_ID.get(name.strip().lower())
    return COLOR_REGISTRY[cid] if cid is not None else None

