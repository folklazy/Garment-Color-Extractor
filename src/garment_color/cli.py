"""Command Line Interface for Garment Color Extractor.

Usage examples:
  python -m garment_color.cli --image person.jpg --mask shirt.png --type upper
  python -m garment_color.cli --image person.jpg --bbox 50 100 200 350 --type lower
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .core import GarmentColorExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Garment Color Extractor (M5 Engine): Extract garment colors from image & mask"
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image file (JPG/PNG)")
    parser.add_argument("--mask", "-m", type=str, default=None, help="Path to binary mask file (PNG)")
    parser.add_argument(
        "--bbox",
        "-b",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Bounding box coordinates (x1 y1 x2 y2) if mask is omitted",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["upper", "lower", "hair", "other"],
        default="upper",
        help="Garment category (default: upper)",
    )
    parser.add_argument("--no-erosion", action="store_true", help="Disable boundary mask erosion")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format only")

    args = parser.parse_args()

    extractor = GarmentColorExtractor()

    mask_arr = None
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
        if m is None:
            sys.exit(f"Error: Could not read mask file from '{args.mask}'")
        mask_arr = m > 0

    try:
        pred = extractor.predict(
            image=args.image,
            mask=mask_arr,
            bbox=tuple(args.bbox) if args.bbox else None,
            garment_type=args.type,
            apply_erosion=not args.no_erosion,
        )
    except Exception as e:
        sys.exit(f"Error processing image: {e}")

    if args.json:
        print(pred.to_json(indent=2))
    else:
        print("\n" + "=" * 55)
        print("👕 GARMENT COLOR EXTRACTION RESULT (M5 Engine)")
        print("=" * 55)
        print(f"  Target Type:       {args.type.upper()}")
        print(f"  Predicted Color:   {pred.color_name} ({pred.color_name_th})")
        print(f"  Color ID (CCIHP):  {pred.color_id}")
        print(f"  Confidence:        {pred.confidence * 100:.1f}%")
        print(f"  HEX Swatch:        {pred.hex_code}")
        print(f"  Is Multicolor:     {pred.is_multicolor}")
        if pred.is_multicolor and pred.secondary_color_name:
            print(f"  Secondary Color:   {pred.secondary_color_name} ({pred.secondary_hex_code})")
        print(f"  Latency:           {pred.latency_ms:.2f} ms")
        print(f"  Reasoning:         {pred.reasoning}")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

