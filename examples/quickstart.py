"""Quickstart Example: 5-line Garment Color Extraction.

Runs completely on CPU, zero setup required.
"""
import numpy as np
from garment_color import GarmentColorExtractor

def main():
    print("=" * 60)
    print("Garment Color Extractor (M5 Engine) - Quickstart Demo")
    print("=" * 60)

    # 1. Initialize Extractor Engine
    extractor = GarmentColorExtractor()

    # 2. Simulate a synthetic 300x300 image with a navy blue t-shirt
    # RGB Navy: (25, 45, 110) with realistic cast shadows
    image_rgb = np.full((300, 300, 3), (200, 200, 200), dtype=np.uint8)  # Grey background
    shirt_mask = np.zeros((300, 300), dtype=bool)
    shirt_mask[50:250, 50:250] = True

    # Fill shirt with Navy blue pixels and some shadow
    image_rgb[shirt_mask] = (30, 60, 140)
    image_rgb[150:250, 50:250] = (20, 40, 95)  # shadow region

    # 3. Predict Garment Color
    result = extractor.predict(image=image_rgb, mask=shirt_mask, garment_type="upper")

    # 4. Display Results
    print(f"👕 Predicted Color:  {result.color_name} ({result.color_name_th})")
    print(f"🔢 CCIHP Color ID:   {result.color_id}")
    print(f"🎨 HEX Swatch:       {result.hex_code}")
    print(f"📊 Confidence:       {result.confidence * 100:.1f}%")
    print(f"⚡ Latency:          {result.latency_ms} ms")
    print(f"💡 Reasoning:        {result.reasoning}")
    print("=" * 60)

if __name__ == "__main__":
    main()

