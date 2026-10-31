"""End-to-End Person Re-ID Integration Example: YOLO Detector + GarmentColorExtractor.

Demonstrates how to process full-frame surveillance images, extract upper/lower
garment colors for each detected person, and produce structured queryable attributes.
"""
from __future__ import annotations

import cv2
import numpy as np
from garment_color import GarmentColorExtractor


def simulate_yolo_detections(image_h: int, image_w: int):
    """Simulate YOLO bounding box detections for 2 people in the scene."""
    return [
        {
            "person_id": 1,
            "bbox": (120, 80, 260, 420),  # (x1, y1, x2, y2)
            "confidence": 0.94,
        },
        {
            "person_id": 2,
            "bbox": (450, 110, 580, 430),
            "confidence": 0.89,
        },
    ]


def main():
    print("=" * 65)
    print("YOLO + GarmentColorExtractor (M5): End-to-End Re-ID Pipeline")
    print("=" * 65)

    extractor = GarmentColorExtractor()

    # Create synthetic surveillance frame (1080x1920)
    frame = np.full((1080, 1920, 3), (210, 210, 210), dtype=np.uint8)

    # Person 1: Red shirt + Black pants
    frame[80:250, 120:260] = (190, 30, 35)    # Red Upper
    frame[250:420, 120:260] = (25, 25, 30)    # Dark Lower

    # Person 2: White shirt + Blue jeans
    frame[110:270, 450:580] = (240, 240, 245) # Light Upper
    frame[270:430, 450:580] = (40, 85, 175)   # Blue Lower

    # Run simulated detector
    detections = simulate_yolo_detections(1080, 1920)

    reid_database_records = []

    for det in detections:
        pid = det["person_id"]
        x1, y1, x2, y2 = det["bbox"]
        h_box = y2 - y1

        # Heuristic upper/lower partition (top 45% = upper body, bottom 55% = lower body)
        y_split = y1 + int(h_box * 0.45)

        upper_bbox = (x1, y1, x2, y_split)
        lower_bbox = (x1, y_split, x2, y2)

        # Extract Upper Color
        pred_upper = extractor.predict(
            image=frame, bbox=upper_bbox, garment_type="upper"
        )

        # Extract Lower Color
        pred_lower = extractor.predict(
            image=frame, bbox=lower_bbox, garment_type="lower"
        )

        record = {
            "person_id": pid,
            "upper_clothing": {
                "color_name": pred_upper.color_name,
                "color_id": pred_upper.color_id,
                "hex": pred_upper.hex_code,
                "confidence": pred_upper.confidence,
            },
            "lower_clothing": {
                "color_name": pred_lower.color_name,
                "color_id": pred_lower.color_id,
                "hex": pred_lower.hex_code,
                "confidence": pred_lower.confidence,
            },
            "total_latency_ms": round(pred_upper.latency_ms + pred_lower.latency_ms, 2),
        }
        reid_database_records.append(record)

        print(f"\n👤 Person #{pid} (BBox: {det['bbox']}, Conf: {det['confidence']}):")
        print(f"  👕 Upper: {pred_upper.color_name} ({pred_upper.color_name_th}) [{pred_upper.hex_code}] - Conf: {pred_upper.confidence:.2f}")
        print(f"  👖 Lower: {pred_lower.color_name} ({pred_lower.color_name_th}) [{pred_lower.hex_code}] - Conf: {pred_lower.confidence:.2f}")
        print(f"  ⚡ Extraction Latency: {record['total_latency_ms']} ms")

    print("\n" + "=" * 65)
    print("Database Ingestion Payload (JSON):")
    import json
    print(json.dumps(reid_database_records, indent=2, ensure_ascii=False))
    print("=" * 65)


if __name__ == "__main__":
    main()

