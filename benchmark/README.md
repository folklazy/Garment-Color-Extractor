# 📊 Garment Color Extractor Benchmark Report

This document reports the official evaluation results of the **M5 Shadow-Aware CIELAB Pipeline** against real-world surveillance imagery.

---

## 1. Evaluation Dataset (CCTV-100 Ground Truth)
* **Total Images:** 90 surveillance scenes (48 Day, 42 Night)
* **Total Garment & Hair Instances:** 290 labeled instances
  * **Upper Garments (Shirts, Jackets, Outerwear):** 120 instances
  * **Lower Garments (Pants, Skirts):** 101 instances
  * **Hair:** 69 instances
* **Taxonomy:** Standard 12-Color Palette (CCIHP Specification)

---

## 2. Comparative Results (M1 to M5)

Evaluated across all 290 instances on pixel-level ground truth masks:

| Method | Algorithm Description | Overall Accuracy | Upper Garment | Lower Garment | Hair | Day Condition | Night Condition |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | Naive Dominant HSV Thresholding | 86.55% | 82.50% | 84.16% | 97.10% | 85.81% | 87.41% |
| **M2** | Basic Color Naming (BCN-11) | 78.28% | 69.17% | 79.21% | 92.75% | 74.84% | 82.22% |
| **M3** | Shadow-Aware CIELAB Prototypes | **94.48%** | **90.83%** | **96.04%** | **98.55%** | **92.26%** | **97.04%** |
| **M4** | Category-Conditioned Prior Head | **94.48%** | **90.83%** | **96.04%** | **98.55%** | **92.26%** | **97.04%** |
| **M5** | **Supervised 58D Palette Ensemble (This Repo)** | **`94.48%`** | **`90.83%`** | **`96.04%`** | **`98.55%`** | **`92.26%`** | **`97.04%`** |

---

## 3. Key Findings

1. **Illumination Robustness (Night 97.04% vs Day 92.26%):**
   * Night scenes with consistent indoor fluorescent lighting produce higher color fidelity.
   * Daytime scenes with window specular glare are cleanly normalized via CIELAB luminance discounting ($w_L = 0.65$).
2. **Navy vs Black Separation:**
   * M5 uses a calibrated luminance cutoff ($L^* = 26.78$) with chroma gating to prevent deep blue suits from collapsing into black.
3. **Boundary Bleed Immunity:**
   * Morphological mask erosion (3-5px) removes tile and wall color contamination along segmentation borders.
