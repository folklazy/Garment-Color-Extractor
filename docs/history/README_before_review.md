# 👕 Garment Color Extractor (M5 Engine)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CCIHP Benchmark](https://img.shields.io/badge/CCIHP%20Benchmark-91.07%25%20(Relaxed)-brightgreen.svg)](#-academic-benchmark--ablation-study-m1-to-m5)
[![Zero-GPU](https://img.shields.io/badge/GPU-0MB%20(100%25%20CPU)-success.svg)](#-key-features)
[![Inference Speed](https://img.shields.io/badge/Speed-%3C%203.2%20ms%2Fgarment-blueviolet.svg)](#-key-features)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20REST-009688.svg)](#-fastapi-microservice)

An interpretable, shadow-aware garment color classification framework for **Human Parsing, Pedestrian Attribute Recognition (PAR), and Video Analytics**.

Evaluated on the academic **CCIHP (Crowd Instance-level Human Parsing with Color Attributes)** benchmark across **1,265 human-annotated validation garments**. Outperforms classical color naming baselines (*van de Weijer et al., IEEE TPAMI 2009*) by **+12.02% in Relaxed Accuracy** and **2.15x in Macro-F1**, while running **100% on CPU** in **< 3.2 ms** with **0 MB GPU VRAM**.

---

## 🔬 Methodology & Architecture

In computer vision literature, extracting clothing color is defined as a mapping from a visual region to a linguistic color label:
$$\mathcal{F}: (\mathbf{I}, \mathbf{M}) \longrightarrow \hat{y} \in \mathcal{C}$$
where $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ is the RGB image, $\mathbf{M} \in \{0, 1\}^{H \times W}$ is the garment segmentation mask, and $\mathcal{C} = \{1, 2, \dots, 12\}$ is the standardized CCIHP color vocabulary.

The **M5 Engine** structures this task into four sequential, physically grounded modules:

```mermaid
flowchart LR
    subgraph S1 ["Stage 1: Spatial Denoising"]
        direction TB
        A["Input: (I, M)"] --> B["Morphological Erosion<br>M_core = M ⊖ K"]
        B --> C["Extract Core Pixels<br>(Eliminate edge bleed)"]
    end

    subgraph S2 ["Stage 2: Photometric Decoupling"]
        direction TB
        D["RGB → CIELAB Space<br>(L*, a*, b*)"] --> E["Separate Lightness (L*)<br>from Chroma (a*, b*)"]
        E --> F["Shadow & Glare<br>Decomposition"]
    end

    subgraph S3 ["Stage 3: Palette Quantization"]
        direction TB
        G["MiniBatch K-Means<br>(K = 4 in LAB)"] --> H["Cluster Centroids (c1, c2)<br>& Area Weights (w1, w2)"]
        H --> I["Form 58D Feature Vector"]
    end

    subgraph S4 ["Stage 4: Metric Decision Head"]
        direction TB
        J["Achromatic Gate<br>(Chroma C* < 11.5)"] --> K["Calibrated Navy Gate<br>(L* cutoff = 26.78)"]
        K --> L["Weighted ΔE Matching<br>(w_L = 0.65)"]
        L --> M["Multicolor Gating<br>(w2 ≥ 27%, ΔE ≥ 22.0)"]
    end

    C --> D
    F --> G
    I --> J
    M --> Out["Predicted Attribute<br>Color ID, Name, HEX"]

    style S1 fill:#f8fafc,stroke:#64748b,stroke-width:1px
    style S2 fill:#f0fdf4,stroke:#16a34a,stroke-width:1px
    style S3 fill:#eff6ff,stroke:#2563eb,stroke-width:1px
    style S4 fill:#faf5ff,stroke:#9333ea,stroke-width:1px
    style Out fill:#fffbeb,stroke:#d97706,stroke-width:2px
```

*Figure 1: End-to-end architecture of the proposed M5 Garment Color Extraction Framework.*

---

### Step-by-Step Pipeline Mechanics

#### Stage 1: Spatial Boundary Denoising ($\mathbf{M} \to \mathbf{M}_{\text{core}}$)
* **The Problem:** Neural network masks (from Mask2Former, YOLO, or SAM) inevitably suffer from boundary blur, bleeding 2–4 pixels into background walls, floor tiles, or human skin.
* **The Formulation:** We apply morphological erosion with an elliptical kernel $\mathbf{K} \in \{3\times3, 5\times5\}$:
  $$\mathbf{M}_{\text{core}} = \mathbf{M} \ominus \mathbf{K}$$
* **Intuition:** Peeling away the outermost pixel boundary isolates the **core fabric region**, completely eliminating background color leakage before any color extraction occurs.

#### Stage 2: Photometric Invariance & Shadow Decoupling ($\text{sRGB} \to \text{CIELAB}$)
* **The Problem:** In sRGB/HSV color spaces, illumination and color information are entangled. A cast shadow across a white t-shirt causes RGB values to plunge, tricking naive algorithms into predicting grey or black.
* **The Formulation:** Pixels are mapped into the device-independent **CIE $L^*a^*b^*$ color space**:
  * $L^* \in [0, 100]$: Achromatic lightness channel.
  * $a^* \in [-128, 127]$: Green–Red opponent chromaticity.
  * $b^* \in [-128, 127]$: Blue–Yellow opponent chromaticity.
* **Intuition:** By decoupling luminance ($L^*$) from chromaticity ($a^*, b^*$), the intrinsic fabric color remains invariant across shaded folds and bright lighting. Extreme specular highlights ($L^* > 98$) and void shadows ($L^* < 12$) are safely isolated.

#### Stage 3: Unsupervised Palette Quantization (MiniBatch $K$-Means, $K=4$)
* **The Problem:** Clothing is rarely uniform; real garments feature folds, secondary logos, zippers, or multicolor patterns. Taking a simple global mean or median blurs distinct colors together.
* **The Formulation:** We fit a MiniBatch $K$-Means clustering model ($K=4$) directly in CIELAB space to minimize within-cluster sum of squares:
  $$\arg\min_{\mathbf{S}} \sum_{k=1}^{4} \sum_{\mathbf{p} \in S_k} \|\mathbf{p} - \mathbf{c}_k\|^2$$
* **Intuition:** Yields dominant cluster centroids $\mathbf{c}_1, \mathbf{c}_2$ and area proportions $w_1, w_2$. Along with multi-channel quantiles and variances, this forms a compact **58-dimensional perceptual feature vector $\mathbf{f}_{58}$**.

#### Stage 4: Calibrated Metric Decision Head ($\mathbf{f}_{58} \to \hat{y}$)
A hierarchical decision network calibrated on 3,017 real-world garments:
1. **Achromatic Gate:** Evaluates metric chroma $C^* = \sqrt{a^{*2} + b^{*2}}$. If $C^* < 11.5$, chromaticity is negligible $\to$ deterministic mapping to **Dark** ($L^* < 28$), **Medium** ($28 \le L^* < 75$), or **Light** ($L^* \ge 75$).
2. **Calibrated Navy Gate:** Solves the notorious navy-vs-black ambiguity under low light using a calibrated decision hyperplane ($L^*_{\text{cutoff}} = 26.78, b^* < -8.0$).
3. **Luminance-Discounted $\Delta E$ Matching:** Chromatic clusters are matched against empirical prototype centers using a modified Euclidean distance that downweights luminance variations ($w_L = 0.65$):
   $$\Delta E_{\text{weighted}} = \sqrt{0.65 \cdot (\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$$
4. **Multicolor Gating:** If the secondary cluster occupies $w_2 \ge 27\%$ of the garment area with perceptual contrast $\Delta E(\mathbf{c}_1, \mathbf{c}_2) \ge 22.0$, the garment is formally classified as **`12 Multicolor`** (capturing stripes, plaid, and color-blocking).

---

## 📊 Academic Benchmark & Ablation Study (M1 to M5)

Evaluated under strict academic protocols on **1,265 Oracle Ground Truth garments** across 200 validation images from the **CCIHP** benchmark (3,017 calibration garments from Training Set, **zero data leakage**):

### Overall Ablation Results (M1 to M5)

| Method ID | Method Description | Color Representation | Single-11 Acc (%) | Relaxed Semantic Acc (%) | CPU Latency |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **M1** | Heuristic HSV Thresholding | Naive RGB/HSV rule-based | 73.12% | 80.00% | 1.1 ms |
| **M2** | Basic Color Naming (*van de Weijer 2009*) | 11-Color pixel lookup table | 66.33% | 79.05% | 4.8 ms |
| **M3** | Shadow-Aware CIELAB Clustering | MiniBatch K-Means ($K=4$) + LAB | 74.39% | 89.25% | 2.5 ms |
| **M4** | Category-Conditioned Prior Head | M3 + Upper/Lower garment priors | 74.57% | 89.17% | 2.6 ms |
| **M5** | **Supervised 58D Palette Ensemble** 🏆 | **Full 58D vector + calibrated decision head** | **`77.47%`** | **`91.07%`** | **`3.2 ms`** |

### Per-Category Performance (M1 to M5 Relaxed Accuracy)

| Garment Category | Instance Count ($N$) | M1 (HSV) | M2 (BCN11) | M3 (CIELAB) | M4 (Prior) | **M5 (Ensemble)** 🏆 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coat (Jackets / Blazers / Outerwear)** | 304 | 80.92% | 80.59% | 91.45% | 90.79% | **`94.08%`** |
| **Pants (Trousers / Jeans / Shorts)** | 397 | 85.46% | 77.69% | 90.98% | 91.98% | **`93.23%`** |
| **UpperClothes (Shirts / T-shirts / Tops)** | 487 | 78.03% | 79.26% | 87.68% | 87.89% | **`89.32%`** |
| **Skirt (Skirts / Dress-bottoms)** | 77 | 60.00% | 78.67% | 81.33% | 76.00% | **`78.67%`** |
| **Hair (Natural Hair Shades)** | — | 90.56% | 91.57% | 88.20% | 91.23% | **`93.42%`** |
| **All Garments Combined** | **1,265** | **80.00%** | **79.05%** | **89.25%** | **89.17%** | **`91.07%`** |

*Complete benchmark breakdowns and confusion matrices are available in [benchmark/README.md](benchmark/README.md).*

---

## 🎨 12-Color Taxonomy (CCIHP Standard)

| ID | Color Name | ชื่อภาษาไทย | Sample Hex | Category Examples |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Dark** | สีดำ / โทนเข้ม | `#1A1A1A` | Black, charcoal, deep navy, dark tone |
| **2** | **Medium** | สีเทา / โทนกลาง | `#808080` | Medium grey, heather grey, slate, mid-tone |
| **3** | **Light** | สีขาว / โทนสว่าง | `#F0F0F0` | White, ivory, cream, off-white, light pastel |
| **4** | **Brown** | สีน้ำตาล / สีกากี | `#7B4B27` | Brown, beige, khaki, tan, coffee, camel |
| **5** | **Red** | สีแดง | `#C42021` | Red, crimson, maroon, burgundy, scarlet |
| **6** | **Pink** | สีชมพู | `#E5859E` | Pink, rose, salmon, peach, magenta |
| **7** | **Yellow** | สีเหลือง | `#E6C82B` | Yellow, gold, mustard, lemon |
| **8** | **Orange** | สีส้ม | `#E36F1E` | Orange, rust, terracotta, tangerine |
| **9** | **Green** | สีเขียว | `#2B8C3E` | Green, olive, emerald, lime, forest green |
| **10** | **Blue** | สีน้ำเงิน / สีฟ้า | `#2D5FB9` | Blue, navy, denim, sky blue, cobalt |
| **11** | **Purple** | สีม่วง | `#7B3798` | Purple, violet, lavender, indigo, lilac |
| **12** | **Multicolor** | หลากสี / มีลวดลาย | `#FF00FF` | Striped, plaid, floral, multi-colored |

---

## 📦 Installation

```bash
git clone https://github.com/folklazy/Garment-Color-Extractor.git
cd Garment-Color-Extractor

# Install core library
pip install -e .

# Or install with REST API dependencies (FastAPI + Uvicorn)
pip install -e ".[api]"
```

---

## 💻 Usage Examples

### 1. Basic Python SDK

```python
import cv2
from garment_color import GarmentColorExtractor

# 1. Initialize engine
extractor = GarmentColorExtractor()

# 2. Load image and binary mask (or bounding box)
image_rgb = cv2.cvtColor(cv2.imread("person.jpg"), cv2.COLOR_BGR2RGB)
shirt_mask = cv2.imread("shirt_mask.png", cv2.IMREAD_GRAYSCALE) > 0

# 3. Extract garment color (< 3.2 ms on CPU)
result = extractor.predict(
    image=image_rgb,
    mask=shirt_mask,
    garment_type="upper"  # 'upper', 'lower', or 'hair'
)

print(f"Color:      {result.color_name} ({result.color_name_th})")
print(f"Color ID:   {result.color_id}")
print(f"Confidence: {result.confidence * 100:.1f}%")
print(f"Hex Code:   {result.hex_code}")
print(f"RGB Swatch: {result.rgb_swatch}")
print(f"Multicolor: {result.is_multicolor}")
print(f"Latency:    {result.latency_ms:.2f} ms")
```

### 2. End-to-End Person Re-ID Pipeline (YOLO + Garment Color)

```python
from garment_color import GarmentColorExtractor

extractor = GarmentColorExtractor()

# Extract attributes from detected bounding boxes
upper_bbox = (120, 80, 260, 230)
lower_bbox = (120, 230, 260, 420)

upper_pred = extractor.predict(image=frame_rgb, bbox=upper_bbox, garment_type="upper")
lower_pred = extractor.predict(image=frame_rgb, bbox=lower_bbox, garment_type="lower")

person_attributes = {
    "upper_color": upper_pred.color_name,
    "lower_color": lower_pred.color_name,
    "upper_hex": upper_pred.hex_code,
    "lower_hex": lower_pred.hex_code,
}
```
*(See [examples/yolo_integration.py](examples/yolo_integration.py) for the complete runnable script).*

### 3. Generate Visual Diagnostic Swatch Card

```python
import cv2
from garment_color import GarmentColorExtractor, create_color_card

extractor = GarmentColorExtractor()
pred = extractor.predict(image_rgb, mask=shirt_mask, garment_type="upper")

# Generate visual inspection card with cropped garment & swatch
diagnostic_card = create_color_card(image_rgb, shirt_mask, pred)
cv2.imwrite("color_diagnostic.png", diagnostic_card)
```

### 4. Command Line Interface (CLI)

```bash
# Using image and mask
garment-color --image person.jpg --mask shirt.png --type upper

# Using bounding box coordinates (x1 y1 x2 y2)
garment-color --image person.jpg --bbox 50 120 220 380 --type lower

# Output raw JSON
garment-color --image person.jpg --bbox 50 120 220 380 --json
```

---

## 🌐 FastAPI REST Microservice

Launch the standalone production server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is available at: `http://localhost:8000/docs`

```bash
curl -X POST "http://localhost:8000/api/v1/extract-color" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "<BASE64_IMAGE_STRING>",
       "mask_base64": "<BASE64_MASK_STRING>",
       "garment_type": "upper"
     }'
```

---

## 📁 Repository Structure

```text
Garment-Color-Extractor/
├── src/
│   └── garment_color/
│       ├── __init__.py         # Package entrypoint (v1.0.0)
│       ├── core.py             # GarmentColorExtractor orchestrator class
│       ├── taxonomy.py         # CCIHP 12-color registry, RGB/HEX, EN/TH names
│       ├── preprocessor.py     # Morphological erosion & shadow decoupling
│       ├── color_spaces.py     # Vectorized sRGB, CIELAB, HSV, Delta-E metrics
│       ├── features.py         # 58D feature extraction & MiniBatchKMeans (K=4)
│       ├── classifier.py       # M5 decision rules (Navy cutoff, multicolor check)
│       ├── visualizer.py       # Visual diagnostic card generator
│       └── cli.py              # CLI runner
├── api/
│   ├── server.py               # FastAPI application with REST endpoints
│   └── schemas.py              # Pydantic request & response models
├── examples/
│   ├── quickstart.py           # 5-line minimal working demo
│   └── yolo_integration.py     # End-to-end Person Re-ID detection pipeline
├── tests/
│   ├── test_extractor.py       # Unit tests for M5 classification
│   ├── test_color_spaces.py    # Unit tests for color conversions
│   └── test_taxonomy.py        # Unit tests for color registry
├── benchmark/
│   ├── README.md               # Detailed evaluation report on CCIHP benchmark
│   ├── eval_results.json       # Structured benchmark statistics (1,265 garments)
│   └── ccihp_benchmark_summary.json # Raw validation pilot log
├── pyproject.toml              # Modern build configuration
├── requirements.txt            # Package dependencies
├── setup.py                    # Legacy installation script
├── LICENSE                     # MIT License
└── README.md                   # Documentation
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

* **folklazy** - [GitHub Profile](https://github.com/folklazy)
