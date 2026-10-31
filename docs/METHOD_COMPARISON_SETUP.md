# Reproduce the selected color-method comparison

Use the branch/commit associated with [current report](../REPORT.md). The public repository contains source, protocol, compact results and hashes; datasets and trained model/feature artifacts are not bundled. Do not treat the imported historical `benchmark/*.json` as current executable validation.

## Required external inputs

- Accepted parent `outputs/t4-pilot-20261101/`: FROZEN_RUN.json, REGIONS.json, DATA_COUNTS.json, SUMMARY.json, validation_predictions.npz, train/calibration/validation_features.npz, learned_color.pt. Expected hashes: [parent manifest](../reports/t4-pilot-20261101/MANIFEST.json). It also identifies dataset annotation/readme and source dependencies. These artifacts are local study outputs, with no public download promised.
- Human-Body-Extraction shared workspace: `datasets/CCIHP_icip.zip` annotations and `datasets/CIHP/{Training,Validation}/Images/`. Dataset source/terms are in the parent's provenance; do not redistribute images or weights automatically. Obtain legitimate access independently; no credentials or signed URLs belong in this repository.
- Parent FROZEN_RUN.json records original local asset locations. If moving machines, preserve that original evidence; create a distinct local artifact copy plus an explicit path-binding amendment rather than editing historical fingerprints to imply unchanged locations. The current runner reads its recorded parent locations; portability of those bindings is a prerequisite, not a claim that `clone` alone reproduces this experiment.

## Environment

Use a compatible existing Python environment. Recorded versions are in each run's ENVIRONMENT.json. Source needs NumPy, OpenCV, SciPy, scikit-learn, Pillow, joblib and threadpoolctl; learned-model replay additionally needs PyTorch. Current execution used Python3.12 and the shared YOLO environment directly without installing or changing it. CPU M1–M5 do not require CUDA. Environment recreation may not reproduce floating-point/library behavior without matching versions.

```bash
# From the Garment-Color-Extractor checkout; set your compatible interpreter.
COLOR_PY=/path/to/existing/environment/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=src "$COLOR_PY" -m unittest discover -s tests -p test_method_comparison.py -v
"$COLOR_PY" scripts/run_method_comparison.py freeze \
  --parent outputs/t4-pilot-20261101 --out outputs/color-methods-YOUR_RUN_ID
"$COLOR_PY" scripts/run_method_comparison.py run \
  --parent outputs/t4-pilot-20261101 --out outputs/color-methods-YOUR_RUN_ID
```

Use a **fresh** output directory. Freeze/check source identity before fitting. No automatic retry or overwriting; investigate contract/resource errors while preserving failed-attempt evidence. For detached server work use the server's validated process supervisor/session and record PID/log paths; a browser terminal alone is not an execution guarantee.

M5 artifacts are `joblib`/pickle and can execute code on loading. Only load locally created or independently trusted files; a SHA256 identifies the local bytes and is not a publisher signature. The study's actual model artifact remains external, so new fits with another library version are not asserted bit-identical.

## What can be verified without external data

The focused synthetic CPU cases and arithmetic from published per-region predictions/confusions can be checked. Original RGB/GT identity, image-level feature extraction, pretrained MLP and trained M5 replay require external inputs. Per-region results use public CCIHP identifiers; no CCTV images, IDs, labels or per-person manifests are published. No inference/training/parsing/full-validation work is implied by this guide.
