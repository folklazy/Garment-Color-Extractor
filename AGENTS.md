# Garment Color Extractor — continuation contract

Start with README.md → reports/COLOR_REVIEW_TH.md → benchmark/EVIDENCE_INDEX.json → CHANGELOG.md. This is the source Git checkout for folklazy/Garment-Color-Extractor, separate from the Human Parsing publication checkout.

- The shipped algorithm is `prototype_rules_v2`; there is no trained M5 ensemble artifact here. Historical imported scores do not validate this executable. Never upgrade its evidence status from a same-named research script.
- Preserve benchmark JSON and docs/history snapshots. Keep algorithm/cache identity changes explicit. `confidence` is a legacy uncalibrated pixel-support score; empty regions have no observed color.
- Never modify original CCTV/CIHP images, labels or historic parsing predictions. Do not publish CCTV filenames, images, crops, base64 thumbnails, annotations or private per-person manifests. Only sanitized aggregate evidence/code/docs may be committed.
- Inspect existing environments first; do not install into or alter the working YOLO/parser environments. Shared CPU imports and bounded tests are acceptable; no new model downloads/training/GPU benchmark is authorized by these instructions.
- Test source/API/input/palette/serialization changes with independent expected cases. Do not alter expected labels to maximize agreement with auto-generated color targets. Grayscale/IR hue observability and label provenance must remain explicit.
- Preserve unrelated/uncommitted changes, inspect staged scope, and use ordinary branch/PR review. Never force-push or rewrite benchmark history. A review commit is not a benchmark launch commit.

## Latest explicit user scope — T4 pilot redesign

User explicitly authorizes designing and running a bounded new color pilot, including a small learned classifier on the available T4. This supersedes the earlier no-training/GPU sentence only for configs/T4_COLOR_PILOT_PROTOCOL.json: 300 train +50 calibration +100 validation images, no downloads or new parsing inference; saved-mask CCTV qualitative check12images only, no CCTV color accuracy claim. Freeze memberships/source before scores; preserve historical artifacts. No additional full benchmark or hyperparameter search is implied.
