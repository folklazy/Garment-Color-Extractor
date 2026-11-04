# Garment Color Extractor — continuation contract

Start with README.md → reports/COLOR_REVIEW_TH.md → benchmark/EVIDENCE_INDEX.json → CHANGELOG.md. This is the source Git checkout for folklazy/Garment-Color-Extractor, separate from the Human Parsing publication checkout.

- The default shipped algorithm is `prototype_rules_v2`. Current method-comparison M5 has a real fitted local artifact under ignored outputs; it is not bundled or the default API. Historical imported scores do not validate this executable. Never upgrade its evidence status from a same-named research script.
- Preserve benchmark JSON and docs/history snapshots. Keep algorithm/cache identity changes explicit. `confidence` is a legacy uncalibrated pixel-support score; empty regions have no observed color.
- Never modify original CCTV/CIHP images, labels or historic parsing predictions. Do not publish CCTV filenames, images, crops, base64 thumbnails, annotations or private per-person manifests. Only sanitized aggregate evidence/code/docs may be committed.
- Inspect existing environments first; do not install into or alter the working YOLO/parser environments. Shared CPU imports and bounded tests are acceptable; no new model downloads/training/GPU benchmark is authorized by these instructions.
- Test source/API/input/palette/serialization changes with independent expected cases. Do not alter expected labels to maximize agreement with auto-generated color targets. Grayscale/IR hue observability and label provenance must remain explicit.
- Preserve unrelated/uncommitted changes, inspect staged scope, and use ordinary branch/PR review. Never force-push or rewrite benchmark history. A review commit is not a benchmark launch commit.

## Latest explicit user scope — T4 pilot redesign

User explicitly authorizes designing and running a bounded new color pilot, including a small learned classifier on the available T4. This supersedes the earlier no-training/GPU sentence only for configs/T4_COLOR_PILOT_PROTOCOL.json: 300 train +50 calibration +100 validation images, no downloads or new parsing inference; saved-mask CCTV qualitative check12images only, no CCTV color accuracy claim. Freeze memberships/source before scores; preserve historical artifacts. No additional full benchmark or hyperparameter search is implied.

## Current explicit user scope — method comparison

User authorizes testing selected techniques and publishing a YOLO-style comparison. Current contract: configs/COLOR_METHOD_COMPARISON.json. Reuse the accepted 300/50/100-image pilot membership, rules predictions and T4 MLP artifact. Fit/evaluate real M1–M5 once with train/calibration separation; no new parsing inference, downloads, CCTV accuracy benchmark or full5000 expansion. New ensemble weights remain external to Git. Historical evidence is preserved.
