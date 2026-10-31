# Canonical color-study tables

Current run: `color-methods-20261101`; protocol `OBSERVED_GARMENT_COLOR_SEVEN_METHODS_V1`. UTF-8 CSV, comma separator, header row. Numeric accuracy/P/R/F1 values are **fractions0–1** in files and percentages0–100 in Markdown/figures. Empty CSV cell / JSON null means unavailable, never zero; nearby reason fields explain omissions. Counts are integers, sizes bytes, memory MiB/GiB binary units, times seconds unless otherwise specified.

Each evaluated unit is one eligible CCIHP person×native garment category in an image. There are614 units/100Validationimages. They are not614 independent images or unique people. Image IDs preserve leading zeros. Source mask/eligibility and color class definitions: [protocol](EXPERIMENT_PROTOCOL.md). All seven configurations use the same exact ordered units. Comparisons refer to checkpoint/algorithm+configuration, not individual component causality.

## METHOD_RESULTS.csv

One row per method, fixed M1,M2,M3,M4,M5,rules,MLP order. `run_id` and `protocol_id` identify this task, avoiding same-named legacy-M5 ambiguity. `technique` is the display description. `expected_regions`/`completed_regions` include all planned units; `correct_regions` is confusion diagonal sum. `strict_accuracy=correct_regions/expected_regions` only when COMPLETE. `failed/missing/invalid` are this comparison's accounting, not a rewrite of historical parser/pilot attempts.

`macroF1_fixed12` is mean of12 class F1, zero for a class with GT_support+pred_support=0. `macroF1_GT_supported` averages classes with GT_support>0; it equals fixed12 here because all12 have GT support. `prediction_origin` distinguishes reused accepted rules/MLP from new M1–M5 execution. `latency_ms` is null with reason `NOT_MEASURED_COMPARABLY`; it is not an estimate derived from total script time.

## PER_CLASS.csv and CONFUSIONS.json

84 rows =7methods×12colors. `color_id/name` are CCIHP native IDs1–12. `GT_support` counts reference regions of that color, `pred_support` predicted regions. TP exact color agreement; FP=pred_support−TP; FN=GT_support−TP. Precision=TP/pred_support (null when0); recall=TP/GT_support (null when0); F1=2TP/(GT_support+pred_support), zero if denominator0. These are **region counts**, not pixels or segmentation IoUs.

CONFUSIONS.json has rowsGT/columnsPrediction, IDs1–12 in that order; each matrix sums614. Marginals agree with PER_CLASS. `CONFUSION_ERRORS.csv` lists nonzero off-diagonal pairs, ordered by decreasing count then numeric IDs within method. No relaxed color-neighbor corrections or removal of failed classes.

## PER_CATEGORY.csv and VALIDATION_PREDICTIONS.csv

28 per-category rows=7×native5/7/9/12, preserving classes instead of merging geometry. `regions`, `correct_regions`, `strict_accuracy` are conditional on that native category. Pooled accuracy is the region-weighted total, not the unweighted category average. Dress parts may be present under CCIHP's category definitions.

614 prediction rows contain public CCIHP `image_id`, `person_id`, `native_category`, `GT_color`, and one native-color label per method. Composite region identity is `(image_id,person_id,native_category)`; never identify a region by an index alone. These are evaluation-derived color results, not CCTV or original dataset images. CSV requires loading image_id as text. Compact data allows independent confusion arithmetic; source-image checks require external artifacts.

## SUMMARY / FIT_EVIDENCE / ENVIRONMENT / MANIFEST / VERIFICATION

SUMMARY retains all full-precision metrics and category breakdowns. `elapsed_seconds` includes extraction/fitting/validation/readback; `max_CPU_RSS_MiB` is whole-process peak. Neither is per-method inference latency or model-memory comparison. `new_GPU_calls=0` describes this work unit, not the original T4 MLP training.

FIT_EVIDENCE binds actual serialized ensemble hash/bytes, per-group training counts/classes/dimensions/iterations/trees and selected calibration thresholds. `joblib_reload_predictions_identical` counts614 regions where all M1–M5 labels match after trusted local export/readback, not probability calibration or independent test accuracy. Library warnings are retained.

MANIFEST links exact source/config/launch commit, parent artifacts/memberships, local artifact hashes and public evidence hashes. The result/report commit occurs **after** execution; it is not the launch commit. Manifest excludes its own hash. SHA256 proves observed local byte identity, not authenticated origin or annotation correctness. Dataset/model download locations are not invented; no credentials/signed URLs included. Trained weights and arrays are LOCAL_ONLY; source weights are not bundled.

VERIFICATION records actual consistency/reload/CPU-contract checks and explicit limitations. Passing contracts means the stated measurement was computed consistently, not that output accuracy or CCTV deployment meets an unprovided threshold. Historical pilot/result files remain unchanged and are cited by hashes rather than edited.
