# Benchmark evidence register

**No accuracy or latency table in this directory currently validates `prototype_rules_v2`.** The package is a rule/prototype implementation, while imported M5 results refer to a separate research program. Retained files are evidence to review, not a reproducibility guarantee.

| Artifact | Actual role | Claim limits |
|---|---|---|
| `ccihp_benchmark_summary.json` | Byte-identical copy of the earlier 200-image A/B pilot with 500 calibration images / 3,017 training garments | Does not describe a trained M1–M5 ensemble or the shipped package |
| `eval_results.json` | Previously assembled presentation table quoting M1–M5 scores | Does not bind executable source, fitted ensemble, per-case outputs, or timing boundaries; calibration metadata mixes the earlier pilot with the later study |
| External `ORACLE_M1_TO_M5_SUMMARY.json` | Existing separate CCIHP research summary: 800 training images / 6,328 target records / 200 validation images | A separate program with fitted HGB/ExtraTrees models; fitted artifacts are not shipped here. No original run rerun during this review |
| `EVIDENCE_INDEX.json` | Hashes of preserved historical files and source commit | Identifies bytes; it does not authenticate data, establish no exposure, or validate the current implementation |

The raw M1–M5 summary has 487 UpperClothes, 304 Coat, **399 Pants and 75 Skirt** observations (1,265 total). The previous README displayed 397/77 while quoting the same aggregate. The original files and [historical table](../docs/history/BENCHMARK_before_review.md) remain preserved; current documentation does not promote them to validated package results.

The old `75.17 seconds` total includes a different workflow. Dividing it by 1,265 gives about 59.4 ms, not 3.2 ms; neither is a clean per-garment inference benchmark without measured boundaries/repetitions. The hand-written nearest-anchor comparator is not an execution of the published van de Weijer learned color-naming lookup table. No literature-baseline superiority claim is supported here.

## CCTV color evidence

The legacy ingestion program assigns explicit colors when available and otherwise runs `classify_color` over RGB pixels. Canonical polygons discard the explicit color suffix, and the metadata does not retain a per-instance human-versus-rule label source. Consequently the retained `Color_ids` must not be presented as entirely independent human color ground truth.

The legacy CCTV evaluation sets `pred_m4 = pred_m3` and `pred_m5 = pred_m3`; these are aliases, not independent evaluations of the category-conditioned method and supervised ensemble. Their high agreement cannot validate this package. See the [technical review](../reports/COLOR_REVIEW_TH.md).

## Required before an accuracy comparison

1. Define whether the target is observed image color or intrinsic material color. Mark hue unobservable in grayscale/IR/ambiguous cases; brightness can be evaluated separately.
2. Retain human/adjudicated versus automatic label provenance per region and prevent rule-generated labels from becoming independent scoring targets for the same method family.
3. Freeze exact train/calibration/evaluation memberships and no-tuning policy. Record RGB/GT identities and transformations, not just image IDs.
4. Bind actual executable source and any trained artifacts to the run. Compare the shipped implementation, not a same-named algorithm in another script.
5. Retain per-region predictions, support, missing/abstained observations, strict confusion and metric definitions. Keep oracle-mask color quality separate from parser-mask-plus-color quality.
6. Report strict accuracy/macro F1 and denominators first. Broad relaxed-neighbor acceptance is a diagnostic, not evidence of strict color correctness.
7. Measure latency separately under a declared CPU profile if a speed claim is needed. The existing historical timing is not reused as a performance guarantee.

Until those bindings exist, measured synthetic behavior and code correctness are available, but real-scene benchmark accuracy remains **NOT_ESTABLISHED_FOR_THIS_IMPLEMENTATION**.
