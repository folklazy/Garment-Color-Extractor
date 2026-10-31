# Seven-method observed garment-color comparison

Authoritative machine contract: [COLOR_METHOD_COMPARISON.json](configs/COLOR_METHOD_COMPARISON.json). The earlier [T4 pilot protocol](configs/T4_COLOR_PILOT_PROTOCOL.json) remains historical and unchanged. This study extends its identical, locked data membership to M1–M5; it does not expand the dataset or parsing workload.

## Question and population

Which selected executable technique classifies the observed color of the same oracle garment regions most accurately? Compare seven **application configurations**, not architecture effects or controlled single-factor ablations. The study is exploratory, designed after earlier results; this is not an independent confirmatory test.

Use the accepted `t4-pilot-20261101` membership: 300 Training images (1,648 regions) for fitting, 50 other Training images (370 regions) for calibration, 100 official Validation images (614 regions) for evaluation. Their RGB hashes are disjoint. Reuse frozen memberships, full-mask geometry, GT labels and exclusion rules. No resampling after scores. This is **100 validation images**, not the full 5,000-image benchmark.

A region is one Human ID × native category 5 UpperClothes, 7 Coat, 9 Pants or 12 Skirt. It needs ≥100 mask pixels, ≥80% characterized Color IDs 1–12, and ≥80% modal consensus among characterized pixels. Color GT is used for eligibility/label only and never to restrict input RGB pixels. All methods use the same complete region after the existing 5×5 ellipse erosion / 3×3 / original fallback. PIL RGB uint8, original geometry, no augmentation, shadow pixel deletion or color correction. Dataset categories may include top/bottom of dresses; this is not parser non-Dress C4-P.

Native color IDs: 1 Dark (includes navy), 2 Medium, 3 Light, 4 Brown, 5 Red, 6 Pink, 7 Yellow, 8 Orange, 9 Green, 10 Blue, 11 Purple, 12 Multicolor. GT0 is uncharacterized. Hair/face are outside the garment-color comparison. Exact eligible units and excluded reasons are inherited; do not remove difficult or misclassified regions.

## Frozen configurations

| ID | Executed technique | Fitting / category information |
|---|---|---|
| M1 | Pixel HSV rules, majority / fixed multicolor rule | No learned parameters; no category |
| M2 | 11 fixed RGB anchors → soft LAB assignment → conditional CCIHP projection | Train conditional projection; calibration multicolor threshold; no category |
| M3 | K=4 LAB palette, global Gaussian prototypes, fixed shadow/contrast rules | Train global means/covariances/priors/navy boundary; calibration threshold |
| M4 | Category-conditioned LAB prototypes with global fallback | Train Upper/Lower heads; calibration thresholds per group |
| M5 | Real HistGradientBoosting + ExtraTrees + prototype probability blend | Two trained Upper/Lower ensemble heads, 96D descriptors; fixed blend/hyperparameters |
| prototype_rules_v2 | Shipped fixed prototype/rule engine | Reuse accepted predictions; Upper/Lower hint |
| T4_MLP_62_64_32_12_v1 | Real exported 62→64→32→12 MLP | Reuse accepted T4 weights/predictions, train-only scaler, calibration epoch selection; native category one-hot |

M2 is a local fixed-anchor approximation, **not** the learned van de Weijer lookup-table implementation. M5 here is an actual fitted ensemble, unlike the historical CCTV alias. Legacy research parameter shapes and deliberate adaptations are recorded in [METHOD_LINEAGE.json](manifests/METHOD_LINEAGE.json); no historical result is overwritten or transplanted into this table.

M1–M5 share their actual selected research algorithms but use the same post-erosion full RGB pixels as both “raw” and “eroded” feature inputs. Their descriptor has **84 base dimensions**, not the 56/58 claimed by old comments. M5 adds 11 prototype values + one secondary mass = **96**, omitting the hair-only branch and its three inputs. The local prototype/rules engine and MLP retain their different original 58/62D descriptors; shared data does not mean identical feature access or category conditioning.

M5: HGB max_iter180, learning_rate.055, max_depth6, min_samples_leaf12, L2=1.5, early_stopping=False; ET250 trees, max_depth16, min_samples_leaf3. Seed42, CPU threads2, ET n_jobs2. Per-training-head weight `(N/(K*n_class))^.22`. Align probabilities using each estimator's actual `classes_`; base p=.60 HGB+.40 ET. Keep p12 unchanged; p1..11=.85 base1..11+.15 prototype1..11×(1−base12). No confidence calibration claim. Fit each head only on its Training rows; no hair/global unused estimator.

M2 threshold grid .40..80/41 points; M3 .16..55/40 points; M4 .18..55/38 points. Choose on the 370 calibration regions only, maximizing `.45*TPR_multi+.55*TNR_single` for M2/3; for M4 use multicolor weight `clip(2.5*calibration_multi_rate,.20,.48)`. First/smallest threshold wins exact ties; if a binary calibration side is absent, use declared fixed default. This is binary multicolor calibration, not VAL search. Global/category prototype fitting, train priors, shrinkage, anchor fallbacks and fixed color rules remain explicit in source. Do not retrain/select MLP again or change defaults after seeing this comparison.

## Metrics and decisions

Primary: strict correct regions / all 614 eligible regions. Secondary: fixed-12 macro F1, supported-GT macro F1, per-color precision/recall/F1/support, per-category accuracy and pooled 12×12 confusion. TP is the confusion diagonal; F1=2TP/(GT_support+prediction_support), with zero denominator →0 for the fixed-12 mean. Precision with no predictions and recall with no GT support are null. Empty population →null. No relaxed-neighbor score or color accuracy against auto-generated CCTV labels.

Report all seven in frozen order; identify the descriptive leader separately for strict accuracy and macro F1, preserving exact ties. No weighted score, practical-tie threshold, significance claim, speed winner or deployment acceptance. A better result is a candidate for further testing, not proof of intrinsic material hue or CCTV robustness. Rare classes and correlated garments within images limit conclusions.

## Execution, integrity and resources

Freeze source/config/parent hashes and exact memberships before the new M1–M5 fit. New source lives in this repository; never import the historical runner (which changes CUDA visibility/global warning filters). Use a fresh output directory, one bounded fit attempt, no downloads/dependency changes/parsing inference/MLP retraining. Check RGB/annotation hashes and regenerated region membership/pixel counts against accepted bindings while reading required inputs; do not rehash the whole dataset/archive.

Use existing compatible environment, CPU thread cap2, free disk≥50GiB, peak RSS guard4GiB checked between images/stages, total wall time guard2h checked at boundaries. HGB/ET are CPU algorithms; their fitting does not require a T4. The reused MLP was genuinely trained on T4. No new GPU calls or CPU-versus-GPU efficiency ranking. Guards are boundary checks, not hard OS quotas; no unlimited retries/fallback. On error retain files/traceback; incomplete configurations have null primary, never successful-subset full scores.

Read back the exported trusted local joblib ensemble and verify all614 method predictions; align reused observations by exact ordered region identity, not counters. Compare reused metrics exactly with the accepted pilot summary. Record scipy clustering warnings and finite/mass checks; do not suppress warnings globally. Joblib is executable pickle: only load locally created/trusted artifacts.

Keep dataset/archive, feature arrays, model artifacts, detailed logs and images outside Git in ignored `outputs/<run_id>/`. Publish compact tables, source, configuration, hashes, metrics, schema and reports. Do not upload CCTV IDs/images/labels or redistribute dataset/trained weights automatically. No new CCTV accuracy evaluation is part of this comparison.

## YOLO format reused

Actually inspected read-only: `YOLO_Instance_Segmentation_MOTS20_Scaling_Study/{MASTER_RESULTS.md,STUDY_STANDARD.md,DATA_SCHEMA.md}` and `YOLO_Large_Seg_MOTS20_Benchmark/{REPORT.md,EXPERIMENT_PROTOCOL.md,RESULTS_SUMMARY_TH.md,PRESENTATION_SUMMARY_TH.md}` under the shared workspace. Reuse fixed roster order, canonical numerical tables, protocol/source provenance, per-unit breakdowns, clear winners/trade-offs and Thai handoff. Adapt AP/instance segmentation to region-color classification and per-color/per-garment tables. Per-sequence MOTS, FPS/VRAM rankings and segmentation mAP are not applicable here. No YOLO files/results/history changed.

## Reproduction

See [setup](docs/METHOD_COMPARISON_SETUP.md). Clone alone cannot reproduce the data-dependent study: the source images/CCIHP annotations, accepted parent artifact set and compatible environment must be supplied separately. Published tables suffice to recompute arithmetic, not to reconstruct missing trained weights or private data.
