# Changes

## 1.1.0 — prototype_rules_v2

This is a behavior revision. Do not reuse old accuracy claims or old cache identities for new outputs.

- Name the shipped engine as a fixed prototype/rule classifier rather than a trained M5 ensemble. `M5Classifier` remains an import alias for compatibility.
- Validate RGB/BGR arrays, binary masks, garment types and half-open XYXY boxes. Invalid/nonintersecting regions raise `ValueError`; supplying both mask and bbox is rejected.
- Empty regions return `status=no_data`, `color_id=0`, null observed HEX/RGB, and an empty palette instead of an invented gray observation.
- Avoid duplicate/zero-weight palette entries; normalize palette mass over all analyzed pixels. Aggregate clusters of the same named color before choosing the dominant family.
- Remove the artificial 65% score floor. Keep `confidence` as a legacy field containing supporting color-family pixel fraction, with explicit `score_kind` and null `calibrated_probability`. Consumers must not display it as correctness probability.
- Use observed representative colors for both HEX and RGB, including multicolor. Add palette, warnings, region source and before/after erosion counts.
- Treat mask borders as exterior during erosion; retain the thin/small region when erosion would lose too much evidence. This changes full-image-mask behavior.
- Update API schema, error status codes, engine identity and packaged optional API module. Diagnostic cards adapt to requested dimensions and explain observed color / uncalibrated support.
- Replace unsupported README accuracy, M5, speed, invariance and Re-ID claims with the actual executable contract. Preserve old benchmark JSON and historical document snapshots.

No fitted model training, segmentation inference, CCIHP/CCTV benchmark rerun, dataset/annotation changes, GPU processing or dependency installation was performed for this revision. CPU synthetic tests and read-only label-provenance checks are not new accuracy results.
