# Observed garment color — current comparison plan

Current authoritative protocol: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) / [COLOR_METHOD_COMPARISON.json](configs/COLOR_METHOD_COMPARISON.json). Current state: [STUDY_STATE.json](STUDY_STATE.json). Start README → PLAN → protocol → current report.

The completed [T4 pilot](reports/T4_COLOR_PILOT_TH.md) and its unchanged [protocol](configs/T4_COLOR_PILOT_PROTOCOL.json) are parent evidence. The user now authorizes comparing each selected technique in the style of the YOLO study, then committing/pushing the deliverables.

1. **Definition/freeze:** reuse accepted exact300train/50calibration/100Validation images and1648/370/614 garment regions. Pin seven configurations M1–M5/rules/MLP, shared inputs, metrics, thresholds and source bindings before new scores. Independent CPU acceptance covers class-column alignment, actual ensemble blend, descriptor shape and calibration ties. No new experiment data or parsing calls.
2. **Execution:** one bounded CPU fit of M2–M5 and evaluation M1–M5; reuse two accepted pilot observations. Export actual ensemble artifact, fit evidence and per-region predictions. Readback verification and exact membership reconciliation; stop on errors, retain history. CPU2threads/4GiB boundary RSS guard/50GiBfree/2h; no GPU work beyond the already completed T4 MLP.
3. **Closure:** compute canonical tables, per-color/category breakdowns and source/artifact manifest. Identify descriptive strict-accuracy/macroF1 leaders independently. Reports explain method/category-conditioning differences, dataset exclusions/rare classes and lack of CCTV/end-to-end/deployment evidence. No speed/VRAM ranking.
4. **Git handoff:** public source/config/compact tables/docs only; data, trained weights, detailed logs and CCTV assets stay outside Git. Use working branch and draft PR; no main merge or historical rewrite. Check staged scope, remote commit and anonymous report access.

Finish when all seven methods account for exact614 validation regions, reused results agree with parent evidence, actual exported ensemble reloads with the same predictions, and report/tables/links/Git are consistent. Accuracy is an experimental result, not a technical acceptance threshold. No automatic model replacement or additional full benchmark follows this work unit.

## Current closure

COMPLETE: all seven configurations account for exact614 validation regions. Actual input450images processed; contributing images294train/49calibration/97validation because some selected images have no eligible garments. New M1–M5 execution completed; accepted rules/MLP unchanged. Four focused CPU contracts, model reload, independent arithmetic and source/parent bindings pass. See [REPORT.md](REPORT.md) for results, diagnostics and separate accuracy/macroF1 leaders. Current work unit ends with Git publication; no automatic extra evaluation/default-model replacement.

## Selected method — user-requested decision

User asks for one best method and paper-style documentation. Selected **T4_MLP_62_64_32_12_v1** for the next phase, using the previously declared primary strict accuracy:422/614=68.73%. Selection does not change metrics, mapping, model weights, default API or frozen protocol. M4 remains the macroF1 leader, recorded as a secondary trade-off rather than a second selected method. See [selected-method report](reports/SELECTED_COLOR_METHOD_TH.md) and [decision](registry/SELECTED_METHOD.json). No new training/inference/testing was needed for this metadata/documentation work unit.
