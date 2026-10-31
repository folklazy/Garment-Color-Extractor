# Observed garment color — T4 pilot plan

Authoritative execution contract: [T4_COLOR_PILOT_PROTOCOL.json](configs/T4_COLOR_PILOT_PROTOCOL.json). This is an exploratory small trial designed after earlier flawed evidence, not a new full parser benchmark.

## Scope

CCIHP oracle masks, person×nativegarment5/7/9/12. Native definitions include top/bottom of dresses; this is not CIHP non-Dress C4-P. Color labels1–12 come from dataset annotations, never from the evaluator's own RGB rules. Face has no suitable color target; Hair is supplementary future work, not silently pooled with clothing.

300Trainingimages forfit,50differentTrainingimages forcalibration,100Validationimages forevaluation. DeterministicSHAorder/RGBdedup acrosspartitions beforefeature/modelscore. Allmodelsuseidenticaleligiblemasks/pixels/erosion andGTlabels. GT0unknown does not filter RGBinputpixels. Minimumregion100px,80%characterized and80%modalcolorconsensus; excludedreasons retained. This defines a subset task, not fullvalidationperformance.

Two configurations: existingprototype_rules_v2baseline and anewsupervised62→64→32→12MLP. Input58descriptor+4nativegarmentonehot. Train-onlystandardization;40epochsAdamW/FP32onTeslaT4;bestcheckpointonlyfromcalibrationcrossentropy. No search/VALtuning/retries/fallback. Strictaccuracy/fixed12macroF1/support/confusion primary; no relaxedscore orcleanlatencyranking.

## Stages

1. Freeze memberships/GT/source/environment and pass independentCPUlabel/metric/artifactcases.
2. Extractsameoraclefeaturesforbothconfigurations,retainallregion/evidence/exclusionrows.
3. Train onceonT4;exportactualweights+normalizer+feature/classorder+sourcehashes,thenevaluateheld-outValidationonce.
4. Replayexportedmodelagainstsavedfeaturestoensureload/readbackparity. BoundedsavedM2FPCCTVqualitativeexamples12(6day/6night) only; no coloraccuracyclaimsagainstunverifiedlabelprovenance. No newparsinginference.
5. Publishcompactreport/tables/source/manifestsandlimits;keepdatasets/checkpoint/featurearrays/CCTVgalleryprivate;closeattemptstatuswithnoautomaticfullrun.

## Resources/stop

CPUthreads2/GPU0T4oneworker/freeVRAM2GiB/disk50GiB/time2h. Actualtrainingpeakrecorded asdiagnostic. FirstOOM/nonfinite/source/GT/contract/resourceerrorstopsattemptandretainsartifacts;noautomaticrestart orprecisionchange. Existingenvironmentsusedwithoutinstallation. Newuserauthorizationpermitsonlythissmallcolortraining/evaluation;no0.8B/1B/parsingrerun/tracking/colorfullbenchmark.

## Acceptance and limits

Exactdeclaredmemberships,disjointRGBhashes,allplannedimages accountedfor,metricarithmetic/confusion/denominatorscorrect,exportedmodelreadbackpredictionsidentical. Accuracydoesnotdecidetechnicalacceptance;reportlowperformancehonestly. Smallvalidationmaylackrareclasses;showsupport. Noindependentfinalconfirmation/CCTVdeploymentacceptance/materialhueclaim. No automaticreplacementofdefaultAPIenginefrompilotscores.
