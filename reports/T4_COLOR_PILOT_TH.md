# ทดลองระบบสีใหม่บน Tesla T4 — ผลนำร่องและข้อจำกัด

ตรวจปิดรอบ **2026-11-01T03:07:22.540243+07:00 (เวลาไทย)** · run `t4-pilot-20261101` · **PILOT_COMPLETE / NOT_FULL_BENCHMARK**

## ผลที่ทำจริง

ออกแบบและรันโมเดลสีใหม่จริงบน **Tesla T4** ใช้FP32/noAMP/noTF32 ไม่ได้เพียงเปลี่ยนชื่อheuristicเป็นMLP โมเดล `T4_MLP_62_64_32_12_v1` มีweights/train-onlynormalizer/classorder/featurebindingที่ส่งออกแล้ว แยกจาก `prototype_rules_v2` เดิม และไม่ใช่historicalM5ensemble

|Configuration|N regions|Strict accuracy %|Macro-F1 fixed12 %|Macro-F1 GT-supported %|
|---|---:|---:|---:|---:|
|prototype_rules_v2|614|58.31|47.47|47.47|
|T4_MLP_62_64_32_12_v1|614|68.73|52.55|52.55|

ในชุดนี้ MLP accuracy **สูงกว่า 10.42 percentage points** เทียบกฎเดิม เป็นผลนำร่องบนoraclemasks ไม่ใช่หลักฐานstatisticalsignificance/ความพร้อมdeployment หรือการรับรองว่าสีวัสดุจริงถูกต้องในทุกแสง การเลือกepochใช้calibrationcrossentropyอย่างเดียว ไม่ดูValidationเพื่อปรับsettings

## ข้อมูลและวิธีทดลอง

- CCIHPTraining300imagesสำหรับfit → 1,648 eligibleperson×garmentrecords; แยกTraining50imagesสำหรับcalibration → 370 records
- CCIHPValidation100images → **614 regions** ใช้evaluateonce ไม่ใช่full5,000. SHA-orderseed20261101และRGBexact-dedupก่อนคะแนน ไม่มีเลือกภาพจากความแม่น
- OracleunitคือHuman_id×nativegarmentID5UpperClothes/7Coat/9Pants/12Skirt. CCIHPนิยามมีtop/bottomของdress จึงไม่เรียกเป็นnon-DressC4-P แบบparserbenchmarkเดิม ไม่ทดสอบFace/Hairรวมเป็นclothingอย่างเงียบ
- Inputใช้fullGTperson/categorymaskก่อนดูColor_ids ไม่กรองRGBตามสีที่ทราบเอง GTcolorใช้modeของcharacterizedpixels; minimumarea100px/characterized≥80%/modalconsensus≥80% เหมือนทุกconfiguration เก็บexcludedreasonsครบในprivateREGIONSและaggregatecounts
- BaselineและMLPใช้RGBเดิม/erosion5×5fallback/descriptor58เหมือนกัน รวม4nativecategoryonehotเป็น62DสำหรับMLP. mean/stdfitจากtrainเท่านั้น; network62→64→32→12,AdamWlr.001/weightdecay.001/batch128/40epochs,seed20261101 ไม่มีsearch/augmentation/VALtuning
- เลือกcheckpointepoch **28** จาก50calibrationimages ไม่มีลดเกณฑ์เพราะคะแนนต่ำ traincalibrationlabelsเป็นdatasetannotations ไม่ใช่กฎRGBเดียวกับevaluatorสร้างให้ตัวเอง

[Frozen protocol](../configs/T4_COLOR_PILOT_PROTOCOL.json) / [plan](../PLAN.md) / [exact memberships/source/artifactmanifest](t4-pilot-20261101/MANIFEST.json). ขอบเขตสีตามREADMEต้นฉบับCCIHP: Darkรวมnavy,Multicolorคือลายหลายสี ไม่อ้างnearestprototypeเท่ากับpublishedlearnedcolor-namingbaseline

## ผลแยก native garments

|Native category|N|Rule accuracy %|T4 MLP accuracy %|
|---|---:|---:|---:|
|5|256|51.56|63.28|
|7|139|60.43|74.10|
|9|183|63.93|71.58|
|12|36|69.44|72.22|


## Support และ F1 รายสี

|Color|GT support|Rule F1 %|MLP F1 %|
|---|---:|---:|---:|
|Dark|208|80.18|82.79|
|Medium|52|32.37|28.17|
|Light|115|63.25|69.01|
|Brown|41|29.03|54.79|
|Red|35|70.59|77.11|
|Pink|17|46.15|60.00|
|Yellow|10|55.56|76.19|
|Orange|3|50.00|0.00|
|Green|17|36.36|66.67|
|Blue|46|30.51|65.88|
|Purple|2|40.00|0.00|
|Multicolor|68|35.59|50.00|

ValidationมีOrangeเพียง3regions/Purple2regions; จึงไม่สรุปว่าคลาสหายากผ่านการยืนยันแล้ว แม้คะแนนรวมดีขึ้น ไม่มีuncertainty/significanceanalysisรอบนี้

Fixed12macroF1นับabsentclassเป็น0ตามกติกาตรึง; supportedmacroแสดงเสริม ไม่แอบเปลี่ยนclasssetให้คะแนนดีขึ้น Pixel-supportไม่ใช่garment-count;ตารางนี้นับregionrecords คะแนนstrictaccuracyไม่ใช่relaxedneighbor/top2หรือofficialcharacterizedparsingAP ไม่มีใช้คำzero-bias/zerotrainingexposure/independentfinalconfirmation

## ตรวจรับ implementation/export

- 4newCPUcasesด้านGTeligibility/handcountmetrics/deterministicmembership/artifactclassorderผ่านก่อนรัน; reusebase22testsที่dependenciesไม่เปลี่ยน ไม่มีใช้testcountแทนquality
- Source/protocol/membershipsfreezeก่อนrun; validationpredictions+GT+confusionเก็บfullprecision. RecomputeaggregateจากsavedpredictionarrayตรงcanonicalMETRICS
- Exportedmodelโหลดใหม่บนCPUแล้วlabels **ตรงกับGPUsavedvalidationทุกregion**, softmaxallcloseatol1e-6/rtol1e-5 ไม่อ้างbitwiseCPU/GPUequivalence
- ActualSDKbaselinechecked3savedoracleexamplesตรงกับbaselineที่ใช้ประเมิน ทุกsourcehash/weightsSHA/normalizer/classorderตรวจแล้ว ไม่มีruntimedefaultAPIถูกเปลี่ยนเป็นtrainedmodelเพราะpilotคะแนนอย่างเดียว
- TrainingGPUpeakallocated **65.06 MiB**, reserved **66.00 MiB** เป็นdiagnosticของMLPนี้ ไม่ใช่VRAM/FPSrankingของparser. ExistingT4ว่างก่อนlaunch ไม่แก้processอื่น/driver/environment ไม่ติดตั้งpackagesหรือดาวน์โหลดweightsเพิ่ม
- FieldSTATE.GPU_model_callsเดิมไม่ได้เพิ่มcounterจึงไม่ใช้เป็นหลักฐานzeroGPUcalls; [verification](t4-pilot-20261101/VERIFICATION.json) แยกtraining/calibration/validationbatchcountsจากactualexecutedepochs มีactualCUDAdevice/losses/VRAM/exportเป็นหลักฐาน

## CCTV check — ไม่สร้าง accuracy จาก pseudo-label

ตรวจเชิงคุณภาพ12ภาพ (6day/6night) เลือกด้วยSHA(seed:RGBhash), reuse **savedM2FPpredictions** ไม่มีparsinginferenceใหม่ รวม19predictedgarmentregions ได้baseline/MLPobservedcolorและwarnings แต่ไม่มีindependenthumanGT จึง **ไม่คำนวณCCTVcoloraccuracy**

ตรวจcropตัวอย่าง8จาก19regionsด้วยตาเพื่อดูgeometryและappearance ไม่ใช่การให้humanaccuracylabel และไม่อ้างว่าดูครบทุกregion

Privategalleryบนเครื่อง: `outputs/t4-pilot-20261101/private_cctv_gallery/index.html` พร้อมprivateMANIFESTและcroppedpredictedmasks เก็บ14low-chromaregionsพร้อมข้อจำกัด: grayscale/IRไม่ยืนยันhueของวัสดุจริง ไม่ให้softmaxสูงกลบปัญหาobservability. บางภาพไม่มีeligiblepredgarmentก็เก็บเหตุผล ไม่สร้างmaskขึ้นมาจากGT

กลุ่มCCTVมีselectionpartialparts/สีบางส่วนauto-generatedตามreviewก่อนหน้า จึงไม่ใช้เป็นtestlabelสำหรับฝึก/เลือกepoch/ปรับthresholdหรืออ้างdeploymentwinner รูป/ชื่อกล้อง/คน/GT/masksไม่ขึ้นpublicGit

## Artifacts และสิ่งที่ยังไม่ปิด

Weightsจริง `outputs/t4-pilot-20261101/learned_color.pt` SHA256 `4145b416f9020422e389210af6eab6adc6d92d6c37af6033f60ffc712969bfde`; ขนาด 30,493 bytes. Savefeaturearrays,per-regionmanifest/predictions/traininghistory,environment/launchcommitและsourcehashesอยู่runเดียวกัน ข้อมูล/weightsที่deriveจากCCIHPNCannotationsไม่เผยแพร่อัตโนมัติ ไม่ประกาศlicenseMITของcodeครอบtrainedartifactsทั้งหมด

ส่งต่อ [metrics CSV](t4-pilot-20261101/METRICS.csv), [per-class CSV](t4-pilot-20261101/PER_CLASS.csv), [per-region public-data predictions](t4-pilot-20261101/VALIDATION_PREDICTIONS.csv), [canonical summary](t4-pilot-20261101/SUMMARY.json). Clonecodeอย่างเดียวไม่reproduceโดยไม่มีdataset/archive/sharedassetsและtrainedartifactที่เตรียมตามmanifest

สิ่งที่ทำสำเร็จคือ **standardizedsmallpilot + actuallearnedmodel + export/readback + held-outscores** ไม่ใช่fullbenchmark/productionreplacement. ยังต้องreviewrareclasses/multicolorและสีที่เห็นในภาพกับคำlabelก่อนขยาย, independentCCTVcolorGT/observabilityและparsermaskend-to-endcoverageแยกงาน ไม่มีเริ่มfulltraining/parserrerun/ทดลองซ้ำเพื่อไล่คะแนนอัตโนมัติ
