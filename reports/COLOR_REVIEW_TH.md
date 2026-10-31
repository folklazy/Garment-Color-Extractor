# ตรวจและแก้ Garment Color Extractor

สถานะ: **CODE_CONTRACT_REPAIRED / REAL_SCENE_ACCURACY_NOT_ESTABLISHED**

ตรวจจาก source repo `folklazy/Garment-Color-Extractor` ฐาน commit `a7a80bc` พร้อมเทียบ research scripts/results ที่มีจริงใน workspace ไม่ได้ตัดสินจาก README อย่างเดียว การแก้รอบนี้ไม่รัน parsing inference, training หรือ full color benchmark ใหม่ ไม่เปลี่ยน dataset/labels/results เดิม

## ข้อสรุป

มีจุดผิดและคำอธิบายเกินหลักฐานจริง จึงไม่ใช่แค่ปรับ README ให้สวยขึ้น แพ็กเกจเดิมเรียกตัวเอง M5 supervised ensemble พร้อมคะแนน91.07% แต่ sourceเป็น fixed prototypes + if/else rules ไม่ได้โหลด HGB/ExtraTrees ที่ research M5ใช้ ความแม่นยำของimplementationที่แจกจึงยังไม่ถูกรับรองด้วยตารางนั้น

แก้ให้เป็นเครื่องมือสกัด **สีที่เห็นในภาพบน region ที่ระบุ** พร้อมพาเลตต์/สัดส่วนและชื่อสีแบบheuristic ระบุข้อจำกัดของscore,low-chroma,แสงมืดและmask input คงทางเข้าหลัก GarmentColorExtractor แต่ใช้ algorithm_id=prototype_rules_v2 และ version1.1.0 แยกcache/behaviorจากอดีต ไม่แอบนำคะแนนเก่ามารับรองเวอร์ชันใหม่

## สิ่งที่พบ → หลักฐาน → การแก้

|ระดับ|สิ่งที่พบและหลักฐาน|ผลกระทบ|สิ่งที่แก้/ข้อจำกัด|
|---|---|---|---|
|สำคัญ|READMEอ้างM5 supervised58Densemble; classifier.pyมีfixedprototypesและrules ไม่มีtrainedmodelโหลด ส่วนresearch FiveMethodColorSuiteมีHGB+ExtraTreesจริง|คนอ่านคิดว่ากำลังใช้ตัวที่มีคะแนนสูงในตาราง ทั้งที่เป็นอีกimplementation|ใช้ชื่อprototype_rules_v2,เก็บM5Classifierเป็นcompatibilityalias,ถอนbenchmarkbadge/ความแม่นยำ/latencyการันตี;ไม่สร้างensembleปลอมแทน|
|สำคัญ|prototypevaluesไม่ตรงcalibrationmeansในimportedpilotJSON ไม่มีartifactผูกการfitกับvalues|คำว่าcalibrated/validatedเกินหลักฐาน|ระบุfixedlegacyheuristicprototypes ไม่ยืนยันว่าฝึกจาก3017records;ถ้าจะใช้calibratedmodelจริงต้องexportweights/fitparameters/source/membershipให้ตรวจได้|
|สำคัญ|CCTV evaluatorเดิมมีpred_m4=pred_m3และpred_m5=pred_m3|M3/M4/M5ที่คะแนนเท่ากันไม่ใช่สามวิธีที่ทดลองอิสระ|แยกlegacyCCTVscoreออกจากpackageevidence ไม่ใช้ผลนี้อ้างM5เหนือกว่า|
|สำคัญ|ingestionscriptใช้explicitcolorsหากมี มิฉะนั้นclassify_colorจากRGB;canonicalexportตัดcolorsuffixและไม่เก็บlabelsourceรายinstance|เกิดความเสี่ยงประเมินกฎสีกับlabelsที่กฎสร้างเอง ไม่มีindependenthumanGTที่ยืนยันได้ทุกชิ้น|read-onlyreplayพบ287/290labelsตรงกฎเดิม;หลักฐานนี้ไม่พิสูจน์ว่า287รายการเป็นอัตโนมัติทั้งหมด ต้องกู้originalexplicitlabels/reviewerrecords;ไม่แก้Color_idsให้ดูเป็นhumanGT|
|สำคัญ|confidence=clipให้ขั้นต่ำ.65สูงสุด.99 หรือคูณsecondaryweight2.2|ดูเหมือนprobabilityที่calibratedทั้งที่ไม่มีการcalibrate|เลิกfloor,ใช้supportingcolorfamilypixelfraction;fieldlegacyconfidenceคงไว้แต่score_kindชัด,calibrated_probability=null,CLI/API/cardแสดงความหมายตรงกัน|
|ข้อผิดพลาด|Multicolorhex='#FF00FF'แต่rgb_swatchเป็นgarmentclusterจริง|HEXและRGBขัดกัน สีม่วงที่เห็นไม่ได้มาจากภาพ|HEX/RGBใช้observedrepresentativeเดียวกัน พาเลตต์ยังเก็บสีทั้งหมด;multicolorไม่ใช่patternclassifier|
|ข้อผิดพลาด|emptyregionได้grayRGB128/HEXแม้color_id0|เหมือนตรวจพบสีเทาทั้งที่ไม่มีข้อมูล|status=no_data,nullHEX/RGB,emptypalette;ไม่สร้างข้อมูลสีขึ้นมา|
|ข้อผิดพลาด|uniformregionสร้างคลัสเตอร์ซ้ำ/weight0;largestclusterคนละสีชนะแม้สีอีกกลุ่มมีมวลรวมมากกว่า|paletteอ่านผิด/ชื่อสีไม่ตรงสัดส่วนรวม|clusterไม่เกินuniqueRGBsample,ไม่duplicate/dropmass,รวมfamilymassก่อนdominant/multicolor;มีsyntheticfixtureblue.3+.3ชนะred.4โดยไม่ปรับจากCCTVคะแนน|
|ข้อผิดพลาด|maskshape/type/bboxintersection/garmenttypoไม่ตรวจแน่นอน|emptyหรือinvalidinputกลายเป็นผลสี หรือ500จากAPI|strictuint8RGB/binarymask,finitepositiveboxclipและfloor/ceil,ตรงgrid,invalidinput400/422;no-dataแยกจากmalformed|
|ข้อผิดพลาด|APIไม่มีmetadataใหม่/installationไม่packapi;CLI/APIเรียกconfidenceคนละความหมาย|ใช้งานจากcleanpackageไม่ได้ครบและตีความผิด|packapiในbuildconfig,metadataแหล่งเดียวในpyproject,APIส่งschemaเดียวกับSDK,validatebinarymaskไม่collapseRGBlabels|
|การสื่อสาร|decompose_shadowsไม่ถูกเรียกแต่READMEอ้างphotometricinvariance/แยกเงาแล้ว|คนอ่านเชื่อว่ากู้สีวัตถุจริงจากshadow/IRได้|อธิบายแค่observedcolor;ไม่ใส่filterใหม่ที่เปลี่ยนสีเพื่อไล่คะแนน;แสดงlowchroma/dark/highlightwarningแทนคำรับรอง|
|การสื่อสาร|58Dcomputedแต่classifierไม่อ่าน;Nearestanchorถูกเรียกvan deWeijerlearnedlookup;demoแบ่งbbox45%เรียกYOLOReID|architectureและbaselineclaimผิด|อธิบายdescriptorเป็นdiagnosticunused,ไม่อ้างpaperreproduction,syntheticexampleระบุว่าไม่รันdetector/tracking/ReID|
|เอกสาร|rawM1–M5summaryฝึก800images/6328targets แต่READMEบอก500/3017;Pants399/Skirt75กลายเป็น397/77;75.17s/1265ไม่เท่ากับ3.2ms|เอาตัวเลขคนละการทดลองมาประกอบเรื่องเดียวกัน|benchmarkregisterแยกimportedpilot/assembledtable/researchsummary;เก็บJSONเดิมและhistoricalsnapshot/SHAs ไม่แก้rawrecordsย้อนหลัง|

## รายละเอียดการตรวจสี CCTV แบบ read-only

อ่าน90images/canonicalpolygons/metadataโดยใช้กฎingestionเดิม ไม่ใช้packageใหม่ปรับlabelให้ดีขึ้น ผลตรงstoredcolor: Lower101/101, Upper118/120, Hair68/69 รวม287/290 ใช้เพื่อตรวจprovenance/consistency **ไม่ใช่accuracyของclassifier** ไม่ใช่จำนวนhumanlabelsที่พิสูจน์แล้ว

Canonicalexportไม่มีexplicit-colorprovenanceรายคน จึงยังแยกไม่ได้ว่าทุกรายการมาจากannotatorหรือfallback ส่วนIR/grayscaleบอกhueของวัตถุในแสงจริงไม่ได้ ต้องมีunknown/unobservableหรือbrightness-onlytask หากจะวัดtruehueต้องมีevidenceอื่น ไม่ให้modelเดาจากgrayแล้วเรียกaccurate

[LEGACY_COLOR_PROVENANCE.json](LEGACY_COLOR_PROVENANCE.json) เก็บเฉพาะaggregate/sourcehashes; privateper-regiontraceอยู่นอกrepo ไม่มีรูป/ชื่อกล้อง/personroles/thumbnails/labelsส่วนบุคคลถูกcommit OriginalCCTVmetadata/metrics/galleryhashesตรวจว่าไม่เปลี่ยน หลังread-onlyตรวจ

## วิธีประเมินที่ควรใช้ถัดไป

1. **Oracle-mask color task:** ให้ทุกวิธีใช้RGB/regionเดียวกัน มีhuman/adjudicatedcolorGTที่มีlabelsource,visibilityและuncertaintyต่อregion แยกobservedimagecolorกับintrinsicmaterialcolor และunknownIR/lowvisibilityไว้ต่างหาก
2. **Parser-mask + color task:** reuseacceptedparseroutputsแล้วผูกownership/matchingตามprotocolใหม่ ประเมินcoverage/missedregions/falsepredictionsร่วมกับcolorquality ไม่เอาoracle-maskaccuracyอ้างเป็นend-to-endผล
3. **Method identity:** ถ้าต้องการM5จริงให้exporttrainedHGB/ExtraTrees+classorder+feature schema/calibrationartifact/trainmanifest แยกจากheuristicตัวนี้ ไม่มีปรับต้นแบบจนดีบนCCTVแล้วเรียกเป็นuntunedtest
4. **Metrics:** strictconfusion/support/accuracy/macroF1พร้อมdenominatorsและunknowncoverageเป็นหลัก top2/relaxedneighborเป็นsupplementaryเท่านั้น ไม่รวมผลคนละpopulation หรือให้ผล0กับmissingdata
5. **Runtime:** แยกCPUprofile/warmup/threadcounts/repetitionsและtimedboundariesตามงานจริง ไม่ยกtotalworkflowtimeหารจำนวนแล้วเรียกcleanlatency ไม่ใช้GPUparsingtimeมาอ้างcolorengine

รอบนี้เตรียมฐานสำหรับงานนั้น โดยยังไม่เริ่มtrainingหรือbenchmarkทั้งชุดและไม่ประกาศว่าv1.1.0แม่นขึ้นจริงจากsynthetictests

## สิ่งที่ตรวจรับ

ดู [VERIFICATION.json](VERIFICATION.json) สำหรับ environment/คำสั่งและผลจริง มีhand-specifiedregressioncasesด้านcolorfamilies,uniform/multicolor/empty/thinregions,RGB/BGR/path/bbox,invalidinputs,JSONnull,APIและcarddimensions ไม่ใช้testsที่ผ่านแทนreal-sceneaccuracy

Visualdiagnosticcardใหม่แสดงregionจริงและweightedobservedpalette,maskoutline,uncalibratedsupport,regionpixels/qualitylimit ไม่โชว์หมวดสีregistryเป็นobservedgarment ไม่อ้างว่ามีThai renderingจากOpenCVfont;ชื่อไทยยังอยู่structuredoutput

Gitworkflow:เก็บbasecommitและhistoricalsnapshots,ทำworkingbranchเพื่อreviewก่อนmerge ไม่forcepush/เปลี่ยนlicense/modelweights/env/dataset เดิม งานCIHP/CCTVparsingresultsในHuman_Parsing_Study_Publicไม่ถูกแก้ในรอบนี้

## การเปลี่ยน API ที่ผู้ใช้เดิมต้องทราบ

เวอร์ชัน1.1.0เปลี่ยนbehaviorอย่างเปิดเผย: invalidbbox/multiclassmaskถูกreject; emptyHEX/RGBเป็นnull; confidenceเดิมเปลี่ยนความหมายเป็นpixelmassและเลิก65%floor; colorfamilyaggregation/explicitimagebordererosionอาจเปลี่ยนlabel. ต้องใช้algorithm_idใหม่ในcache/metadataและอ่านCHANGELOG ไม่อ้างย้อนหลังว่าoldscoresเป็นผลเวอร์ชันนี้
