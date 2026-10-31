"""Read finished pilot artifacts, verify load parity, and report; never train."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from collections import Counter
import argparse,csv,hashlib,html,json,os,subprocess,sys
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from garment_color.learned import LearnedColorPredictor
from garment_color import GarmentColorExtractor
from garment_color.preprocessor import MaskPreprocessor
from garment_color.features import extract_features
from garment_color.classifier import PrototypeRuleClassifier
from run_t4_pilot import metrics

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def percent(v):return 'N/A' if v is None else f'{v*100:.2f}'

def qualitative(out,workspace,model):
 from PIL import Image
 import cv2
 parent=workspace/'CCTV_Human_Parsing_Benchmark/outputs/cctv-partial-20261031T1800Z';cfg=read(parent/'CONFIG.json')
 sample=[]
 for group in ['day','night']:
  pool=[m for m in cfg['members'] if m['condition']==group]
  sample+=sorted(pool,key=lambda m:hashlib.sha256(('20261101:'+m['RGB_sha256']).encode()).digest())[:6]
 gallery=out/'private_cctv_gallery';gallery.mkdir(exist_ok=False);observations=[];pre=MaskPreprocessor();base=GarmentColorExtractor();cards=[]
 for m in sample:
  packet=parent/'M2FP_R101_CIHP/packets'/m['id'];meta=read(packet/'packet.json');assert meta['input_sha256']==m['RGB_sha256'];assert sha(packet/'prediction.npz')==meta['packet_sha256']
  image=workspace/cfg['dataset_root']/'Images'/(m['stem']+'.jpg');assert sha(image)==m['RGB_sha256'];rgb=np.array(Image.open(image).convert('RGB'))
  with np.load(packet/'prediction.npz',allow_pickle=False) as z:planes=z['person_semantic_masks']
  rendered=[]
  for person,plane in enumerate(planes):
   for cat in (5,7,9,12):
    mask=plane==cat
    if mask.sum()<100:continue
    pixels=pre.extract_clean_pixels(rgb,mask,True);prediction=model.predict_pixels(pixels,cat)
    baseline=base.predict(rgb,mask=mask,garment_type='upper' if cat in (5,7) else 'lower')
    flags=list(baseline.warnings)
    if 'low_chroma_no_material_hue_recovery' in flags:flags.append('Hue observability unverified; no intrinsic material-color claim')
    obs=dict(image_id=m['id'],source_private_stem=m['stem'],condition=m['condition'],person_index=person,native_category=cat,
      source_packet_sha256=meta['packet_sha256'],predicted_pixels=int(mask.sum()),analyzed_pixels=len(pixels),learned=prediction,baseline=baseline.to_dict(),
      human_color_reference=None,accuracy=None,quality_flags=flags)
    observations.append(obs)
    if len(rendered)<4:
     ys,xs=np.where(mask);x0,x1=max(0,int(xs.min())-8),min(rgb.shape[1],int(xs.max())+9);y0,y1=max(0,int(ys.min())-8),min(rgb.shape[0],int(ys.max())+9)
     crop=rgb[y0:y1,x0:x1].copy();crop_mask=mask[y0:y1,x0:x1];crop[~crop_mask]=(crop[~crop_mask]*.2+235*.8).astype(np.uint8)
     scale=min(220/crop.shape[1],260/crop.shape[0]);crop=cv2.resize(crop,(max(1,int(crop.shape[1]*scale)),max(1,int(crop.shape[0]*scale))))
     name=f'{m["id"]}_{person}_{cat}.png';Image.fromarray(crop).save(gallery/name)
     rendered.append(f'<article><img src="{name}" alt="Saved parser region"><p>Category {cat} / {int(mask.sum())} px</p><p>Rule: {html.escape(baseline.color_name)} / learned ID: {prediction["color_id"]}</p><small>Qualitative prediction only; no independent color GT</small></article>')
  cards.append(f'<section><h2>{m["id"]} · {m["condition"]}</h2><div>'+(''.join(rendered) or '<p>No eligible predicted garment region; no crop or result invented.</p>')+'</div></section>')
 (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Private CCTV color pilot</title><style>body{font:16px system-ui;background:#f5f6f8;color:#203040;max-width:1100px;margin:30px auto}section{background:white;padding:20px;margin:18px 0;border-radius:12px}section>div{display:flex;gap:20px;flex-wrap:wrap}article{width:230px}img{max-width:220px}small{color:#596775}</style><h1>Private CCTV color pilot</h1><p>12 deterministic images / saved M2FP masks / no new parsing inference. Observed-color suggestions; independent human color labels unavailable. Grayscale/IR material hue is not recoverable from these predictions.</p>'+''.join(cards))
 atomic(gallery/'MANIFEST.json',dict(images=sample,observations=observations,no_GT_accuracy=True,selection='6day+6night SHA256(seed:RGBhash), no score selection'))
 return dict(images=len(sample),regions=len(observations),day_images=6,night_images=6,saved_model='M2FP_R101_CIHP',new_parsing_calls=0,
  low_chroma_region_count=sum('low_chroma_no_material_hue_recovery' in r['quality_flags'] for r in observations),accuracy=None,
  reason='Mixed/unknown color-label provenance and observability; not independent accuracy evidence',gallery='outputs/'+out.name+'/private_cctv_gallery/index.html')

def finish(out,workspace):
 import torch
 p=read(out/'FROZEN_RUN.json');summary=read(out/'SUMMARY.json');state=read(out/'STATE.json')
 assert summary['status']==state['status']=='COMPLETE'
 for rel,h in p['source_hashes'].items():assert sha(R/rel)==h,('frozen source drift',rel)
 assert sha(out/'learned_color.pt')==summary['artifact']['sha256']
 model=LearnedColorPredictor(out/'learned_color.pt',device='cpu',expected_sha256=summary['artifact']['sha256'])
 artifact=model.artifact;assert artifact['frozen_run_sha256']==sha(out/'FROZEN_RUN.json')
 with np.load(out/'validation_features.npz',allow_pickle=False) as z:x=z['X'];gt=z['y']
 with np.load(out/'validation_predictions.npz',allow_pickle=False) as z:pred=z['prediction'];baseline=z['baseline'];saved_prob=z['softmax'];assert np.array_equal(gt,z['gt'])
 with torch.inference_mode():cpu_prob=torch.softmax(model.model(torch.from_numpy((x-model.mean)/model.scale)),dim=1).numpy()
 assert np.array_equal(cpu_prob.argmax(1)+1,pred)
 assert np.allclose(cpu_prob,saved_prob,atol=1e-6,rtol=1e-5)
 assert metrics(gt,pred)==summary['results']['T4_MLP_62_64_32_12_v1']
 assert metrics(gt,baseline)==summary['results']['prototype_rules_v2']
 regions=read(out/'REGIONS.json');eligible=[r for r in regions['accepted'] if r['partition']=='validation'];assert len(eligible)==len(gt)
 # Verify the public SDK path uses exactly the same baseline preprocessing/decision.
 import zipfile
 from PIL import Image
 from run_t4_pilot import png
 with zipfile.ZipFile(p['dataset_archive']['path']) as z:
  for r in eligible[:3]:
   prefix=f'CCIHP_icip/dataset/Validation';cat=png(z,f'{prefix}/Category_ids/{r["image_id"]}.png');hum=png(z,f'{prefix}/Human_ids/{r["image_id"]}.png');rgb=np.array(Image.open(workspace/'datasets/CIHP/Validation/Images'/(r['image_id']+'.jpg')).convert('RGB'))
   mask=(cat==r['category'])&(hum==r['person_id'])
   sdk=GarmentColorExtractor().predict(rgb,mask,garment_type='upper' if r['category'] in (5,7) else 'lower');assert sdk.color_id==r['baseline_prediction']
 q=qualitative(out,workspace,model)
 record=R/'reports'/out.name;record.mkdir(exist_ok=True)
 atomic(record/'SUMMARY.json',summary);atomic(record/'DATA_COUNTS.json',read(out/'DATA_COUNTS.json'));atomic(record/'CCTV_QUALITATIVE.json',q)
 compact_members={k:[{field:v for field,v in m.items()} for m in members] for k,members in p['memberships'].items()}
 atomic(record/'MANIFEST.json',dict(run_id=out.name,protocol_id=p['protocol_id'],protocol_sha256=sha(R/'configs/T4_COLOR_PILOT_PROTOCOL.json'),
  frozen_run_sha256=sha(out/'FROZEN_RUN.json'),source_hashes=p['source_hashes'],memberships=compact_members,duplicate_exclusions=p['duplicate_exclusions'],
  dataset_archive_identity='Existing shared CCIHP_icip.zip; original README SHA '+p['CCIHP_README_sha256'],
  launch=read(out/'EXECUTION_STARTED.json'),trained_artifact=summary['artifact'],model_redistribution='Not published automatically; CCIHP annotations carry noncommercial terms; code MIT unchanged',
  external_artifact_directory='outputs/'+out.name,artifact_files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in out.iterdir() if f.is_file() and f.suffix in ['.json','.npz','.pt']}))
 rows=[]
 for method,result in summary['results'].items():
  rows.append(dict(method=method,group='ALL_GARMENTS',count=result['count'],strict_accuracy=result['strict_accuracy'],macroF1_fixed12=result['macroF1_fixed12'],macroF1_GT_supported=result['macroF1_GT_supported']))
  for cat,methods in summary['per_category'].items():
   m=methods[method];rows.append(dict(method=method,group='native_category_'+cat,count=m['count'],strict_accuracy=m['strict_accuracy'],macroF1_fixed12=m['macroF1_fixed12'],macroF1_GT_supported=m.get('macroF1_GT_supported')))
 csvout(record/'METRICS.csv',rows)
 predrows=[dict(image_id=r['image_id'],person_id=r['person_id'],category=r['category'],GT=int(gt[i]),prototype_rules_v2=int(baseline[i]),T4_MLP=int(pred[i])) for i,r in enumerate(eligible)]
 csvout(record/'VALIDATION_PREDICTIONS.csv',predrows)
 perclass=[]
 for name,result in summary['results'].items():perclass += [dict(method=name,**r) for r in result['per_class']]
 csvout(record/'PER_CLASS.csv',perclass)
 verification=dict(status='PASS',no_GPU_training_rerun=True,exact_validation_regions=len(gt),saved_exported_CPU_vs_GPU_prediction_labels='IDENTICAL',
  saved_exported_CPU_vs_GPU_softmax='allclose(atol1e-6,rtol1e-5), not claimed bitwise',metric_arithmetic='PASS',source_and_artifact_bindings='PASS',
  baseline_actual_SDK_examples=3,validation_used_for_fit_or_selection=False,
  membership_image_counts={k:len(v) for k,v in p['memberships'].items()},
  GPU_forward_accounting=dict(training_batches=int(np.ceil(summary['populations']['train']['regions']/128))*40,calibration_batches=40,validation_batches=1),
  counter_correction='STATE.GPU_model_calls was an unpopulated orchestration field; not a measured zero-forward count. Actual T4 epochs/losses/export/VRAM and forward accounting are recorded here.',
  GPU_training=summary['GPU'],CCTV=q,limitations=summary['caveats'])
 atomic(record/'VERIFICATION.json',verification)
 atomic(record/'ENVIRONMENT.json',read(out/'ENVIRONMENT.json'))
 a=summary['results']['prototype_rules_v2'];b=summary['results']['T4_MLP_62_64_32_12_v1'];gap=100*(b['strict_accuracy']-a['strict_accuracy'])
 direction='สูงกว่า' if gap>=0 else 'ต่ำกว่า'
 table='|Configuration|N regions|Strict accuracy %|Macro-F1 fixed12 %|Macro-F1 GT-supported %|\n|---|---:|---:|---:|---:|\n'
 for name,r in summary['results'].items():table+=f'|{name}|{r["count"]}|{percent(r["strict_accuracy"])}|{percent(r["macroF1_fixed12"])}|{percent(r["macroF1_GT_supported"])}|\n'
 cat_table='|Native category|N|Rule accuracy %|T4 MLP accuracy %|\n|---|---:|---:|---:|\n'
 for cat,methods in summary['per_category'].items():cat_table+=f'|{cat}|{methods["prototype_rules_v2"]["count"]}|{percent(methods["prototype_rules_v2"]["strict_accuracy"])}|{percent(methods["T4_MLP_62_64_32_12_v1"]["strict_accuracy"])}|\n'
 names=['Dark','Medium','Light','Brown','Red','Pink','Yellow','Orange','Green','Blue','Purple','Multicolor']
 ct='|Color|GT support|Rule F1 %|MLP F1 %|\n|---|---:|---:|---:|\n'
 for i,name in enumerate(names):ct+=f'|{name}|{a["per_class"][i]["GT_support"]}|{percent(a["per_class"][i]["F1"])}|{percent(b["per_class"][i]["F1"])}|\n'
 now=datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Bangkok')).isoformat()
 report=f'''# ทดลองระบบสีใหม่บน Tesla T4 — ผลนำร่องและข้อจำกัด

ตรวจปิดรอบ **{now} (เวลาไทย)** · run `{out.name}` · **PILOT_COMPLETE / NOT_FULL_BENCHMARK**

## ผลที่ทำจริง

ออกแบบและรันโมเดลสีใหม่จริงบน **{summary['GPU']['name']}** ใช้FP32/noAMP/noTF32 ไม่ได้เพียงเปลี่ยนชื่อheuristicเป็นMLP โมเดล `T4_MLP_62_64_32_12_v1` มีweights/train-onlynormalizer/classorder/featurebindingที่ส่งออกแล้ว แยกจาก `prototype_rules_v2` เดิม และไม่ใช่historicalM5ensemble

{table}
ในชุดนี้ MLP accuracy **{direction} {abs(gap):.2f} percentage points** เทียบกฎเดิม เป็นผลนำร่องบนoraclemasks ไม่ใช่หลักฐานstatisticalsignificance/ความพร้อมdeployment หรือการรับรองว่าสีวัสดุจริงถูกต้องในทุกแสง การเลือกepochใช้calibrationcrossentropyอย่างเดียว ไม่ดูValidationเพื่อปรับsettings

## ข้อมูลและวิธีทดลอง

- CCIHPTraining300imagesสำหรับfit → {summary['populations']['train']['regions']:,} eligibleperson×garmentrecords; แยกTraining50imagesสำหรับcalibration → {summary['populations']['calibration']['regions']:,} records
- CCIHPValidation100images → **{len(gt):,} regions** ใช้evaluateonce ไม่ใช่full5,000. SHA-orderseed20261101และRGBexact-dedupก่อนคะแนน ไม่มีเลือกภาพจากความแม่น
- OracleunitคือHuman_id×nativegarmentID5UpperClothes/7Coat/9Pants/12Skirt. CCIHPนิยามมีtop/bottomของdress จึงไม่เรียกเป็นnon-DressC4-P แบบparserbenchmarkเดิม ไม่ทดสอบFace/Hairรวมเป็นclothingอย่างเงียบ
- Inputใช้fullGTperson/categorymaskก่อนดูColor_ids ไม่กรองRGBตามสีที่ทราบเอง GTcolorใช้modeของcharacterizedpixels; minimumarea100px/characterized≥80%/modalconsensus≥80% เหมือนทุกconfiguration เก็บexcludedreasonsครบในprivateREGIONSและaggregatecounts
- BaselineและMLPใช้RGBเดิม/erosion5×5fallback/descriptor58เหมือนกัน รวม4nativecategoryonehotเป็น62DสำหรับMLP. mean/stdfitจากtrainเท่านั้น; network62→64→32→12,AdamWlr.001/weightdecay.001/batch128/40epochs,seed20261101 ไม่มีsearch/augmentation/VALtuning
- เลือกcheckpointepoch **{summary['selected_epoch']}** จาก50calibrationimages ไม่มีลดเกณฑ์เพราะคะแนนต่ำ traincalibrationlabelsเป็นdatasetannotations ไม่ใช่กฎRGBเดียวกับevaluatorสร้างให้ตัวเอง

[Frozen protocol](../configs/T4_COLOR_PILOT_PROTOCOL.json) / [plan](../PLAN.md) / [exact memberships/source/artifactmanifest]({out.name}/MANIFEST.json). ขอบเขตสีตามREADMEต้นฉบับCCIHP: Darkรวมnavy,Multicolorคือลายหลายสี ไม่อ้างnearestprototypeเท่ากับpublishedlearnedcolor-namingbaseline

## ผลแยก native garments

{cat_table}

## Support และ F1 รายสี

{ct}
Fixed12macroF1นับabsentclassเป็น0ตามกติกาตรึง; supportedmacroแสดงเสริม ไม่แอบเปลี่ยนclasssetให้คะแนนดีขึ้น Pixel-supportไม่ใช่garment-count;ตารางนี้นับregionrecords คะแนนstrictaccuracyไม่ใช่relaxedneighbor/top2หรือofficialcharacterizedparsingAP ไม่มีใช้คำzero-bias/zerotrainingexposure/independentfinalconfirmation

## ตรวจรับ implementation/export

- 4newCPUcasesด้านGTeligibility/handcountmetrics/deterministicmembership/artifactclassorderผ่านก่อนรัน; reusebase22testsที่dependenciesไม่เปลี่ยน ไม่มีใช้testcountแทนquality
- Source/protocol/membershipsfreezeก่อนrun; validationpredictions+GT+confusionเก็บfullprecision. RecomputeaggregateจากsavedpredictionarrayตรงcanonicalMETRICS
- Exportedmodelโหลดใหม่บนCPUแล้วlabels **ตรงกับGPUsavedvalidationทุกregion**, softmaxallcloseatol1e-6/rtol1e-5 ไม่อ้างbitwiseCPU/GPUequivalence
- ActualSDKbaselinechecked3savedoracleexamplesตรงกับbaselineที่ใช้ประเมิน ทุกsourcehash/weightsSHA/normalizer/classorderตรวจแล้ว ไม่มีruntimedefaultAPIถูกเปลี่ยนเป็นtrainedmodelเพราะpilotคะแนนอย่างเดียว
- TrainingGPUpeakallocated **{summary['GPU']['allocated_peak_MiB']:.2f} MiB**, reserved **{summary['GPU']['reserved_peak_MiB']:.2f} MiB** เป็นdiagnosticของMLPนี้ ไม่ใช่VRAM/FPSrankingของparser. ExistingT4ว่างก่อนlaunch ไม่แก้processอื่น/driver/environment ไม่ติดตั้งpackagesหรือดาวน์โหลดweightsเพิ่ม
- FieldSTATE.GPU_model_callsเดิมไม่ได้เพิ่มcounterจึงไม่ใช้เป็นหลักฐานzeroGPUcalls; [verification]({out.name}/VERIFICATION.json) แยกtraining/calibration/validationbatchcountsจากactualexecutedepochs มีactualCUDAdevice/losses/VRAM/exportเป็นหลักฐาน

## CCTV check — ไม่สร้าง accuracy จาก pseudo-label

ตรวจเชิงคุณภาพ12ภาพ (6day/6night) เลือกด้วยSHA(seed:RGBhash), reuse **savedM2FPpredictions** ไม่มีparsinginferenceใหม่ รวม{q['regions']}predictedgarmentregions ได้baseline/MLPobservedcolorและwarnings แต่ไม่มีindependenthumanGT จึง **ไม่คำนวณCCTVcoloraccuracy**

Privategalleryบนเครื่อง: `outputs/{out.name}/private_cctv_gallery/index.html` พร้อมprivateMANIFESTและcroppedpredictedmasks เก็บ{q['low_chroma_region_count']}low-chromaregionsพร้อมข้อจำกัด: grayscale/IRไม่ยืนยันhueของวัสดุจริง ไม่ให้softmaxสูงกลบปัญหาobservability. บางภาพไม่มีeligiblepredgarmentก็เก็บเหตุผล ไม่สร้างmaskขึ้นมาจากGT

กลุ่มCCTVมีselectionpartialparts/สีบางส่วนauto-generatedตามreviewก่อนหน้า จึงไม่ใช้เป็นtestlabelสำหรับฝึก/เลือกepoch/ปรับthresholdหรืออ้างdeploymentwinner รูป/ชื่อกล้อง/คน/GT/masksไม่ขึ้นpublicGit

## Artifacts และสิ่งที่ยังไม่ปิด

Weightsจริง `outputs/{out.name}/learned_color.pt` SHA256 `{summary['artifact']['sha256']}`; ขนาด {summary['artifact']['bytes']:,} bytes. Savefeaturearrays,per-regionmanifest/predictions/traininghistory,environment/launchcommitและsourcehashesอยู่runเดียวกัน ข้อมูล/weightsที่deriveจากCCIHPNCannotationsไม่เผยแพร่อัตโนมัติ ไม่ประกาศlicenseMITของcodeครอบtrainedartifactsทั้งหมด

ส่งต่อ [metrics CSV]({out.name}/METRICS.csv), [per-class CSV]({out.name}/PER_CLASS.csv), [per-region public-data predictions]({out.name}/VALIDATION_PREDICTIONS.csv), [canonical summary]({out.name}/SUMMARY.json). Clonecodeอย่างเดียวไม่reproduceโดยไม่มีdataset/archive/sharedassetsและtrainedartifactที่เตรียมตามmanifest

สิ่งที่ทำสำเร็จคือ **standardizedsmallpilot + actuallearnedmodel + export/readback + held-outscores** ไม่ใช่fullbenchmark/productionreplacement. ยังต้องreviewrareclasses/multicolorและสีที่เห็นในภาพกับคำlabelก่อนขยาย, independentCCTVcolorGT/observabilityและparsermaskend-to-endcoverageแยกงาน ไม่มีเริ่มfulltraining/parserrerun/ทดลองซ้ำเพื่อไล่คะแนนอัตโนมัติ
'''
 (R/'reports/T4_COLOR_PILOT_TH.md').write_text(report)
 atomic(out/'POSTFLIGHT.json',verification)
 status=read(R/'STUDY_STATE.json');status.update(status='T4_PILOT_COMPLETE_REQUIRES_REVIEW',current_report='reports/T4_COLOR_PILOT_TH.md',learned_color_artifact_present_locally=True,
  learned_algorithm='T4_MLP_62_64_32_12_v1',default_API_engine='prototype_rules_v2',pilot_validation_accuracy=b['strict_accuracy'],CCTV_accuracy=None,
  next_action='Review pilot evidence/rare classes before expanding or enabling learned model; no automatic full benchmark.')
 atomic(R/'STUDY_STATE.json',status)
 (R/'README.md').write_text((R/'README.md').read_text().replace('New work: bounded T4 color pilot prepared.','New work: bounded T4 color pilot COMPLETE.'))
 with (R/'WORK_LOG.md').open('a') as f:f.write('\n## '+now+' — T4 pilot closure\n\nActualT4MLP40epochs/calibrationselection/exportcomplete; train300/cal50/val100sourcebound. Baselineaccuracy'+percent(a['strict_accuracy'])+'%,MLP'+percent(b['strict_accuracy'])+'%; noVALtuning/defaultAPIchange. CPUreloadlabelsidenticaltoGPUallregions,softmaxallclose; exactsavedconfusionarithmetic/source/artifactchecksPASS. SavedM2FPCCTV12qualitativeonly,noindependentcolorGTaccuracy. No parserinference/downloads/GTmutation. STATEGPUcallcounterunuseddisclosedandderivedtrainingbatchaccountingrecorded. Nextpublishcompactevidenceonlyandhandoff,dontautomaticallyexpand.\n')
 print(json.dumps(dict(status='PILOT_COMPLETE',baseline_accuracy=a['strict_accuracy'],MLP_accuracy=b['strict_accuracy'],CCTV=q)))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--workspace',type=Path,required=True);args=a.parse_args();finish(args.output.resolve(),args.workspace.resolve())
