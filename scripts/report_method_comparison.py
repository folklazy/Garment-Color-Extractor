"""Verify saved results and render canonical tables/reports; never fit or infer."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import sys
import numpy as np

R = Path(__file__).resolve().parents[1]
COLORS = ['Dark','Medium','Light','Brown','Red','Pink','Yellow','Orange','Green','Blue','Purple','Multicolor']
NAMES = {'M1':'HSV rules','M2':'11-anchor soft projection','M3':'Global LAB prototypes',
         'M4':'Category LAB prototypes','M5':'HGB + ExtraTrees + prototype',
         'prototype_rules_v2':'Library rules v2','T4_MLP_62_64_32_12_v1':'T4 MLP 62–64–32–12'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvwrite(p, rows):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def pc(x):
    return 'N/A' if x is None else f'{100*x:.2f}'


def main(out):
    s = read(out/'SUMMARY.json')
    if s['status'] != 'COMPLETE':
        raise ValueError('Cannot render complete results from an incomplete attempt')
    frozen = read(out/'FROZEN_RUN.json')
    methods = frozen['protocol']['methods']
    if methods != s['methods'] or len(methods) != 7:
        raise ValueError('Frozen method roster mismatch')
    parent = Path(frozen['parent_location'])
    for name,h in frozen['parent_sha256'].items():
        if sha(parent/name) != h:
            raise ValueError('Parent changed: '+name)
    for name,h in frozen['source_sha256'].items():
        if sha(R/name) != h:
            raise ValueError('Frozen source changed: '+name)
    rows = [r for r in read(out/'REGIONS.json')['accepted'] if r['partition']=='validation']
    identities = [(r['image_id'],r['person_id'],r['category']) for r in rows]
    assert len(set(identities)) == len(rows) == 614
    selected_validation = {r['image_id'] for r in frozen['memberships']['validation']}
    contributing_validation = {r['image_id'] for r in rows}
    assert len(selected_validation) == 100 and contributing_validation <= selected_validation
    with np.load(out/'validation_predictions.npz', allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    assert np.array_equal(arrays['gt'], [r['label'] for r in rows])
    gt = arrays['gt']
    summary_rows, per_class, per_category, confusions, errors = [],[],[],{},[]
    for m in methods:
        pred = arrays[m]
        assert len(pred)==len(gt) and np.all((pred>=1)&(pred<=12))
        # Independent Python counts, not the evaluator's bincount implementation.
        pairs = Counter(zip(map(int,gt),map(int,pred)))
        confusion = [[pairs[(a,b)] for b in range(1,13)] for a in range(1,13)]
        metric = s['results'][m]
        assert confusion == metric['confusion']
        correct = sum(pairs[(c,c)] for c in range(1,13))
        f1s=[]
        for c in range(1,13):
            support = sum(pairs[(c,b)] for b in range(1,13))
            predicted = sum(pairs[(a,c)] for a in range(1,13))
            tp = pairs[(c,c)]
            f1 = 2*tp/(support+predicted) if support+predicted else 0
            f1s.append(f1)
            actual = metric['per_class'][c-1]
            assert (support,predicted)==(actual['GT_support'],actual['pred_support'])
            assert math.isclose(f1,actual['F1'],abs_tol=1e-15)
            per_class.append(dict(run_id=out.name,method=m,color_id=c,color_name=COLORS[c-1],GT_support=support,
                pred_support=predicted,TP=tp,FP=predicted-tp,FN=support-tp,
                precision=tp/predicted if predicted else None,recall=tp/support if support else None,F1=f1))
        assert math.isclose(correct/len(gt),metric['strict_accuracy'],abs_tol=1e-15)
        assert math.isclose(sum(f1s)/12,metric['macroF1_fixed12'],abs_tol=1e-15)
        summary_rows.append(dict(run_id=out.name,protocol_id=s['protocol_id'],method=m,technique=NAMES[m],status='COMPLETE',
            expected_regions=len(gt),completed_regions=len(pred),correct_regions=correct,failed=0,missing=0,invalid=0,
            strict_accuracy=metric['strict_accuracy'],macroF1_fixed12=metric['macroF1_fixed12'],
            macroF1_GT_supported=metric['macroF1_GT_supported'],prediction_origin='REUSED_ACCEPTED_PILOT' if m in methods[-2:] else 'NEW_METHOD_COMPARISON',
            latency_ms=None,latency_reason='NOT_MEASURED_COMPARABLY'))
        category_correct=0
        for cat in (5,7,9,12):
            take=arrays['category']==cat
            n=int(take.sum());right=int((gt[take]==pred[take]).sum());category_correct+=right
            assert s['per_category'][str(cat)][m]['count']==n
            assert math.isclose(s['per_category'][str(cat)][m]['strict_accuracy'],right/n,abs_tol=1e-15)
            per_category.append(dict(run_id=out.name,method=m,native_category=cat,
                category_name={5:'UpperClothes',7:'Coat',9:'Pants',12:'Skirt'}[cat],regions=n,correct_regions=right,strict_accuracy=right/n))
        assert category_correct==correct
        confusions[m]=confusion
        errors.extend(dict(method=m,GT_id=a,GT_name=COLORS[a-1],prediction_id=b,prediction_name=COLORS[b-1],count=count)
                      for (a,b),count in sorted(pairs.items(),key=lambda kv:(-kv[1],kv[0])) if a!=b)
    with np.load(parent/'validation_predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(z['baseline'],arrays[methods[-2]]) and np.array_equal(z['prediction'],arrays[methods[-1]])
    fit = read(out/'FIT_EVIDENCE.json')
    assert fit['model_sha256']==sha(out/'classic_m2_m5.joblib')
    assert fit['joblib_reload_predictions_identical']==614
    assert all(h['HGB_iterations']==180 and h['ExtraTrees_estimators']==250 and h['feature_dimensions']==96 for h in fit['heads'].values())
    with np.load(out/'classic_descriptors.npz', allow_pickle=False) as features:
        X = features['X']
        weights = X[:, [60,65,70,75]]
        assert X.shape == (2632,84) and np.isfinite(X).all()
        assert np.all(weights >= 0) and np.allclose(weights.sum(axis=1),1,atol=1e-12)
    public = R/'reports'/out.name
    public.mkdir(parents=True,exist_ok=True)
    csvwrite(public/'METHOD_RESULTS.csv',summary_rows)
    csvwrite(public/'PER_CLASS.csv',per_class)
    csvwrite(public/'PER_CATEGORY.csv',per_category)
    csvwrite(public/'CONFUSION_ERRORS.csv',errors)
    predictions = [dict(run_id=out.name,image_id=r['image_id'],person_id=r['person_id'],native_category=r['category'],
        GT_color=r['label'],**{m:int(arrays[m][i]) for m in methods}) for i,r in enumerate(rows)]
    csvwrite(public/'VALIDATION_PREDICTIONS.csv',predictions)
    write(public/'CONFUSIONS.json',dict(axis_labels=list(range(1,13)),rows='GT',columns='prediction',matrices=confusions))
    for name in ['SUMMARY.json','FIT_EVIDENCE.json','ENVIRONMENT.json','EXECUTION_STARTED.json']:
        write(public/name,read(out/name))
    verification = dict(status='PASS',verified_utc=datetime.now(timezone.utc).isoformat(),
        selected_validation_images=100,contributing_validation_images=len(contributing_validation),exact_unique_regions=614,methods_complete=7,
        independent_counter_confusion_arithmetic=True,per_category_reconciles_global=True,
        accepted_parent_sha256_unchanged=True,source_sha256_unchanged=True,
        rules_MLP_reused_predictions_identical=True,M5_is_fitted_ensemble_not_alias=True,
        exported_ensemble_readback_all614=True,all2632_descriptor_palette_masses_valid=True,
        targeted_CPU_tests=dict(command='PYTHONPATH=src python -m unittest discover -s tests -p test_method_comparison.py -v',cases=4,status='PASS'),
        legacy_results_and_source_not_modified=True,new_parsing_inference_calls=0,new_GPU_calls=0,
        limitations=['No independent image-label audit','No full5000 or CCTV accuracy evaluation','No clean timing/VRAM comparison'])
    write(public/'VERIFICATION.json',verification)
    all_regions = read(out/'REGIONS.json')['accepted']
    accounting = {}
    for part,members in frozen['memberships'].items():
        selected = {r['image_id'] for r in members}
        contributed = {r['image_id'] for r in all_regions if r['partition']==part}
        accounting[part] = dict(selected_images=len(selected),contributing_images=len(contributed),
            images_without_eligible_regions=sorted(selected-contributed),eligible_regions=sum(r['partition']==part for r in all_regions))
    write(public/'POPULATION_ACCOUNTING.json',accounting)
    write(public/'REPORT_GENERATION.json',dict(reporter_sha256=sha(Path(__file__)),
        generated_utc=datetime.now(timezone.utc).isoformat(),checks='saved artifacts only; no fit/inference'))
    # Data-free figures only; no dataset or CCTV imagery is copied to Git.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':out.name,'font.size':10})
    x=np.arange(len(methods));a=[s['results'][m]['strict_accuracy']*100 for m in methods];f=[s['results'][m]['macroF1_fixed12']*100 for m in methods]
    fig,ax=plt.subplots(figsize=(10,4.8));ax.bar(x-.18,a,.36,label='Strict accuracy');ax.bar(x+.18,f,.36,label='Macro F1 (fixed12)')
    ax.set_xticks(x, ['M1','M2','M3','M4','M5','Rules v2','T4 MLP']);ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.set_title('Observed garment color — 614 oracle regions / 100 selected validation images (97 contribute)');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for k,v in enumerate(a):ax.text(k-.18,v+.8,f'{v:.1f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(public/'METHOD_COMPARISON.svg',metadata={'Date':None});fig.savefig(out/'METHOD_COMPARISON.png',dpi=130);plt.close(fig)
    per=np.array([[s['results'][m]['per_class'][c]['F1'] for c in range(12)] for m in methods])
    fig,ax=plt.subplots(figsize=(12,4.2));im=ax.imshow(per,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(12),COLORS,rotation=35,ha='right');ax.set_yticks(range(7),['M1','M2','M3','M4','M5','Rules v2','T4 MLP'])
    ax.set_title('Per-color F1; validation support: '+', '.join(str(int((gt==c).sum())) for c in range(1,13)))
    for i in range(7):
        for j in range(12):ax.text(j,i,f'{per[i,j]:.2f}',ha='center',va='center',color='white' if per[i,j]>.6 else 'black',fontsize=8)
    fig.colorbar(im,ax=ax,label='F1');fig.tight_layout();fig.savefig(public/'PER_COLOR_F1.svg',metadata={'Date':None});plt.close(fig)
    for svg in public.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    artifacts=['FROZEN_RUN.json','REGIONS.json','classic_m2_m5.joblib','classic_descriptors.npz','validation_predictions.npz','SUMMARY.json','FIT_EVIDENCE.json','STATE.json','RUN.log','PROCESS.json','ENVIRONMENT.json','EXECUTION_STARTED.json']
    write(public/'MANIFEST.json',dict(run_id=out.name,protocol_id=s['protocol_id'],protocol_sha256=sha(R/'configs/COLOR_METHOD_COMPARISON.json'),
        launch=read(out/'EXECUTION_STARTED.json'),source_sha256=frozen['source_sha256'],
        parent=dict(run_id=parent.name,accepted_bindings=frozen['parent_sha256'],manifest=f'../{parent.name}/MANIFEST.json'),
        memberships=frozen['memberships'],logical_artifact_root=f'Garment-Color-Extractor/outputs/{out.name}',
        artifacts={name:dict(sha256=sha(out/name),bytes=(out/name).stat().st_size,distribution='LOCAL_ONLY') for name in artifacts},
        weights_redistribution='NOT_AUTHORIZED_FOR_AUTOMATIC_PUBLICATION',
        public_evidence={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(public.iterdir()) if p.name!='MANIFEST.json'},
        hash_meaning='Local byte identities, not creator-signed authentication or annotation correctness'))
    table='\n'.join(f"| {m} — {NAMES[m]} | {next(r['correct_regions'] for r in summary_rows if r['method']==m)}/614 | {pc(s['results'][m]['strict_accuracy'])} | {pc(s['results'][m]['macroF1_fixed12'])} |" for m in methods)
    ranked=sorted(methods,key=lambda m:s['results'][m]['strict_accuracy'],reverse=True)
    best,second=ranked[:2];bestf=max(methods,key=lambda m:s['results'][m]['macroF1_fixed12'])
    gap=(s['results'][best]['strict_accuracy']-s['results'][second]['strict_accuracy'])*100
    category_table='\n'.join('| '+m+' | '+' | '.join(pc(s['per_category'][str(c)][m]['strict_accuracy'])+f" ({s['per_category'][str(c)][m]['count']})" for c in (5,7,9,12))+' |' for m in methods)
    color_table='\n'.join('| '+COLORS[c]+' | '+str(int((gt==c+1).sum()))+' | '+' | '.join(pc(s['results'][m]['per_class'][c]['F1']) for m in methods)+' |' for c in range(12))
    thresholds='\n'.join(f"| {name} | {v['rows']} | {v['multicolor']} | {v['threshold']:.3f} |" for name,v in fit['calibration'].items())
    time_th=datetime.fromisoformat(s['finished_utc']).astimezone(ZoneInfo('Asia/Bangkok')).isoformat()
    text=f'''# ผลเปรียบเทียบเทคนิคสีเสื้อผ้า 7 วิธี

สถานะ **COMPLETE 7/7 configurations**, ตรวจครบวิธีละ **614 garment regions** จาก Validationที่เลือก100ภาพ โดย97ภาพมีregionsผ่านเกณฑ์; ปิด execution {time_th} เวลาไทย. Run `{out.name}` / protocol `{s['protocol_id']}`. เป็น exploratory subset study ไม่ใช่ CIHP5000ภาพ, parsing benchmark หรือ CCTV accuracy.

คะแนน strict accuracy สูงสุดคือ **{best} ({NAMES[best]}) {pc(s['results'][best]['strict_accuracy'])}%**; สูงกว่า {second} **{gap:.2f} percentage points**. ตัวนำ fixed12 macroF1 คือ **{bestf} {pc(s['results'][bestf]['macroF1_fixed12'])}%**. ตัวเลขนี้เป็นผลที่สังเกตบนชุดนี้ ไม่ใช่ statistical significance หรือ final deployment acceptance.

## วัตถุประสงค์และสิ่งที่ทดสอบจริง

เลือกวิธีจำแนกสีที่สังเกตได้ภายใน mask เสื้อผ้า โดยให้ข้อมูล/GT/eligibility/erosion/metric เดียวกัน เปรียบเทียบ M1 HSV, M2 fixed-anchor projection, M3 global LAB prototypes, M4 category LAB prototypes, **M5 ensemble HGB+ExtraTrees+prototype ที่ fit จริง**, rules v2 และ MLP ที่ฝึกบนT4แล้ว. ไม่ใช้ M4/M5 aliasจากสคริปต์ CCTVเก่า ไม่เอาคะแนนประวัติคนละเงื่อนไขมารวม. M2ไม่ได้ใช้ learned color-name LUT ของงานตีพิมพ์.

อ่าน [protocol](EXPERIMENT_PROTOCOL.md) สำหรับสูตร/พารามิเตอร์และ [lineage](manifests/METHOD_LINEAGE.json) สำหรับต้นทาง/การดัดแปลง. นี่คือ comparison ของเทคนิคพร้อม configuration; feature/category information ต่างกัน จึงแยกผล architecture หรือองค์ประกอบทีละชิ้นไม่ได้.

## ข้อมูลและเงื่อนไข

CCIHP Color_ids1–12 เป็น reference จาก annotationเดิม ไม่สร้าง GTจากกฎสีของตัวประเมินเอง. Training300ภาพ/1,648regions สำหรับfit, Trainingอีก50ภาพ/370regions สำหรับcalibration และ Validation100ภาพ/614regions สำหรับทดสอบ. ภาพที่ให้regionsผ่านเกณฑ์จริงมี294/49/97ภาพตามลำดับ ส่วน6/1/3ภาพไม่มีeligiblegarmentregions แต่ยังคงในselectedimageaccounting ไม่ตัดตามprediction. [Population accounting](reports/{out.name}/POPULATION_ACCOUNTING.json). Membershipตรึงตามpilotเดิมและ RGBhashข้ามpartitionไม่ซ้ำ. กลุ่ม native5 UpperClothes,7Coat,9Pants,12Skirt; ขอบเขตอาจรวมส่วนบน/ล่างของdress ไม่ใช่ non-Dress C4-P. หน่วยคือ person×nativecategory ไม่ใช่หนึ่งภาพหรือหนึ่งคน.

Mask≥100pixels, characterized≥80%, modalcolor≥80%ของcharacterized. กรองก่อน predictions; inputใช้maskเต็ม ไม่ตัดpixelตามColorGT. Exclusionเดิมรวม334regionsจาก450ภาพ: เล็ก86/characterizedไม่พอ248. ทุกวิธีใช้ RGBและerosionเดียวกัน. ไม่ประเมินHair/Face colorในชุดนี้.

M2–M5 fitบนTrainingเท่านั้น; thresholdเลือกบนcalibrationแยก ไม่ใช้Validation. MLP reuse checkpoint epoch28และ predictions ที่ตรวจรับแล้ว ไม่ฝึก/เลือกใหม่. Rules reuse ผลเดิมที่ผูก region orderตรงกัน. Data-dependentfittingต่างกันตามเทคนิค โดยไม่มีsearchหลังเห็นคะแนนรอบนี้.

## ผลรวม — frozen method order

| วิธี | ถูก/ทั้งหมด | Strict accuracy % | Macro F1 fixed12 % |
|---|---:|---:|---:|
{table}

Canonical [METHOD_RESULTS.csv](reports/{out.name}/METHOD_RESULTS.csv) เก็บfullprecision/completeness/origin; [predictions](reports/{out.name}/VALIDATION_PREDICTIONS.csv) และ [confusions](reports/{out.name}/CONFUSIONS.json) ตรวจเลขย้อนกลับได้. ไม่มีmissing/failed/invalid. ค่า N/Aในruntimeหมายถึงยังไม่ได้วัด ไม่ใช่0.

![Method comparison](reports/{out.name}/METHOD_COMPARISON.svg)

## ราย garment category — strict accuracy % (จำนวนregions)

| วิธี | UpperClothes5 | Coat7 | Pants9 | Skirt12 |
|---|---:|---:|---:|---:|
{category_table}

ตารางนี้ไม่ใช่คะแนนtarget upper/lower parsing; เป็นสีของoracle regionตามnativegarment. ค่ารวมถ่วงตามจำนวนregions ไม่ใช่เฉลี่ยสี่categoryเท่ากัน. รายละเอียด [PER_CATEGORY.csv](reports/{out.name}/PER_CATEGORY.csv).

## รายสี — F1 %

| Color | GT support | M1 | M2 | M3 | M4 | M5 | Rules v2 | T4 MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{color_table}

![Per-color F1](reports/{out.name}/PER_COLOR_F1.svg)

Darkรวมnavyตามdataset; Mediumคือgray/Lightคือwhiteตามreference. Multicolorคือlabelของregion ไม่ใช่paletteหลายclusterโดยอัตโนมัติ. ValidationOrangeมีเพียง3/Purple2regions จึงยังสรุปความทนทานสีหายากไม่ได้. Accuracyให้น้ำหนักตามจำนวนregions; fixed12macroF1ให้น้ำหนักแต่ละสีเท่ากัน ไม่สร้างweightedscoreใหม่. [PER_CLASS.csv](reports/{out.name}/PER_CLASS.csv) มีTP/FP/FN/P/R/F1และsupport; [CONFUSION_ERRORS.csv](reports/{out.name}/CONFUSION_ERRORS.csv) แสดงคู่ผิดทั้งหมดตามจำนวน.

## การตรวจและหลักฐาน execution

- Four independent CPU acceptance cases: HSVsingle/multicolor/empty, M5 class-column alignmentและhand-calculatedblend, calibrationtie, descriptor84D/finite/palette mass. ไม่ใช้จำนวนtestsแทนsemanticaccuracy.
- ตรวจ RGB/GTshaและfull-mask/erosion/label/pixelcountsขณะอ่าน450ภาพเท่าที่ใช้; exactregionsตรงparentทุกpartition. ไม่audit/hasharchiveทั้งหมดใหม่.
- M5มี Upper/Lower headsอย่างละHGB180iterations/ET250trees, feature96D. Export `classic_m2_m5.joblib` **{fit['model_bytes']:,}bytes**, SHA256 `{fit['model_sha256']}`. โหลดกลับแล้วpredictionsM1–M5ตรงกันทั้งหมด614regions. ชื่อนี้เป็น configurationของprotocolใหม่ ไม่อ้างว่าweightเดียวกับงานวิจัยเก่า.
- IndependentPythonCounterยืนยันconfusion, correct/N, macroF1, per-categoryรวมกลับglobal และreuseสองวิธีตรงparentทุกprediction. Frozen source/parent hashesตรง; nativeparserresults/GT/legacycoloroutputsไม่เปลี่ยน.
- CPU threads2; เวลาทั้งกระบวนการ {s['elapsed_seconds']:.1f}s/peakRSS {s['max_CPU_RSS_MiB']:.1f}MiB เป็นoperationaldiagnostics ไม่ใช่latencyของวิธีใด. M1–M5ใช้CPU; MLPใช้ผลartifactที่ฝึกT4แล้ว ไม่ทำGPUcallsใหม่หรือฝึกซ้ำ.
- [VERIFICATION.json](reports/{out.name}/VERIFICATION.json), [FIT_EVIDENCE.json](reports/{out.name}/FIT_EVIDENCE.json), [ENVIRONMENT.json](reports/{out.name}/ENVIRONMENT.json), [MANIFEST.json](reports/{out.name}/MANIFEST.json) ผูกsource/launch/parent/model/ผล. Hashระบุbytesในเครื่อง ไม่ได้ยืนยันauthenticityหรือannotationcorrectnessจากผู้สร้าง.

Calibration thresholds (ตรึงสูตรก่อนfit ไม่เปลี่ยนตามValidation):

| Head | Calibration regions | Multicolor regions | Threshold |
|---|---:|---:|---:|
{thresholds}

Clustering warnings ที่บันทึก: `{fit['library_warnings']}`. Finite/descriptor/palette checksผ่าน; ไม่ซ่อนwarningทั่วprocess. [Schema](DATA_SCHEMA.md) อธิบายdenominators/units/null.

## ข้อสรุปและสิ่งที่ยังกล่าวไม่ได้

MLP ถูกมากกว่า M4 เพียง7regionsจาก614 หรือ1.14percentagepoints แต่ macroF1 ของ M4 สูงกว่า MLP5.52percentagepoints. จึงเสนอให้เก็บทั้งสองเป็นshortlist: MLPเมื่อให้ความสำคัญกับจำนวนregionที่ถูกโดยรวม และM4เมื่อให้ความสำคัญกับความสมดุลรายสี. ไม่ประกาศว่าความต่างมีนัยสำคัญ.

M5ไม่ได้ชนะโดยรวม แม้มีความซับซ้อนกว่า แต่ทำCoatได้76.98%สูงสุดในชุดนี้; M4นำPants74.86%; MLPนำUpperClothes63.28%และSkirt72.22%. เป็นผลเชิงพรรณนา ไม่ได้พิสูจน์สาเหตุของความต่าง. M2มีmacroF1ต่ำและบางสีไม่มีTP; ไม่เปลี่ยนthresholdเพื่อช่วยผลValidation.

ใช้ {best}เป็น **candidateที่นำด้านstrictaccuracyบนชุดนี้** และพิจารณา {bestf}เมื่อสนใจความสมดุลรายสี; ดูผลรายcategory/rareclassesประกอบ. ไม่เปลี่ยนdefaultAPIengineโดยอัตโนมัติ. ถ้าทั้งสองเป้าหมายชี้คนละตัวต้องรายงานtrade-off ไม่บังคับรวมคะแนนเพื่อให้มีผู้ชนะเดียว.

ผลยังพึ่งoraclemaskและknownnativecategory ซึ่งใช้งานจริงอาจผิดจากparser. ความต่างอาจมาจากfeatures/categoryconditioning/calibration/prior/algorithmร่วมกัน ไม่พิสูจน์เหตุเชิงarchitecture. ชุดValidationนี้เคยเปิดดูผลrules/MLPแล้ว จึงไม่ใช่ independentfinalconfirmation; ไม่มีuncertaintytest และหลายregionจากภาพเดียวกันมีความสัมพันธ์. ไม่อ้างว่าโมเดลชนะบนCCTV/nightIR หรือคืนสีวัสดุที่แท้จริงจากแสงได้.

ไม่ได้จัดอันดับlatency/FPS/VRAM เพราะไม่มีcleanmeasurementเดียวกันทุกวิธี. ไม่เอาเวลาทั้งสคริปต์หรือT4trainingpeakไปเป็นinferencecost. CCTVgallery12ภาพจากpilotเดิมเป็นqualitative-only ไม่มีindependentcolorGTaccuracy และไม่ขยายประเมินใหม่ในรอบนี้. ภาพ/ชื่อไฟล์/labelsCCTVไม่ขึ้นpublicGit.

## โครงรายงานแบบ YOLO และการทำต่อ

ตรวจอ่านต้นแบบจริงตาม [protocol](EXPERIMENT_PROTOCOL.md#yolo-format-reused): ใช้fixedorder/protocol/canonicaltables/schema/provenance/ข้อจำกัด/Thaihandoff. แทนMOTSsequence/APด้วยcolor/category/regionclassification; ไม่คัดลอกFPSrankingหรือแก้YOLOrepo.

Artifactsภายนอก: `<Garment-Color-Extractor>/outputs/{out.name}/` เก็บjoblib,descriptors,predictions,regions,log/process/runmanifest; parentmodelอยู่ `outputs/t4-pilot-20261101/`. ไม่แจกdataset/weightsอัตโนมัติ. [Setup/reproduction](docs/METHOD_COMPARISON_SETUP.md) ระบุinput/env/คำสั่ง; cloneอย่างเดียวไม่พอ. Publictablesใช้ตรวจarithmeticได้โดยไม่ต้องมีภาพ.

ขั้นต่อไปที่มีประโยชน์คือทบทวนความต้องการdeploymentและreferenceCCTVcolorที่เชื่อถือได้ แล้วอนุมัติการประเมินปลายทางแยก; รอบนี้หยุดหลังส่งมอบ ไม่เปิดfullbenchmarkใหม่หรือparsinginference. [Thai summary](RESULTS_SUMMARY_TH.md) / [presentation](PRESENTATION_SUMMARY_TH.md).
'''
    (R/'REPORT.md').write_text(text)
    (R/'RESULTS_SUMMARY_TH.md').write_text(f'''# สรุปผลสีเสื้อผ้า 7 วิธี\n\nทดสอบครบ7/7วิธี บน614oracle garment regionsจาก100Validationภาพที่เลือก (97ภาพให้regionsผ่านเกณฑ์) โดยใช้Training300ภาพและcalibration50ภาพชุดเดียวกัน. [รายงานหลัก](REPORT.md) / [กติกา](EXPERIMENT_PROTOCOL.md).\n\n| วิธี | ถูก/ทั้งหมด | Strict accuracy % | Macro F1 fixed12 % |\n|---|---:|---:|---:|\n{table}\n\nStrictaccuracyนำ: **{best} {pc(s['results'][best]['strict_accuracy'])}%** สูงกว่า{second} {gap:.2f}percentagepoints. MacroF1นำ: **{bestf} {pc(s['results'][bestf]['macroF1_fixed12'])}%**. MLPมีaccuracyสูงกว่าM4 7regionsแต่macroF1ต่ำกว่า5.52percentagepoints จึงเสนอshortlistทั้งสองตามวัตถุประสงค์. M5คือensembleที่fitจริง ไม่ใช่aliasM3. Rules/MLP reuseหลักฐานเดิมโดยไม่ฝึกซ้ำ.\n\nเป็นexploratorysubsetcolorclassification ไม่ใช่CIHP5000/parsingaccuracy/CCTVaccuracy; Orange3/Purple2regions. ไม่มีspeedrankingหรือdeploymentacceptance. ไม่มีการเปลี่ยนdefaultAPIอัตโนมัติ. Canonicaltablesและsource/artifacthashesอยู่ [run evidence](reports/{out.name}/MANIFEST.json).\n''')
    (R/'PRESENTATION_SUMMARY_TH.md').write_text(f'''# สรุปสำหรับนำเสนอ — เปรียบเทียบวิธีสีเสื้อผ้า\n\n1. **โจทย์:** เมื่อmaskเสื้อผ้าถูกต้องแล้ว วิธีใดจำแนกสีที่เห็นในภาพได้ดีกว่า ใช้CCIHPannotationเป็นreference ไม่ใช้กฎของเราเองสร้างคำตอบ.\n2. **วิธี:** M1HSV/M2anchor/M3LABglobal/M4LABcategory/M5HGB+ExtraTrees+prototype เทียบrulesและT4MLP; ข้อมูลและmaskตรงกัน7วิธี. ฝึก300ภาพ/calibration50ภาพ/validation100ภาพที่เลือก/97ภาพมีeligiblegarments รวม614regions.\n3. **ผล:** {best}นำstrictaccuracy {pc(s['results'][best]['strict_accuracy'])}% และ {bestf}นำmacroF1 {pc(s['results'][bestf]['macroF1_fixed12'])}%. M5เป็นmodelที่train/export/readbackจริง ไม่ใช้ผลM3แทน. ดู [ตาราง/กราฟหลัก](REPORT.md#ผลรวม--frozen-method-order).\n4. **ความน่าเชื่อถือ:** ตรึงสมาชิก/สูตรก่อนfitรอบใหม่; ไม่tuneจากValidation; ตรวจexactmembership/GT/sourcehashes/metrics/reload. ไม่ลบตัวอย่างที่ทายผิด.\n5. **ขอบเขต:** เป็นผลย่อยบนoraclemask ไม่ยืนยันCCTV/nightIR/end-to-endparsingหรือสีวัสดุจริง. ข้อมูลหายากน้อย ไม่มีstatisticalsignificance/speedranking. ตัวนำเป็นcandidateสำหรับงานต่อ ไม่ใช่finaldeploymentacceptance.\n\nใช้กราฟที่ไม่มีภาพdatasetได้จาก [method comparison](reports/{out.name}/METHOD_COMPARISON.svg) และ [per-color F1](reports/{out.name}/PER_COLOR_F1.svg). ตัวเลขเต็มและข้อจำกัดอยู่ [REPORT.md](REPORT.md); ไม่อัปโหลดCCTVassets.\n''')
    print(json.dumps(dict(report=str(R/'REPORT.md'),strict_leader=best,macroF1_leader=bestf,accuracy=s['results'][best]['strict_accuracy'],methods=summary_rows),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
