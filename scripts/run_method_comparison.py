"""Freeze, fit and evaluate seven color configurations on existing pilot regions.

Never imports the legacy experiment runner or changes segmentation/GT assets.
All data/model artifacts remain in an ignored fresh output directory.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
import warnings
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'src'), str(R / 'scripts')]
import cv2
import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from garment_color.preprocessor import MaskPreprocessor
from garment_color.research_methods import ResearchColorSuite
from run_t4_pilot import eligible, metrics, png

METHODS = ['M1', 'M2', 'M3', 'M4', 'M5', 'prototype_rules_v2', 'T4_MLP_62_64_32_12_v1']
PARENT_FILES = ['FROZEN_RUN.json', 'REGIONS.json', 'DATA_COUNTS.json', 'SUMMARY.json',
                'validation_predictions.npz', 'train_features.npz', 'calibration_features.npz',
                'validation_features.npz', 'learned_color.pt']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def freeze(parent, out):
    protocol = read(R / 'configs/COLOR_METHOD_COMPARISON.json')
    previous = read(parent / 'FROZEN_RUN.json')
    manifest = read(R / 'reports' / parent.name / 'MANIFEST.json')['artifact_files']
    identities = {}
    for name in PARENT_FILES:
        identities[name] = sha(parent / name)
        if identities[name] != manifest[name]['sha256']:
            raise ValueError('Accepted pilot artifact identity changed: ' + name)
    for name, digest in previous['source_hashes'].items():
        if sha(R / name) != digest:
            raise ValueError('Pilot source dependency changed: ' + name)
    sources = ['scripts/run_method_comparison.py', 'src/garment_color/research_methods.py',
               'tests/test_method_comparison.py', 'configs/COLOR_METHOD_COMPARISON.json',
               'scripts/run_t4_pilot.py', 'src/garment_color/preprocessor.py', 'src/garment_color/learned.py']
    out.mkdir(parents=True, exist_ok=False)
    atomic(out / 'FROZEN_RUN.json', dict(protocol=protocol, parent_run=parent.name,
           parent_location=str(parent), run_id=out.name, frozen_utc=utc(), parent_sha256=identities,
           source_sha256={s: sha(R / s) for s in sources},
           base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip(),
           memberships=previous['memberships']))
    atomic(out / 'STATE.json', dict(status='FROZEN_PENDING_EXECUTION', run_id=out.name))
    print(out / 'FROZEN_RUN.json')


def run(parent, out):
    frozen = read(out / 'FROZEN_RUN.json')
    if read(out / 'STATE.json')['status'] != 'FROZEN_PENDING_EXECUTION':
        raise ValueError('Fresh frozen attempt required; no automatic retry/overwrite')
    for rel, h in frozen['source_sha256'].items():
        if sha(R / rel) != h:
            raise ValueError('Frozen source changed: ' + rel)
    for rel, h in frozen['parent_sha256'].items():
        if sha(parent / rel) != h:
            raise ValueError('Parent binding changed: ' + rel)
    start = time.monotonic()
    atomic(out / 'EXECUTION_STARTED.json', dict(utc=utc(), pid=os.getpid(),
        launch_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip()))
    def check():
        if time.monotonic() - start > 7200:
            raise RuntimeError('TWO_HOUR_GUARD')
        if shutil.disk_usage(out).free < 50 * 2**30:
            raise RuntimeError('FREE_DISK_GUARD_50GiB')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 2**30:
            raise RuntimeError('CPU_RSS_GUARD_4GiB')
    previous = read(parent / 'FROZEN_RUN.json')
    workspace = Path(previous['workspace'])
    expected = read(parent / 'REGIONS.json')['accepted']
    def key(r):
        return (r['partition'], r['image_id'], r['person_id'], r['category'])
    if len({key(r) for r in expected}) != len(expected):
        raise ValueError('Duplicate parent region identity')
    expected_by_key = {key(r): r for r in expected}
    partitions = {p: [] for p in frozen['memberships']}
    observed = []
    feature_tables = []
    suite = ResearchColorSuite()
    pre = MaskPreprocessor()
    cv2.setNumThreads(2)
    warning_counts = Counter()
    with zipfile.ZipFile(previous['dataset_archive']['path']) as archive:
        for part, members in frozen['memberships'].items():
            for index, member in enumerate(members):
                check()
                iid, split = member['image_id'], member['split']
                rgb_path = workspace / 'datasets/CIHP' / split / 'Images' / (iid + '.jpg')
                if sha(rgb_path) != member['RGB_sha256']:
                    raise ValueError('RGB identity changed')
                rgb = np.array(Image.open(rgb_path).convert('RGB'))
                prefix = f'CCIHP_icip/dataset/{split}'
                gt = {k: png(archive, f'{prefix}/{k}/{iid}.png') for k in ['Category_ids', 'Human_ids', 'Color_ids']}
                if any(v.shape != rgb.shape[:2] for v in gt.values()):
                    raise ValueError('GT geometry changed')
                hashes = {k: hashlib.sha256(archive.read(f'{prefix}/{k}/{iid}.png')).hexdigest() for k in gt}
                for pid in sorted(set(np.unique(gt['Human_ids'])) - {0}):
                    for cat in (5, 7, 9, 12):
                        mask = (gt['Human_ids'] == pid) & (gt['Category_ids'] == cat)
                        if not mask.any():
                            continue
                        label, reason = eligible(gt['Color_ids'], mask)
                        if reason:
                            continue
                        k = (part, iid, int(pid), cat)
                        row = expected_by_key[k]
                        pixels = pre.extract_clean_pixels(rgb, mask, True)
                        if (label != row['label'] or row['GT_sha256'] != hashes or
                                int(mask.sum()) != row['original_pixels'] or len(pixels) != row['analyzed_pixels']):
                            raise ValueError('Pilot region/pixel/label binding changed')
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            f = suite.extract_kmeans_and_features(pixels, pixels)
                        warning_counts.update(str(w.message) for w in caught)
                        if f['feat_vec'].shape != (84,) or any(not np.isfinite(v).all() for v in f.values()):
                            raise ValueError('Invalid classic descriptor')
                        partitions[part].append(dict(target='UPPER' if cat in (5, 7) else 'LOWER', gt_color=label,
                                                     features=f, key=k, native_category=cat))
                        feature_tables.append(f['feat_vec'])
                        observed.append(row)
                if index % 25 == 0:
                    print('FEATURES', part, index + 1, '/', len(members), flush=True)
                atomic(out / 'STATE.json', dict(status='FEATURE_EXTRACTION', partition=part, completed_images=index+1,
                                               expected_images=len(members), completed_regions=len(observed), updated_utc=utc()))
    if [key(r) for r in observed] != [key(r) for r in expected]:
        raise ValueError('Exact ordered region membership mismatch')
    atomic(out / 'STATE.json', dict(status='FITTING_M2_M3_M4_M5', updated_utc=utc()))
    check()
    print('FIT', {p: len(v) for p, v in partitions.items()}, flush=True)
    with threadpool_limits(limits=2):
        suite.fit(partitions['train'], partitions['calibration'])
    check()
    model_path = out / 'classic_m2_m5.joblib'
    joblib.dump(suite, model_path, compress=3)
    # Local artifacts created by this process only. Never load untrusted pickle/joblib files.
    loaded = joblib.load(model_path)
    validation = partitions['validation']
    predictions = {m: [] for m in METHODS}
    with np.load(parent / 'validation_predictions.npz', allow_pickle=False) as z:
        gt = z['gt'].copy()
        predictions['prototype_rules_v2'] = z['baseline'].tolist()
        predictions['T4_MLP_62_64_32_12_v1'] = z['prediction'].tolist()
    if not np.array_equal(gt, [r['gt_color'] for r in validation]):
        raise ValueError('Reused observations are not in the same ordered population')
    for i, row in enumerate(validation):
        check()
        with threadpool_limits(limits=2):
            result = suite.predict_all(row['features'], row['target'])
            replay = loaded.predict_all(row['features'], row['target'])
        if result != replay:
            raise ValueError('Reload prediction mismatch')
        for method, prediction in result.items():
            predictions[method].append(prediction)
        if i % 100 == 0:
            print('VALIDATION', i + 1, '/', len(validation), flush=True)
    if any(len(v) != len(gt) for v in predictions.values()):
        raise ValueError('Incomplete method result')
    scores = {m: metrics(gt, predictions[m]) for m in METHODS}
    parent_summary = read(parent / 'SUMMARY.json')
    for m in METHODS[-2:]:
        if scores[m] != parent_summary['results'][m]:
            raise ValueError('Accepted pilot score reuse mismatch')
    cats = np.array([r['native_category'] for r in validation])
    per_category = {str(cat): {m: metrics(gt[cats == cat], np.asarray(predictions[m])[cats == cat]) for m in METHODS} for cat in (5,7,9,12)}
    np.savez_compressed(out / 'validation_predictions.npz', gt=gt, category=cats, **{m: np.array(p) for m,p in predictions.items()})
    np.savez_compressed(out / 'classic_descriptors.npz', X=np.array(feature_tables), region_key=np.array([':'.join(map(str, key(r))) for r in observed]))
    atomic(out / 'REGIONS.json', dict(accepted=observed))
    atomic(out / 'FIT_EVIDENCE.json', dict(model_sha256=sha(model_path), model_bytes=model_path.stat().st_size,
        heads=suite.fit_evidence, calibration=suite.calibration_log, seed=42,
        no_validation_fitting=True, no_hair_head=True, descriptor_base=84, M5_dimensions=96,
        library_warnings=dict(warning_counts), joblib_reload_predictions_identical=len(gt)))
    atomic(out / 'SUMMARY.json', dict(status='COMPLETE', run_id=out.name, protocol_id=frozen['protocol']['protocol_id'],
        methods=METHODS, populations={p:len(v) for p,v in partitions.items()}, expected_validation=len(gt),
        results=scores, per_category=per_category, missing=0, invalid=0, failed=0, new_parsing_calls=0,
        new_GPU_calls=0, reused_T4_model=True, elapsed_seconds=time.monotonic()-start,
        max_CPU_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, finished_utc=utc()))
    atomic(out / 'ENVIRONMENT.json', dict(python=sys.version, executable=sys.executable,
        packages={p: importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','opencv-python-headless','pillow','joblib','threadpoolctl']},
        threads=2, device='CPU new M1–M5 / reuse prior T4 MLP artifact',
        prediction_timing_ranking='NOT_MEASURED', environment_modified=False))
    atomic(out / 'STATE.json', dict(status='COMPLETE', validation_regions=len(gt), methods=len(METHODS), updated_utc=utc()))
    print('COMPLETE', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','run'])
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.parent, args.out)
    else:
        try:
            run(args.parent, args.out)
        except Exception:
            atomic(args.out / 'FAILURE.json', dict(utc=utc(), traceback=traceback.format_exc()))
            atomic(args.out / 'STATE.json', dict(status='STOPPED_ERROR', updated_utc=utc()))
            raise


if __name__ == '__main__':
    main()
