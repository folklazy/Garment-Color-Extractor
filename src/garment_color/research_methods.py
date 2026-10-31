"""Clothing-only M1–M5 comparison, not the default library engine.

Adapted from the first-party Clothing_Color_Extraction_Study research script.
Its source identity and deliberate deviations are in manifests/METHOD_LINEAGE.json.
M2 is a fixed-anchor approximation, not the published learned color-name LUT.
Hair is outside this comparison; no synthetic hair samples/head are fitted.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
import cv2
from scipy.cluster.vq import kmeans2
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


def blend_m5(hgb_prob, hgb_classes, et_prob, et_classes, prototype11):
    """Align actual classifier columns, then apply the frozen .60/.40/.15 blend."""
    p = np.zeros(13, dtype=np.float64)
    for weight, values, classes in ((.60, hgb_prob, hgb_classes), (.40, et_prob, et_classes)):
        values = np.asarray(values, float)
        classes = np.asarray(classes, int)
        if len(set(classes)) != len(classes) or np.any((classes < 1) | (classes > 12)):
            raise ValueError('Invalid trained class columns')
        if values.shape != classes.shape or not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1):
            raise ValueError('Invalid classifier probabilities')
        p[classes] += weight * values
    prototype11 = np.asarray(prototype11, float)
    if prototype11.shape != (11,) or not np.isfinite(prototype11).all() or np.any(prototype11 < 0) or not np.isclose(prototype11.sum(), 1):
        raise ValueError('Invalid prototype distribution')
    p[1:12] = .85 * p[1:12] + .15 * prototype11 * (1 - p[12])
    if not np.isclose(p.sum(), 1):
        raise ValueError('Ensemble mass is not conserved')
    return p

BCN11_SRGB = {'black': (20, 20, 20), 'grey': (128, 128, 128), 'white': (235, 235, 235), 'brown': (120, 72, 42), 'red': (195, 35, 35), 'pink': (230, 135, 165), 'yellow': (235, 205, 45), 'orange': (230, 115, 35), 'green': (45, 145, 65), 'blue': (45, 95, 185), 'purple': (125, 55, 155)}

BCN11_TO_CCIHP = {'black': 1, 'grey': 2, 'white': 3, 'brown': 4, 'red': 5, 'pink': 6, 'yellow': 7, 'orange': 8, 'green': 9, 'blue': 10, 'purple': 11}

def srgb_to_cielab(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert (N, 3) uint8 RGB to standard CIELAB: L* in [0, 100], a*, b* in [-128, 127]."""
    if len(rgb_uint8) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(rgb_uint8, dtype=np.uint8).reshape(-1, 1, 3)
    lab_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = lab_cv[:, 0] * (100.0 / 255.0)
    a = lab_cv[:, 1] - 128.0
    b = lab_cv[:, 2] - 128.0
    return np.stack([L, a, b], axis=1)

def srgb_to_hsv(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert (N, 3) uint8 RGB to HSV: H in [0, 360), S in [0, 1], V in [0, 1]."""
    if len(rgb_uint8) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(rgb_uint8, dtype=np.uint8).reshape(-1, 1, 3)
    hsv_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    H = hsv_cv[:, 0] * 2.0
    S = hsv_cv[:, 1] / 255.0
    V = hsv_cv[:, 2] / 255.0
    return np.stack([H, S, V], axis=1)

def predict_m1_garment(hsv_pixels: np.ndarray) -> dict:
    """M1: Naive pixel-wise HSV rule lookup + majority vote + secondary ratio for Multicolor."""
    if len(hsv_pixels) == 0:
        raise ValueError('Empty color region')
    counts = np.zeros(13, dtype=np.int64)
    H, S, V = (hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2])
    c = np.zeros(len(H), dtype=np.int64)
    achro = (V < 0.25) | (S < 0.16) | (V < 0.35) & (S < 0.24)
    c[achro & (V < 0.38)] = 1
    c[achro & (V >= 0.38) & (V < 0.68)] = 2
    c[achro & (V >= 0.68)] = 3
    rem = ~achro
    m_red = rem & ((H < 14.0) | (H >= 342.0))
    c[m_red & (V > 0.72) & (S < 0.55)] = 6
    c[m_red & (V < 0.42)] = 4
    c[m_red & (c == 0)] = 5
    m_org = rem & (H >= 14.0) & (H < 40.0)
    c[m_org & ((V < 0.58) | (S < 0.55) & (V < 0.7))] = 4
    c[m_org & (c == 0)] = 8
    m_yel = rem & (H >= 40.0) & (H < 70.0)
    c[m_yel & (V < 0.52)] = 4
    c[m_yel & (c == 0)] = 7
    m_grn = rem & (H >= 70.0) & (H < 168.0)
    c[m_grn] = 9
    m_blu = rem & (H >= 168.0) & (H < 258.0)
    c[m_blu & (V < 0.32)] = 1
    c[m_blu & (c == 0)] = 10
    m_pur = rem & (H >= 258.0) & (H < 308.0)
    c[m_pur] = 11
    m_pnk = rem & (H >= 308.0) & (H < 342.0)
    c[m_pnk & (V > 0.55)] = 6
    c[m_pnk & (c == 0)] = 11
    bc = np.bincount(c, minlength=13)[1:12]
    probs = bc / np.maximum(bc.sum(), 1)
    order = np.argsort(probs)[::-1]
    single = int(order[0] + 1)
    second = int(order[1] + 1)
    is_achro_pair = {single, second} in ({1, 2}, {2, 3})
    pred = 12 if probs[order[0]] < 0.5 and probs[order[1]] >= 0.22 and (not is_achro_pair) else single
    return {'pred': int(pred), 'single': single, 'top2': [single, second], 'probs11': probs}

class ResearchColorSuite:

    def __init__(self):
        self.calibration_log = {}
        self.bcn_names = list(BCN11_SRGB.keys())
        bcn_rgb = np.array([BCN11_SRGB[k] for k in self.bcn_names], dtype=np.uint8)
        self.bcn_lab = srgb_to_cielab(bcn_rgb)
        self.bcn_to_ccihp = np.zeros((11, 12), dtype=np.float64)
        self.tau_multi_m2 = 0.72
        self.cluster_means: dict[int, np.ndarray] = {}
        self.cluster_inv_covs: dict[int, np.ndarray] = {}
        self.cluster_log_dets: dict[int, float] = {}
        self.cluster_priors: dict[int, float] = {}
        self.navy_L_cutoff = 30.0
        self.tau_multi_m3 = 0.27
        self.delta_e_multi = 22.0
        self.cat_priors: dict[str, dict[int, float]] = {}
        self.cat_means: dict[str, dict[int, np.ndarray]] = {}
        self.cat_inv_covs: dict[str, dict[int, np.ndarray]] = {}
        self.cat_log_dets: dict[str, dict[int, float]] = {}
        self.cat_tau_multi: dict[str, float] = {'UPPER': 0.25, 'LOWER': 0.32}
        self.ml_models: dict[str, tuple[HistGradientBoostingClassifier, ExtraTreesClassifier, list[int]]] = {}

    def _bcn11_hist(self, lab_pixels: np.ndarray) -> np.ndarray:
        diff = lab_pixels[:, None, :] - self.bcn_lab[None, :, :]
        dist2 = 0.65 * diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2
        probs = np.exp(-dist2 / (2.0 * 22.0 ** 2))
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return probs.mean(axis=0)

    def _score_lab_m3(self, lab: np.ndarray, cat_group: str | None=None) -> np.ndarray:
        L, a, b = (float(lab[0]), float(lab[1]), float(lab[2]))
        C = float(np.hypot(a, b))
        x = np.array([L, a, b, C], dtype=np.float64)
        log_scores = np.full(12, -1000000000.0, dtype=np.float64)
        means = self.cat_means.get(cat_group, self.cluster_means) if cat_group else self.cluster_means
        inv_covs = self.cat_inv_covs.get(cat_group, self.cluster_inv_covs) if cat_group else self.cluster_inv_covs
        log_dets = self.cat_log_dets.get(cat_group, self.cluster_log_dets) if cat_group else self.cluster_log_dets
        priors = self.cat_priors.get(cat_group, self.cluster_priors) if cat_group else self.cluster_priors
        for c in range(1, 12):
            diff = x - means[c]
            maha = float(diff @ inv_covs[c] @ diff)
            log_scores[c] = -0.5 * (maha + log_dets[c]) + np.log(max(priors[c], 1e-08))
        if C < 9.5:
            for c in range(4, 12):
                log_scores[c] -= 3.5
        elif C >= 20.0:
            log_scores[2] -= 2.5
            log_scores[3] -= 2.5
            if L >= self.navy_L_cutoff + 4.0:
                log_scores[1] -= 1.8
        if b < -10.0 and a < 15.0:
            if L < self.navy_L_cutoff - 2.0:
                log_scores[1] += 1.2
            elif L > self.navy_L_cutoff + 4.0 and C >= 15.0:
                log_scores[10] += 1.2
        m = log_scores[1:12].max()
        probs = np.zeros(12, dtype=np.float64)
        probs[1:12] = np.exp(log_scores[1:12] - m)
        probs[1:12] /= np.maximum(probs[1:12].sum(), 1e-12)
        return probs

    def extract_kmeans_and_features(self, rgb_eroded: np.ndarray, rgb_raw: np.ndarray) -> dict:
        """84D LAB/HSV/anchor/cluster descriptor, fitted on common eroded pixels."""
        if len(rgb_eroded) == 0:
            raise ValueError('Empty color region')
        lab = srgb_to_cielab(rgb_eroded)
        hsv = srgb_to_hsv(rgb_eroded)
        lab_raw = srgb_to_cielab(rgb_raw)
        hsv_raw = srgb_to_hsv(rgb_raw)
        n = len(lab)
        k_clusters = 4
        if n < k_clusters * 4:
            med = np.median(lab, axis=0) if n else np.zeros(3, dtype=np.float32)
            centers = np.tile(med.astype(np.float64), (k_clusters, 1))
            weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            order = np.argsort(lab[:, 0])
            init_idx = [order[int((i + 0.5) * n / k_clusters)] for i in range(k_clusters)]
            init_centers = lab[init_idx].astype(np.float64)
            centers, labels = kmeans2(lab.astype(np.float64), init_centers, iter=12, minit='matrix')
            counts = np.bincount(labels, minlength=k_clusters)
            weights = counts.astype(np.float64) / float(n)
            sort_idx = np.argsort(weights)[::-1]
            centers = centers[sort_idx]
            weights = weights[sort_idx]
        L, a, b = (lab[:, 0], lab[:, 1], lab[:, 2])
        C = np.hypot(a, b)
        H, S, V = (hsv[:, 0], hsv[:, 1], hsv[:, 2])
        q_L = np.percentile(L, [10, 25, 50, 75, 90])
        q_a = np.percentile(a, [10, 25, 50, 75, 90])
        q_b = np.percentile(b, [10, 25, 50, 75, 90])
        q_C = np.percentile(C, [10, 25, 50, 75, 90])
        q_S = np.percentile(S, [25, 50, 75, 90])
        q_V = np.percentile(V, [10, 25, 50, 75, 90])
        std_stats = np.array([np.mean(L), np.std(L), np.mean(a), np.std(a), np.mean(b), np.std(b), np.mean(C), np.std(C), np.mean(S), np.std(S), np.mean(V), np.std(V), float(np.mean(C < 9.5)), float(np.mean(C >= 20.0)), float(np.mean(L < 28.0)), float(np.mean(L > 72.0))], dtype=np.float64)
        bcn_hist = self._bcn11_hist(lab)
        cl_feats = []
        for k in range(4):
            cL, ca, cb = centers[k]
            cC = float(np.hypot(ca, cb))
            cl_feats.extend([cL, ca, cb, cC, weights[k]])
        cl_feats = np.array(cl_feats, dtype=np.float64)
        d_ab12 = float(np.hypot(centers[0, 1] - centers[1, 1], centers[0, 2] - centers[1, 2]))
        d_L12 = abs(float(centers[0, 0] - centers[1, 0]))
        d_E12 = float(np.linalg.norm(centers[0] - centers[1]))
        d_ab13 = float(np.hypot(centers[0, 1] - centers[2, 1], centers[0, 2] - centers[2, 2]))
        d_E13 = float(np.linalg.norm(centers[0] - centers[2]))
        pair_feats = np.array([d_ab12, d_L12, d_E12, d_ab13, d_E13, weights[1] * d_ab12, weights[1] * d_E12, (weights[1] + weights[2]) * max(d_ab12, d_ab13)], dtype=np.float64)
        feat_vec = np.concatenate([q_L, q_a, q_b, q_C, q_S, q_V, std_stats, bcn_hist, cl_feats, pair_feats])
        return {'lab': lab, 'lab_raw': lab_raw, 'hsv_raw': hsv_raw, 'bcn_hist': bcn_hist, 'centers': centers, 'weights': weights, 'feat_vec': feat_vec}

    def _eval_clusters_m3(self, feat_dict: dict, cat_group: str | None=None) -> dict:
        centers = feat_dict['centers']
        weights = feat_dict['weights']
        raw_clusters = []
        for k in range(len(weights)):
            w = float(weights[k])
            if w <= 1e-06:
                continue
            c_lab = centers[k]
            probs = self._score_lab_m3(c_lab, cat_group=cat_group)
            c_id = int(np.argmax(probs[1:12]) + 1)
            raw_clusters.append({'lab': c_lab, 'weight': w, 'probs': probs, 'ccihp_class': c_id})
        dom = raw_clusters[0]
        class_weights: dict[int, float] = defaultdict(float)
        soft_probs = np.zeros(12, dtype=np.float64)
        for cl in raw_clusters:
            d_ab = float(np.hypot(cl['lab'][1] - dom['lab'][1], cl['lab'][2] - dom['lab'][2]))
            d_L = abs(float(cl['lab'][0] - dom['lab'][0]))
            is_shadow = d_ab < 9.5 and d_L < 28.0
            target_c = dom['ccihp_class'] if is_shadow else cl['ccihp_class']
            class_weights[target_c] += cl['weight']
            soft_probs += cl['weight'] * cl['probs']
        sorted_cw = sorted(class_weights.items(), key=lambda kv: kv[1], reverse=True)
        single_pred = int(sorted_cw[0][0])
        top2 = [int(kv[0]) for kv in sorted_cw[:2]]
        if len(top2) < 2:
            second_best = int(np.argsort(soft_probs[1:12])[::-1][1] + 1)
            top2.append(second_best)
        eff_sec = 0.0
        for cl in raw_clusters[1:]:
            if cl['ccihp_class'] != dom['ccihp_class']:
                d_ab = float(np.hypot(cl['lab'][1] - dom['lab'][1], cl['lab'][2] - dom['lab'][2]))
                d_E = float(np.linalg.norm(cl['lab'] - dom['lab']))
                if d_ab >= 12.0 or d_E >= self.delta_e_multi:
                    eff_sec += cl['weight']
        tau = self.cat_tau_multi.get(cat_group, self.tau_multi_m3) if cat_group else self.tau_multi_m3
        pred_c = 12 if eff_sec >= tau else single_pred
        return {'pred': int(pred_c), 'single': int(single_pred), 'top2': top2, 'eff_sec': float(eff_sec), 'soft_probs11': soft_probs[1:12], 'raw_clusters': raw_clusters}

    def fit(self, train_items, calibration_items):
        """Fit prototypes/estimators on train, thresholds only on calibration."""
        if not train_items or not calibration_items:
            raise ValueError('Explicit train and calibration populations required')
        for row in train_items + calibration_items:
            if row['target'] not in ('UPPER', 'LOWER') or row['gt_color'] not in range(1, 13):
                raise ValueError('This study supports clothing color IDs 1..12 only')
        grouped = defaultdict(list)
        counts = np.full((11, 12), .5, dtype=np.float64)
        for row in train_items:
            c = row['gt_color']
            if c <= 11:
                counts[:, c] += row['features']['bcn_hist']
                lab = np.median(row['features']['lab'], axis=0)
                grouped[c].append(np.r_[lab, np.hypot(lab[1], lab[2])])
        n = np.array([max(len(grouped[c]), 1) for c in range(1, 12)], float)
        weighted = counts[:, 1:12] * (n.sum() / (11 * n)) ** .55
        self.bcn_to_ccihp[:, 1:12] = weighted / weighted.sum(axis=1, keepdims=True)
        total = max(sum(len(v) for v in grouped.values()), 1)
        for c in range(1, 12):
            samples = np.asarray(grouped[c])
            if len(samples) < 5:
                anchor = self.bcn_lab[c - 1]
                mu = np.r_[anchor, np.hypot(anchor[1], anchor[2])]
                cov = np.diag([225., 225., 225., 144.])
            else:
                mu = samples.mean(axis=0)
                cov = np.cov(samples, rowvar=False) + np.eye(4) * 18
            self.cluster_means[c] = mu
            self.cluster_inv_covs[c] = np.linalg.inv(cov)
            self.cluster_log_dets[c] = float(np.linalg.slogdet(cov)[1])
            self.cluster_priors[c] = (max(len(samples), 1) / total) ** .35
        blue = [x[0] for x in grouped[10]]
        navy = [x[0] for x in grouped[1] if x[2] < -8]
        if blue and navy:
            self.navy_L_cutoff = float((np.percentile(navy, 80) + np.percentile(blue, 20)) / 2)
        for group in ('UPPER', 'LOWER'):
            sub = [r for r in train_items if r['target'] == group]
            if not sub:
                raise ValueError('Missing training garment group')
            by_class = defaultdict(list)
            for r in sub:
                if r['gt_color'] <= 11:
                    lab = np.median(r['features']['lab'], axis=0)
                    by_class[r['gt_color']].append(np.r_[lab, np.hypot(lab[1], lab[2])])
            total_cat = max(sum(len(v) for v in by_class.values()), 1)
            for name in ('cat_means', 'cat_inv_covs', 'cat_log_dets', 'cat_priors'):
                getattr(self, name)[group] = {}
            for c in range(1, 12):
                samples = np.asarray(by_class[c])
                if len(samples) >= 12:
                    mu = .75 * samples.mean(axis=0) + .25 * self.cluster_means[c]
                    cov = np.cov(samples, rowvar=False) + np.eye(4) * 16
                    inv, ld = np.linalg.inv(cov), float(np.linalg.slogdet(cov)[1])
                else:
                    mu, inv, ld = self.cluster_means[c], self.cluster_inv_covs[c], self.cluster_log_dets[c]
                self.cat_means[group][c], self.cat_inv_covs[group][c] = mu, inv
                self.cat_log_dets[group][c] = ld
                self.cat_priors[group][c] = (max(len(samples), 1) / total_cat) ** .55
        self.calibration_log = {}
        self.tau_multi_m2 = self._calibrate(calibration_items, 'M2', None, np.linspace(.40, .80, 41), .72)
        self.tau_multi_m3 = self._calibrate(calibration_items, 'M3', None, np.linspace(.16, .55, 40), .27)
        for group in ('UPPER', 'LOWER'):
            rows = [r for r in calibration_items if r['target'] == group]
            self.cat_tau_multi[group] = self._calibrate(rows, 'M4', group, np.linspace(.18, .55, 38), self.tau_multi_m3)
        self.fit_evidence = {}
        for group in ('UPPER', 'LOWER'):
            rows = [r for r in train_items if r['target'] == group]
            X = np.array([self._build_m5_feature_vector(r['features'], group) for r in rows])
            y = np.array([r['gt_color'] for r in rows], dtype=np.int64)
            counts = Counter(y)
            weights = np.array([(len(y) / (len(counts) * counts[c])) ** .22 for c in y])
            hgb = HistGradientBoostingClassifier(max_iter=180, learning_rate=.055, max_depth=6,
                    min_samples_leaf=12, l2_regularization=1.5, random_state=42, early_stopping=False)
            et = ExtraTreesClassifier(n_estimators=250, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=2)
            hgb.fit(X, y, sample_weight=weights)
            et.fit(X, y, sample_weight=weights)
            self.ml_models[group] = (hgb, et, sorted(int(c) for c in counts))
            self.fit_evidence[group] = dict(training_rows=len(y), feature_dimensions=X.shape[1],
                classes=[int(c) for c in hgb.classes_], HGB_iterations=int(hgb.n_iter_),
                ExtraTrees_estimators=len(et.estimators_), sample_weight_exponent=.22)

    def _calibrate(self, rows, method, group, grid, default):
        values, multi = [], []
        for r in rows:
            f = r['features']
            if method == 'M2':
                p = (f['bcn_hist'] @ self.bcn_to_ccihp)[1:12]
                value = 1 - float((p / p.sum()).max())
            else:
                value = self._eval_clusters_m3(f, group)['eff_sec']
            values.append(value)
            multi.append(r['gt_color'] == 12)
        values, multi = np.asarray(values), np.asarray(multi, bool)
        if multi.any() and (~multi).any():
            weight = min(max(float(multi.mean()) * 2.5, .20), .48) if method == 'M4' else .45
            scores = [weight * np.mean(values[multi] >= t) + (1 - weight) * np.mean(values[~multi] < t) for t in grid]
            threshold = float(grid[int(np.argmax(scores))])  # first/smallest threshold in a tie
        else:
            weight, threshold = None, float(default)
        self.calibration_log[f'{method}:{group or "GLOBAL"}'] = dict(rows=len(rows), multicolor=int(multi.sum()),
                multicolor_weight=weight, threshold=threshold, fallback='FIXED_DEFAULT' if weight is None else None)
        return threshold

    def _build_m5_feature_vector(self, features, group):
        result = self._eval_clusters_m3(features, group)
        x = np.r_[features['feat_vec'], result['soft_probs11'], result['eff_sec']]
        if x.shape != (96,) or not np.isfinite(x).all():
            raise ValueError('Invalid 96D clothing descriptor')
        return x

    def predict_all(self, features, group):
        if group not in ('UPPER', 'LOWER'):
            raise ValueError('No hair/face head is trained in this comparison')
        m1 = predict_m1_garment(features['hsv_raw'])['pred']
        p2 = (features['bcn_hist'] @ self.bcn_to_ccihp)[1:12]
        p2 /= p2.sum()
        m2 = 12 if 1 - p2.max() >= self.tau_multi_m2 else int(p2.argmax() + 1)
        m3 = self._eval_clusters_m3(features)
        m4 = self._eval_clusters_m3(features, group)
        hgb, et, _ = self.ml_models[group]
        x = self._build_m5_feature_vector(features, group).reshape(1, -1)
        p5 = blend_m5(hgb.predict_proba(x)[0], hgb.classes_, et.predict_proba(x)[0], et.classes_, m4['soft_probs11'])
        return dict(M1=int(m1), M2=int(m2), M3=int(m3['pred']), M4=int(m4['pred']), M5=int(p5[1:].argmax() + 1))
