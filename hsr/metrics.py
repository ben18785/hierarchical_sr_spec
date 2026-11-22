"""Evaluation metrics (spec section 6)."""
from __future__ import annotations

import numpy as np
from scipy.special import comb


def adjusted_rand_index(a, b) -> float:
    """Adjusted Rand index between two labelings (Hubert & Arabie 1985)."""
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)
    n = len(a)
    sum_comb = comb(table, 2).sum()
    sa = comb(table.sum(axis=1), 2).sum()
    sb = comb(table.sum(axis=0), 2).sum()
    total = comb(n, 2)
    expected = sa * sb / total if total else 0.0
    max_index = 0.5 * (sa + sb)
    if max_index == expected:
        return 1.0  # both partitions trivial and identical in structure
    return float((sum_comb - expected) / (max_index - expected))


def form_recovery(top_keys, true_keys) -> float:
    return float(np.mean([t == s for t, s in zip(top_keys, true_keys)]))


def weight_error(est: dict, true_keys, true_weights) -> float:
    """Mean absolute error of the estimated population weight of each true form."""
    return float(np.mean([abs(est.get(k, 0.0) - w) for k, w in zip(true_keys, true_weights)]))


def reliability_pairs(probs: np.ndarray, true_idx, threshold=0.01):
    """(p, is_true) pairs for every (dataset, candidate) with posterior > threshold,
    plus the true candidate of each dataset even when its probability is small."""
    out_p, out_y = [], []
    for j, t in enumerate(true_idx):
        row = probs[j]
        idx = set(np.flatnonzero(row > threshold).tolist())
        if t is not None and t >= 0:
            idx.add(int(t))
        for e in idx:
            out_p.append(float(row[e]))
            out_y.append(int(e == t))
    return np.array(out_p), np.array(out_y)


def reliability_curve(p, y, bins=10):
    """Binned (mean predicted, observed frequency, count)."""
    p = np.asarray(p)
    y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def expected_calibration_error(p, y, bins=10) -> float:
    rows = reliability_curve(p, y, bins)
    n = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[0] - r[1]) for r in rows) / max(n, 1))
