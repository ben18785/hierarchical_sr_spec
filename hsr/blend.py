"""Blended (mixture) priors over model forms.

Every tree prior (fnprior, hpy, the hpyv variants) is a proper distribution
over one shared space of coded trees (docs/prior_models.tex), so a convex
combination of them is proper too:

    P_w(f) = sum_k w_k P_k(f),   w_k >= 0, sum_k w_k = 1,

computed per form in log space as logsumexp_k(log w_k + log P_k(f)). For two
components, P_w = w P_A + (1 - w) P_B.

The weights are chosen on leave-one-source-out (LOSO) runs only: maximise the
mean over left-out sources of the mean log P_w per model-form pair of that
source (each source counts equally, whatever its size). The objective is
concave in w (a mean of logs of functions linear in w), so a grid search
(``select_w``, two components) and EM (``em_weights``, any number) find the
same global maximum.
"""
import numpy as np


def mix_logp(logps, weights):
    """log sum_k w_k P_k per form. ``logps``: K arrays of log P (same shape;
    -inf for zero probability); ``weights``: K non-negative weights summing
    to 1. Components with weight 0 are dropped, so w = 0 or 1 returns a
    component exactly."""
    weights = np.asarray(weights, float)
    if weights.ndim != 1 or len(weights) != len(logps):
        raise ValueError("one weight per component")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"weights must be non-negative and sum to 1: {weights}")
    terms = [np.log(w) + np.asarray(lp, float) for w, lp in zip(weights, logps) if w > 0]
    with np.errstate(invalid="ignore"):
        return np.logaddexp.reduce(np.stack(terms), axis=0)


def mix2(lA, lB, w):
    """log(w P_A + (1 - w) P_B) per form."""
    return mix_logp([lA, lB], [w, 1.0 - w])


def w_grid(step=0.005, logit_step=0.05, logit_max=12.0):
    """Weights in [0, 1]: 0, 1, a linear grid of spacing ``step`` and a grid of
    spacing ``logit_step`` in logit(w) out to +-``logit_max`` (fine near the
    ends, where the linear grid is coarse in log w)."""
    t = np.arange(-logit_max, logit_max + logit_step / 2, logit_step)
    g = np.concatenate([[0.0, 1.0], np.arange(step, 1.0, step), 1.0 / (1.0 + np.exp(-t))])
    return np.unique(np.round(g, 15))


def source_mean(groups, w, mean_over="sources"):
    """Mean log P_w per pair, averaged equally over the groups (``mean_over``
    "sources") or over all pairs ("pairs"). ``groups``: {name: (lA, lB)}."""
    vals = [mix2(a, b, w) for a, b in groups.values()]
    if mean_over == "pairs":
        return float(np.mean(np.concatenate(vals)))
    return float(np.mean([v.mean() for v in vals]))


def objective_curve(groups, grid=None, mean_over="sources"):
    grid = w_grid() if grid is None else np.asarray(grid, float)
    return grid, np.array([source_mean(groups, w, mean_over) for w in grid])


def select_w(groups, grid=None, mean_over="sources"):
    """(w maximising ``source_mean`` on the grid, its objective, (grid, curve))."""
    grid, curve = objective_curve(groups, grid, mean_over)
    i = int(np.nanargmax(np.where(np.isfinite(curve), curve, -np.inf)))
    return float(grid[i]), float(curve[i]), (grid, curve)


def em_weights(groups, mean_over="sources", iters=5000, tol=1e-12, init=None):
    """Mixture weights maximising the mean log P_w (equal over groups, or
    over pairs) by EM. ``groups``: {name: [l_1, ..., l_K]} (log P per pair of
    each component). Returns (weights, objective)."""
    names = list(groups)
    L = np.concatenate([np.stack([np.asarray(x, float) for x in groups[n]]) for n in names], axis=1)  # K x N
    n = np.array([len(groups[g][0]) for g in names])
    if mean_over == "pairs":
        a = np.full(L.shape[1], 1.0 / L.shape[1])
    else:
        a = np.concatenate([np.full(k, 1.0 / (len(n) * k)) for k in n])
    K = L.shape[0]
    w = np.full(K, 1.0 / K) if init is None else np.asarray(init, float)
    prev = -np.inf
    for _ in range(iters):
        with np.errstate(divide="ignore"):
            T = np.log(w)[:, None] + L
        m = np.logaddexp.reduce(T, axis=0)
        obj = float(a @ m)
        R = np.exp(T - m)
        w = R @ a
        w = w / w.sum()
        if obj - prev < tol:
            break
        prev = obj
    with np.errstate(divide="ignore"):
        obj = float(a @ np.logaddexp.reduce(np.log(w)[:, None] + L, axis=0))
    return w, obj
