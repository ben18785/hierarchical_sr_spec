"""MCMC convergence diagnostics for the DP samplers.

Cluster labels are skeletons, so some summaries are label-invariant and can be
compared across chains directly:

  log_lik      sum of the cluster marginals M_e(S) (log p(y | partition, labels))
  n_forms      number of distinct skeletons in use
  gamma        DP concentration
  share[e]     fraction of datasets assigned to skeleton e, for the skeletons
               with the largest posterior population weight

For each, the rank-normalised split R-hat (with the folded version for the
tails) and the bulk effective sample size of Vehtari, Gelman, Simpson,
Carpenter & Buerkner (2021). Chain disagreement over assignment posteriors is
reported as both the mean and the max total-variation distance.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def _split(x):
    """(chains, draws) -> (2*chains, draws//2)."""
    x = np.asarray(x, float)
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, -n:]], axis=0)


def _rank_normalise(x):
    r = rankdata(x, method="average").reshape(x.shape)
    S = x.size
    return norm.ppf((r - 3 / 8) / (S + 1 / 4))


def _rhat_basic(x):
    m, n = x.shape
    if n < 2:
        return np.nan
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    if W <= 0:
        return 1.0 if B <= 0 else np.inf
    var = (n - 1) / n * W + B / n
    return float(np.sqrt(var / W))


def split_rhat(x) -> float:
    """Rank-normalised split R-hat (max of bulk and folded/tail versions).
    A quantity that is constant within every chain gets 1.0 if all chains agree
    and inf if they do not."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] < 4 or not np.all(np.isfinite(x)):
        return np.nan
    if np.all(x.std(axis=1) == 0):
        return 1.0 if np.ptp(x.mean(axis=1)) == 0 else np.inf
    s = _split(x)
    bulk = _rhat_basic(_rank_normalise(s))
    fold = _rhat_basic(_rank_normalise(np.abs(s - np.median(s))))
    return float(max(bulk, fold))


def ess_bulk(x) -> float:
    """Bulk effective sample size (rank-normalised split chains, Geyer's
    initial monotone sequence), as in Stan."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] < 4 or not np.all(np.isfinite(x)):
        return np.nan
    if np.all(x.std(axis=1) == 0):
        return float(x.size) if np.ptp(x.mean(axis=1)) == 0 else 0.0
    z = _rank_normalise(_split(x))
    m, n = z.shape
    # autocovariance per chain by FFT
    zc = z - z.mean(axis=1, keepdims=True)
    f = np.fft.rfft(zc, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    chain_var = acov[:, 0] * n / (n - 1)
    W = chain_var.mean()
    var_plus = W * (n - 1) / n + z.mean(axis=1).var(ddof=1) if m > 1 else W * (n - 1) / n
    if var_plus <= 0:
        return np.nan
    rho = 1 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sum adjacent pairs while positive, enforce monotone decrease
    t = 0
    pairs = []
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        pairs.append(p)
        t += 2
    pairs = np.minimum.accumulate(np.array(pairs)) if pairs else np.array([1.0])
    tau = -1 + 2 * pairs.sum()
    tau = max(tau, 1 / np.log10(m * n))
    return float(m * n / tau)


def chain_tv(chain_assign_probs):
    """(mean, max) over datasets and chain pairs of the total-variation
    distance between chain-specific assignment posteriors."""
    P = np.asarray(chain_assign_probs)
    vals = []
    for a in range(P.shape[0]):
        for b in range(a + 1, P.shape[0]):
            vals.append(0.5 * np.abs(P[a] - P[b]).sum(axis=1))
    if not vals:
        return np.nan, np.nan
    v = np.concatenate(vals)
    return float(v.mean()), float(v.max())


def summary(res, top_k=3, rhat_max=1.05, ess_min=100) -> dict:
    """Diagnostics dict for a GibbsResult / ThemeResult."""
    q = {"n_forms": res.n_forms, "gamma": res.gamma}
    if getattr(res, "log_lik", None) is not None:
        q["log_lik"] = res.log_lik
    z = res.z_samples                                   # (chains, draws, J)
    J = z.shape[-1]
    for e in np.argsort(res.pop_weight)[::-1][:top_k]:
        q[f"share_{int(e)}"] = (z == e).sum(axis=-1) / J
    rh = {k: split_rhat(v) for k, v in q.items()}
    es = {k: ess_bulk(v) for k, v in q.items()}
    tv_mean, tv_max = chain_tv(res.chain_assign_probs)
    finite_rh = [v for v in rh.values() if np.isfinite(v)]
    finite_es = [v for v in es.values() if np.isfinite(v)]
    max_rhat = max(finite_rh) if finite_rh else np.nan
    if any(v == np.inf for v in rh.values()):
        max_rhat = np.inf
    min_ess = min(finite_es) if finite_es else np.nan
    return {
        "rhat": {k: (None if not np.isfinite(v) and v != np.inf else float(v)) for k, v in rh.items()},
        "ess_bulk": {k: (None if not np.isfinite(v) else float(v)) for k, v in es.items()},
        "max_rhat": None if np.isnan(max_rhat) else float(max_rhat),
        "min_ess_bulk": None if np.isnan(min_ess) else float(min_ess),
        "chain_tv_mean": tv_mean, "chain_tv_max": tv_max,
        "converged": bool(np.isfinite(max_rhat) and max_rhat < rhat_max and np.isfinite(min_ess) and min_ess >= ess_min),
    }
