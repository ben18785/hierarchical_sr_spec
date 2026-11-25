"""How much of a chosen form is decided by the data and how much by the prior.

For each dataset j, compare the form the method chose (the posterior mode,
``e_post``) with the form the data alone favour (``e_data``, the maximum of the
dataset's log evidence L[j, :]):

* data cost   = L[j, e_data] - L[j, e_post]  (>= 0; nats of evidence given up)
* base gain   = log P0(e_post) - log P0(e_data)  (what the base measure, e.g.
  the corpus prior, contributes)
* sharing     = data cost - base gain  (the rest: the posterior's preference
  that comes from other datasets, through the shared forms of the DP mixture
  and, for pooled models, the shared constants)

When the method keeps the data's choice (e_post == e_data), the data margin
over the best rival, L[j, e_post] - max_{e != e_post} L[j, e], and the base
measure's margin over that rival are reported instead.

For a PCFG base measure whose rule probabilities are sampled (Grammar with
update_theta), log P0 is taken at the prior mean of theta: an approximation.
For fixed priors (hsr.fixed_prior.FixedPrior) it is exact.
"""
from __future__ import annotations

import numpy as np


def _finite(a):
    a = np.array(a, dtype=float)
    a[~np.isfinite(a)] = -np.inf
    return a


def evidence_decomposition(probs, L, log_p0, keys=None, max_list=None):
    """Summary dict (keys prefixed ``dec_``) for a (J, K) posterior ``probs``,
    the (J, K) log-evidence matrix ``L`` and the (K,) base-measure log
    probabilities ``log_p0``. ``keys`` (form names) adds per-dataset detail."""
    probs = np.asarray(probs, float)
    L = _finite(L)
    lp = _finite(log_p0)
    J, K = probs.shape
    L = L[:, :K]
    lp = lp[:K]
    e_post = np.argmax(probs, axis=1)
    e_data = np.argmax(L, axis=1)
    rows = []
    for j in range(J):
        ep, ed = int(e_post[j]), int(e_data[j])
        if ep != ed:
            cost = float(L[j, ed] - L[j, ep])
            base = float(lp[ep] - lp[ed])
            rows.append({"changed": True, "data_cost": cost, "base_gain": base, "sharing": cost - base,
                         "rival": ed})
        else:
            Lr = L[j].copy()
            Lr[ep] = -np.inf
            r = int(np.argmax(Lr))
            rows.append({"changed": False, "data_margin": float(L[j, ep] - L[j, r]),
                         "base_margin": float(lp[ep] - lp[r]), "rival": r})
    ch = [r for r in rows if r["changed"]]
    un = [r for r in rows if not r["changed"]]

    def mean(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else None

    def med(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else None

    out = {
        "dec_frac_changed": len(ch) / J if J else None,
        "dec_data_cost_mean": mean([r["data_cost"] for r in ch]),
        "dec_base_gain_mean": mean([r["base_gain"] for r in ch]),
        "dec_sharing_mean": mean([r["sharing"] for r in ch]),
        "dec_data_margin_median": med([r["data_margin"] for r in un]),
        "dec_base_margin_median": med([r["base_margin"] for r in un]),
        # datasets where the base measure alone outweighs the data's preference
        "dec_frac_base_decides": (sum(1 for r in ch if r["base_gain"] >= r["data_cost"]) / J) if J else None,
    }
    if keys is not None:
        det = []
        for j, r in enumerate(rows[: max_list or J]):
            d = {"chosen": keys[int(e_post[j])], "rival": keys[r["rival"]]}
            d.update({k: round(v, 3) for k, v in r.items() if k not in ("rival",) and isinstance(v, float)})
            d["changed"] = r["changed"]
            det.append(d)
        out["dec_datasets"] = det
    return out
