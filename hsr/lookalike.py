"""Look-alike discrimination test (paper 1): when data barely distinguish
candidate forms, does the prior put its mass on the true one?

Design (``make_design``; settings in config ``lookalike``):

true forms
    A stratified sample of distinct forms, uniform within each stratum:
    ``holdout_seen`` / ``holdout_novel``: forms of the hold-out sources (rnls,
    ratelaws, ecocat2) that do / do not occur in the training sources;
    ``mech_<class>``: forms of each mechanism class (hsr.mechanisms; seen or
    novel recorded); ``temporal_novel``: forms of models dated from the
    temporal cutoff on that no earlier model uses (hsr.temporal). Only forms
    in the candidate set (below) with at least one constant, and whose
    constants can be drawn (hsr.simulate.draw_constants), are eligible.
candidate set
    Every form of the support of the mechanism hold-out run (training,
    candidate, hold-out and mechanism forms) with at most ``max_consts``
    constants and at most ``max_nodes`` nodes in the tree coding (the true
    forms obey the same limits, so a true form is never the only candidate of
    its size). For temporal true forms the candidate set is restricted to the
    support of the temporal run (the priors trained before the cutoff score
    only that support; it holds every corpus and candidate form).
datasets
    For every true form, every (n, noise) setting and every replicate: fresh
    constants from hsr.simulate.draw_constants (N(0, 1.5^2), rejected unless
    the curve is finite, bounded and non-constant on [0.1, 6]) and data from
    hsr.simulate._make_dataset (x uniform on [0.1, 5], Gaussian noise with sd
    = noise x range of the curve). No rescaling: the data are on the scale on
    which the constants were drawn, and the evidence prior on the constants
    (N(0, 3^2), hsr.likelihood) contains the generating one.
evidence
    Laplace log evidence of every candidate on every dataset
    (hsr.likelihood.fit_skeleton), computed in shards of the candidate list
    (``evidence_shard``; GitHub Actions, .github/workflows/lookalike.yml).
scores (``dataset_scores``, ``summarise``)
    Look-alike set A_D: candidates whose log evidence is within D nats of the
    best. Given the truth is in A_D: its rank within A_D under the prior
    (1 = the prior's favourite; ties count half), its prior mass renormalised
    within A_D, and whether it is the prior's favourite. Over the whole
    candidate set: posterior probability of the truth (evidence + log prior)
    and whether it is the MAP form. The flat prior (evidence only) and sym0
    (symbol frequencies only) are the baselines.
tempering (``temper``, ``log_p_true_taus``, ``select_tau_loto``)
    The prior raised to a power tau and renormalised over the candidate set,
    P(form)^tau / Z (tau = 0: flat; tau = 1: the prior). tau is chosen by
    leave-one-truth-form-out on the mean log posterior probability of the truth.
"""
from __future__ import annotations

import re
import signal

import numpy as np
from scipy.special import logsumexp

CONST_RE = re.compile(r"\bc\d+\b")


def n_consts(key):
    return len(set(CONST_RE.findall(key)))


def candidate_set(support, trees, max_consts, max_nodes):
    """Support forms with a tree coding, <= max_consts constants and <= max_nodes nodes."""
    return sorted(k for k in support if trees.get(k) and n_consts(k) <= max_consts and len(trees[k]) <= max_nodes)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def with_timeout(fn, seconds):
    """fn() or None if it raises or takes longer than ``seconds`` (SIGALRM; main thread only)."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(seconds))
    try:
        return fn()
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def compile_key(key, timeout=60):
    from .candidates import skeleton_from_key
    from .likelihood import CompiledSkeleton
    return with_timeout(lambda: CompiledSkeleton(skeleton_from_key(key)), timeout)


# ------------------------------------------------------------------ design
def strata_pools(mech_folds, temporal_folds, holdouts, mech_names):
    """{stratum: sorted distinct test forms} and {(stratum, key): seen}."""
    fold = mech_folds["folds"][0]
    train = set(fold["train"])
    pools, seen = {}, {}
    hold = sorted({k for k, lab in zip(fold["test"], fold["test_sources"]) if lab in holdouts})
    pools["holdout_seen"] = [k for k in hold if k in train]
    pools["holdout_novel"] = [k for k in hold if k not in train]
    for name in mech_names:
        pools[name] = sorted({k for k, lab in zip(fold["test"], fold["test_sources"]) if lab == name})
    tf = temporal_folds["folds"][0]
    ttrain = set(tf["train"])
    pools["temporal_novel"] = sorted({k for k in tf["test"] if k not in ttrain})
    for s, keys in pools.items():
        for k in keys:
            seen[(s, k)] = (k in ttrain) if s == "temporal_novel" else (k in train)
    return pools, seen


def make_design(mech_folds, temporal_folds, trees, holdouts, mech_names, cfg):
    """True forms, datasets and candidate set of the look-alike test (see the module docstring)."""
    from .simulate import _make_dataset, draw_constants
    cands = candidate_set(mech_folds["support"], trees, cfg["max_consts"], cfg["max_nodes"])
    cset = set(cands)
    tsup = set(temporal_folds["support"])
    pools, seen = strata_pools(mech_folds, temporal_folds, holdouts, mech_names)
    rng = np.random.default_rng(cfg["seed"])
    true_forms, skipped = [], {}
    for stratum, quota in cfg["strata"].items():
        pool = [k for k in pools.get(stratum, []) if k in cset and n_consts(k) >= 1
                and (stratum != "temporal_novel" or k in tsup)]
        order = rng.permutation(len(pool))
        got, bad = 0, 0
        for i in order:
            if got >= quota:
                break
            key = pool[i]
            cs = compile_key(key)
            if cs is None or draw_constants(np.random.default_rng(0), cs) is None:
                bad += 1
                continue
            true_forms.append({"key": key, "stratum": stratum, "seen": bool(seen[(stratum, key)]),
                               "pool": len(pool), "nodes": len(trees[key]), "n_consts": n_consts(key)})
            got += 1
        skipped[stratum] = {"eligible": len(pool), "not_simulable": bad, "chosen": got}
    datasets = []
    for f_i, tf in enumerate(true_forms):
        cs = compile_key(tf["key"])
        for s_i, (n, noise) in enumerate(cfg["settings"]):
            for r in range(cfg["reps"]):
                rng_d = np.random.default_rng([cfg["seed"], f_i, s_i, r])
                phi = draw_constants(rng_d, cs)
                ds = _make_dataset(rng_d, cs, phi, int(n), float(noise))
                datasets.append({"form": f_i, "n": int(n), "noise": float(noise), "rep": r,
                                 "phi": phi.tolist(), "x": ds.x.tolist(), "y": ds.y.tolist(),
                                 "noise_sd": float(ds.noise_sd)})
    return {"config": cfg, "candidates": cands, "in_temporal": [k in tsup for k in cands],
            "true_forms": true_forms, "datasets": datasets, "selection": skipped,
            "pool_sizes": {s: len(v) for s, v in pools.items()}}


def padded(design):
    """(X (J, nmax, 1), Y (J, nmax), M (J, nmax)) of the design's datasets."""
    ds = design["datasets"]
    nmax = max(len(d["x"]) for d in ds)
    J = len(ds)
    X = np.ones((J, nmax, 1))
    Y = np.zeros((J, nmax))
    M = np.zeros((J, nmax))
    for j, d in enumerate(ds):
        n = len(d["x"])
        X[j, :n, 0] = d["x"]
        Y[j, :n] = d["y"]
        M[j, :n] = 1.0
    return X, Y, M


# ---------------------------------------------------------------- evidence
def evidence_shard(design, shard, n_shards, n_starts=10, timeout=60, progress=True):
    """Laplace log evidence (J, n_cand_shard) of candidates shard::n_shards on every dataset."""
    import time

    from .likelihood import fit_skeleton
    keys = design["candidates"][shard::n_shards]
    X, Y, M = padded(design)
    out = np.full((Y.shape[0], len(keys)), -np.inf)
    secs = np.zeros(len(keys))
    for e, key in enumerate(keys):
        t0 = time.time()
        cs = compile_key(key, timeout)
        if cs is not None:
            seed = int(design["config"]["seed"]) * 100_003 + design["candidates"].index(key)
            res = with_timeout(lambda: fit_skeleton(cs, X, Y, M, n_starts=n_starts, seed=seed), 20 * timeout)
            if res is not None:
                out[:, e] = res.log_ml
        secs[e] = time.time() - t0
        if progress and e % 25 == 0:
            print(f"shard {shard}/{n_shards}: {e}/{len(keys)} candidates, {secs[:e + 1].sum():.0f} s", flush=True)
    return keys, out, secs


# ------------------------------------------------------------------ scores
def dataset_scores(L, lp, t, deltas):
    """Scores of one dataset under one prior.

    L: (C,) log evidence over the dataset's candidate set (-inf: failed fit),
    lp: (C,) log prior (-inf: zero probability), t: index of the truth."""
    out = {}
    fin = np.isfinite(L)
    best = L[fin].max() if fin.any() else -np.inf
    for D in deltas:
        A = fin & (L >= best - D)
        inA = bool(fin[t] and A[t])
        o = {"in_set": inA, "set_size": int(A.sum())}
        if inA:
            a = lp[A]
            higher = np.sum(a > lp[t])
            ties = np.sum(a == lp[t]) - 1
            o["rank"] = float(1 + higher + 0.5 * ties)
            o["pct"] = float((o["rank"] - 1) / (A.sum() - 1)) if A.sum() > 1 else 0.0
            o["top"] = bool(higher == 0 and ties == 0)
            o["mass"] = float(np.exp(lp[t] - logsumexp(a))) if np.isfinite(a).any() and np.isfinite(lp[t]) else 0.0
            o["gain"] = float(np.log(max(o["mass"], 1e-300) * A.sum()))   # log of mass relative to flat
        out[D] = o
    post = np.where(fin, L, -np.inf) + lp
    ok = np.isfinite(post)
    if ok.any() and np.isfinite(post[t]):
        out["log_p_post"] = float(post[t] - logsumexp(post[ok]))
        out["p_post"] = float(np.exp(out["log_p_post"]))
        out["map"] = bool(np.argmax(np.where(ok, post, -np.inf)) == t)
        # rank of the truth under the posterior over the whole candidate set (1 = MAP; ties count half)
        out["post_rank"] = float(1 + np.sum(post[ok] > post[t]) + 0.5 * (np.sum(post[ok] == post[t]) - 1))
    else:
        out["p_post"], out["log_p_post"], out["map"], out["post_rank"] = 0.0, -np.inf, False, float(ok.sum() + 1)
    return out


# --------------------------------------------------------------- tempering
def temper(lp, tau):
    """log of the tempered prior P(form)^tau / Z over the given candidate set.

    lp: (C,) log prior over the candidate set (-inf: zero probability). The
    result is a proper distribution on the candidate set: Z = sum_c P(c)^tau
    is a finite sum of non-negative terms with at least one positive one.
    tau = 0 is the flat prior 1/C on every candidate (including any the prior
    gives zero probability); tau = 1 is the prior renormalised over the set;
    tau > 1 sharpens it."""
    lp = np.asarray(lp, float)
    if tau < 0:
        raise ValueError("tau must be >= 0")
    if tau == 0:
        return np.full(lp.shape, -np.log(lp.size))
    if not np.isfinite(lp).any():
        raise ValueError("prior gives zero probability to every candidate")
    a = tau * lp
    return a - logsumexp(a[np.isfinite(a)])


def temper_all(lp, taus):
    """(K, C) tempered log priors temper(lp, tau) for each tau in taus."""
    return np.stack([temper(lp, tau) for tau in taus])


def log_p_true_taus(L, lp_taus, t):
    """(K,) log posterior probability of the truth t under each row of lp_taus
    (temper_all). L: (C,) log evidence (-inf: failed fit)."""
    if not np.isfinite(L[t]):
        return np.full(lp_taus.shape[0], -np.inf)
    fin = np.isfinite(L)
    post = L[fin][None, :] + lp_taus[:, fin]
    tt = int(np.sum(fin[:t]))
    return post[:, tt] - logsumexp(post, axis=1)


def select_tau_loto(logp, folds, taus):
    """Leave-one-truth-form-out choice of tau.

    logp: (J, K) log P(true) of dataset j at tau_k (rows with any -inf are
    dropped by the caller), folds: (J,) truth-form index of each dataset.
    For every form f, tau_f maximises the mean of logp over the datasets of the
    other forms; the held-out score of dataset j is logp[j, tau_{f(j)}].
    Returns {"tau_full": argmax over all datasets, "tau_folds": {f: tau_f},
    "cv": (J,) held-out log P(true)}."""
    logp = np.asarray(logp, float)
    folds = np.asarray(folds)
    taus = np.asarray(taus, float)
    full = logp.mean(axis=0)
    tot, n = logp.sum(axis=0), logp.shape[0]
    tau_f, cv = {}, np.empty(n)
    for f in np.unique(folds):
        m = folds == f
        rest = (tot - logp[m].sum(axis=0)) / max(n - m.sum(), 1)
        k = int(np.argmax(rest))
        tau_f[int(f)] = float(taus[k])
        cv[m] = logp[m, k]
    return {"tau_full": float(taus[int(np.argmax(full))]), "mean_full": full.tolist(), "tau_folds": tau_f, "cv": cv}


def summarise(rows, deltas):
    """Means over the rows (each the output of dataset_scores) of one prior and group."""
    s = {"n": len(rows), "p_post": float(np.mean([r["p_post"] for r in rows])) if rows else None,
         "map": float(np.mean([r["map"] for r in rows])) if rows else None}
    # log P(true) and posterior rank: over the datasets whose true-form fit succeeded
    lg = [r["log_p_post"] for r in rows if "log_p_post" in r and np.isfinite(r["log_p_post"])]
    s["log_p_post"] = float(np.mean(lg)) if lg else None
    s["n_log_p"] = len(lg)
    pr = [r["post_rank"] for r in rows if "log_p_post" in r and np.isfinite(r["log_p_post"])]
    s["post_rank_median"] = float(np.median(pr)) if pr else None
    for D in deltas:
        inA = [r[D] for r in rows if r[D]["in_set"]]
        s[f"in_set_{D}"] = float(np.mean([r[D]["in_set"] for r in rows])) if rows else None
        s[f"set_size_{D}"] = float(np.median([r[D]["set_size"] for r in rows])) if rows else None
        for f in ("rank", "pct", "top", "mass"):
            s[f"{f}_{D}"] = float(np.mean([o[f] for o in inA])) if inA else None
        # median: a prior that gives the truth zero probability would dominate a mean of logs
        s[f"gain_{D}"] = float(np.median([o["gain"] for o in inA])) if inA else None
        s[f"n_in_{D}"] = len(inA)
    return s
