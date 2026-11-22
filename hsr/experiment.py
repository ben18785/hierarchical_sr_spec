"""Experiment runner for the synthetic benchmark (spec section 6).

    python -m hsr.experiment --setting mixture --n 10 --noise 0.05 --reps 0:20 \
        --candidates data/candidates.json --out results/mixture_n10_s05.jsonl

Each replicate writes one JSON line holding every method's metrics for both
marginal-likelihood kinds (Laplace and BIC), plus the calibration pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
from scipy.special import logsumexp

from . import baselines
from .candidates import CandidateSet, expand_candidates, from_skeletons
from .canonical import canonicalise, counts_array
from .gibbs import GibbsConfig, run_gibbs
from .grammar import Grammar
from .likelihood import CompiledSkeleton, compute_likelihoods, predictive_logpdf, scale_population
from .metrics import adjusted_rand_index, form_recovery, reliability_pairs, weight_error
from .simulate import all_A_population, benchmark_population, grammar_population

_COMPILED = {}
SETTING_SEED = {"mixture": 11, "all_A": 23, "grammar": 37}


def make_population(setting, rng, n, noise, cands, J=30):
    if setting == "mixture":
        return benchmark_population(rng, n, noise, J)
    if setting == "all_A":
        return all_A_population(rng, n, noise, J)
    if setting == "grammar":
        return grammar_population(rng, n, noise, cands, J)
    raise ValueError(setting)


def scaled_arrays(pop, mode="pooled"):
    X, Y, sx, sy = scale_population([d.x for d in pop.datasets], [d.y for d in pop.datasets], mode)
    return np.array(X), np.array(Y), np.array(sx), np.array(sy)


def _compiled(sk):
    cs = _COMPILED.get(sk.key)
    if cs is None:
        cs = CompiledSkeleton(sk)
        _COMPILED[sk.key] = cs
    return cs


def lpd(pop, probs, skeletons, table, sx, sy, which="in", min_prob=1e-3, n_draws=200, seed=0):
    """Mean pointwise log predictive density over datasets, mixing over the
    skeletons that carry posterior mass (columns of ``probs`` refer to
    ``skeletons`` and to ``table`` columns)."""
    per_ds = []
    for j, ds in enumerate(pop.datasets):
        xn, yn = (ds.x_in, ds.y_in) if which == "in" else (ds.x_out, ds.y_out)
        row = probs[j]
        idx = np.flatnonzero(row >= min_prob)
        if len(idx) == 0:
            idx = np.array([int(np.argmax(row))])
        w = row[idx] / row[idx].sum()
        comps = []
        for e in idx:
            cs = _compiled(skeletons[e])
            comps.append(predictive_logpdf(cs, xn, yn, table.phi[e][j], table.log_sigma[j, e], table.cov[e][j],
                                           sx[j], sy[j], n_draws=n_draws, seed=seed + j))
        comps = np.array(comps)
        mix = logsumexp(comps + np.log(w)[:, None], axis=0)
        per_ds.append(np.mean(mix))
    return float(np.mean(per_ds)), per_ds


def one_hot(idx, n_cols):
    P = np.zeros((len(idx), n_cols))
    P[np.arange(len(idx)), idx] = 1.0
    return P


def evaluate(name, probs, pop, skeletons, table, sx, sy, true_idx, weights_est=None, extras=None,
             calibration=True):
    top = np.argmax(probs, axis=1)
    top_keys = [skeletons[e].key for e in top]
    out = {
        "method": name,
        "recovery": form_recovery(top_keys, pop.true_keys),
        "ari": adjusted_rand_index(top, pop.labels),
        "n_distinct_top": int(len(set(top.tolist()))),
    }
    if weights_est is None:
        weights_est = {}
        for k in top_keys:
            weights_est[k] = weights_est.get(k, 0.0) + 1.0 / len(top_keys)
    out["weight_error"] = weight_error(weights_est, pop.form_keys, pop.form_weights)
    out["lpd_in"], _ = lpd(pop, probs, skeletons, table, sx, sy, "in")
    out["lpd_out"], _ = lpd(pop, probs, skeletons, table, sx, sy, "out")
    out["p_true_mean"] = float(np.mean([probs[j, t] if t >= 0 else 0.0 for j, t in enumerate(true_idx)]))
    if calibration:
        p, y = reliability_pairs(probs, true_idx)
        out["calib_p"] = np.round(p, 4).tolist()
        out["calib_y"] = y.tolist()
    if extras:
        out.update(extras)
    return out


def gibbs_summary(res):
    rh = res.rhat(res.n_forms.astype(float))
    return {
        "n_forms_mean": float(res.n_forms.mean()),
        "gamma_mean": float(res.gamma.mean()),
        "rhat_n_forms": None if not np.isfinite(rh) else float(rh),
        "chain_disagreement": res.chain_disagreement(),
    }


SENSITIVITY = {
    "alpha_x0.25": {"alpha_scale": 0.25},
    "alpha_x4": {"alpha_scale": 4.0},
    "gamma_prior_1_0.1": {"gamma_a": 1.0, "gamma_b": 0.1},
    "gamma_prior_1_10": {"gamma_a": 1.0, "gamma_b": 10.0},
    "gamma_fixed_1": {"update_gamma": False, "gamma_init": 1.0},
}


def _scaled_grammar(g: Grammar, scale: float) -> Grammar:
    return Grammar(n_vars=g.n_vars, alpha={k: v * scale for k, v in g.alpha.items()},
                   max_size=g.max_size, max_consts=g.max_consts)


def run_replicate(setting, n, noise, rep, cands: CandidateSet, cfg: GibbsConfig, expand=False,
                  sensitivity=False, kinds=("laplace", "bic"), J=30, n_starts=10, seed_base=12345,
                  scaling="pooled"):
    t0 = time.time()
    seed = seed_base + 1_000_003 * rep + int(1000 * noise) * 7919 + n * 104729 + SETTING_SEED[setting]
    rng = np.random.default_rng(abs(seed))
    pop = make_population(setting, rng, n, noise, cands, J)
    X, Y, sx, sy = scaled_arrays(pop, scaling)
    grammar = cands.grammar
    cset = from_skeletons(list(cands.skeletons), grammar, dict(cands.stats))
    table = compute_likelihoods(cset.skeletons, X, Y, n_starts=n_starts, seed=rep, compiled_cache=_COMPILED)
    n_added = 0
    if expand:
        post = baselines.no_pooling_posterior(table.laplace, cset.counts, grammar)
        top = set()
        for j in range(J):
            top.update(np.argsort(post[j])[-3:].tolist())
        new_idx = expand_candidates(cset, sorted(top))
        if new_idx:
            extra = compute_likelihoods([cset.skeletons[i] for i in new_idx], X, Y, n_starts=n_starts,
                                        seed=rep + 7, compiled_cache=_COMPILED)
            table = table.extend(extra)
        n_added = len(new_idx)
    # true forms: make sure every one has an L column (append if missing from C)
    true_in_C = {k: (k in set(cset.keys)) for k in pop.form_keys}
    missing = [k for k, ok in true_in_C.items() if not ok]
    skeletons = list(cset.skeletons)
    counts = cset.counts
    n_C = len(skeletons)
    if missing:
        sks = [canonicalise(k) for k in missing]
        extra = compute_likelihoods(sks, X, Y, n_starts=n_starts, seed=rep + 11, compiled_cache=_COMPILED)
        table = table.extend(extra)
        skeletons += sks
        counts = np.vstack([counts, np.array([counts_array(s.expr, grammar) for s in sks])])
    key_to_idx = {s.key: i for i, s in enumerate(skeletons)}
    true_idx_all = np.array([key_to_idx[k] for k in pop.true_keys])
    # columns outside C may only be used by the oracle
    in_C = np.arange(n_C)
    true_idx_C = np.array([t if t < n_C else -1 for t in true_idx_all])
    form_idx = [key_to_idx[k] for k in pop.form_keys]
    t_lik = time.time() - t0

    rows = []
    for kind in kinds:
        L = table.matrix(kind)[:, in_C]
        cnt = counts[in_C]
        sk_C = skeletons[:n_C]
        tabC = table  # columns 0..n_C-1 coincide with C

        res = run_gibbs(L, cnt, grammar, cfg)
        pop_w = {sk_C[e].key: float(res.pop_weight[e]) for e in range(n_C)}
        rows.append(evaluate(f"hier", res.assign_probs, pop, sk_C, tabC, sx, sy, true_idx_C, pop_w,
                             gibbs_summary(res)) | {"kind": kind})

        z = baselines.no_pooling(L, cnt, grammar)
        rows.append(evaluate("no_pool", one_hot(z, n_C), pop, sk_C, tabC, sx, sy, true_idx_C,
                             calibration=False) | {"kind": kind})

        P = baselines.no_pooling_posterior(L, cnt, grammar)
        w_np = {sk_C[e].key: float(P[:, e].mean()) for e in range(n_C)}
        rows.append(evaluate("no_pool_bma", P, pop, sk_C, tabC, sx, sy, true_idx_C, w_np) | {"kind": kind})

        e_cp = baselines.complete_pooling(L)
        rows.append(evaluate("complete_pool", one_hot(np.full(J, e_cp), n_C), pop, sk_C, tabC, sx, sy,
                             true_idx_C, calibration=False) | {"kind": kind})

        ores, _ = baselines.oracle(table.matrix(kind), counts, grammar, form_idx, cfg)
        # map oracle posteriors back to full columns for evaluation
        Pfull = np.zeros((J, len(skeletons)))
        Pfull[:, form_idx] = ores.assign_probs
        w_or = {skeletons[e].key: float(ores.pop_weight[i]) for i, e in enumerate(form_idx)}
        rows.append(evaluate("oracle", Pfull, pop, skeletons, table, sx, sy, true_idx_all, w_or,
                             gibbs_summary(ores)) | {"kind": kind})

        if sensitivity and kind == "laplace":
            for label, opts in SENSITIVITY.items():
                g2 = grammar
                c2 = cfg
                if "alpha_scale" in opts:
                    g2 = _scaled_grammar(grammar, opts["alpha_scale"])
                else:
                    c2 = replace(cfg, **opts)
                r2 = run_gibbs(L, cnt, g2, c2)
                pw = {sk_C[e].key: float(r2.pop_weight[e]) for e in range(n_C)}
                rows.append(evaluate(f"hier[{label}]", r2.assign_probs, pop, sk_C, tabC, sx, sy, true_idx_C,
                                     pw, gibbs_summary(r2), calibration=False) | {"kind": kind})

    meta = {
        "setting": setting, "n": n, "noise": noise, "rep": rep, "J": J,
        "n_candidates": n_C, "scaling": scaling, "n_added_by_edits": n_added, "expand": expand,
        "true_forms_in_C": true_in_C, "frac_datasets_true_in_C": float(np.mean(true_idx_C >= 0)),
        "seconds_likelihood": t_lik, "seconds_total": time.time() - t0,
    }
    for r in rows:
        r.update(meta)
    return rows


def parse_reps(s):
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b)))
    return [int(v) for v in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="mixture", choices=["mixture", "all_A", "grammar"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--reps", default="0:20")
    ap.add_argument("--J", type=int, default=30)
    ap.add_argument("--candidates", default=os.path.join(os.path.dirname(__file__), "data", "candidates.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweeps", type=int, default=2000)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--scaling", default="pooled", choices=["pooled", "per_dataset", "none"])
    ap.add_argument("--expand", action="store_true", help="add data-driven one-step edits to C")
    ap.add_argument("--sensitivity", action="store_true", help="also run alpha / gamma-prior variants")
    args = ap.parse_args(argv)

    cands = CandidateSet.from_json(args.candidates)
    cfg = GibbsConfig(n_sweeps=args.sweeps, burn_in=args.burn, n_chains=args.chains)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fh:
        for rep in parse_reps(args.reps):
            rows = run_replicate(args.setting, args.n, args.noise, rep, cands, cfg, expand=args.expand,
                                 sensitivity=args.sensitivity, J=args.J, n_starts=args.starts, scaling=args.scaling)
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            hier = [r for r in rows if r["method"] == "hier" and r["kind"] == "laplace"][0]
            print(f"[{args.setting} n={args.n} noise={args.noise} rep={rep}] "
                  f"hier recovery={hier['recovery']:.2f} lpd_out={hier['lpd_out']:.2f} "
                  f"t={hier['seconds_total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
