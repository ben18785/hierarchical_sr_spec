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
from .candidates import CandidateSet, default_n_jobs, edit_graph, expand_candidates, from_skeletons
from .canonical import canonicalise, counts_array
from .families import SETTINGS as FAMILY_SETTINGS
from .families import family_metrics, family_population, theme_metrics
from .gibbs import GibbsConfig, run_gibbs
from .grammar import Grammar
from .likelihood import (CompiledSkeleton, LikelihoodTable, compute_likelihoods, predictive_logpdf,
                         scale_population)
from .pooled import PooledModel
from .themes import EditKernel, run_gibbs_themes
from .metrics import adjusted_rand_index, form_recovery, reliability_pairs, weight_error
from .simulate import all_A_population, benchmark_population, grammar_population

_COMPILED = {}
SETTING_SEED = {"mixture": 11, "all_A": 23, "grammar": 37}
SETTING_SEED.update({s: 101 + 13 * i for i, s in enumerate(FAMILY_SETTINGS)})


def make_population(setting, rng, n, noise, cands, J=30):
    if setting == "mixture":
        return benchmark_population(rng, n, noise, J)
    if setting == "all_A":
        return all_A_population(rng, n, noise, J)
    if setting == "grammar":
        return grammar_population(rng, n, noise, cands, J)
    if setting in FAMILY_SETTINGS:
        return family_population(rng, setting, n, noise, J)
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
        "top_keys": top_keys,
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
    if hasattr(pop, "parent_key"):
        out.update(family_metrics(pop, top_keys))
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
                  scaling="pooled", tau=3.0, grow_rounds=0, pooled=False, themes=False, theme_top_k=10,
                  theme_min_mass=0.02, scale_invariant=False):
    t0 = time.time()
    seed = seed_base + 1_000_003 * rep + int(1000 * noise) * 7919 + n * 104729 + SETTING_SEED[setting]
    rng = np.random.default_rng(abs(seed))
    pop = make_population(setting, rng, n, noise, cands, J)
    X, Y, sx, sy = scaled_arrays(pop, scaling)
    grammar = cands.grammar
    base = list(cands.skeletons)
    admit = None
    if scale_invariant:
        from .properties import ScaleFilter
        admit = ScaleFilter()
        base = admit.filter(base)
    cset = from_skeletons(base, grammar, dict(cands.stats))
    cset.admit = admit
    table = compute_likelihoods(cset.skeletons, X, Y, n_starts=n_starts, tau=tau, seed=rep, compiled_cache=_COMPILED)
    n_added = 0
    for _ in range(grow_rounds if grow_rounds else (1 if expand else 0)):
        # data-driven growth: one-step edits of each dataset's top-3 skeletons
        # under the independent (no-pooling) posterior
        post = baselines.no_pooling_posterior(table.laplace, cset.counts, grammar)
        top = set()
        for j in range(J):
            top.update(np.argsort(post[j])[-3:].tolist())
        new_idx = expand_candidates(cset, sorted(top))
        if not new_idx:
            break
        extra = compute_likelihoods([cset.skeletons[i] for i in new_idx], X, Y, n_starts=n_starts,
                                    tau=tau, seed=rep + 7 + n_added, compiled_cache=_COMPILED)
        table = table.extend(extra)
        n_added += len(new_idx)
    # themes: a second growth round around the skeletons that matter, which
    # also records the edit graph (parent -> one-step edit) for the kernel
    edges = set()
    n_added_themes = 0
    if themes:
        L0 = table.laplace
        post = baselines.no_pooling_posterior(L0, cset.counts, grammar)
        sources = set()
        for j in range(J):
            sources.update(np.argsort(post[j])[-theme_top_k:].tolist())
        pm0 = PooledModel(table, [sk.n_consts for sk in cset.skeletons], tau_ref=tau, tau_mu=tau)
        short = replace(cfg, n_sweeps=300, burn_in=100, n_chains=2)
        r0 = run_gibbs(L0, cset.counts, grammar, short, model=pm0)
        sources.update(np.flatnonzero(r0.assign_probs.max(axis=0) >= theme_min_mass).tolist())
        new_idx, edges = edit_graph(cset, sorted(sources), n_jobs=default_n_jobs())
        if new_idx:
            extra = compute_likelihoods([cset.skeletons[i] for i in new_idx], X, Y, n_starts=n_starts,
                                        tau=tau, seed=rep + 31, compiled_cache=_COMPILED)
            table = table.extend(extra)
        n_added_themes = len(new_idx)
    # true forms: make sure every one has an L column (append if missing from C)
    true_in_C = {k: (k in set(cset.keys)) for k in pop.form_keys}
    missing = [k for k, ok in true_in_C.items() if not ok]
    skeletons = list(cset.skeletons)
    counts = cset.counts
    n_C = len(skeletons)
    if missing:
        sks = [canonicalise(k) for k in missing]
        extra = compute_likelihoods(sks, X, Y, n_starts=n_starts, tau=tau, seed=rep + 11,
                                    compiled_cache=_COMPILED)
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

        if (pooled or themes) and kind == "laplace":
            tab_in = LikelihoodTable(table.laplace[:, :n_C], table.bic[:, :n_C], table.phi[:n_C],
                                   table.log_sigma[:, :n_C], table.cov[:n_C])
            pm = PooledModel(tab_in, [s.n_consts for s in sk_C], tau_ref=tau, tau_mu=tau)
            resp = run_gibbs(L, cnt, grammar, cfg, model=pm)
            pop_wp = {sk_C[e].key: float(resp.pop_weight[e]) for e in range(n_C)}
            rows.append(evaluate("hier_pooled", resp.assign_probs, pop, sk_C, table, sx, sy, true_idx_C, pop_wp,
                                 gibbs_summary(resp)) | {"kind": kind})

        if themes and kind == "laplace":
            kernel = EditKernel(n_C, edges, log_w=grammar.log_prob(cnt, grammar.prior_mean()))
            rest = run_gibbs_themes(L, cnt, grammar, kernel, cfg, model=pm)
            pop_wt = {sk_C[e].key: float(rest.pop_weight[e]) for e in range(n_C)}
            theme_keys = [sk_C[t].key for t in rest.theme_top]
            ext = gibbs_summary(rest) | {"theme_top_keys": theme_keys,
                                         "n_themes_mean": float(rest.n_themes.mean()),
                                         "rho_mean": float(rest.rho.mean()),
                                         "n_kernel_edges": kernel.n_edges}
            if hasattr(pop, "parent_key"):
                ext |= theme_metrics(pop, theme_keys)
            rows.append(evaluate("hier_themes", rest.assign_probs, pop, sk_C, table, sx, sy, true_idx_C, pop_wt,
                                 ext) | {"kind": kind})

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
        "n_candidates": n_C, "scaling": scaling, "tau": tau, "grow_rounds": grow_rounds, "n_added_by_edits": n_added, "expand": expand,
        "themes": themes, "n_added_by_themes": n_added_themes, "scale_invariant": scale_invariant,
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
    ap.add_argument("--setting", default="mixture", choices=["mixture", "all_A", "grammar"] + FAMILY_SETTINGS)
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
    ap.add_argument("--tau", type=float, default=3.0, help="prior sd of every constant")
    ap.add_argument("--grow", type=int, default=0, help="rounds of data-driven candidate growth")
    ap.add_argument("--pooled", action="store_true", help="also run the pooled-constants model")
    ap.add_argument("--themes", action="store_true",
                    help="grow candidates around the posterior and run the themes-and-variations model")
    ap.add_argument("--expand", action="store_true", help="add data-driven one-step edits to C")
    ap.add_argument("--sensitivity", action="store_true", help="also run alpha / gamma-prior variants")
    ap.add_argument("--scale-invariant", action="store_true",
                    help="keep only skeletons whose constants absorb changes of units in x and y (hsr.properties)")
    ap.add_argument("--corpus", default=None,
                    help="corpus JSON (hsr.corpus): set the grammar's Dirichlet prior from its rule frequencies")
    ap.add_argument("--kappa", type=float, default=50.0, help="prior sample size of the corpus prior")
    ap.add_argument("--corpus-heldout", action="store_true",
                    help="drop every benchmark-family key (parents, variants, specialisations) from the corpus")
    args = ap.parse_args(argv)

    grammar, prior_label = None, "default"
    if args.corpus:
        from .corpus import corpus_alpha, load as load_corpus
        from .families import FAMILIES, family_keys
        exclude = [k for fam in FAMILIES for k in family_keys(fam)] if args.corpus_heldout else []
        grammar = Grammar(alpha=corpus_alpha(load_corpus(args.corpus, exclude), Grammar(), kappa=args.kappa))
        prior_label = f"corpus{'_heldout' if args.corpus_heldout else ''}_k{args.kappa:g}"
        print("grammar prior", prior_label, {k: np.round(v, 2).tolist() for k, v in grammar.alpha.items()}, flush=True)
    cands = CandidateSet.from_json(args.candidates, grammar)
    cfg = GibbsConfig(n_sweeps=args.sweeps, burn_in=args.burn, n_chains=args.chains)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fh:
        for rep in parse_reps(args.reps):
            rows = run_replicate(args.setting, args.n, args.noise, rep, cands, cfg, expand=args.expand,
                                 sensitivity=args.sensitivity, J=args.J, n_starts=args.starts, scaling=args.scaling, tau=args.tau,
                                 grow_rounds=args.grow, pooled=args.pooled, themes=args.themes,
                                 scale_invariant=args.scale_invariant)
            for r in rows:
                r["grammar_prior"] = prior_label
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            hier = [r for r in rows if r["method"] == "hier" and r["kind"] == "laplace"][0]
            print(f"[{args.setting} n={args.n} noise={args.noise} rep={rep}] "
                  f"hier recovery={hier['recovery']:.2f} lpd_out={hier['lpd_out']:.2f} "
                  f"t={hier['seconds_total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
