"""Hierarchical symbolic regression on real grouped data.

    python -m hsr.realdata --data hsr/data/real/indometh.csv --split full --out r.jsonl
    python -m hsr.realdata --data hsr/data/real/forage_replaced.csv --scaling per_dataset \
        --split interp:0:3 --grow 1 --themes --out r.jsonl

The CSV has columns group,x,y (one row per observation). There is no ground
truth, so methods are compared by held-out log predictive density:

  --split full         fit every point; report the forms found (no held-out score)
  --split interp:k:K   per group, the interior distinct x values (not the smallest
                       or largest) are dealt into K folds; fold k is held out
  --split extrap       per group, the points at the largest x are held out

Methods: hier (Laplace and BIC), hier_pooled, hier_themes (with --themes),
no_pool_bma, complete_pool (one form for all groups, constants per group),
single_curve (one form and one set of constants: all points as one dataset),
and ``reference``: the hierarchical model restricted to the
textbook form(s) for the dataset (REFERENCE below), i.e. what a modeller who
fixed the literature form would get.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np

from . import baselines
from .candidates import CandidateSet, default_n_jobs, edit_graph, expand_candidates, from_skeletons
from .canonical import canonicalise
from .experiment import _COMPILED, gibbs_summary
from .equivalence import Equivalence, class_summary
from .fixed_prior import FixedPrior, fnprior_log_prior, with_log_prior
from .gibbs import GibbsConfig, run_gibbs
from .grammar import Grammar
from .likelihood import LikelihoodTable, compute_likelihoods, predictive_logpdf, scale_population
from .constraints import Spec, apply_truncation, check_grid, constrained_candidates_path, prepare, satisfied, unscale_y
from .experiment import _compiled
from .pooled import PooledModel
from .themes import EditKernel, run_gibbs_themes

# Textbook forms (Pinheiro & Bates 2000 for the R datasets; Holling types I-III
# for functional responses), with c<k> constants (a repeated index is shared).
REFERENCE = {
    "indometh": ["c0*exp(c1*x1) + c2*exp(c3*x1)"],        # bi-exponential
    "theoph": ["c0*(exp(c1*x1) - exp(c2*x1))"],           # first-order absorption and elimination
    "orange": ["c0/(1 + c1*exp(c2*x1))"],                 # logistic
    "dnase": ["c0*x1**c1/(c2 + x1**c1)"],                 # logistic in log(conc) = Hill in conc
    "co2": ["c0*exp(c1*x1) + c2"],                        # asymptotic regression with offset
    "loblolly": ["c0*exp(c1*x1) + c2"],                   # asymptotic regression
    "forage_replaced": ["c0*x1", "c0*x1/(c1 + x1)", "c0*x1**2/(c1 + x1**2)"],  # Holling I, II, III
}


def reference_skeletons(forms):
    return [canonicalise(f) for f in forms]


# ------------------------------------------------------------------- data
def load_groups(path):
    groups = OrderedDict()
    with open(path) as fh:
        for r in csv.DictReader(fh):
            x, y = float(r["x"]), float(r["y"])
            if np.isfinite(x) and np.isfinite(y):
                groups.setdefault(r["group"], []).append((x, y))
    return OrderedDict((g, (np.array([p[0] for p in v]), np.array([p[1] for p in v]))) for g, v in groups.items())


def split_group(x, y, split):
    """(fit mask, held-out mask) for one group."""
    held = np.zeros(len(x), bool)
    ux = np.unique(x)
    if split == "full":
        pass
    elif split == "extrap":
        if len(ux) >= 5:
            held = x == ux[-1]
    elif split.startswith("interp"):
        _, k, K = split.split(":")
        k, K = int(k), int(K)
        interior = ux[1:-1]
        chosen = interior[np.arange(len(interior)) % K == k]
        if len(ux) - len(chosen) >= 4:
            held = np.isin(x, chosen)
    else:
        raise ValueError(split)
    return ~held, held


def pad(arrs):
    n = max(len(a) for a in arrs)
    out = np.zeros((len(arrs), n))
    M = np.zeros((len(arrs), n))
    for j, a in enumerate(arrs):
        out[j, :len(a)] = a
        M[j, :len(a)] = 1.0
    return out, M


# --------------------------------------------------------------- scoring
_ACCEPT = None   # set by run() when a property spec truncates the prior
_EQUIV = None    # behaviour classes for reporting (hsr.equivalence), set by run()


def heldout_lpd(probs, skeletons, table, held, sx, sy, min_prob=1e-3, n_draws=200):
    """Pointwise log predictive density of the held-out points, mixing over the
    skeletons with posterior mass. Returns (mean over all held-out points,
    per-group means (None where a group has no held-out point))."""
    from scipy.special import logsumexp
    from .experiment import _compiled
    per_group, allv = [], []
    for j, (xh, yh) in enumerate(held):
        if len(xh) == 0:
            per_group.append(None)
            continue
        row = probs[j]
        idx = np.flatnonzero(row >= min_prob)
        if len(idx) == 0:
            idx = np.array([int(np.argmax(row))])
        w = row[idx] / row[idx].sum()
        comps = []
        for e in idx:
            cs = _compiled(skeletons[e])
            acc = None if _ACCEPT is None else (lambda ph, cs=cs, j=j: _ACCEPT(cs, j, ph))
            comps.append(predictive_logpdf(cs, xh, yh, table.phi[e][j], table.log_sigma[j, e], table.cov[e][j],
                                           sx[j], sy[j], n_draws=n_draws, seed=j, accept=acc))
        comps = np.array(comps)
        v = logsumexp(comps + np.log(w)[:, None], axis=0)
        per_group.append(float(np.mean(v)))
        allv += v.tolist()
    return (float(np.mean(allv)) if allv else None), per_group


def summarise(name, probs, skeletons, table, held, sx, sy, groups, pop_weight=None, extras=None, top_n=15,
              res=None, group_top=5, group_min=0.01):
    """One output row. Besides point summaries it records posterior
    uncertainty: each group's posterior over forms (up to ``group_top`` forms
    with probability >= ``group_min``), and for sampler results the (J, J)
    co-assignment matrix and the posterior over the number of forms."""
    top = np.argmax(probs, axis=1)
    out = {"method": name, "top_keys": [skeletons[e].key for e in top],
           "top_prob": [float(probs[j, e]) for j, e in enumerate(top)], "groups": list(groups)}
    gp = []
    for j in range(probs.shape[0]):
        idx = np.argsort(probs[j])[::-1][:group_top]
        gp.append([[skeletons[e].key, round(float(probs[j, e]), 4)] for e in idx if probs[j, e] >= group_min])
    out["group_posterior"] = gp
    if res is not None:
        out["coassignment"] = np.round(res.coassignment(), 3).tolist()
        out["n_forms_posterior"] = {int(k): round(v, 4) for k, v in res.n_forms_distribution().items()}
    if pop_weight is None:
        pop_weight = probs.mean(axis=0)
    order = np.argsort(pop_weight)[::-1][:top_n]
    out["pop_weight"] = {skeletons[e].key: float(pop_weight[e]) for e in order if pop_weight[e] > 1e-3}
    out["lpd_heldout"], out["lpd_group"] = heldout_lpd(probs, skeletons, table, held, sx, sy)
    if _EQUIV is not None:
        t_cls = time.time()
        out.update(class_summary(probs, _EQUIV.classes(probs), skeletons,
                                 z_samples=None if res is None else res.z_samples))
        out.update(_EQUIV.end_summary(probs))
        out["seconds_classes"] = time.time() - t_cls
    if extras:
        out.update(extras)
    return out


# ------------------------------------------------------------------- run
def run(data, split, cands: CandidateSet, cfg: GibbsConfig, reference=(), scaling="pooled", n_starts=10, tau=3.0,
        grow_rounds=0, themes=False, theme_top_k=10, theme_min_mass=0.02, seed=0, scale_invariant=False, spec=None,
        classes=True, prior=None):
    global _ACCEPT, _EQUIV
    t0 = time.time()
    groups = load_groups(data)
    names = list(groups)
    fit, held = [], []
    for g in names:
        x, y = groups[g]
        mf, mh = split_group(x, y, split)
        fit.append((x[mf], y[mf]))
        held.append((x[mh], y[mh]))
    Xs, Ys, sx, sy = scale_population([f[0] for f in fit], [f[1] for f in fit], scaling)
    if spec is not None and spec.fixes_y_scale:
        Ys, sy = unscale_y(Ys, sy)
    X, M = pad([x[:, 0] for x in Xs])
    Y, _ = pad(Ys)
    X = X[..., None]
    J = len(names)
    grammar = cands.grammar
    base, admit = prepare(list(cands.skeletons), spec, scale_invariant)
    cset = from_skeletons(base, grammar, dict(cands.stats))
    cset.admit = admit
    lik = lambda sks, s: compute_likelihoods(sks, X, Y, M, n_starts=n_starts, tau=tau, seed=s,
                                             compiled_cache=_COMPILED)
    table = lik(cset.skeletons, seed)
    n_added = 0
    for r in range(grow_rounds):
        post = baselines.no_pooling_posterior(table.laplace, cset.counts, grammar)
        top = set()
        for j in range(J):
            top.update(np.argsort(post[j])[-3:].tolist())
        new_idx = expand_candidates(cset, sorted(top))
        if not new_idx:
            break
        table = table.extend(lik([cset.skeletons[i] for i in new_idx], seed + 7 + r))
        n_added += len(new_idx)
    edges = set()
    if themes:
        post = baselines.no_pooling_posterior(table.laplace, cset.counts, grammar)
        sources = set()
        for j in range(J):
            sources.update(np.argsort(post[j])[-theme_top_k:].tolist())
        pm0 = PooledModel(table, [sk.n_consts for sk in cset.skeletons], tau_ref=tau, tau_mu=tau)
        r0 = run_gibbs(table.laplace, cset.counts, grammar, replace(cfg, n_sweeps=300, burn_in=100, n_chains=2),
                       model=pm0)
        sources.update(np.flatnonzero(r0.assign_probs.max(axis=0) >= theme_min_mass).tolist())
        new_idx, edges = edit_graph(cset, sorted(sources), n_jobs=default_n_jobs())
        if new_idx:
            table = table.extend(lik([cset.skeletons[i] for i in new_idx], seed + 31))
    # reference forms join C (as a modeller's prior guesses, they are fair game for every method)
    ref_sks = reference_skeletons(reference)
    have = {k: i for i, k in enumerate(cset.keys)}
    missing = [s for s in ref_sks if s.key not in have]
    if missing:
        before = len(cset.keys)
        cset.add(missing)
        table = table.extend(lik(cset.skeletons[before:], seed + 11))
        have = {k: i for i, k in enumerate(cset.keys)}
    ref_idx = sorted({have[s.key] for s in ref_sks})
    sks = list(cset.skeletons)
    cnt = cset.counts
    n_C = len(sks)
    if prior == "fnprior":
        # fixed corpus-derived base measure (hsr.fixed_prior); candidate growth above used the grammar
        cnt = with_log_prior(cnt, fnprior_log_prior([s.key for s in sks]))
        grammar = FixedPrior(grammar, "fnprior")
    _ACCEPT = None
    trunc_logq = None
    if spec is not None:
        trunc_logq = apply_truncation(table, sks, X, M, spec, tau, sy, _compiled, seed=seed)
        grids = [check_grid(X[j][M[j] > 0, 0]) for j in range(J)]
        _ACCEPT = lambda cs, j, ph: satisfied(cs, ph, grids[j], spec, sy[j])
    # single-curve baseline: all points as one dataset (one form, one set of constants)
    X1 = X[M > 0][None]
    Y1 = Y[M > 0][None]
    table1 = compute_likelihoods(sks, X1, Y1, n_starts=n_starts, tau=tau, seed=seed + 97, compiled_cache=_COMPILED)
    if spec is not None:
        apply_truncation(table1, sks, X1, np.ones(Y1.shape), spec, tau, [sy[0]], _compiled, seed=seed)
    tab1 = baselines.broadcast_table(table1, J)
    t_lik = time.time() - t0
    _EQUIV = Equivalence(sks, table, X, M > 0, compiled=_compiled, seed=seed) if classes else None

    rows = []
    for kind in ("laplace", "bic"):
        L = table.matrix(kind)
        res = run_gibbs(L, cnt, grammar, cfg)
        rows.append(summarise("hier", res.assign_probs, sks, table, held, sx, sy, names, res.pop_weight,
                              gibbs_summary(res, [s.key for s in sks]), res=res) | {"kind": kind})
        P = baselines.no_pooling_posterior(L, cnt, grammar)
        rows.append(summarise("no_pool_bma", P, sks, table, held, sx, sy, names) | {"kind": kind})
        Pc = baselines.complete_pooling_posterior(L, cnt, grammar)
        rows.append(summarise("complete_pool", np.tile(Pc, (J, 1)), sks, table, held, sx, sy, names, Pc)
                    | {"kind": kind})
        P1 = baselines.single_curve_posterior(table1.matrix(kind)[0], cnt, grammar)
        rows.append(summarise("single_curve", np.tile(P1, (J, 1)), sks, tab1, held, sx, sy, names, P1)
                    | {"kind": kind})
        if ref_idx:
            ores, _ = baselines.oracle(L, cnt, grammar, ref_idx, cfg)
            Pf = np.zeros((J, n_C))
            Pf[:, ref_idx] = ores.assign_probs
            pw = np.zeros(n_C)
            pw[ref_idx] = ores.pop_weight
            rows.append(summarise("reference", Pf, sks, table, held, sx, sy, names, pw) | {"kind": kind})
        if kind == "laplace":
            pm = PooledModel(table, [s.n_consts for s in sks], tau_ref=tau, tau_mu=tau, trunc_logq=trunc_logq)
            resp = run_gibbs(L, cnt, grammar, cfg, model=pm)
            rows.append(summarise("hier_pooled", resp.assign_probs, sks, table, held, sx, sy, names,
                                  resp.pop_weight, gibbs_summary(resp, [s.key for s in sks]), res=resp) | {"kind": kind})
            if themes:
                kernel = EditKernel(n_C, edges, log_w=grammar.log_prob(cnt, grammar.prior_mean()))
                rest = run_gibbs_themes(L, cnt, grammar, kernel, cfg, model=pm)
                ext = gibbs_summary(rest, [s.key for s in sks]) | {"theme_top_keys": [sks[t].key for t in rest.theme_top],
                                             "n_themes_mean": float(rest.n_themes.mean()),
                                             "rho_mean": float(rest.rho.mean())}
                rows.append(summarise("hier_themes", rest.assign_probs, sks, table, held, sx, sy, names,
                                      rest.pop_weight, ext, res=rest) | {"kind": kind})
    meta = {"data": os.path.basename(data), "split": split, "J": J, "n_fit": int(M.sum()),
            "n_heldout": int(sum(len(h[0]) for h in held)), "scaling": scaling, "tau": tau,
            "grow_rounds": grow_rounds, "n_added_by_edits": n_added, "themes": themes, "n_candidates": n_C,
            "scale_invariant": scale_invariant, "constraints": None if spec is None else spec.label,
            "reference_keys": [sks[i].key for i in ref_idx],
            "seconds_likelihood": t_lik, "seconds_total": time.time() - t0}
    for r in rows:
        r.update(meta)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", nargs="*", default=None,
                    help="textbook forms (default: REFERENCE[<csv name>])")
    ap.add_argument("--candidates", default=os.path.join(os.path.dirname(__file__), "data", "candidates.json"))
    ap.add_argument("--scaling", default="pooled", choices=["pooled", "per_dataset", "none"])
    ap.add_argument("--sweeps", type=int, default=1000)
    ap.add_argument("--burn", type=int, default=250)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--temps", type=int, default=1, help="> 1: non-reversible parallel tempering replicas per chain")
    ap.add_argument("--split-merge", type=int, default=3, help="split-merge proposals per sweep")
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--grow", type=int, default=0)
    ap.add_argument("--themes", action="store_true")
    ap.add_argument("--scale-invariant", action="store_true",
                    help="keep only skeletons whose constants absorb changes of units in x and y (hsr.properties)")
    ap.add_argument("--constrain", default=None,
                    help='known properties of f, e.g. "range=0,1;limit_inf=1;monotone=increasing" (hsr.constraints)')
    ap.add_argument("--corpus", default=None, help="corpus JSON: corpus-informed grammar prior")
    ap.add_argument("--kappa", type=float, default=50.0)
    ap.add_argument("--prior", default="grammar", choices=["grammar", "fnprior"],
                    help="DP base measure: the learned grammar (default) or fnprior trained on the BioModels corpus")
    args = ap.parse_args(argv)
    grammar, prior_label = None, "default"
    if args.corpus:
        from .corpus import corpus_alpha, load as load_corpus
        grammar = Grammar(alpha=corpus_alpha(load_corpus(args.corpus), Grammar(), kappa=args.kappa))
        prior_label = f"corpus_k{args.kappa:g}"
    if args.prior == "fnprior":
        prior_label = "fnprior_biomodels"
    spec = Spec.parse(args.constrain)
    if spec is not None and spec.exact and args.candidates == ap.get_default("candidates"):
        args.candidates = constrained_candidates_path(spec)
    cands = CandidateSet.from_json(args.candidates, grammar)
    cfg = GibbsConfig(n_sweeps=args.sweeps, burn_in=args.burn, n_chains=args.chains, n_temps=args.temps,
                      split_merge=args.split_merge)
    name = os.path.splitext(os.path.basename(args.data))[0]
    reference = args.reference if args.reference is not None else REFERENCE.get(name, [])
    rows = run(args.data, args.split, cands, cfg, reference, args.scaling, args.starts, args.tau, args.grow,
               args.themes, scale_invariant=args.scale_invariant, spec=spec,
               prior=None if args.prior == "grammar" else args.prior)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as fh:
        for r in rows:
            r["grammar_prior"] = prior_label
            fh.write(json.dumps(r) + "\n")
    for r in rows:
        lp = r["lpd_heldout"]
        print(f"[{name} {args.split}] {r['method']:12s} {r['kind']:7s} lpd={'-' if lp is None else f'{lp:.3f}'} "
              f"top={sorted(r['pop_weight'].items(), key=lambda kv: -kv[1])[:3]}", flush=True)


if __name__ == "__main__":
    main()
