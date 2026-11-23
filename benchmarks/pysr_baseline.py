"""PySR baseline: non-hierarchical symbolic regression fitted to each dataset
separately, on exactly the populations / splits the hsr methods see.

    # synthetic families (same seeds as hsr.experiment)
    python benchmarks/pysr_baseline.py --setting saturation_mixture --n 20 --noise 0.05 --reps 0:5 --out p.jsonl
    # real data (same splits as hsr.realdata)
    python benchmarks/pysr_baseline.py --data hsr/data/real/indometh.csv --split interp:0:3 --out p.jsonl

PySR uses the same operator set as the grammar and the same scaling as the hsr
run. Its chosen equation (model_selection="best") is turned into a skeleton:
fitted numbers within 1e-6 of an integer stay numbers, every other number
becomes a free constant, and the result is canonicalised, so form recovery is
scored exactly as for the hsr methods. PySR gives point predictions, so its
held-out log density is a plug-in Gaussian with sigma = in-sample residual SD
(labelled ``lpd_*_plugin``); it is optimistic relative to a full predictive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hsr.canonical import canonicalise  # noqa: E402
from hsr.likelihood import LPD_FLOOR  # noqa: E402


def pysr_model(seed, niterations, maxsize):
    from pysr import PySRRegressor
    return PySRRegressor(
        niterations=niterations, maxsize=maxsize, populations=15,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["exp", "log", "sqrt", "sin", "tanh"],
        constraints={"^": (-1, 1)},
        nested_constraints={"exp": {"exp": 1}, "log": {"log": 0}, "tanh": {"tanh": 0}, "sin": {"sin": 0}},
        model_selection="best", random_state=seed, deterministic=True, parallelism="serial",
        progress=False, verbosity=0, temp_equation_file=True)


def to_key(expr):
    """PySR sympy expression -> canonical skeleton key (or None)."""
    x1 = [s for s in expr.free_symbols if s.name == "x1"]
    k = [0]
    reps = {}
    for f in expr.atoms(sp.Float):
        v = float(f)
        if abs(v - round(v)) < 1e-6:
            reps[f] = sp.Integer(round(v))
        else:
            reps[f] = sp.Symbol(f"c{k[0]}", real=True)
            k[0] += 1
    e = expr.xreplace(reps)
    if not x1:
        return None
    try:
        sk = canonicalise(str(e), 1, max_consts=None)
    except Exception:
        return None
    return None if sk is None else sk.key


def fit_one(x, y, seed, niterations, maxsize):
    m = pysr_model(seed, niterations, maxsize)
    m.fit(x.reshape(-1, 1), y, variable_names=["x1"])
    expr = m.sympy()
    resid = y - m.predict(x.reshape(-1, 1))
    sigma = float(np.sqrt(np.mean(resid ** 2))) or 1e-12
    return m, expr, sigma


def plugin_lpd(m, sigma, x, y, y_scale):
    """Pointwise log density in original units (x, y scaled), floored as in
    hsr.likelihood.predictive_logpdf."""
    if len(x) == 0:
        return None
    with np.errstate(all="ignore"):
        mu = m.predict(x.reshape(-1, 1))
        v = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((y - mu) / sigma) ** 2 - np.log(y_scale)
    v = np.where(np.isfinite(v), np.maximum(v, LPD_FLOOR), LPD_FLOOR)
    return v.tolist()


def run_synthetic(args):
    from hsr.candidates import CandidateSet
    from hsr.experiment import SETTING_SEED, make_population, scaled_arrays
    from hsr.families import family_metrics
    cands = CandidateSet.from_json(os.path.join(os.path.dirname(__file__), "..", "hsr", "data", "candidates.json"))
    a, b = map(int, args.reps.split(":"))
    for rep in range(a, b):
        t0 = time.time()
        seed = 12345 + 1_000_003 * rep + int(1000 * args.noise) * 7919 + args.n * 104729 + SETTING_SEED[args.setting]
        pop = make_population(args.setting, np.random.default_rng(abs(seed)), args.n, args.noise, cands, args.J)
        X, Y, sx, sy = scaled_arrays(pop, "pooled")
        keys, lin, lout, exprs = [], [], [], []
        for j, ds in enumerate(pop.datasets):
            m, expr, sig = fit_one(X[j][:, 0], Y[j], rep * 1000 + j, args.niterations, args.maxsize)
            exprs.append(str(expr))
            keys.append(to_key(expr))
            # held-out points in original units -> scaled
            for which, store in (("in", lin), ("out", lout)):
                xn, yn = (ds.x_in, ds.y_in) if which == "in" else (ds.x_out, ds.y_out)
                v = plugin_lpd(m, sig, np.asarray(xn, float).ravel() / sx[j][0], np.asarray(yn, float) / sy[j], sy[j])
                store.append(float(np.mean(v)))
        row = {"method": "pysr", "kind": "pysr", "setting": args.setting, "n": args.n, "noise": args.noise,
               "rep": rep, "J": args.J, "top_keys": keys, "pysr_exprs": exprs,
               "recovery": float(np.mean([k == t for k, t in zip(keys, pop.true_keys)])),
               "lpd_in_plugin": float(np.mean(lin)), "lpd_out_plugin": float(np.mean(lout)),
               "niterations": args.niterations, "maxsize": args.maxsize, "seconds_total": time.time() - t0}
        if hasattr(pop, "parent_key"):
            row.update(family_metrics(pop, [k or "" for k in keys]))
        yield row


def run_real(args):
    from hsr.likelihood import scale_population
    from hsr.realdata import load_groups, split_group
    groups = load_groups(args.data)
    names = list(groups)
    fit, held = [], []
    for g in names:
        x, y = groups[g]
        mf, mh = split_group(x, y, args.split)
        fit.append((x[mf], y[mf]))
        held.append((x[mh], y[mh]))
    Xs, Ys, sx, sy = scale_population([f[0] for f in fit], [f[1] for f in fit], args.scaling)
    t0 = time.time()
    keys, exprs, lpd_group, allv = [], [], [], []
    for j in range(len(names)):
        m, expr, sig = fit_one(Xs[j][:, 0], Ys[j], j, args.niterations, args.maxsize)
        exprs.append(str(expr))
        keys.append(to_key(expr))
        v = plugin_lpd(m, sig, held[j][0] / sx[j][0], held[j][1] / sy[j], sy[j])
        if v is None:
            lpd_group.append(None)
        else:
            lpd_group.append(float(np.mean(v)))
            allv += v
    yield {"method": "pysr", "kind": "pysr", "data": os.path.basename(args.data), "split": args.split,
           "scaling": args.scaling, "groups": names, "top_keys": keys, "pysr_exprs": exprs,
           "lpd_heldout_plugin": float(np.mean(allv)) if allv else None, "lpd_group_plugin": lpd_group,
           "niterations": args.niterations, "maxsize": args.maxsize, "seconds_total": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--reps", default="0:5")
    ap.add_argument("--J", type=int, default=30)
    ap.add_argument("--data")
    ap.add_argument("--split", default="full")
    ap.add_argument("--scaling", default="pooled")
    ap.add_argument("--niterations", type=int, default=40)
    ap.add_argument("--maxsize", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    gen = run_real(args) if args.data else run_synthetic(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for row in gen:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print({k: row[k] for k in row if k in ("setting", "data", "split", "rep", "recovery", "family_recovery",
                                                 "lpd_in_plugin", "lpd_heldout_plugin", "seconds_total")}, flush=True)


if __name__ == "__main__":
    main()
