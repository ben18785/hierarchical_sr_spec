"""Calibration and properness of one tree prior trained on every BioModels
form (the training set of the transfer runs). Runs in the fnprior
environment.

  * properness: every reachable position's symbol distribution sums to one
    (or the mass outside the one-variable vocabulary is reported) and P(the
    generated tree is finite) (hsr.treecheck.termination)
  * forward samples: node-count and depth distributions against the
    training forms; the fraction of sampled trees that are canonical codes
    (re-coding the sampled expression through fnprior's sympy parser gives
    back the same tree) and that contain the variable. Prior mass on
    non-canonical trees is mass that no corpus form can ever receive.

    python hpyv_calib.py folds.json trees.json variant.json out.json sweeps keep n_samples
    (variant.json {"name": "fnprior_kn"} for fnprior itself)
"""
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np
import sympy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr import treecheck  # noqa: E402
from hsr import treecode as tc  # noqa: E402
from hsr.adaptor import TreeAdaptor  # noqa: E402
from hsr.fnprior_tools import to_fn, train  # noqa: E402
from hsr.pyp import ContextSpec, TreeHPY  # noqa: E402

SPEC_ARGS = ("n", "drop_L", "drop_R", "graph", "depth_cap", "source")


def fnprior_dist(kp):
    code = kp.coder.code
    cv = [code[s] for s in tc.V]
    none = code["None"]
    cache = {}

    def c(s):
        return none if s is None else code[s]

    def nd(pos):
        if pos.side == "L":
            ctx = (c(pos.anc[-2]), c(pos.anc[-1]))
            m = kp.backoff_left
        else:
            ctx = (c(pos.anc[-2]), c(pos.anc[-1]), c(pos.sib))
            m = kp.backoff_right
        if ctx not in cache:
            cache[(pos.side, ctx)] = np.array([m.get_pbo(w, ctx) for w in cv])
        return [], 1.0, cache[(pos.side, ctx)]

    def key(pos):
        return (pos.side, pos.anc[-2:], pos.sib if pos.side == "R" else None)

    return nd, key


def hist(xs, top):
    c = Counter(min(x, top) for x in xs)
    n = len(xs)
    return [c.get(i, 0) / n for i in range(1, top + 1)]


def main(folds_path, trees_path, variant_path, out_path, sweeps=20, keep=10, n_samples=4000):
    sweeps, keep, n_samples = int(sweeps), int(keep), int(n_samples)
    with open(folds_path) as fh:
        F = json.load(fh)
    with open(trees_path) as fh:
        Tr = json.load(fh)["trees"]
    with open(variant_path) as fh:
        var = json.load(fh)
    keys = F["folds"][0]["train"]
    trees = [tc.from_prefix(Tr[k]) for k in keys if Tr.get(k)]
    rng = np.random.default_rng(0)
    out = {"variant": var, "n_train": len(trees)}
    with tempfile.TemporaryDirectory() as tmp:
        kp = train(keys, False, 2, tmp, "calib")
    if var["name"].startswith("fnprior"):
        nd, key = fnprior_dist(kp)
        dists, keyf = [nd], key
        # fnprior's own conditionals over its full vocabulary (with the second
        # variable y) sum to one in every context; P(y) is mass outside the
        # one-variable space
        code = kp.coder.code
        ops = [None] + list(tc.UNARY + tc.BINARY)
        cd = lambda s: code["None"] if s is None else code[s]    # noqa: E731
        dev, py = [], []
        for g in ops:
            for p in ops:
                dev.append(abs(sum(kp.backoff_left.get_pbo(w, (cd(g), cd(p))) for w in kp.vocab_left) - 1))
            for p in tc.BINARY:
                for sib in tc.V:
                    ctx = (cd(g), cd(p), cd(sib))
                    dev.append(abs(sum(kp.backoff_right.get_pbo(w, ctx) for w in kp.vocab_right) - 1))
                    py.append(kp.backoff_right.get_pbo(code["y"], ctx))
        out["fnprior_full_vocab_max_abs_sum_minus_1"] = float(max(dev))
        out["fnprior_max_p_y"] = float(max(py))
    else:
        spec = ContextSpec(**{k: var[k] for k in SPEC_ARGS if k in var})
        src = ["biomodels"] * len(trees)
        if var.get("model", "hpy") == "adaptor":
            m = TreeAdaptor(spec, mode=var["mode"], seed=0).fit(trees, src, sweeps, keep)
            dists = [(lambda s: (lambda pos: s.node_dist(pos, "biomodels")))(s) for s in m.samples]
            keyf = lambda pos: (spec.key(pos, "biomodels"), m.adapted(pos))      # noqa: E731
        else:
            m = TreeHPY(spec, seed=0).fit(trees, src, sweeps, keep)
            dists = [m.cond_dist(s, "biomodels") for s in m.samples]
            keyf = lambda pos: spec.key(pos, "biomodels")                         # noqa: E731
    out["termination"] = [treecheck.termination(d, keyf) for d in dists[:: max(1, len(dists) // 3)]]
    samples = [treecheck.sample_tree(dists[i % len(dists)], rng, cap=200) for i in range(n_samples)]
    fin = [t for t in samples if t is not None]
    out["frac_finite_or_in_vocab"] = len(fin) / n_samples
    # canonical round trip through fnprior's parser
    x = sympy.symbols([f"x{i}" for i in range(2)], real=True)
    a = sympy.symbols([f"a{i}" for i in range(200)], real=True)
    locs = {"x0": x[0], **{f"a{i}": a[i] for i in range(200)}}
    canon = {}
    for t in set(fin):
        try:
            canon[t] = tc.from_labels(kp.coder.equation2labels(tc.to_string(t), locs)) == t
        except Exception:
            canon[t] = False
    out["frac_canonical"] = float(np.mean([canon[t] for t in fin])) if fin else float("nan")
    out["frac_has_x"] = float(np.mean([tc.has_x(t) for t in fin])) if fin else float("nan")
    out["frac_canonical_with_x"] = float(np.mean([canon[t] and tc.has_x(t) for t in fin])) if fin else float("nan")
    out["frac_in_train"] = float(np.mean([t in set(trees) for t in fin])) if fin else float("nan")
    top = 30
    out["size_hist_prior"] = hist([tc.size(t) for t in fin], top)
    out["size_hist_train"] = hist([tc.size(t) for t in trees], top)
    out["depth_hist_prior"] = hist([tc.depth(t) for t in fin], 15)
    out["depth_hist_train"] = hist([tc.depth(t) for t in trees], 15)
    for nm, f in (("size", tc.size), ("depth", tc.depth)):
        p, q = [f(t) for t in fin], [f(t) for t in trees]
        out[f"{nm}_mean_prior"], out[f"{nm}_mean_train"] = float(np.mean(p)), float(np.mean(q))
        out[f"{nm}_q90_prior"], out[f"{nm}_q90_train"] = float(np.quantile(p, 0.9)), float(np.quantile(q, 0.9))
    print(json.dumps({k: v for k, v in out.items() if "hist" not in k}, indent=1))
    with open(out_path, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main(*sys.argv[1:])
