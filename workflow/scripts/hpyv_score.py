"""Score every support form of one comparison run under one variant of the
tree priors of hsr.pyp / hsr.adaptor, trained on each fold's training forms.
Same output format as fnprior_score.py (raw log P(tree) per support key, so
the table can report it both renormalised over the support and as is).

    python hpyv_score.py folds.json trees.json sources.json variant.json out.json sweeps keep

variant.json: {"name": ..., "model": "hpy" | "adaptor", "mode": "root" | "all",
ContextSpec keyword arguments (n, drop_L, drop_R, graph, depth_cap, source)}.

Sources: every training form carries the corpus it came from (BioModels for
the CV and transfer runs; for leave-one-source-out runs the labels are
rebuilt from sources.json and checked against the fold's training list).
Test forms are scored as coming from the run's test source; a source not
seen in training has empty source-specific restaurants, whose predictive
is that of the shared levels (the prior predictive of a new source).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr import treecheck  # noqa: E402
from hsr import treecode as tc  # noqa: E402
from hsr.adaptor import TreeAdaptor  # noqa: E402
from hsr.pyp import ContextSpec, TreeHPY  # noqa: E402

SPEC_ARGS = ("n", "drop_L", "drop_R", "graph", "depth_cap", "source")


def train_sources(F, fold, sources):
    """Source label of every training form, and the test source."""
    test = F.get("test_corpus") or ""
    if "train_sizes" in F:                       # leave one source out
        test_src = test.split(" ")[0]
        weighting = "equal" if "; equal)" in test else "pooled"
        parts = {n: sources[n] for n in F["train_sizes"]}
        if weighting == "equal":
            big = max(len(v) for v in parts.values())
            reps = {n: max(1, round(big / len(v))) for n, v in parts.items()}
        else:
            reps = {n: 1 for n in parts}
        keys, labels = [], []
        for n, v in parts.items():
            keys += v * reps[n]
            labels += [n] * (len(v) * reps[n])
        if keys != fold["train"]:
            raise ValueError("cannot rebuild the source labels of the training forms")
        return labels, test_src
    if test:                                      # transfer: train on BioModels, test on another corpus
        return ["biomodels"] * len(fold["train"]), test.replace("corpus_", "").replace(".json", "")
    return ["biomodels"] * len(fold["train"]), "biomodels"


def main(folds_path, trees_path, sources_path, variant_path, out_path, sweeps=20, keep=10):
    sweeps, keep = int(sweeps), int(keep)
    with open(folds_path) as fh:
        F = json.load(fh)
    with open(trees_path) as fh:
        T = {k: (tc.from_prefix(v) if v is not None else None) for k, v in json.load(fh)["trees"].items()}
    with open(sources_path) as fh:
        sources = json.load(fh)
    with open(variant_path) as fh:
        var = json.load(fh)
    name = var["name"]
    spec = ContextSpec(**{k: var[k] for k in SPEC_ARGS if k in var})
    support = F["support"]
    res = {"settings": {"variant": var, "spec": spec.describe(), "sweeps": sweeps, "keep": keep},
           "logprior": {name: []}, "diag": []}
    for f, fold in enumerate(F["folds"]):
        t0 = time.time()
        labels, test_src = train_sources(F, fold, sources)
        pairs = [(T[k], s) for k, s in zip(fold["train"], labels) if T.get(k) is not None]
        trees, srcs = [p[0] for p in pairs], [p[1] for p in pairs]
        if var.get("model", "hpy") == "adaptor":
            m = TreeAdaptor(spec, mode=var["mode"], seed=f).fit(trees, srcs, sweeps, keep)
            snaps = m.samples
            logp_s = lambda s, t: float(np.log(s.p_tree(tc.ROOT, test_src, t)))   # noqa: E731
            node_dist = lambda s: (lambda pos: s.node_dist(pos, test_src))         # noqa: E731
            key = lambda pos: (spec.key(pos, test_src), m.adapted(pos))             # noqa: E731
        else:
            m = TreeHPY(spec, seed=f).fit(trees, srcs, sweeps, keep)
            snaps = m.samples
            logp_s = lambda s, t: m.logp_sample(s, t, test_src)                    # noqa: E731
            node_dist = lambda s: m.cond_dist(s, test_src)                           # noqa: E731
            key = lambda pos: spec.key(pos, test_src)                                # noqa: E731
        fit_s = time.time() - t0
        lp = []
        for k in support:
            t = T.get(k)
            if t is None:
                lp.append(None)
                continue
            v = np.array([logp_s(s, t) for s in snaps])
            v = float(np.logaddexp.reduce(v) - np.log(len(v)))
            lp.append(v if np.isfinite(v) else None)
        res["logprior"][name].append(lp)
        # properness of the predictive at the test source: symbol distributions
        # sum to one in every reachable position, and P(finite tree)
        term = [treecheck.termination(node_dist(s), key) for s in (snaps[0], snaps[-1])]
        res["diag"].append({"fold": f, "n_train_trees": len(trees), "test_source": test_src,
                            "fit_seconds": fit_s, "hyper": m.hyper_summary(), "termination": term})
        print(name, "fold", f, f"{fit_s:.0f}s", "P(finite)", [round(x["p_finite"], 6) for x in term], flush=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main(*sys.argv[1:])
