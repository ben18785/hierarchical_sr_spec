"""Held-out tables for one comparison run with two scores per prior:

  renorm    log P(form) renormalised over the run's common support (as in
            grammar_steps.py table)
  raw       log P(form) as the prior's own probability of the form's
            derivation, with no renormalisation (a proper distribution over
            all expressions of the prior's space)

The tree priors (fnprior, hpy, hpyv_*) share one space, fnprior's coded
trees (hsr.treecode), so their raw scores are directly comparable. The PCFG
grammars (cf, parent, ..., term_root) generate canonical derivations of a
different space, so a raw comparison across the two families is only
approximate. "uniform symbols" draws every node's symbol uniformly from the
13 symbols: it is improper (only 40% of its mass is on finite trees) and
is listed only as a reference point.

    python hpyv_table.py folds.json trees.json out.json out.md score.json ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr import treecode as tc  # noqa: E402

def main(folds_path, trees_path, out_json, out_md, *score_files):
    with open(folds_path) as fh:
        F = json.load(fh)
    with open(trees_path) as fh:
        trees = json.load(fh)["trees"]
    lp, fam = {}, {"uniform symbols": "tree"}
    for p in score_files:
        with open(p) as fh:
            d = json.load(fh)["logprior"]
        lp.update(d)
        # ours.json holds the PCFG grammars; every other score file a tree prior
        fam.update({k: "pcfg" if os.path.basename(p) == "ours.json" else "tree" for k in d})
    sup = F["support"]
    idx = {s: i for i, s in enumerate(sup)}
    lp["uniform symbols"] = [[-len(trees[k]) * np.log(len(tc.V)) if trees.get(k) else None for k in sup]
                             for _ in F["folds"]]
    rows = [{"prior": "uniform over support", "family": "-", "renorm": -float(np.log(len(sup))), "raw": None,
             "renorm_folds": [-float(np.log(len(sup)))] * len(F["folds"]), "raw_folds": [], "n_zero": 0}]
    for name, per_fold in lp.items():
        rn, rw, zeros, rnf, rwf, rws, rwn = [], [], 0, [], [], [], []
        for fold, v in zip(F["folds"], per_fold):
            # a test form is "seen" if its tree is the tree of some training form
            seen_trees = {json.dumps(trees.get(k)) for k in fold["train"]}
            seen = np.array([json.dumps(trees.get(k)) in seen_trees for k in fold["test"]])
            v = np.array([-np.inf if x is None else x for x in v], float)
            z = v - np.logaddexp.reduce(v[np.isfinite(v)])
            a = np.array([z[idx[k]] for k in fold["test"]])
            b = np.array([v[idx[k]] for k in fold["test"]])
            ok = np.isfinite(a)
            zeros += int((~ok).sum())
            rn += a[ok].tolist()
            rw += b[ok].tolist()
            rws += b[ok & seen].tolist()
            rwn += b[ok & ~seen].tolist()
            rnf.append(float(a[ok].mean()) if ok.any() else float("nan"))
            rwf.append(float(b[ok].mean()) if ok.any() else float("nan"))
        rows.append({"prior": name, "family": fam[name], "renorm": float(np.mean(rn)), "raw": float(np.mean(rw)),
                     "renorm_folds": rnf, "raw_folds": rwf, "n_zero": zeros,
                     "raw_seen": float(np.mean(rws)) if rws else None,
                     "raw_novel": float(np.mean(rwn)) if rwn else None,
                     "frac_novel": len(rwn) / max(1, len(rw))})
    rows.sort(key=lambda r: -(r["raw"] if r["raw"] is not None and r["family"] == "tree" and
                              r["prior"] != "uniform symbols" else -1e9 + r["renorm"]))
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=1)
    n_test = sum(len(f["test"]) for f in F["folds"])
    with open(out_md, "w") as fh:
        what = (f"tested on the {n_test} forms of {F['test_corpus']}" if F.get("test_corpus") else
                f"{F['k']}-fold CV by BioModels model (seed {F['seed']}), {n_test} held-out model-form pairs")
        fh.write(f"Held-out mean log probability per form; {what}. renorm: renormalised over a common support "
                 f"of {len(sup)} forms; raw: the prior's own log P (no renormalisation; tree priors share one "
                 f"space, PCFG grammars generate a different one).\n\n")
        fh.write("| prior | family | raw log P | renorm log P | raw spread over folds | zero-probability forms |\n"
                 "|---|---|---|---|---|---|\n")
        for r in rows:
            raw = "" if r["raw"] is None else f"{r['raw']:.3f}"
            fm = [x for x in r["raw_folds"] if x == x]
            spread = f"{min(fm):.2f} to {max(fm):.2f}" if fm else ""
            fh.write(f"| {r['prior']} | {r['family']} | {raw} | {r['renorm']:.3f} | {spread} | {r['n_zero']} |\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
