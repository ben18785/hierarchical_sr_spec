"""Paper-1 data extract (rule paper1_data; hsr environment).

Reads the per-run outputs of the LOSO, hold-out and temporal runs (folds,
score files and tables; gitignored, rebuilt by the workflow or downloaded
from GitHub Actions) and the corpus files, and writes small JSON files under
results/paper1/data/ (committed). Every table and figure of paper 1 is drawn
from these and from the committed reports (results/grammar/*.json) by
paper1_report.py, so the plots can be restyled without the run directories.

    python paper1_data.py OUTDIR --gen generalised.json --trees trees.json \\
        --corpora role:name=path ... --holdout trees_holdout.json run_dir ... \\
        --temporal trees.json temporal.json run_dir ... --loso run_dir ... \\
        --variants v1 v2 ... --calib calib_fnprior_kn.json

Outputs
  corpus.json       per source (training or hold-out): models with at least one
                    form, model-form pairs, distinct generalised forms (indices
                    into ``keys``) and the node count of every form
  logp_size.json    raw log P by node count (n, mean, sd per node count), seen
                    and novel pairs, per prior and run (hold-out and temporal);
                    per-pair records of the hold-out runs
  loso.json         per left-out source and weighting: pairs, fraction novel and
                    every prior's raw log P on all / seen / novel pairs
  temporal.json     per temporal run: test-set sizes and every prior's raw log P
                    (all test pairs and without ModelDB), gap diagnostics
  diagnostics.json  per tree-prior variant and run: fit time, P(finite tree) at
                    the first and last kept snapshot, adaptor acceptance per
                    sweep, final hyperparameters; Teh's HPY hyperparameters;
                    fnprior's termination check
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr.grammar_compare import source_items  # noqa: E402


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=None, separators=(",", ":"), sort_keys=False)
        fh.write("\n")


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def split_options(argv):
    """{option: [values]} for ``--name v1 v2 ...`` style arguments."""
    out, cur = {}, None
    for a in argv:
        if a.startswith("--"):
            cur = a[2:]
            out[cur] = []
        else:
            out[cur].append(a)
    return out


def run_scores(run_dir, variants):
    """{prior: (family, raw log P per support key)}; hpyv_ or hpyvt_ variant files."""
    out = {}
    pre = "hpyvt_" if os.path.exists(os.path.join(run_dir, f"hpyvt_{variants[0]}.json")) else "hpyv_"
    files = [("ours.json", "pcfg"), ("fnprior.json", "tree"), ("hpy.json", "tree")]
    files += [(f"{pre}{v}.json", "tree") for v in variants]
    for name, fam in files:
        for prior, per_fold in _load(os.path.join(run_dir, name))["logprior"].items():
            if len(per_fold) != 1:
                raise ValueError(f"{run_dir}/{name}: expected one fold")
            out[prior] = (fam, np.array([-np.inf if x is None else x for x in per_fold[0]], float))
    return out, pre


def by_node(logp, nodes, seen):
    """{"seen"|"novel": {node count: [n, mean, sd]}} over finite pairs with a known size."""
    out = {}
    for lab, m in (("seen", seen), ("novel", ~seen)):
        d = {}
        ok = m & np.isfinite(logp) & np.isfinite(nodes)
        for n in np.unique(nodes[ok]).astype(int):
            v = logp[ok & (nodes == n)]
            d[str(n)] = [int(v.size), _r(v.mean()), _r(v.std(ddof=1) if v.size > 1 else 0.0)]
        out[lab] = d
    return out


def corpus_data(gen_path, trees_path, specs):
    gen = _load(gen_path)
    trees = _load(trees_path)["trees"]
    keys, idx, sources = [], {}, {}
    for spec in specs:
        role, rest = spec.split(":", 1)
        name, path = rest.split("=", 1)
        pm = _load(path)["per_model"]
        items = source_items(path, gen)
        forms = sorted(set(items))
        for k in forms:
            if k not in idx:
                idx[k] = len(keys)
                keys.append(k)
        sources[name] = {"role": role, "models": sum(1 for ks in pm.values() if ks), "pairs": len(items),
                         "distinct": len(forms), "forms": sorted(idx[k] for k in forms)}
    nodes = [len(trees[k]) if trees.get(k) else None for k in keys]
    return {"sources": sources, "keys": keys, "nodes": nodes,
            "note": "forms: indices into keys; nodes: tree node count (hsr.treecode) of each key, null if not coded"}


def holdout_data(trees_path, run_dirs, variants):
    trees = _load(trees_path)["trees"]
    res = {}
    pairs = {}
    for d in run_dirs:
        F = _load(os.path.join(d, "folds.json"))
        w = "equal" if F["test_corpus"].endswith("; equal)") else "pooled"
        sc, _ = run_scores(d, variants)
        fold = F["folds"][0]
        idx = {s: i for i, s in enumerate(F["support"])}
        train = set(fold["train"])
        test = fold["test"]
        ti = np.array([idx[k] for k in test])
        seen = np.array([k in train for k in test], bool)
        nodes = np.array([len(trees[k]) if trees.get(k) else np.nan for k in test], float)
        res[w] = {p: {"family": fam, **by_node(v[ti], nodes, seen)} for p, (fam, v) in sc.items()}
        pairs[w] = {"source": fold["test_sources"], "nodes": [None if np.isnan(n) else int(n) for n in nodes],
                    "seen": seen.tolist(),
                    "logp": {p: [_r(x) for x in v[ti]] for p, (fam, v) in sc.items() if fam == "tree"}}
    return res, pairs


def temporal_data(trees_path, report_path, run_dirs, variants):
    trees = _load(trees_path)["trees"]
    rep = _load(report_path)["runs"]
    size, summ = {}, {}
    for d in run_dirs:
        F = _load(os.path.join(d, "folds.json"))
        run = f"{F['cutoff']}_{F['weighting']}"
        sc, _ = run_scores(d, variants)
        fold = F["folds"][0]
        idx = {s: i for i, s in enumerate(F["support"])}
        ti = np.array([idx[k] for k in fold["test"]])
        seen = ~np.array(fold["novel"], bool)
        nodes = np.array([len(trees[k]) if trees.get(k) else np.nan for k in fold["test"]], float)
        modeldb = np.array([s == "modeldb" for s in fold["test_sources"]], bool)
        size[run] = {p: {"family": fam, **by_node(v[ti], nodes, seen)} for p, (fam, v) in sc.items()}
        size[run + "_no_modeldb"] = {p: {"family": fam, **by_node(v[ti][~modeldb], nodes[~modeldb], seen[~modeldb])}
                                    for p, (fam, v) in sc.items() if fam == "tree"}
        r = rep[run]
        nm = r["no_modeldb"]
        strip = lambda g: {p: {k: v for k, v in a.items() if k != "by_size"} for p, a in g.items()}  # noqa: E731
        summ[run] = {"cutoff": r["cutoff"], "weighting": r["weighting"], "n_train": r["n_train"],
                     "n_test": r["n_test"], "n_test_models": r["n_test_models"], "frac_novel": r["frac_novel"],
                     "n_novel_pairs": r["novel"]["n_novel_pairs"], "n_novel_forms": r["novel"]["n_novel_forms"],
                     "test_pairs_by_source": {s: v["test_pairs"] for s, v in r["sizes"].items()},
                     "priors": r["priors"], "gap": strip(r["gap"]),
                     "redundant_novel_pairs": r["gap_counts"]["redundant_novel_pairs"],
                     "no_modeldb": {"n_test": nm["n_test"], "n_novel": nm["n_novel"], "priors": nm["priors"],
                                    "gap": strip(nm["gap"])}}
    return size, summ


def loso_data(run_dirs):
    out = defaultdict(dict)
    for d in run_dirs:
        F = _load(os.path.join(d, "folds.json"))
        src, w = F["test_corpus"].split(" (left out; ")
        w = w.rstrip(")")
        rows = _load(os.path.join(d, "table_v.json"))
        fold = F["folds"][0]
        train = set(fold["train"])
        test = fold["test"]
        out[src][w] = {"pairs": len(test), "distinct": len(set(test)),
                       "novel_pairs": sum(k not in train for k in test),
                       "novel_distinct": len({k for k in test if k not in train}),
                       "priors": {r["prior"]: {"family": r["family"], "raw": r["raw"], "raw_seen": r.get("raw_seen"),
                                               "raw_novel": r.get("raw_novel"), "renorm": r.get("renorm"),
                                               "n_zero": r.get("n_zero")} for r in rows}}
    return dict(out)


def diagnostics(groups, variants, calib_path):
    """groups: {label: [run_dir]}."""
    out = {"variants": defaultdict(list), "hpy": [], "fnprior_kn": None}
    for label, dirs in groups.items():
        for d in dirs:
            run = os.path.basename(os.path.normpath(d))
            pre = "hpyvt_" if os.path.exists(os.path.join(d, f"hpyvt_{variants[0]}.json")) else "hpyv_"
            for v in variants:
                S = _load(os.path.join(d, f"{pre}{v}.json"))
                for g in S.get("diag") or []:
                    h = g.get("hyper", {})
                    out["variants"][v].append({
                        "split": label, "run": run, "fit_seconds": _r(g.get("fit_seconds"), 1),
                        "n_train_trees": g.get("n_train_trees"),
                        "p_finite": [t["p_finite"] for t in g.get("termination", [])],
                        "sym_sum_min": min((t["sym_sum_min"] for t in g.get("termination", [])), default=None),
                        "acceptance": [_r(a, 6) for a in h.get("acceptance", [])],
                        "hyper": {k: h[k] for k in ("d", "theta", "a", "b") if k in h}})
            H = _load(os.path.join(d, "hpy.json"))
            for h in H.get("hyper") or []:
                out["hpy"].append({"split": label, "run": run, "hyper": h})
    cal = _load(calib_path)
    out["fnprior_kn"] = {"n_train": cal["n_train"], "termination": cal["termination"],
                         "max_abs_sum_minus_1": cal.get("fnprior_full_vocab_max_abs_sum_minus_1"),
                         "max_p_outside_vocab": cal.get("fnprior_max_p_y")}
    out["variants"] = dict(out["variants"])
    return out


def main(outdir, *argv):
    o = split_options(argv)
    variants = o["variants"]
    os.makedirs(outdir, exist_ok=True)
    _dump(corpus_data(o["gen"][0], o["trees"][0], o["corpora"]), os.path.join(outdir, "corpus.json"))
    hsize, hpairs = holdout_data(o["holdout"][0], o["holdout"][1:], variants)
    tsize, tsumm = temporal_data(o["temporal"][0], o["temporal"][1], o["temporal"][2:], variants)
    _dump({"holdout": hsize, "temporal": tsize, "holdout_pairs": hpairs}, os.path.join(outdir, "logp_size.json"))
    _dump(loso_data(o["loso"]), os.path.join(outdir, "loso.json"))
    _dump(tsumm, os.path.join(outdir, "temporal.json"))
    groups = {"loso": o["loso"], "holdout": o["holdout"][1:], "temporal": o["temporal"][2:]}
    _dump(diagnostics(groups, variants, o["calib"][0]), os.path.join(outdir, "diagnostics.json"))


if __name__ == "__main__":
    main(*sys.argv[1:])
