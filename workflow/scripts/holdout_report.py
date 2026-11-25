"""Hold-out report (rule holdout_report): every prior trained on all training
sources (hsr.grammar_compare.SOURCES; equally weighted or pooled) and scored
on the forms of each hold-out source (hsr.grammar_compare.HOLDOUT_SOURCES)
and on their union.

Scores per prior:
  raw     the prior's own log P(form), no renormalisation (sym0, the tree
          prior with no context, is the honest baseline for the tree priors;
          PCFG grammars generate a different space, so raw scores compare
          only approximately across the two families)
  renorm  log P renormalised over the run's common support (training forms,
          candidate forms and every hold-out form)
Means are over model-form pairs (each hold-out model contributes each of its
distinct forms once) and over distinct forms; "seen" forms are generalised
keys present in the training data, "novel" ones are not. Forms a prior gives
zero probability are left out of its means and counted.

Diagnostics of the seen/novel gap (section "Why novel forms score low"):
  size       mean log P by node count, and per prior an OLS fit
             log P = a + b * nodes + g * novel over model-form pairs, so g is
             the novelty penalty at fixed size
  vs sym0    log P minus sym0's log P (sym0 depends only on size and symbol
             use), seen / novel
  redundant  novel forms whose constants are not identifiable: the Jacobian
             of f(x; c) in c, on a grid of x at random c, is rank deficient
             (e.g. c0*x1 + c1*x1), the signature of a form the canonicaliser
             failed to simplify; seen / novel means are repeated without them

    python holdout_report.py out.md out.json trees.json run_dir [run_dir ...] [--title T] [--target rule] \\
        --variants v1 v2 ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr import treecode as tc  # noqa: E402
from hsr.gap_diagnostics import (FIXED_SIZE_NOTE, by_size_table, fixed_size_table,  # noqa: E402,F401
                                  gap_stats, redundant)
from hsr.gap_diagnostics import fmt as _f  # noqa: E402
from hsr.gap_diagnostics import mean as _mean  # noqa: E402


def _scores(run_dir, variants, trees, support):
    lp, fam = {}, {}
    for name in ["ours", "fnprior", "hpy"] + [f"hpyv_{v}" for v in variants]:
        with open(os.path.join(run_dir, f"{name}.json")) as fh:
            d = json.load(fh)["logprior"]
        for k, v in d.items():
            if len(v) != 1:
                raise ValueError(f"{run_dir}/{name}.json: expected one fold")
            lp[k] = np.array([-np.inf if x is None else x for x in v[0]], float)
            fam[k] = "pcfg" if name == "ours" else "tree"
    lp["uniform symbols"] = np.array([-len(trees[k]) * np.log(len(tc.V)) if trees.get(k) else -np.inf
                                      for k in support])
    fam["uniform symbols"] = "tree"
    return lp, fam


def summarise(F, lp, fam, nodes_of=None, redund=frozenset()):
    """{prior: {group: stats}} for each hold-out and "all"."""
    nodes_of = nodes_of or {}
    fold = F["folds"][0]
    sup = F["support"]
    idx = {s: i for i, s in enumerate(sup)}
    train = set(fold["train"])
    test, labels = fold["test"], fold["test_sources"]
    groups = {g: [k for k, lab in zip(test, labels) if lab == g] for g in dict.fromkeys(labels)}
    groups["all"] = list(test)
    out, counts = {}, {}
    for g, keys in groups.items():
        dist = sorted(set(keys))
        counts[g] = {"pairs": len(keys), "distinct": len(dist),
                     "novel_pairs": sum(k not in train for k in keys),
                     "novel_distinct": sum(k not in train for k in dist),
                     "redundant_novel_pairs": sum(k not in train and k in redund for k in keys),
                     "redundant_novel_distinct": sum(k not in train and k in redund for k in dist)}
    sym0 = lp.get("sym0")
    for name, v in lp.items():
        finite = np.isfinite(v)
        z = v - np.logaddexp.reduce(v[finite])
        out[name] = {"family": fam[name]}
        for g, keys in groups.items():
            dist = sorted(set(keys))
            raw = np.array([v[idx[k]] for k in keys])
            ren = np.array([z[idx[k]] for k in keys])
            seen = np.array([k in train for k in keys], bool)
            rawd = np.array([v[idx[k]] for k in dist])
            rend = np.array([z[idx[k]] for k in dist])
            seend = np.array([k in train for k in dist], bool)
            out[name][g] = {"raw": _mean(raw), "renorm": _mean(ren), "raw_seen": _mean(raw[seen]),
                            "raw_novel": _mean(raw[~seen]), "raw_distinct": _mean(rawd),
                            "renorm_distinct": _mean(rend), "raw_distinct_seen": _mean(rawd[seend]),
                            "raw_distinct_novel": _mean(rawd[~seend]),
                            "n_zero": int((~np.isfinite(raw)).sum())}
            nodes = np.array([nodes_of.get(k, np.nan) for k in keys], float)
            red = np.array([k in redund for k in keys], bool)
            s0 = None if sym0 is None else np.array([sym0[idx[k]] for k in keys])
            gs = gap_stats(raw, nodes, seen, red, s0, by_size=(g == "all"))
            o = out[name][g]
            o.update({k: gs[k] for k in ("per_node", "novel_penalty_fixed_size", "raw_novel_nonredundant")})
            if sym0 is not None:
                o["vs_sym0_seen"], o["vs_sym0_novel"] = gs["vs_sym0_seen"], gs["vs_sym0_novel"]
            if g == "all":
                o["by_size"] = gs["by_size"]
    return out, counts


def why(L, res, ws, w0, priors, groups):
    tree = [p for p in priors if res[w0]["stats"][p]["family"] == "tree"]
    L.append("## Why novel forms score low\n")
    L.append("Novel forms are larger than seen ones, and every prior pays per node. sym0 (symbol frequencies "
             "only) measures what size and symbol use alone cost.\n")
    L.append("### Log P by node count (all hold-outs; seen / novel; pairs in brackets)\n")
    for w in ws:
        L.append(f"{w} weighting:\n")
        L += by_size_table([(p, res[w]["stats"][p]["all"]["by_size"]) for p in tree])
        L.append("")
    L.append("### Novelty at fixed size, relative to sym0, and without redundant forms (all hold-outs)\n")
    L.append(FIXED_SIZE_NOTE + "\n")
    L += fixed_size_table([(p, w, res[w]["stats"][p]["all"]) for p in tree for w in ws])
    L.append("")
    c = res[w0]["counts"]
    L.append("Redundant novel forms (constants not identifiable; Jacobian rank test): " + ", ".join(
        f"{g} {c[g]['redundant_novel_distinct']} of {c[g]['novel_distinct']} distinct "
        f"({c[g]['redundant_novel_pairs']} of {c[g]['novel_pairs']} pairs)" for g in groups) + ".\n")
    if res[w0]["redundant_novel"]:
        L.append("Redundant novel keys: " + "; ".join(f"`{k}`" for k in res[w0]["redundant_novel"]) + ".\n")


def _option(rest, name, default):
    """Remove ``name value`` from the argument list ``rest``; return (value, rest)."""
    rest = list(rest)
    if name in rest:
        i = rest.index(name)
        val = rest[i + 1]
        del rest[i:i + 2]
        return val, rest
    return default, rest


def main(out_md, out_json, trees_path, *rest):
    title, rest = _option(rest, "--title", "Hold-out sources")
    target, rest = _option(rest, "--target", "holdout")
    i = rest.index("--variants")
    run_dirs, variants = rest[:i], rest[i + 1:]
    with open(trees_path) as fh:
        trees = json.load(fh)["trees"]
    res = {}
    for d in run_dirs:
        with open(os.path.join(d, "folds.json")) as fh:
            F = json.load(fh)
        w = "equal" if F["test_corpus"].endswith("; equal)") else "pooled"
        lp, fam = _scores(d, variants, trees, F["support"])
        fold = F["folds"][0]
        train = set(fold["train"])
        nodes_of = {k: len(trees[k]) for k in set(fold["test"]) if trees.get(k)}
        redund = frozenset(k for k in set(fold["test"]) if k not in train and redundant(k))
        stats, counts = summarise(F, lp, fam, nodes_of, redund)
        F_models = F.get("holdout_models", {})
        res[w] = {"stats": stats, "counts": counts, "train_sizes": F["train_sizes"],
                  "redundant_novel": sorted(redund),
                  "support": len(F["support"]), "n_train": len(F["folds"][0]["train"])}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)

    ws = [w for w in ("equal", "pooled") if w in res]
    w0 = ws[0]
    groups = list(res[w0]["counts"])
    priors = list(res[w0]["stats"])
    # tree priors first (by raw log P on all hold-outs, equal weighting), then the PCFG grammars
    priors.sort(key=lambda p: (res[w0]["stats"][p]["family"] != "tree", p == "uniform symbols",
                               -(res[w0]["stats"][p]["all"]["raw"] or -1e9)))
    cols = [(g, w) for g in groups for w in ws]
    L = []
    L.append(f"# {title}\n")
    L.append("Every prior is trained on all twelve training sources (`hsr.grammar_compare.SOURCES`; "
             "`equal`: each source replicated to the size of the largest, `pooled`: every model-form pair once) "
             "and scored on the forms of each hold-out source, which no prior ever trains on. Mean log "
             "probability per model-form pair. raw: the prior's own log P (no renormalisation; tree priors share "
             "one space and sym0, the tree prior without context, is the honest baseline; PCFG grammars generate "
             "a different space, so raw scores compare only approximately across the two families); renorm: "
             f"renormalised over a common support of {res[w0]['support']} forms (training, candidate and hold-out "
             f"forms). Built by `snakemake --use-conda --cores 2 {target}`.\n")
    L.append("| hold-out | models | model-form pairs | distinct forms | novel pairs | novel distinct forms |")
    L.append("|---|---|---|---|---|---|")
    for g in groups:
        c = res[w0]["counts"][g]
        nm = F_models.get(g, sum(F_models.values())) if F_models else ""
        L.append(f"| {g} | {nm} | {c['pairs']} | {c['distinct']} | "
                 f"{c['novel_pairs']} | {c['novel_distinct']} |")
    L.append("\nNovel: generalised form absent from the training sources.\n")

    def table(title, field, note=""):
        L.append(f"## {title}\n")
        if note:
            L.append(note + "\n")
        L.append("| prior | family | " + " | ".join(f"{g} {w}" for g, w in cols) + " |")
        L.append("|---|---|" + "---|" * len(cols))
        for p in priors:
            st = {w: res[w]["stats"][p] for w in ws}
            L.append(f"| {p} | {st[w0]['family']} | " + " | ".join(_f(st[w][g][field]) for g, w in cols) + " |")
        L.append("")

    table("Raw log P per model-form pair", "raw")
    L.append("## Seen and novel forms (raw log P per model-form pair, seen / novel)\n")
    L.append("| prior | " + " | ".join(f"{g} {w}" for g, w in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for p in priors:
        L.append(f"| {p} | " + " | ".join(f"{_f(res[w]['stats'][p][g]['raw_seen'])} / "
                                         f"{_f(res[w]['stats'][p][g]['raw_novel'])}" for g, w in cols) + " |")
    L.append("")
    why(L, res, ws, w0, priors, groups)
    table("Raw log P per distinct form", "raw_distinct")
    table("Renormalised log P per model-form pair", "renorm",
          f"uniform over support: {-np.log(res[w0]['support']):.2f}.")
    zeros = {p: {(g, w): res[w]["stats"][p][g]["n_zero"] for g, w in cols} for p in priors}
    nz = [p for p in priors if any(zeros[p].values())]
    if nz:
        L.append("Zero-probability pairs (left out of the means): " + "; ".join(
            f"{p}: " + ", ".join(f"{g} {w} {n}" for (g, w), n in zeros[p].items() if n) for p in nz) + ".\n")
    L.append("Training sizes (model-form pairs before weighting): " + ", ".join(
        f"{n} {v}" for n, v in res[w0]["train_sizes"].items()) + ".\n")
    with open(out_md, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
