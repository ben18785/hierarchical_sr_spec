"""Summary of the tree-prior variants (rule hpyv_report): held-out tables
over every run (CV seeds, transfers, leave-one-source-out), properness
checks and calibration.

    python hpyv_report.py out.md trees.json [--notes notes.md] --tables ... --scores ... --calib ...
"""
import argparse
import json
import os
import re

import numpy as np

PCFG_REF = "term_root"


def run_name(path):
    return os.path.basename(os.path.dirname(path))


def fmt(x, nd=3):
    return "" if x is None or x != x else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("trees")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--tables", nargs="+")
    ap.add_argument("--scores", nargs="+")
    ap.add_argument("--calib", nargs="+")
    a = ap.parse_args()
    T = {}
    for p in a.tables:
        with open(p) as fh:
            T[run_name(p)] = {r["prior"]: r for r in json.load(fh)}
    seeds = sorted(r for r in T if r.startswith("seed"))
    transfers = sorted(r for r in T if r.startswith("transfer_"))
    loso_src = sorted({re.match(r"loso_(.+)_(pooled|equal)$", r).group(1) for r in T if r.startswith("loso_")})
    tree_priors = [p for p, r in T[seeds[0]].items() if r["family"] == "tree"]
    variants = [p for p in tree_priors if p not in ("uniform symbols",)]

    def cv(prior, col):
        v = [T[s][prior][col] for s in seeds if prior in T[s] and T[s][prior][col] is not None]
        return float(np.mean(v)) if v else None

    order = sorted(variants, key=lambda p: -cv(p, "raw"))
    L = ["# Tree-prior variants: held-out comparison\n"]
    if a.notes and os.path.exists(a.notes):
        with open(a.notes) as fh:
            L.append(fh.read().rstrip() + "\n")
    L.append("## Held-out mean log P per form: CV and transfer\n")
    L.append("raw: the prior's own log P(tree) (no renormalisation; a proper distribution over fnprior-coded "
             "trees, comparable across all tree priors); renorm: renormalised over the run's common support. "
             f"CV: mean over seeds {', '.join(s[4:] for s in seeds)} (5-fold by BioModels model). Transfers: "
             "trained on all BioModels. d(hpy2): raw difference from hpy2 (positive = better).\n")
    hdr = "| prior | CV raw | d(hpy2) | CV renorm | " + " | ".join(f"{t[9:]} raw" for t in transfers) + " | " + \
          " | ".join(f"{t[9:]} renorm" for t in transfers) + " |"
    L += [hdr, "|" + "---|" * (4 + 2 * len(transfers))]
    base = cv("hpy2", "raw") if "hpy2" in variants else None
    for p in order + [PCFG_REF, "uniform over support", "uniform symbols"]:
        if p not in T[seeds[0]]:
            continue
        craw = cv(p, "raw")
        d = None if craw is None or base is None or T[seeds[0]][p]["family"] != "tree" else craw - base
        row = [p + (" (PCFG)" if T[seeds[0]][p]["family"] == "pcfg" else ""), fmt(craw), fmt(d, 2),
               fmt(cv(p, "renorm"))]
        row += [fmt(T[t][p]["raw"]) if p in T[t] else "" for t in transfers]
        row += [fmt(T[t][p]["renorm"]) if p in T[t] else "" for t in transfers]
        L.append("| " + " | ".join(row) + " |")
    for col, title in (("raw", "raw log P"), ("renorm", "renormalised log P")):
        L.append(f"\n## Leave one source out: {title} (equal / pooled source weights)\n")
        L += ["| prior | " + " | ".join(loso_src) + " | mean |", "|" + "---|" * (len(loso_src) + 2)]
        for p in order + [PCFG_REF, "uniform over support"]:
            if col == "raw" and p == "uniform over support":
                continue
            cells, eqs = [], []
            for s in loso_src:
                e, po = T.get(f"loso_{s}_equal", {}).get(p), T.get(f"loso_{s}_pooled", {}).get(p)
                ev = e[col] if e else None
                pv = po[col] if po else None
                cells.append(f"{fmt(ev, 2)} / {fmt(pv, 2)}")
                eqs.append((ev, pv))
            me = np.mean([x[0] for x in eqs if x[0] is not None]) if any(x[0] is not None for x in eqs) else None
            mp = np.mean([x[1] for x in eqs if x[1] is not None]) if any(x[1] is not None for x in eqs) else None
            L.append(f"| {p} | " + " | ".join(cells) + f" | {fmt(me, 2)} / {fmt(mp, 2)} |")

    # properness diagnostics from the score files
    diag = {}
    for p in a.scores:
        with open(p) as fh:
            d = json.load(fh)
        name = d["settings"]["variant"]["name"]
        g = diag.setdefault(name, {"pf": [], "dev": [], "acc": [], "sec": [], "hyper": []})
        for x in d["diag"]:
            for t in x["termination"]:
                g["pf"].append(t["p_finite"])
                g["dev"].append(max(abs(t["total_min"] - 1), abs(t["total_max"] - 1)))
            g["sec"].append(x["fit_seconds"])
            if "acceptance" in x["hyper"]:
                g["acc"].append(np.mean(x["hyper"]["acceptance"][-5:]))
            if run_name(p) == "transfer_sabiork":
                g["hyper"] = x["hyper"]
    C = {}
    for p in a.calib:
        with open(p) as fh:
            c = json.load(fh)
        C[c["variant"]["name"]] = c
    L.append("\n## Properness and calibration\n")
    L.append("P(finite): probability that the prior generates a finite tree (minimal fixed point of the "
             "branching equations), min over every fold and two retained samples; max |sum - 1|: largest "
             "deviation from one of a position's total next-node probability over every reachable position. "
             "Calibration: trained on all BioModels; forward samples (4000) against the training forms: mean "
             "and 90% quantile of the node count and depth, the fraction of sampled trees that are canonical "
             "codes (re-parsing the sampled expression gives the same tree) and the fraction that contain the "
             "variable.\n")
    L += ["| prior | P(finite) min | max abs(sum-1) | MH acceptance | fit s (mean) | size mean (train) | size q90 "
          "(train) | depth mean (train) | canonical | has x | canonical and has x | sampled tree in training set |",
          "|" + "---|" * 12]
    for p in sorted(set(diag) | set(C), key=lambda p: (p not in order, order.index(p) if p in order else 0)):
        g, c = diag.get(p), C.get(p)
        cells = [p]
        cells += [fmt(min(g["pf"]), 6), f"{max(g['dev']):.1e}", fmt(np.mean(g["acc"]), 3) if g["acc"] else "",
                  f"{np.mean(g['sec']):.0f}"] if g else [fmt(c["termination"][0]["p_finite"], 6),
                                                       f"{max(abs(t['total_min'] - 1) for t in c['termination']):.1e}"
                                                       " (y)", "", ""]
        if c:
            cells += [f"{c['size_mean_prior']:.2f} ({c['size_mean_train']:.2f})",
                      f"{c['size_q90_prior']:.0f} ({c['size_q90_train']:.0f})",
                      f"{c['depth_mean_prior']:.2f} ({c['depth_mean_train']:.2f})",
                      fmt(c["frac_canonical"]), fmt(c["frac_has_x"]), fmt(c["frac_canonical_with_x"]),
                      fmt(c["frac_in_train"])]
        else:
            cells += [""] * 7
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n## Learned hyperparameters (trained on all BioModels: run transfer_sabiork)\n")
    for p in order:
        if p in diag and diag[p]["hyper"]:
            h = diag[p]["hyper"]
            parts = []
            for k in ("d", "theta", "a", "b"):
                if h.get(k):
                    parts.append(f"{k}: " + ", ".join(f"{g} {v:.2f}" for g, v in h[k].items()))
            if h.get("lambda"):
                parts.append("lambda: " + ", ".join(f"{g} [{', '.join(f'{x:.2f}' for x in v)}]"
                                                    for g, v in h["lambda"].items()))
            L.append(f"- **{p}**: " + "; ".join(parts))
    with open(a.trees) as fh:
        chk = json.load(fh)["check"]
    L.append("\n## Tree coding\n")
    L.append(f"{chk['n_coded']} of {chk['n_keys']} support forms coded; events differ from fnprior's own coding "
             f"(n = 2 and 3) for {chk['event_mismatches_vs_fnprior']} forms. The forms map to "
             f"{chk['n_distinct_trees']} distinct trees ({chk['n_keys_sharing_a_tree']} forms share their tree "
             "with another form, e.g. x^2, x^c0 and x^(1/3) are all pow(x, a)), so a tree prior's probabilities "
             "summed over the support exceed one, and the renormalised score divides a tree's mass among the "
             "forms that share it.")
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
