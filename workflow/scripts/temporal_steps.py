"""Snakemake entry points for the temporal hold-out (hsr.temporal; hsr environment).

    python temporal_steps.py folds generalised.json model_dates.json cutoff weighting out.json
    python temporal_steps.py report out.md out.json model_dates.json trees.json variants run_dir [run_dir ...]

The report adds, as holdout.md does, diagnostics of the seen/novel gap
(hsr.gap_diagnostics: log P by node count, OLS novelty penalty at fixed size,
gain over sym0, forms with redundant constants), the summary table with
ModelDB left out of the test set (ModelDB supplies most test pairs and most
novel pairs) and the novel-pair scores per test source.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from hsr import gap_diagnostics as gd
from hsr import temporal

# priors shown in the per-source and headline gap tables
MAIN = ["tree_ad3", "form_ad3_depth", "hpy3_depth", "hpy2", "fnprior_kn", "sym0", "term_root"]
EXCLUDE = "modeldb"
N_KEYS = 10  # redundant novel keys listed per cutoff in the markdown


def folds(gen_path, dates_path, cutoff, weighting, out):
    with open(gen_path) as fh:
        gen = json.load(fh)
    with open(dates_path) as fh:
        dates = json.load(fh)
    F = temporal.make_temporal(gen, dates, int(cutoff), weighting)
    fold = F["folds"][0]
    print(f"cutoff {cutoff} ({weighting}): {len(fold['train'])} training pairs, {len(fold['test'])} test pairs "
          f"({sum(fold['novel'])} novel)", flush=True)
    with open(out, "w") as fh:
        json.dump(F, fh)


def run_scores(run_dir, variants):
    """{prior: (family, raw log P per support key)} of one run."""
    out = {}
    files = [("ours.json", "pcfg"), ("fnprior.json", "tree"), ("hpy.json", "tree")]
    files += [(f"hpyvt_{v}.json", "tree") for v in variants]
    for name, fam in files:
        with open(os.path.join(run_dir, name)) as fh:
            for prior, per_fold in json.load(fh)["logprior"].items():
                out[prior] = (fam, per_fold[0])
    return out


def run_metrics(F, scores, keep=None):
    """Per prior: mean raw log P over all / seen / novel test pairs (only the
    pairs where ``keep`` is true, if given)."""
    fold = F["folds"][0]
    idx = {s: i for i, s in enumerate(F["support"])}
    novel = np.array(fold["novel"], bool)
    ti = np.array([idx[k] for k in fold["test"]])
    if keep is not None:
        keep = np.asarray(keep, bool)
        novel, ti = novel[keep], ti[keep]
    rows = {}
    for prior, (fam, v) in scores.items():
        v = np.array([-np.inf if x is None else x for x in v], float)
        b = v[ti]
        ok = np.isfinite(b)
        z = v - np.logaddexp.reduce(v[np.isfinite(v)])

        def mean(m):
            return float(b[m].mean()) if m.any() else None
        rows[prior] = {"family": fam, "raw": mean(ok), "raw_seen": mean(ok & ~novel), "raw_novel": mean(ok & novel),
                       "renorm": float(z[ti][ok].mean()) if ok.any() else None, "n_zero": int((~ok).sum()),
                       "n_zero_novel": int((~ok & novel).sum())}
    return rows


def _arrays(F, scores):
    fold = F["folds"][0]
    idx = {s: i for i, s in enumerate(F["support"])}
    ti = np.array([idx[k] for k in fold["test"]])
    return {p: np.array([-np.inf if x is None else x for x in v], float)[ti] for p, (fam, v) in scores.items()}


def gap_metrics(F, scores, nodes_of, is_redundant, keep=None):
    """hsr.gap_diagnostics.gap_stats of every tree prior on the test pairs
    (those where ``keep`` is true, if given), plus redundant-form counts."""
    fold = F["folds"][0]
    keys = list(fold["test"])
    keep = np.ones(len(keys), bool) if keep is None else np.asarray(keep, bool)
    keys = [k for k, m in zip(keys, keep) if m]
    novel = np.array(fold["novel"], bool)[keep]
    nodes = np.array([nodes_of.get(k, np.nan) for k in keys], float)
    red = np.array([bool(nv and is_redundant(k)) for k, nv in zip(keys, novel)], bool)
    lp = {p: a[keep] for p, a in _arrays(F, scores).items()}
    sym0 = lp.get("sym0")
    out = {p: gd.gap_stats(lp[p], nodes, ~novel, red, sym0) for p, (fam, _) in scores.items() if fam == "tree"}
    nk = {k for k, nv in zip(keys, novel) if nv}
    counts = {"novel_pairs": int(novel.sum()), "novel_distinct": len(nk),
              "redundant_novel_pairs": int(red.sum()),
              "redundant_novel_distinct": int(sum(is_redundant(k) for k in nk)),
              "redundant_novel": sorted(k for k in nk if is_redundant(k))}
    return out, counts


def novel_by_source(F, scores, nodes_of, priors):
    """Per test source: novel pairs, distinct novel forms, mean node count of
    seen and novel pairs, and each prior's mean raw log P on its novel pairs."""
    fold = F["folds"][0]
    keys, src = fold["test"], np.array(fold["test_sources"])
    novel = np.array(fold["novel"], bool)
    nodes = np.array([nodes_of.get(k, np.nan) for k in keys], float)
    lp = _arrays(F, scores)
    out = {}
    for s in sorted(set(src)):
        m = src == s
        out[s] = {"pairs": int(m.sum()), "novel_pairs": int((m & novel).sum()),
                  "novel_distinct": len({k for k, mm in zip(keys, m & novel) if mm}),
                  "nodes_seen": gd.mean(nodes[m & ~novel]), "nodes_novel": gd.mean(nodes[m & novel]),
                  "novel": {p: gd.mean(lp[p][m & novel]) for p in priors if p in lp},
                  "seen": {p: gd.mean(lp[p][m & ~novel]) for p in priors if p in lp}}
    return out


def novel_examples(F, n=10):
    fold = F["folds"][0]
    users, srcs, first = Counter(), defaultdict(set), {}
    for k, nv, s, y in zip(fold["test"], fold["novel"], fold["test_sources"], fold["test_years"]):
        if nv:
            users[k] += 1
            srcs[k].add(s)
            first[k] = min(y, first.get(k, y))
    return {"n_novel_forms": len(users), "n_novel_pairs": sum(users.values()),
            "examples": [{"form": k, "models": c, "sources": sorted(srcs[k]), "first_year": first[k]}
                         for k, c in users.most_common(n)]}


def _f(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def _summary(L, runs, cutoffs, w, metrics="rows"):
    """The all / seen / novel table of one weighting (``metrics``: which
    per-prior metrics of each run to show)."""
    first = runs[(cutoffs[0], w)][metrics]
    order = sorted(first, key=lambda p: (first[p]["family"] != "tree",
                                         -np.mean([runs[(c, w)][metrics][p]["raw_novel"] or -1e9 for c in cutoffs])))
    L += ["| prior | family | " + " | ".join(f"{c} all | {c} seen | {c} novel" for c in cutoffs)
          + " | zero-probability pairs |", "|---|---|" + "---|---|---|" * len(cutoffs) + "---|"]
    for p in order:
        cells, zeros = [], []
        for c in cutoffs:
            m = runs[(c, w)][metrics][p]
            cells += [_f(m["raw"]), _f(m["raw_seen"]), _f(m["raw_novel"])]
            zeros.append(str(m["n_zero"]))
        L.append(f"| {p} | {first[p]['family']} | " + " | ".join(cells) + " | " + "/".join(zeros) + " |")
    return order


def _w_main(weightings):
    return "pooled" if "pooled" in weightings else weightings[0]


def _excluded(L, runs, cutoffs, weightings):
    L += ["", f"## Excluding {EXCLUDE} from the test set", "",
          f"{EXCLUDE} supplies most test pairs and most novel pairs (many of them large, multi-gate channel "
          f"kinetics), so the summary is repeated with every {EXCLUDE} test pair dropped. The priors are the same "
          f"(trained before Y, {EXCLUDE} included in training); only the test set changes.", "",
          f"| cutoff Y | test pairs | without {EXCLUDE} | novel pairs | novel without {EXCLUDE} |",
          "|---|---|---|---|---|"]
    for c in cutoffs:
        r = runs[(c, _w_main(weightings))]
        nv = int(np.sum(r["F"]["folds"][0]["novel"]))
        L.append(f"| {c} | {len(r['F']['folds'][0]['test'])} | {r['n_test_x']} | {nv} | {r['n_novel_x']} |")
    for w in weightings:
        L += ["", f"### Raw log P per test pair without {EXCLUDE} ({w} source weighting)", ""]
        _summary(L, runs, cutoffs, w, "rows_x")


def _why(L, runs, cutoffs, weightings):
    w0 = _w_main(weightings)
    tree = [p for p in runs[(cutoffs[0], w0)]["rows"]
            if runs[(cutoffs[0], w0)]["rows"][p]["family"] == "tree"]
    tree.sort(key=lambda p: -(runs[(cutoffs[0], w0)]["rows"][p]["raw_novel"] or -1e9))
    main = [p for p in MAIN if p in tree]
    L += ["", "## Why novel forms score low", "",
          "As in holdout.md (hsr.gap_diagnostics): novel forms are larger than seen ones and every prior pays per "
          "node; sym0 (symbol frequencies only) measures what size and symbol use alone cost. " + gd.FIXED_SIZE_NOTE,
          "", "### Size against genuine novelty", "",
          "Per cutoff: seen-minus-novel gap in mean log P per pair, and the OLS novelty penalty at fixed node "
          f"count (gap / penalty; the penalty is the OLS coefficient, negative when novel forms score lower than "
          f"seen forms of the same size; roughly, gap + penalty is the part of the gap due to size), on all test "
          f"pairs and without {EXCLUDE}.", "",
          "| prior | weighting | " + " | ".join(f"{c} gap / penalty | {c} no {EXCLUDE}" for c in cutoffs) + " |",
          "|---|---|" + "---|---|" * len(cutoffs)]

    def gp(a):
        return f"{_f(a['gap'])} / {_f(a['novel_penalty_fixed_size'])}"
    for p in main:
        for w in weightings:
            L.append(f"| {p} | {w} | " + " | ".join(f"{gp(runs[(c, w)]['gap'][p])} | {gp(runs[(c, w)]['gap_x'][p])}"
                                                   for c in cutoffs) + " |")
    a0 = {c: runs[(c, w0)]["gap"][main[0]] for c in cutoffs}
    ax = {c: runs[(c, w0)]["gap_x"][main[0]] for c in cutoffs}
    L += ["", "Mean node count, seen / novel pairs: " + "; ".join(
        f"{c}: {_f(a0[c]['nodes_seen'], 1)} / {_f(a0[c]['nodes_novel'], 1)} (without {EXCLUDE} "
        f"{_f(ax[c]['nodes_seen'], 1)} / {_f(ax[c]['nodes_novel'], 1)})" for c in cutoffs) + ".", ""]
    for c in cutoffs:
        L += [f"### Y = {c}", "", f"Log P by node count (seen / novel; {w0} weighting):", ""]
        L += gd.by_size_table([(p, runs[(c, w0)]["gap"][p]["by_size"]) for p in tree])
        L += ["", "Novelty at fixed size, relative to sym0, and without redundant forms:", ""]
        L += gd.fixed_size_table([(p, w, runs[(c, w)]["gap"][p]) for p in tree for w in weightings])
        g = runs[(c, w0)]["gap_counts"]
        L += ["", f"Redundant novel forms (constants not identifiable; Jacobian rank test): "
              f"{g['redundant_novel_distinct']} of {g['novel_distinct']} distinct ({g['redundant_novel_pairs']} of "
              f"{g['novel_pairs']} pairs)."
              + (" Examples (all in temporal.json): " + "; ".join(f"`{k}`" for k in g["redundant_novel"][:N_KEYS]) + "."
                 if g["redundant_novel"] else ""),
              ""]


def _by_source(L, runs, cutoffs, weightings):
    w0 = _w_main(weightings)
    pri = [p for p in MAIN if p in runs[(cutoffs[0], w0)]["rows"]]
    L += ["## Novel pairs by test source", "",
          f"Mean raw log P of each source's novel test pairs ({w0} weighting; the other weighting is in "
          "temporal.json), with the mean node count of the source's seen and novel pairs.", ""]
    for c in cutoffs:
        bs = runs[(c, w0)]["by_source"]
        L += [f"### Y = {c}", "", "| source | test pairs | novel pairs | distinct novel | nodes seen / novel | "
              + " | ".join(pri) + " |", "|---|---|---|---|---|" + "---|" * len(pri)]
        for src, b in bs.items():
            if not b["novel_pairs"]:
                continue
            L.append(f"| {src} | {b['pairs']} | {b['novel_pairs']} | {b['novel_distinct']} | "
                     f"{_f(b['nodes_seen'], 1)} / {_f(b['nodes_novel'], 1)} | "
                     + " | ".join(_f(b["novel"][p]) for p in pri) + " |")
        L.append("")


def report(out_md, out_json, dates_path, trees_path, variants, *run_dirs):
    variants = [v for v in variants.split(",") if v]
    with open(dates_path) as fh:
        dates = json.load(fh)
    with open(trees_path) as fh:
        trees = json.load(fh)["trees"]
    nodes_of = {k: len(t) for k, t in trees.items() if t}
    is_red = gd.RedundancyCache()
    runs = {}
    for d in run_dirs:
        with open(os.path.join(d, "folds.json")) as fh:
            F = json.load(fh)
        sc = run_scores(d, variants)
        keep = np.array([s != EXCLUDE for s in F["folds"][0]["test_sources"]], bool)
        gap, gcounts = gap_metrics(F, sc, nodes_of, is_red)
        gap_x, gcounts_x = gap_metrics(F, sc, nodes_of, is_red, keep)
        runs[(F["cutoff"], F["weighting"])] = {
            "F": F, "rows": run_metrics(F, sc), "rows_x": run_metrics(F, sc, keep), "gap": gap,
            "gap_counts": gcounts, "gap_x": gap_x, "gap_counts_x": gcounts_x,
            "by_source": novel_by_source(F, sc, nodes_of, MAIN),
            "n_test_x": int(keep.sum()), "n_novel_x": int((keep & np.array(F["folds"][0]["novel"], bool)).sum())}
    cutoffs = sorted({c for c, _ in runs})
    weightings = [w for w in ("equal", "pooled") if any(ww == w for _, ww in runs)]
    res = {"coverage": dates["coverage"], "how": dates.get("how", {}),
           "year_distribution": temporal.year_distribution(dates), "runs": {}}
    for (c, w), r in runs.items():
        F = r["F"]
        fold = F["folds"][0]
        res["runs"][f"{c}_{w}"] = {"cutoff": c, "weighting": w, "n_train": len(fold["train"]),
                                   "n_test": len(fold["test"]), "n_test_models": len(fold["test_models"]),
                                   "frac_novel": float(np.mean(fold["novel"])) if fold["novel"] else None,
                                   "sizes": F["sizes"], "priors": r["rows"], "novel": novel_examples(F),
                                   "gap": r["gap"], "gap_counts": r["gap_counts"],
                                   f"no_{EXCLUDE}": {"n_test": r["n_test_x"], "n_novel": r["n_novel_x"],
                                                     "priors": r["rows_x"], "gap": r["gap_x"],
                                                     "gap_counts": r["gap_counts_x"]},
                                   "novel_by_source": r["by_source"]}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)

    L = ["# Temporal hold-out: do the priors anticipate forms not yet used?", "",
         "Each prior is trained only on the models published before a cutoff year Y and scored on the "
         "model-form pairs of the models published from Y on (raw log P of the form under the prior, mean per "
         "pair; no renormalisation). Test pairs are split into forms already used by some model before Y "
         "(\"seen\": the generalised key is in the training set) and forms first used at or after Y "
         "(\"novel\"). Years: the publication year of the paper describing the model where the source "
         "records it, otherwise the first submission (hsr/data/model_dates.json; hsr.model_dates). Sources "
         "without per-model dates (" + ", ".join(dates["undated_sources"]) + ") are always in the training "
         "set; undated models of dated sources are dropped. Tree priors (fnprior, hpy, hpyv variants) share "
         "one space, so their raw scores are comparable; the PCFG grammars generate a different space. "
         "Source-aware variants score test forms as a new, unseen source. Per-run tables, including the "
         "scores renormalised over the common support: results/grammar/temporal_<Y>_<weighting>/table_v.md.", ""]
    L += ["## Date coverage", "", "| source | dated models | undated models | year from |", "|---|---|---|---|"]
    for s, cv in dates["coverage"].items():
        hw = ", ".join(f"{k} {v}" for k, v in dates.get("how", {}).get(s, {}).items())
        L.append(f"| {s} | {cv['dated']} | {cv['undated']} | {hw} |")
    for s in dates["undated_sources"]:
        L.append(f"| {s} | - | all (always training) | |")
    L += ["", "Coverage counts every model of the corpus file; the year distribution below counts models "
          "with at least one form.", "", "## Year distribution (models with forms)", ""]
    yd = res["year_distribution"]
    cols = list(next(iter(yd.values())))
    L += ["| source | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s, r in yd.items():
        L.append(f"| {s} | " + " | ".join(str(r[c]) for c in cols) + " |")
    tot = {c: sum(r[c] for r in yd.values()) for c in cols}
    L.append("| all | " + " | ".join(str(tot[c]) for c in cols) + " |")
    L += ["", "## Test sets", "", "| cutoff Y | training pairs (pooled) | test models | test pairs | novel pairs "
          "| fraction novel | distinct novel forms | test pairs by source |", "|---|---|---|---|---|---|---|---|"]
    for c in cutoffs:
        r = res["runs"].get(f"{c}_pooled") or res["runs"][f"{c}_{weightings[0]}"]
        by = ", ".join(f"{s} {v['test_pairs']}" for s, v in r["sizes"].items() if v["test_pairs"])
        L.append(f"| {c} | {r['n_train']} | {r['n_test_models']} | {r['n_test']} | {r['novel']['n_novel_pairs']} "
                 f"| {_f(r['frac_novel'], 3)} | {r['novel']['n_novel_forms']} | {by} |")
    for w in weightings:
        L += ["", f"## Raw log P per test pair ({w} source weighting)", "",
              "all / seen / novel test pairs; rows ordered by the mean over cutoffs of the novel-form score "
              "(tree priors first). Zero-probability pairs are left out of the means (counted in the last column).",
              ""]
        order = _summary(L, runs, cutoffs, w)
        L += ["", f"Rank of each tree prior on novel forms ({w}; 1 = highest log P):", "",
              "| prior | " + " | ".join(f"{c} seen | {c} novel" for c in cutoffs) + " |",
              "|---|" + "---|---|" * len(cutoffs)]
        trees_ = [p for p in order if runs[(cutoffs[0], w)]["rows"][p]["family"] == "tree"]
        rk = {}
        for c in cutoffs:
            for what in ("raw_seen", "raw_novel"):
                srt = sorted(trees_, key=lambda p: -(runs[(c, w)]["rows"][p][what] or -1e9))
                for i, p in enumerate(srt):
                    rk[(p, c, what)] = i + 1
        for p in trees_:
            L.append(f"| {p} | " + " | ".join(f"{rk[(p, c, 'raw_seen')]} | {rk[(p, c, 'raw_novel')]}"
                                              for c in cutoffs) + " |")
    _excluded(L, runs, cutoffs, weightings)
    _why(L, runs, cutoffs, weightings)
    _by_source(L, runs, cutoffs, weightings)
    L += ["## Novel forms", "", "Forms first used by a model dated at or after the cutoff (generalised keys; "
          "the ten used by the most test models).", ""]
    for c in cutoffs:
        r = res["runs"].get(f"{c}_pooled") or res["runs"][f"{c}_{weightings[0]}"]
        nv = r["novel"]
        L += [f"### Y = {c}: {nv['n_novel_forms']} novel forms in {nv['n_novel_pairs']} test pairs", "",
              "| form | test models | sources | first year |", "|---|---|---|---|"]
        for e in nv["examples"]:
            L.append(f"| `{e['form']}` | {e['models']} | {', '.join(e['sources'])} | {e['first_year']} |")
        L.append("")
    with open(out_md, "w") as fh:
        fh.write("\n".join(L) + "\n")



if __name__ == "__main__":
    {"folds": folds, "report": report}[sys.argv[1]](*sys.argv[2:])
