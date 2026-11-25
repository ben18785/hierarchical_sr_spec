"""Snakemake entry points for the temporal hold-out (hsr.temporal; hsr environment).

    python temporal_steps.py folds generalised.json model_dates.json cutoff weighting out.json
    python temporal_steps.py report out.md out.json model_dates.json variants run_dir [run_dir ...]
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from hsr import temporal


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


def run_metrics(F, scores):
    fold = F["folds"][0]
    idx = {s: i for i, s in enumerate(F["support"])}
    novel = np.array(fold["novel"], bool)
    ti = np.array([idx[k] for k in fold["test"]])
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


def report(out_md, out_json, dates_path, variants, *run_dirs):
    variants = [v for v in variants.split(",") if v]
    with open(dates_path) as fh:
        dates = json.load(fh)
    runs = {}
    for d in run_dirs:
        with open(os.path.join(d, "folds.json")) as fh:
            F = json.load(fh)
        runs[(F["cutoff"], F["weighting"])] = (F, run_metrics(F, run_scores(d, variants)))
    cutoffs = sorted({c for c, _ in runs})
    weightings = [w for w in ("equal", "pooled") if any(ww == w for _, ww in runs)]
    res = {"coverage": dates["coverage"], "how": dates.get("how", {}),
           "year_distribution": temporal.year_distribution(dates), "runs": {}}
    for (c, w), (F, rows) in runs.items():
        fold = F["folds"][0]
        res["runs"][f"{c}_{w}"] = {"cutoff": c, "weighting": w, "n_train": len(fold["train"]),
                                   "n_test": len(fold["test"]), "n_test_models": len(fold["test_models"]),
                                   "frac_novel": float(np.mean(fold["novel"])) if fold["novel"] else None,
                                   "sizes": F["sizes"], "priors": rows, "novel": novel_examples(F)}
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
        priors = list(runs[(cutoffs[0], w)][1])
        order = sorted(priors, key=lambda p: (runs[(cutoffs[0], w)][1][p]["family"] != "tree",
                                              -np.mean([runs[(c, w)][1][p]["raw_novel"] or -1e9 for c in cutoffs])))
        L += ["", f"## Raw log P per test pair ({w} source weighting)", "",
              "all / seen / novel test pairs; rows ordered by the mean over cutoffs of the novel-form score "
              "(tree priors first). Zero-probability pairs are left out of the means (counted in the last column).",
              "", "| prior | family | " + " | ".join(f"{c} all | {c} seen | {c} novel" for c in cutoffs)
              + " | zero-probability pairs |", "|---|---|" + "---|---|---|" * len(cutoffs) + "---|"]
        for p in order:
            cells, zeros = [], []
            for c in cutoffs:
                m = runs[(c, w)][1][p]
                cells += [_f(m["raw"]), _f(m["raw_seen"]), _f(m["raw_novel"])]
                zeros.append(str(m["n_zero"]))
            L.append(f"| {p} | {runs[(cutoffs[0], w)][1][p]['family']} | " + " | ".join(cells) + " | "
                     + "/".join(zeros) + " |")
        L += ["", f"Rank of each tree prior on novel forms ({w}; 1 = highest log P):", "",
              "| prior | " + " | ".join(f"{c} seen | {c} novel" for c in cutoffs) + " |",
              "|---|" + "---|---|" * len(cutoffs)]
        trees = [p for p in order if runs[(cutoffs[0], w)][1][p]["family"] == "tree"]
        rk = {}
        for c in cutoffs:
            for what in ("raw_seen", "raw_novel"):
                srt = sorted(trees, key=lambda p: -(runs[(c, w)][1][p][what] or -1e9))
                for i, p in enumerate(srt):
                    rk[(p, c, what)] = i + 1
        for p in trees:
            L.append(f"| {p} | " + " | ".join(f"{rk[(p, c, 'raw_seen')]} | {rk[(p, c, 'raw_novel')]}"
                                              for c in cutoffs) + " |")
    L += ["", "## Novel forms", "", "Forms first used by a model dated at or after the cutoff (generalised keys; "
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
