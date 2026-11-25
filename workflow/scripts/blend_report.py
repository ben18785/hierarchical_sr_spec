"""Blended priors (rule blend_report): mixtures P_w = sum_k w_k P_k of tree
priors (hsr.blend), weights chosen on the leave-one-source-out runs only, then
evaluated, untouched, on the hold-out sources and the temporal hold-out.

Selection: for each source weighting (equal, pooled) separately, maximise the
mean over the twelve left-out sources of the mean raw log P_w per model-form
pair of that source, using the priors trained with that weighting. Two
components: grid search over w (hsr.blend.w_grid); three or more: EM (the
objective is concave, so both find the global maximum). The jackknife range
of w (the selection repeated leaving each LOSO source out of the objective)
shows how stable the choice is.

Evaluation: mean raw log P per test pair on all, seen and novel pairs, and
the OLS novelty penalty at fixed node count (hsr.gap_diagnostics.ols_novel),
for every component and blend. The "oracle" weights maximise the mean log P
over all pairs of a test set itself; they are shown for reference only and
are not a valid selection.

    python blend_report.py out.md out.json --variants v1 v2 ... --pairs A:B ... --triples A:B:C ...
        --loso dir ... --holdout dir ... --holdout-trees trees.json --temporal dir ... --temporal-trees trees.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr import blend  # noqa: E402
from hsr.gap_diagnostics import fmt as _f  # noqa: E402
from hsr.gap_diagnostics import mean as _mean  # noqa: E402
from hsr.gap_diagnostics import ols_novel  # noqa: E402

ALL_TREE = "all tree priors"


def parse_args(argv):
    out_md, out_json, rest = argv[0], argv[1], argv[2:]
    opts, key = {}, None
    for a in rest:
        if a.startswith("--"):
            key = a[2:]
            opts[key] = []
        else:
            opts[key].append(a)
    return out_md, out_json, opts


def tree_scores(run_dir, variants):
    """{prior: log P over the run's support} of every tree prior of a run
    (hpyv_<v>.json, or hpyvt_<v>.json for temporal runs)."""
    out = {}
    names = ["fnprior.json", "hpy.json"]
    for v in variants:
        names.append(f"hpyv_{v}.json" if os.path.exists(os.path.join(run_dir, f"hpyv_{v}.json"))
                     else f"hpyvt_{v}.json")
    for name in names:
        with open(os.path.join(run_dir, name)) as fh:
            for prior, per_fold in json.load(fh)["logprior"].items():
                if len(per_fold) != 1:
                    raise ValueError(f"{run_dir}/{name}: expected one fold")
                out[prior] = np.array([-np.inf if x is None else x for x in per_fold[0]], float)
    return out


def load_run(run_dir, variants, nodes_of=None):
    """Test pairs of a run: log P of every tree prior, seen mask, node counts, source labels."""
    with open(os.path.join(run_dir, "folds.json")) as fh:
        F = json.load(fh)
    fold = F["folds"][0]
    idx = {s: i for i, s in enumerate(F["support"])}
    ti = np.array([idx[k] for k in fold["test"]])
    lp = {p: v[ti] for p, v in tree_scores(run_dir, variants).items()}
    if "novel" in fold:
        seen = ~np.array(fold["novel"], bool)
    else:
        train = set(fold["train"])
        seen = np.array([k in train for k in fold["test"]], bool)
    nodes = np.array([(nodes_of or {}).get(k, np.nan) for k in fold["test"]], float)
    weighting = F.get("weighting") or ("equal" if F["test_corpus"].endswith("; equal)") else "pooled")
    return {"lp": lp, "seen": seen, "nodes": nodes, "weighting": weighting, "cutoff": F.get("cutoff"),
            "source": F["test_corpus"].split(" ")[0], "n": len(ti)}


def mix(run, comps, w):
    return blend.mix_logp([run["lp"][c] for c in comps], w)


def metrics(y, run):
    seen = run["seen"]
    slope, pen = ols_novel(y, run["nodes"], ~seen)
    return {"all": _mean(y), "seen": _mean(y[seen]), "novel": _mean(y[~seen]), "per_node": slope,
            "novel_penalty_fixed_size": pen, "n_zero": int((~np.isfinite(y)).sum())}


def fit(groups, comps, mean_over="sources"):
    """Weights maximising the mean log P over ``groups`` ({name: run}):
    grid search for two components, EM otherwise."""
    g = {s: [r["lp"][c] for c in comps] for s, r in groups.items()}
    if len(comps) == 2:
        w, obj, _ = blend.select_w(g, mean_over=mean_over)
        return np.array([w, 1.0 - w]), obj
    return blend.em_weights(g, mean_over=mean_over)


def loso_objective(groups, comps, w):
    return float(np.mean([_mean(mix(r, comps, w)) for r in groups.values()]))


def wstr(comps, w):
    if len(comps) == 2:
        return f"{w[0]:.3f}"
    return ", ".join(f"{c} {x:.3f}" for c, x in sorted(zip(comps, w), key=lambda t: -t[1]) if x >= 0.0005)


def main(out_md, out_json, opts):
    variants = opts["variants"]
    blends = [tuple(p.split(":")) for p in opts.get("pairs", [])] + [tuple(p.split(":")) for p in
                                                                    opts.get("triples", [])]
    with open(opts["holdout-trees"][0]) as fh:
        ho_nodes = {k: len(t) for k, t in json.load(fh)["trees"].items() if t}
    with open(opts["temporal-trees"][0]) as fh:
        te_nodes = {k: len(t) for k, t in json.load(fh)["trees"].items() if t}
    loso = {}
    for d in opts["loso"]:
        r = load_run(d, variants)
        loso.setdefault(r["weighting"], {})[r["source"]] = r
    holdout = {}
    for d in opts["holdout"]:
        r = load_run(d, variants, ho_nodes)
        holdout[r["weighting"]] = r
    temporal = {}
    for d in opts["temporal"]:
        r = load_run(d, variants, te_nodes)
        temporal[(r["cutoff"], r["weighting"])] = r
    weightings = [w for w in ("equal", "pooled") if w in loso]
    cutoffs = sorted({c for c, _ in temporal})
    priors = list(next(iter(loso[weightings[0]].values()))["lp"])
    blends.append(tuple(priors))  # every tree prior
    names = {b: " + ".join(b) for b in blends}
    names[blends[-1]] = ALL_TREE
    comps_all = list(dict.fromkeys(c for b in blends[:-1] for c in b))

    res = {"blends": {}, "components": {}, "weightings": weightings, "cutoffs": cutoffs}
    for wt in weightings:
        groups = loso[wt]
        for c in priors:
            res["components"].setdefault(c, {})[wt] = {
                "loso": float(np.mean([_mean(r["lp"][c]) for r in groups.values()])),
                "loso_by_source": {s: _mean(r["lp"][c]) for s, r in groups.items()},
                "holdout": metrics(holdout[wt]["lp"][c], holdout[wt]),
                "temporal": {str(cu): metrics(temporal[(cu, wt)]["lp"][c], temporal[(cu, wt)]) for cu in cutoffs}}
        for b in blends:
            w, obj = fit(groups, b)
            jack = [fit({s: r for s, r in groups.items() if s != drop}, b)[0] for drop in groups]
            d = {"components": list(b), "weights": [float(x) for x in w], "loso": obj,
                 "loso_by_source": {s: _mean(mix(r, b, w)) for s, r in groups.items()},
                 "jackknife_weights": [[float(x) for x in j] for j in jack]}
            if len(b) == 2:
                wem, _ = blend.em_weights({s: [r["lp"][c] for c in b] for s, r in groups.items()})
                d["loso_em_check"] = float(wem[0])
            hr = holdout[wt]
            d["holdout"] = metrics(mix(hr, b, w), hr)
            wo, oo = fit({"all": hr}, b, "pairs")
            d["holdout_oracle"] = {"weights": [float(x) for x in wo], "all": oo}
            d["temporal"], d["temporal_oracle"] = {}, {}
            for cu in cutoffs:
                tr = temporal[(cu, wt)]
                d["temporal"][str(cu)] = metrics(mix(tr, b, w), tr)
                wo, oo = fit({"all": tr}, b, "pairs")
                d["temporal_oracle"][str(cu)] = {"weights": [float(x) for x in wo], "all": oo}
            res["blends"].setdefault(names[b], {})[wt] = d
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)

    L = ["# Blended priors", "",
         "P_w(f) = sum_k w_k P_k(f) over tree priors, computed per form as logsumexp_k(log w_k + log P_k(f)) "
         "(hsr.blend). Every tree prior is proper over the one shared space of coded trees "
         "(docs/prior_models.tex), so every blend is proper. For two components P_w = w P_A + (1 - w) P_B and w "
         "is the weight of the first.", "",
         "Selection uses the leave-one-source-out runs only: for each source weighting, w maximises the mean over "
         f"the {len(loso[weightings[0]])} left-out sources of the mean raw log P_w per model-form pair of that "
         "source (each source counts equally), with the priors trained under that weighting. Two components: grid "
         "over w in [0, 1] (spacing 0.005, plus spacing 0.05 in logit(w) out to |logit w| = 12); three or more: "
         "EM (the objective is concave in w, so both reach the global maximum; the EM fit of every pair agrees to within the grid spacing "
         "with its grid choice, see blend.json `loso_em_check`). Jackknife: w re-chosen with each LOSO source "
         "left out of the objective in turn. The chosen w is then applied, untouched, to the hold-out sources "
         "(holdout.md) and the temporal hold-out (temporal.md), with the priors trained under the same weighting. "
         f"\"{ALL_TREE}\": EM mixture of every tree prior. Built by `snakemake --cores 1 blend`.", ""]
    L += ["## Weights chosen on LOSO", "",
          "LOSO columns: mean over left-out sources of mean raw log P per pair (the selection objective).", "",
          "| blend | weighting | chosen w | jackknife w range | LOSO first component | LOSO second component "
          "| LOSO blend | gain over best component |", "|---|---|---|---|---|---|---|---|"]
    for b in blends:
        for wt in weightings:
            d = res["blends"][names[b]][wt]
            cl = [res["components"][c][wt]["loso"] for c in b]
            if len(b) == 2:
                jk = [j[0] for j in d["jackknife_weights"]]
                jr = f"{min(jk):.3f} - {max(jk):.3f}"
                c1, c2 = _f(cl[0]), _f(cl[1])
            else:
                jr = "see blend.json"
                c1, c2 = "", ""
            L.append(f"| {names[b]} | {wt} | {wstr(b, d['weights'])} | {jr} | {c1} | {c2} | {_f(d['loso'])} | "
                     f"{_f(d['loso'] - max(cl))} |")
    L += ["", "Tree priors alone on LOSO (same objective): " + "; ".join(
        f"{c} " + " / ".join(_f(res["components"][c][wt]["loso"]) for wt in weightings) for c in
        sorted(priors, key=lambda c: -res["components"][c][weightings[0]]["loso"])) +
        f" ({' / '.join(weightings)}).", ""]

    def rows(wt):
        out = [(c, "", res["components"][c][wt]) for c in comps_all]
        out += [(names[b], wstr(b, res["blends"][names[b]][wt]["weights"]), res["blends"][names[b]][wt])
                for b in blends]
        return out

    L += ["## Hold-out sources", "",
          "All hold-out pairs pooled (holdout.md \"all\"): mean raw log P per pair on all / seen / novel pairs, "
          "and the novelty penalty at fixed node count (OLS coefficient of the novel indicator; negative = novel "
          "forms score lower than seen forms of the same size). Blends use the LOSO weights above.", ""]
    for wt in weightings:
        L += [f"### {wt} weighting", "", "| prior | weights | all | seen | novel | novel penalty |",
              "|---|---|---|---|---|---|"]
        for name, ws, d in rows(wt):
            m = d["holdout"]
            L.append(f"| {name} | {ws} | {_f(m['all'])} | {_f(m['seen'])} | {_f(m['novel'])} | "
                     f"{_f(m['novel_penalty_fixed_size'])} |")
        L.append("")
    L += ["## Temporal hold-out", "",
          "Per cutoff Y (temporal.md): all / seen / novel / novelty penalty at fixed size. Blends use the LOSO "
          "weights above.", ""]
    for wt in weightings:
        L += [f"### {wt} weighting", "", "| prior | " + " | ".join(
            f"{c} all | {c} seen | {c} novel | {c} penalty" for c in cutoffs) + " |",
            "|---|" + "---|---|---|---|" * len(cutoffs)]
        for name, ws, d in rows(wt):
            cells = []
            for c in cutoffs:
                m = d["temporal"][str(c)]
                cells += [_f(m["all"]), _f(m["seen"]), _f(m["novel"]), _f(m["novel_penalty_fixed_size"])]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Oracle weights (for reference only; NOT a valid selection)", "",
          "Weights maximising the mean raw log P over all pairs of each test set itself, and that maximum minus "
          "the score at the LOSO-chosen weights (the most any re-tuning on the test set could gain). Two "
          "components: weight of the first.", "",
          "| blend | weighting | LOSO w | hold-out oracle w (gain) | " + " | ".join(
              f"temporal {c} oracle w (gain)" for c in cutoffs) + " |", "|---|---|---|---|" + "---|" * len(cutoffs)]
    for b in blends:
        for wt in weightings:
            d = res["blends"][names[b]][wt]
            o = d["holdout_oracle"]
            cells = [f"{wstr(b, o['weights'])} ({_f(o['all'] - d['holdout']['all'])})"]
            for c in cutoffs:
                o = d["temporal_oracle"][str(c)]
                cells.append(f"{wstr(b, o['weights'])} ({_f(o['all'] - d['temporal'][str(c)]['all'])})")
            L.append(f"| {names[b]} | {wt} | {wstr(b, d['weights'])} | " + " | ".join(cells) + " |")
    L.append("")
    with open(out_md, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main(*parse_args(sys.argv[1:]))
