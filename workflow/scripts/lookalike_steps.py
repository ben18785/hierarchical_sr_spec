"""Look-alike discrimination test (hsr.lookalike): design, evidence shards, report.

    python lookalike_steps.py design out.json mech_folds.json temporal_folds.json trees.json config.json
    python lookalike_steps.py shard design.json out.npz shard n_shards n_starts
    python lookalike_steps.py report out.md out.json design.json trees.json mech_run_dir temporal_run_dir \
        variants(comma-separated) shard.npz ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from hsr import lookalike as la  # noqa: E402
from hsr import treecode as tc  # noqa: E402

HOLDOUTS = ["rnls", "ratelaws", "ecocat2"]


def design(out, mech_folds, temporal_folds, trees, cfg_json):
    with open(mech_folds) as fh:
        MF = json.load(fh)
    with open(temporal_folds) as fh:
        TF = json.load(fh)
    with open(trees) as fh:
        T = json.load(fh)["trees"]
    cfg = json.loads(cfg_json) if cfg_json.lstrip().startswith("{") else json.load(open(cfg_json))
    labels = set(MF["folds"][0]["test_sources"])
    mech_names = sorted(lab for lab in labels if lab.startswith("mech_"))
    d = la.make_design(MF, TF, T, [h for h in HOLDOUTS if h in labels], mech_names, cfg)
    with open(out, "w") as fh:
        json.dump(d, fh)
    print(f"{len(d['true_forms'])} true forms, {len(d['datasets'])} datasets, {len(d['candidates'])} candidates")
    print(json.dumps(d["selection"], indent=1))


def shard(design_path, out, i, n, n_starts):
    with open(design_path) as fh:
        D = json.load(fh)
    keys, L, secs = la.evidence_shard(D, int(i), int(n), n_starts=int(n_starts))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, keys=np.array(keys, dtype=object), laplace=L, seconds=secs, allow_pickle=True)
    print(f"shard {i}/{n}: {len(keys)} candidates in {secs.sum():.0f} s; "
          f"{int(np.isfinite(L).all(axis=0).sum())} fitted on every dataset")


# ------------------------------------------------------------------ report
def _priors(run_dir, prefix, variants, trees, support):
    lp, fam = {}, {}
    for name in ["ours", "fnprior", "hpy"] + [f"{prefix}_{v}" for v in variants]:
        with open(os.path.join(run_dir, f"{name}.json")) as fh:
            d = json.load(fh)["logprior"]
        for k, v in d.items():
            lp[k] = dict(zip(support, (-np.inf if x is None else x for x in v[0])))
            fam[k] = "pcfg" if name == "ours" else "tree"
    lp["uniform symbols"] = {k: -len(trees[k]) * np.log(len(tc.V)) for k in support if trees.get(k)}
    fam["uniform symbols"] = "tree"
    return lp, fam


def _f(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def report(out_md, out_json, design_path, trees_path, mech_dir, temporal_dir, variants, *shards):
    variants = [v for v in variants.split(",") if v]
    with open(design_path) as fh:
        D = json.load(fh)
    with open(trees_path) as fh:
        trees = json.load(fh)["trees"]
    cfg = D["config"]
    deltas = [float(x) for x in cfg["deltas"]]
    cands = D["candidates"]
    col = {k: i for i, k in enumerate(cands)}
    J = len(D["datasets"])
    L = np.full((J, len(cands)), np.nan)
    secs = 0.0
    for p in shards:
        z = np.load(p, allow_pickle=True)
        for e, k in enumerate(z["keys"]):
            L[:, col[str(k)]] = z["laplace"][:, e]
        secs += float(z["seconds"].sum())
    missing = int(np.isnan(L).all(axis=0).sum())
    if missing:
        raise ValueError(f"{missing} candidates have no evidence (missing shards?)")
    with open(os.path.join(mech_dir, "folds.json")) as fh:
        msup = json.load(fh)["support"]
    with open(os.path.join(temporal_dir, "folds.json")) as fh:
        tsup = json.load(fh)["support"]
    lp_m, fam = _priors(mech_dir, "hpyv", variants, trees, msup)
    lp_t, _ = _priors(temporal_dir, "hpyvt", variants, trees, tsup)
    priors = ["flat"] + list(lp_m)
    in_t = np.array(D["in_temporal"], bool)
    all_idx = np.arange(len(cands))
    t_idx = all_idx[in_t]
    lp_arr = {}
    for p in priors:
        if p == "flat":
            lp_arr[("m", p)] = np.zeros(len(cands))
            lp_arr[("t", p)] = np.zeros(len(t_idx))
            continue
        lp_arr[("m", p)] = np.array([lp_m[p].get(k, -np.inf) for k in cands])
        lp_arr[("t", p)] = np.array([lp_t[p].get(cands[i], -np.inf) for i in t_idx])
    rows = {p: [] for p in priors}
    meta = []
    for j, ds in enumerate(D["datasets"]):
        tf = D["true_forms"][ds["form"]]
        temporal = tf["stratum"] == "temporal_novel"
        idx = t_idx if temporal else all_idx
        Lj = np.where(np.isfinite(L[j, idx]), L[j, idx], -np.inf)
        t = int(np.flatnonzero(idx == col[tf["key"]])[0])
        meta.append({"stratum": tf["stratum"], "seen": tf["seen"], "setting": f"n={ds['n']}, noise={ds['noise']}",
                     "true_fit": bool(np.isfinite(Lj[t]))})
        for p in priors:
            rows[p].append(la.dataset_scores(Lj, lp_arr[("t" if temporal else "m", p)], t, deltas))

    def groups():
        g = {"all": [True] * J}
        for s in dict.fromkeys(m["stratum"] for m in meta):
            g[s] = [m["stratum"] == s for m in meta]
        g["seen"] = [m["seen"] for m in meta]
        g["novel"] = [not m["seen"] for m in meta]
        for s in dict.fromkeys(m["setting"] for m in meta):
            g[s] = [m["setting"] == s for m in meta]
        return g
    G = groups()
    summ = {p: {g: la.summarise([r for r, keep in zip(rows[p], mask) if keep], deltas) for g, mask in G.items()}
            for p in priors}
    # order: flat, then tree priors by mean posterior probability of the truth, then PCFG grammars
    order = ["flat"] + sorted([p for p in priors if p != "flat"],
                              key=lambda p: (fam[p] != "tree", -summ[p]["all"]["p_post"]))
    res = {"config": cfg, "n_datasets": J, "n_candidates": len(cands), "n_candidates_temporal": int(in_t.sum()),
           "evidence_cpu_seconds": secs, "true_fit_fraction": float(np.mean([m["true_fit"] for m in meta])),
           "selection": D["selection"], "pool_sizes": D["pool_sizes"], "true_forms": D["true_forms"],
           "summary": summ, "family": {p: ("-" if p == "flat" else fam[p]) for p in priors}}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)

    strata = [g for g in G if g in {m["stratum"] for m in meta}]
    settings = [g for g in G if g.startswith("n=")]
    Lm = []
    Lm.append("# Look-alike discrimination test\n")
    Lm.append("When the data barely distinguish candidate forms, does the prior put its mass on the true one? "
              "True forms are drawn from the hold-out sources (seen and novel relative to the training sources), "
              "the mechanism classes (hsr.mechanisms) and the forms first used from the temporal cutoff "
              f"{cfg['temporal_cutoff']} on; datasets are simulated from each (hsr.simulate: constants N(0, 1.5^2) "
              "with rejection, x uniform on [0.1, 5], noise sd = noise x range of the curve); every candidate's "
              "Laplace log evidence is computed on every dataset (hsr.likelihood; GitHub Actions, "
              ".github/workflows/lookalike.yml). Design and scores: hsr.lookalike. Priors: trained on all training "
              "sources with equal source weighting (runs mechholdout_equal; for temporal true forms the priors "
              f"trained on models dated before {cfg['temporal_cutoff']}, run temporal_{cfg['temporal_cutoff']}_equal),"
              " raw log P(form).\n")
    Lm.append(f"Candidate set: the {len(cands)} support forms of the mechanism hold-out run with at most "
              f"{cfg['max_consts']} constants and {cfg['max_nodes']} nodes ({int(in_t.sum())} of them in the "
              f"temporal run's support, the candidate set for temporal true forms). {J} datasets: "
              f"{len(D['true_forms'])} true forms x {len(cfg['settings'])} (n, noise) settings x {cfg['reps']} "
              f"replicates. The true form's fit succeeded on {res['true_fit_fraction']:.1%} of datasets. "
              f"Evidence: {secs / 3600:.1f} CPU hours.\n")
    Lm.append("Look-alike set A_D: candidates within D nats of the best log evidence. rank: rank of the truth in A_D "
              "under the prior (1 = the prior's favourite, ties count half), given the truth is in A_D; top: "
              "fraction of those datasets where the truth is the prior's strict favourite in A_D; mass: prior "
              "probability of the truth renormalised within A_D (flat: 1/|A_D|); gain: mean log(mass x |A_D|), the "
              "log factor by which the prior beats the flat prior on the truth; P(true): posterior probability of "
              "the truth over the whole candidate set (log evidence + log prior); MAP: fraction of datasets where "
              "the truth is the posterior mode.\n")
    Lm.append("## True forms\n")
    Lm.append("| stratum | eligible distinct forms | chosen | seen | not simulable (skipped) |")
    Lm.append("|---|---|---|---|---|")
    for s in strata:
        sel = D["selection"][s]
        n_seen = sum(t["seen"] for t in D["true_forms"] if t["stratum"] == s)
        Lm.append(f"| {s} | {sel['eligible']} | {sel['chosen']} | {n_seen} | {sel['not_simulable']} |")
    Lm.append("")
    Lm.append("## How often is the truth a look-alike?\n")
    Lm.append("Fraction of datasets whose true form is in A_D, and the median size of A_D.\n")
    Lm.append("| group | datasets | " + " | ".join(f"truth in A_{D:g} | median A_{D:g}" for D in deltas) + " |")
    Lm.append("|---|---|" + "---|---|" * len(deltas))
    for g in ["all"] + strata + ["seen", "novel"] + settings:
        s = summ["flat"][g]
        Lm.append(f"| {g} | {s['n']} | " + " | ".join(f"{_f(s[f'in_set_{D}'])} | {_f(s[f'set_size_{D}'], 0)}"
                                                       for D in deltas) + " |")
    Lm.append("")
    for D in deltas:
        Lm.append(f"## All datasets, D = {D:g}\n")
        s0 = summ["flat"]["all"]
        Lm.append(f"Truth in A_{D:g} on {s0[f'n_in_{D}']} of {s0['n']} datasets.\n")
        Lm.append("| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |")
        Lm.append("|---|---|---|---|---|---|---|---|---|")
        for p in order:
            s = summ[p]["all"]
            Lm.append(f"| {p} | {res['family'][p]} | {_f(s[f'rank_{D}'])} | {_f(s[f'pct_{D}'])} | "
                      f"{_f(s[f'top_{D}'])} | {_f(s[f'mass_{D}'], 3)} | {_f(s[f'gain_{D}'])} | "
                      f"{_f(s['p_post'], 3)} | {_f(s['map'])} |")
        Lm.append("")
    D0 = deltas[-1]

    def split(title, gs, field, nd=2):
        Lm.append(f"## {title}\n")
        Lm.append("| prior | " + " | ".join(gs) + " |")
        Lm.append("|---|" + "---|" * len(gs))
        Lm.append("| datasets | " + " | ".join(str(summ["flat"][g]["n"]) for g in gs) + " |")
        for p in order:
            Lm.append(f"| {p} | " + " | ".join(_f(summ[p][g][field], nd) for g in gs) + " |")
        Lm.append("")
    split(f"Renormalised prior mass on the truth within A_{D0:g}, by stratum", strata + ["seen", "novel"],
          f"mass_{D0}", 3)
    split(f"Mean rank of the truth within A_{D0:g}, by stratum", strata + ["seen", "novel"], f"rank_{D0}")
    split("MAP accuracy (posterior mode = truth), by stratum", strata + ["seen", "novel"], "map")
    split("Posterior probability of the truth, by stratum", strata + ["seen", "novel"], "p_post", 3)
    split(f"Renormalised prior mass on the truth within A_{D0:g}, by (n, noise)", settings, f"mass_{D0}", 3)
    split("MAP accuracy, by (n, noise)", settings, "map")
    Lm.append("## True forms used\n")
    Lm.append("| stratum | form | seen | constants | nodes |")
    Lm.append("|---|---|---|---|---|")
    for t in D["true_forms"]:
        Lm.append(f"| {t['stratum']} | `{t['key']}` | {'yes' if t['seen'] else 'no'} | {t['n_consts']} | {t['nodes']} |")
    Lm.append("")
    with open(out_md, "w") as fh:
        fh.write("\n".join(Lm))


if __name__ == "__main__":
    {"design": design, "shard": shard, "report": report}[sys.argv[1]](*sys.argv[2:])
