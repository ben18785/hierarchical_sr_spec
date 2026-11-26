"""Look-alike discrimination test (hsr.lookalike): design, evidence shards, reports.

    python lookalike_steps.py design out.json mech_folds.json temporal_folds.json trees.json config.json
    python lookalike_steps.py shard design.json out.npz shard n_shards n_starts
    python lookalike_steps.py report out.md out.json spec.json
    python lookalike_steps.py tempering out.md out.json spec.json

spec (a JSON string or file; built by the Snakefile):
    designs: [{name, title, design, shards: [npz, ...], run_id: file}, ...]
    trees: tree codings; variants: hpyv variants; deltas: look-alike thresholds D
    runs: {weighting: {"mech": run dir, "temporal": run dir}}
    taus, tau_grid, priors_md (tempering only)
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
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
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


# ------------------------------------------------------------- shared inputs
def _spec(s):
    return json.loads(s) if s.lstrip().startswith("{") else json.load(open(s))


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


def _evidence(DES, shards):
    """(J, C) Laplace log evidence of the design's candidates (-inf: failed fit) and the CPU seconds."""
    cands = DES["candidates"]
    col = {k: i for i, k in enumerate(cands)}
    L = np.full((len(DES["datasets"]), len(cands)), np.nan)
    secs = 0.0
    for p in shards:
        z = np.load(p, allow_pickle=True)
        for e, k in enumerate(z["keys"]):
            L[:, col[str(k)]] = z["laplace"][:, e]
        secs += float(z["seconds"].sum())
    missing = int(np.isnan(L).all(axis=0).sum())
    if missing:
        raise ValueError(f"{missing} candidates have no evidence (missing shards?)")
    return np.where(np.isfinite(L), L, -np.inf), secs


def _prior_arrays(DES, trees, run, variants):
    """{("m"|"t", prior): (C_m,) or (C_t,) log prior over the design's candidate sets} and the families."""
    cands = DES["candidates"]
    in_t = np.array(DES["in_temporal"], bool)
    with open(os.path.join(run["mech"], "folds.json")) as fh:
        msup = json.load(fh)["support"]
    with open(os.path.join(run["temporal"], "folds.json")) as fh:
        tsup = json.load(fh)["support"]
    lp_m, fam = _priors(run["mech"], "hpyv", variants, trees, msup)
    lp_t, _ = _priors(run["temporal"], "hpyvt", variants, trees, tsup)
    t_keys = [k for k, b in zip(cands, in_t) if b]
    arr = {("m", "flat"): np.zeros(len(cands)), ("t", "flat"): np.zeros(len(t_keys))}
    for p in lp_m:
        arr[("m", p)] = np.array([lp_m[p].get(k, -np.inf) for k in cands])
        arr[("t", p)] = np.array([lp_t[p].get(k, -np.inf) for k in t_keys])
    return arr, fam


class Design:
    """A design with its evidence: per dataset, the candidate set, its log evidence and the truth's index."""

    def __init__(self, spec_d, trees):
        with open(spec_d["design"]) as fh:
            self.DES = json.load(fh)
        self.name, self.title = spec_d["name"], spec_d.get("title", spec_d["name"])
        self.run_id = open(spec_d["run_id"]).read().strip() if spec_d.get("run_id") else ""
        self.L, self.secs = _evidence(self.DES, spec_d["shards"])
        cands = self.DES["candidates"]
        col = {k: i for i, k in enumerate(cands)}
        in_t = np.array(self.DES["in_temporal"], bool)
        all_idx = np.arange(len(cands))
        t_idx = all_idx[in_t]
        self.n_cand, self.n_cand_t = len(cands), int(in_t.sum())
        self.items, self.meta = [], []
        for j, ds in enumerate(self.DES["datasets"]):
            tf = self.DES["true_forms"][ds["form"]]
            kind = "t" if tf["stratum"] == "temporal_novel" else "m"
            idx = t_idx if kind == "t" else all_idx
            t = int(np.flatnonzero(idx == col[tf["key"]])[0])
            Lj = self.L[j, idx]
            self.items.append((kind, Lj, t))
            self.meta.append({"form": ds["form"], "stratum": tf["stratum"], "seen": tf["seen"],
                              "setting": f"n={ds['n']}, noise={ds['noise']}", "true_fit": bool(np.isfinite(Lj[t]))})
        self.true_fit = float(np.mean([m["true_fit"] for m in self.meta]))

    def groups(self):
        meta = self.meta
        g = {"all": [True] * len(meta)}
        for s in dict.fromkeys(m["stratum"] for m in meta):
            g[s] = [m["stratum"] == s for m in meta]
        g["seen"] = [m["seen"] for m in meta]
        g["novel"] = [not m["seen"] for m in meta]
        for s in dict.fromkeys(m["setting"] for m in meta):
            g[s] = [m["setting"] == s for m in meta]
        return g

    def rows(self, lp_arr, prior, deltas, tau=None):
        """dataset_scores of every dataset under one prior (tempered by tau if given)."""
        cache = {}
        out = []
        for kind, Lj, t in self.items:
            if kind not in cache:
                lp = lp_arr[(kind, prior)]
                cache[kind] = lp if tau is None else la.temper(lp, tau)
            out.append(la.dataset_scores(Lj, cache[kind], t, deltas))
        return out

    def summaries(self, rows, deltas):
        return {g: la.summarise([r for r, keep in zip(rows, mask) if keep], deltas) for g, mask in self.groups().items()}


def _f(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


# ------------------------------------------------------------------ report
def _design_md(Lm, d, summ, order, family, deltas, cfg):
    """The per-design tables (one weighting: summ[prior][group])."""
    DES = d.DES
    G = d.groups()
    strata = [g for g in G if g in {m["stratum"] for m in d.meta}]
    settings = [g for g in G if g.startswith("n=")]
    Lm.append(f"Candidate set: the {d.n_cand} support forms of the mechanism hold-out run with at most "
              f"{cfg['max_consts']} constants and {cfg['max_nodes']} nodes ({d.n_cand_t} of them in the "
              f"temporal run's support, the candidate set for temporal true forms). {len(DES['datasets'])} datasets: "
              f"{len(DES['true_forms'])} true forms x {len(cfg['settings'])} (n, noise) settings x {cfg['reps']} "
              f"replicates. The true form's fit succeeded on {d.true_fit:.1%} of datasets. "
              f"Evidence: {d.secs / 3600:.1f} CPU hours (GitHub Actions run {d.run_id}).\n")
    Lm.append("### True forms\n")
    Lm.append("| stratum | eligible distinct forms | chosen | seen | not simulable (skipped) |")
    Lm.append("|---|---|---|---|---|")
    for s in strata:
        sel = DES["selection"][s]
        n_seen = sum(t["seen"] for t in DES["true_forms"] if t["stratum"] == s)
        Lm.append(f"| {s} | {sel['eligible']} | {sel['chosen']} | {n_seen} | {sel['not_simulable']} |")
    Lm.append("")
    Lm.append("### How often is the truth a look-alike?\n")
    Lm.append("Fraction of datasets whose true form is in A_D, and the median size of A_D.\n")
    Lm.append("| group | datasets | " + " | ".join(f"truth in A_{D:g} | median A_{D:g}" for D in deltas) + " |")
    Lm.append("|---|---|" + "---|---|" * len(deltas))
    for g in ["all"] + strata + ["seen", "novel"] + settings:
        s = summ["flat"][g]
        Lm.append(f"| {g} | {s['n']} | " + " | ".join(f"{_f(s[f'in_set_{D}'])} | {_f(s[f'set_size_{D}'], 0)}"
                                                       for D in deltas) + " |")
    Lm.append("")
    for D in deltas:
        Lm.append(f"### All datasets, D = {D:g}\n")
        s0 = summ["flat"]["all"]
        Lm.append(f"Truth in A_{D:g} on {s0[f'n_in_{D}']} of {s0['n']} datasets.\n")
        Lm.append("| prior | family | mean rank | rank percentile | top | mass | gain | P(true) | MAP |")
        Lm.append("|---|---|---|---|---|---|---|---|---|")
        for p in order:
            s = summ[p]["all"]
            Lm.append(f"| {p} | {family[p]} | {_f(s[f'rank_{D}'])} | {_f(s[f'pct_{D}'])} | "
                      f"{_f(s[f'top_{D}'])} | {_f(s[f'mass_{D}'], 3)} | {_f(s[f'gain_{D}'])} | "
                      f"{_f(s['p_post'], 3)} | {_f(s['map'])} |")
        Lm.append("")
    D0 = deltas[len(deltas) // 2]          # the middle threshold (5 nats by default) for the splits

    def split(title, gs, field, nd=2):
        Lm.append(f"### {title}\n")
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
    Lm.append("### True forms used\n")
    Lm.append("| stratum | form | seen | constants | nodes |")
    Lm.append("|---|---|---|---|---|")
    for t in DES["true_forms"]:
        Lm.append(f"| {t['stratum']} | `{t['key']}` | {'yes' if t['seen'] else 'no'} | {t['n_consts']} | {t['nodes']} |")
    Lm.append("")
    return D0


def _weighting_md(Lm, summ_w, order, D0):
    """Equal against pooled source weighting: one row per (prior, weighting)."""
    ws = list(summ_w)
    Lm.append("### Source weighting: " + " vs ".join(ws) + "\n")
    Lm.append(f"P(true): mean posterior probability of the truth; log P: mean log P(true) over datasets whose "
              f"true-form fit succeeded; mass: renormalised prior mass on the truth within A_{D0:g}.\n")
    Lm.append("| prior | weighting | P(true) all | P(true) seen | P(true) novel | log P seen | log P novel | "
              f"MAP seen | MAP novel | mass seen | mass novel |")
    Lm.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in order:
        for w in (ws if p != "flat" else ws[:1]):
            s = summ_w[w][p]
            Lm.append(f"| {p} | {w if p != 'flat' else '-'} | {_f(s['all']['p_post'], 3)} | "
                      f"{_f(s['seen']['p_post'], 3)} | {_f(s['novel']['p_post'], 3)} | "
                      f"{_f(s['seen']['log_p_post'])} | {_f(s['novel']['log_p_post'])} | {_f(s['seen']['map'])} | "
                      f"{_f(s['novel']['map'])} | {_f(s['seen'][f'mass_{D0}'], 3)} | {_f(s['novel'][f'mass_{D0}'], 3)} |")
    Lm.append("")


def report(out_md, out_json, spec):
    S = _spec(spec)
    with open(S["trees"]) as fh:
        trees = json.load(fh)["trees"]
    deltas = [float(x) for x in S["deltas"]]
    weightings = list(S["runs"])
    res = {"designs": {}, "weightings": weightings}
    Lm = ["# Look-alike discrimination test\n"]
    Lm.append("When the data barely distinguish candidate forms, does the prior put its mass on the true one? "
              "True forms are drawn from the hold-out sources (seen and novel relative to the training sources), "
              "the mechanism classes (hsr.mechanisms) and the forms first used from the temporal cutoff on; "
              "datasets are simulated from each (hsr.simulate: constants N(0, 1.5^2) with rejection, x uniform on "
              "[0.1, 5], noise sd = noise x range of the curve); every candidate's Laplace log evidence is computed "
              "on every dataset (hsr.likelihood; GitHub Actions, .github/workflows/lookalike.yml). Design and "
              "scores: hsr.lookalike. Priors: trained on all training sources (runs mechholdout_<weighting>; for "
              "temporal true forms the priors trained on models dated before the cutoff, run "
              "temporal_<cutoff>_<weighting>), raw log P(form). The main tables use equal source weighting; "
              "pooled weighting is compared at the end of each design's section. Prior tempering: "
              "lookalike_tempering.md.\n")
    Lm.append("Look-alike set A_D: candidates within D nats of the best log evidence. rank: rank of the truth in A_D "
              "under the prior (1 = the prior's favourite, ties count half), given the truth is in A_D; top: "
              "fraction of those datasets where the truth is the prior's strict favourite in A_D; mass: prior "
              "probability of the truth renormalised within A_D (flat: 1/|A_D|); gain: median over datasets of "
              "log(mass x |A_D|), the log factor by which the prior beats the flat prior on the truth; P(true): "
              "posterior probability of the truth over the whole candidate set (log evidence + log prior); MAP: "
              "fraction of datasets where the truth is the posterior mode.\n")
    Lm.append("## Designs\n")
    Lm.append("| design | max constants | max nodes | candidates | true forms | datasets | true fit | "
              "evidence CPU h | Actions run |")
    Lm.append("|---|---|---|---|---|---|---|---|---|")
    designs = [Design(sd, trees) for sd in S["designs"]]
    for d in designs:
        c = d.DES["config"]
        Lm.append(f"| {d.title} | {c['max_consts']} | {c['max_nodes']} | {d.n_cand} | {len(d.DES['true_forms'])} | "
                  f"{len(d.DES['datasets'])} | {d.true_fit:.1%} | {d.secs / 3600:.1f} | {d.run_id} |")
    Lm.append("")
    for d in designs:
        cfg = d.DES["config"]
        summ_w, fam = {}, {}
        for w in weightings:
            lp_arr, fam = _prior_arrays(d.DES, trees, S["runs"][w], S["variants"])
            priors = ["flat"] + [p for p in fam]
            summ_w[w] = {p: d.summaries(d.rows(lp_arr, p, deltas), deltas) for p in priors}
        family = {"flat": "-", **fam}
        w0 = weightings[0]
        # order: flat, then tree priors by mean posterior probability of the truth, then PCFG grammars
        order = ["flat"] + sorted(fam, key=lambda p: (fam[p] != "tree", -summ_w[w0][p]["all"]["p_post"]))
        Lm.append(f"## {d.title}\n")
        D0 = _design_md(Lm, d, summ_w[w0], order, family, deltas, cfg)
        _weighting_md(Lm, summ_w, order, D0)
        res["designs"][d.name] = {
            "title": d.title, "config": cfg, "run_id": d.run_id, "n_datasets": len(d.DES["datasets"]),
            "n_candidates": d.n_cand, "n_candidates_temporal": d.n_cand_t, "evidence_cpu_seconds": d.secs,
            "true_fit_fraction": d.true_fit, "selection": d.DES["selection"], "pool_sizes": d.DES["pool_sizes"],
            "true_forms": d.DES["true_forms"], "family": family, "summary": summ_w}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    with open(out_md, "w") as fh:
        fh.write("\n".join(Lm))


# --------------------------------------------------------------- tempering
def tempering(out_md, out_json, spec):
    """P(form)^tau / Z over the candidate set; tau chosen by leave-one-truth-form-out (hsr.lookalike)."""
    S = _spec(spec)
    with open(S["trees"]) as fh:
        trees = json.load(fh)["trees"]
    deltas = [float(x) for x in S["deltas"]]
    D0 = deltas[len(deltas) // 2]
    taus = [float(t) for t in S["taus"]]
    grid = [float(t) for t in S["tau_grid"]]
    weightings = list(S["runs"])
    res = {"taus": taus, "tau_grid": grid, "weightings": weightings, "designs": {}}
    Lm = ["# Prior tempering in the look-alike test\n"]
    Lm.append("The prior over forms is raised to a power tau and renormalised over the look-alike candidate set: "
              "P_tau(form) = P(form)^tau / Z, Z = sum over candidates of P(c)^tau (hsr.lookalike.temper). Z is a "
              "finite sum with positive terms, so P_tau is a proper distribution on the candidate set for every "
              "tau >= 0. tau = 0 is the flat prior, tau = 1 the prior itself (renormalised over the set), tau > 1 "
              "sharpens it. Datasets, candidates and Laplace evidence are those of lookalike.md; no new fits.\n")
    Lm.append(f"P(true): mean posterior probability of the truth over the whole candidate set; log P: mean log "
              f"P(true) over the datasets whose true-form fit succeeded (the others have P(true) = 0 for every "
              f"tau); post. rank: median rank of the truth under the posterior (1 = MAP); mass: prior mass on the "
              f"truth renormalised within A_{D0:g} (the candidates within {D0:g} nats of the best evidence), mean "
              f"over datasets with the truth in A_{D0:g}; MAP: fraction of datasets where the truth is the "
              f"posterior mode. The rank of the truth within A_{D0:g} under the prior (lookalike.md) does not "
              f"depend on tau > 0, since P^tau is monotone in P.\n")
    Lm.append(f"Choice of tau: leave-one-truth-form-out over the grid {grid[0]:g}, {grid[1]:g}, ..., {grid[-1]:g}. "
              "For each true form, tau maximises the mean log P(true) on the datasets of all other true forms "
              "(datasets whose true-form fit failed are left out); the held-out score is the mean log P(true) on "
              "the left-out form's datasets at that tau. tau* (all): the maximiser on all datasets; tau* seen / "
              "novel: the maximiser on seen / novel truths only (in-sample).\n")
    designs = [Design(sd, trees) for sd in S["designs"]]
    for d in designs:
        G = d.groups()
        seen = np.array(G["seen"])
        ok = np.array([m["true_fit"] for m in d.meta])
        forms = np.array([m["form"] for m in d.meta])
        out_d = {"by_tau": {}, "loto": {}}
        Lm.append(f"## {d.title}\n")
        Lm.append(f"{len(d.DES['datasets'])} datasets ({int(ok.sum())} with a successful true-form fit), "
                  f"{len(d.DES['true_forms'])} true forms, {d.n_cand} candidates.\n")
        loto_rows, tau_rows = {}, {}
        for w in weightings:
            lp_arr, fam = _prior_arrays(d.DES, trees, S["runs"][w], S["variants"])
            priors = [p for p in fam]
            flat_rows = d.rows(lp_arr, "flat", deltas)
            flat_s = d.summaries(flat_rows, deltas)
            out_d["by_tau"].setdefault("flat", flat_s)
            for p in priors:
                by = {}
                for tau in taus:
                    by[f"{tau:g}"] = d.summaries(d.rows(lp_arr, p, deltas, tau=tau), deltas)
                out_d["by_tau"].setdefault(p, {})[w] = by
                # LOTO on the fine grid
                tl = {kind: la.temper_all(lp_arr[(kind, p)], grid) for kind in ("m", "t")}
                logp = np.stack([la.log_p_true_taus(Lj, tl[kind], t) for kind, Lj, t in d.items])
                lg, fo, se = logp[ok], forms[ok], seen[ok]
                sel = la.select_tau_loto(lg, fo, grid)
                cv = sel["cv"]
                tf = np.array(list(sel["tau_folds"].values()))
                k1, k0 = grid.index(1.0), grid.index(0.0)
                r = {"tau_full": sel["tau_full"], "tau_seen": float(grid[int(np.argmax(lg[se].mean(axis=0)))]),
                     "tau_novel": float(grid[int(np.argmax(lg[~se].mean(axis=0)))]),
                     "tau_folds": sel["tau_folds"], "tau_folds_median": float(np.median(tf)),
                     "tau_folds_min": float(tf.min()), "tau_folds_max": float(tf.max()),
                     "cv_log_p": float(cv.mean()), "cv_log_p_seen": float(cv[se].mean()),
                     "cv_log_p_novel": float(cv[~se].mean()),
                     "log_p_tau1": float(lg[:, k1].mean()), "log_p_tau0": float(lg[:, k0].mean()),
                     "log_p_tau1_seen": float(lg[se, k1].mean()), "log_p_tau1_novel": float(lg[~se, k1].mean()),
                     "mean_log_p_grid": sel["mean_full"], "n": int(ok.sum())}
                out_d["loto"].setdefault(p, {})[w] = r
                loto_rows.setdefault(p, {})[w] = r
                tau_rows.setdefault(p, {})[w] = by
        out_d["family"] = fam
        res["designs"][d.name] = out_d
        w0 = weightings[0]
        order = sorted(loto_rows, key=lambda p: (fam[p] != "tree", -loto_rows[p][w0]["cv_log_p"]))
        fl = flat_s
        Lm.append("### Leave-one-truth-form-out choice of tau\n")
        Lm.append(f"Flat prior (tau = 0): mean log P(true) {_f(fl['all']['log_p_post'])} (seen "
                  f"{_f(fl['seen']['log_p_post'])}, novel {_f(fl['novel']['log_p_post'])}).\n")
        Lm.append("| prior | weighting | tau* (all) | tau* seen | tau* novel | LOTO tau median [range] | "
                  "LOTO log P | LOTO log P seen | LOTO log P novel | log P at tau=1 | log P at tau=0 |")
        Lm.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in order:
            for w in weightings:
                r = loto_rows[p][w]
                Lm.append(f"| {p} | {w} | {r['tau_full']:g} | {r['tau_seen']:g} | {r['tau_novel']:g} | "
                          f"{r['tau_folds_median']:g} [{r['tau_folds_min']:g}, {r['tau_folds_max']:g}] | "
                          f"{_f(r['cv_log_p'])} | {_f(r['cv_log_p_seen'])} | {_f(r['cv_log_p_novel'])} | "
                          f"{_f(r['log_p_tau1'])} | {_f(r['log_p_tau0'])} |")
        Lm.append("")
        show = [p for p in S["priors_md"] if p in tau_rows]
        for w in weightings:
            Lm.append(f"### By tau, {w} source weighting\n")
            Lm.append("| prior | tau | P(true) seen | P(true) novel | log P seen | log P novel | post. rank seen | "
                      f"post. rank novel | mass seen | mass novel | MAP seen | MAP novel |")
            Lm.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

            def row(name, tau, s):
                a, b = s["seen"], s["novel"]
                return (f"| {name} | {tau} | {_f(a['p_post'], 3)} | {_f(b['p_post'], 3)} | {_f(a['log_p_post'])} | "
                        f"{_f(b['log_p_post'])} | {_f(a['post_rank_median'], 0)} | {_f(b['post_rank_median'], 0)} | "
                        f"{_f(a[f'mass_{D0}'], 3)} | {_f(b[f'mass_{D0}'], 3)} | {_f(a['map'])} | {_f(b['map'])} |")
            Lm.append(row("flat", "-", fl))
            for p in show:
                for tau in taus:
                    Lm.append(row(p, f"{tau:g}", tau_rows[p][w][f"{tau:g}"]))
            Lm.append("")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    with open(out_md, "w") as fh:
        fh.write("\n".join(Lm))


if __name__ == "__main__":
    {"design": design, "shard": shard, "report": report, "tempering": tempering}[sys.argv[1]](*sys.argv[2:])
