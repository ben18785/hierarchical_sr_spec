"""Effect of a change of canonicalisation on the corpora (results/grammar/canon_changes.md).

    python canon_changes.py OLD_DIR NEW_DIR OLD_GEN NEW_GEN out.md [out.json]

OLD_DIR / NEW_DIR hold corpus_*.json (and corpus.json) before and after a
rebuild; OLD_GEN / NEW_GEN are the matching corpus_generalised.json files
(BioModels curated keys keep their literals; their generalised forms are what
the priors see). Per source: distinct forms before and after, keys that
disappeared or appeared, models whose forms changed, and forms merged (old
keys that the new canonicalisation maps to one key, via
hsr.canonical.reduce_key). Redundant forms are counted with the Jacobian-rank
test of workflow/scripts/holdout_report.py (as it is) and with the
row-equilibrated test of hsr.canonical.is_redundant. Changed pairs are
checked numerically: the new form is fitted to curves of the old one and
conversely, at random positive constants.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from hsr.candidates import skeleton_from_key  # noqa: E402
from hsr.canonical import consts_of, is_redundant, reduce_key  # noqa: E402

TRAIN = ["biomodels", "sabiork", "cellml", "curvelib", "biocat", "biomodels_nc", "thermal", "ecocurves", "modeldb",
         "nlmixr2lib", "allodb", "jws"]
HOLDOUT = ["rnls", "ratelaws", "ecocat2"]


def _path(d, name):
    return os.path.join(d, "corpus.json" if name == "biomodels" else f"corpus_{name}.json")


def _load(d, name, gen=None):
    """{model: set of forms}; BioModels keys mapped through the generalised map."""
    with open(_path(d, name)) as fh:
        pm = json.load(fh)["per_model"]
    out = {}
    for m, ks in pm.items():
        if gen is not None:
            ks = [gen.get(k) for k in ks]
        out[m] = {k for k in ks if k}
    return out


def _redundant_report(key):
    from holdout_report import redundant
    return bool(redundant(key))


def _redundant_eq(key):
    try:
        return is_redundant(skeleton_from_key(key).expr)
    except Exception:
        return False


def _n_consts(key):
    try:
        return len(consts_of(skeleton_from_key(key).expr))
    except Exception:
        return None


def _fit_residual(src_key, dst_key, seed=0, n_tries=2):
    """Worst relative RMS misfit of dst fitted to curves of src (None if no curve could be drawn)."""
    from hsr.likelihood import CompiledSkeleton, _lm
    rng = np.random.default_rng(seed)
    try:
        a, b = CompiledSkeleton(skeleton_from_key(src_key)), CompiledSkeleton(skeleton_from_key(dst_key))
    except Exception:
        return None
    x = np.linspace(0.3, 3.0, 40)
    worst, n_ok = 0.0, 0
    for _ in range(30):
        if n_ok == n_tries:
            break
        phi = rng.uniform(0.5, 1.5, a.k)
        with np.errstate(all="ignore"):
            y = a.f(x[None, :, None], phi[None, :])[0]
        if not np.all(np.isfinite(y)) or np.ptp(y) < 1e-3 or np.abs(y).max() > 1e6:
            continue
        S = 48
        starts = np.abs(rng.normal(1.0, 1.0, (S, b.k))) + 0.05
        X = np.broadcast_to(x[None, :, None], (S, 40, 1))
        Y = np.broadcast_to(y[None, :], (S, 40))
        with np.errstate(all="ignore"):
            phib, _, _ = _lm(b, X, Y, np.ones((S, 40)), starts, np.full(S, 40.0), 3.0, False, 800, 1e-15)
            rms = np.sqrt(np.mean((b.f(X, phib) - Y) ** 2, axis=1))
        r = np.nanmin(rms) / (np.std(y) + 1e-12) if np.any(np.isfinite(rms)) else np.inf
        worst = max(worst, r)
        n_ok += 1
    return worst if n_ok else None


def main(old_dir, new_dir, old_gen_path, new_gen_path, out_md, out_json=None, n_check=40):
    with open(old_gen_path) as fh:
        old_gen = json.load(fh)["map"]
    with open(new_gen_path) as fh:
        new_gen = json.load(fh)["map"]
    rows, examples, all_old, all_new = [], {}, {}, {}
    per_source = {}
    red_cache = {}

    def red(k):
        if k not in red_cache:
            red_cache[k] = (_redundant_report(k), _redundant_eq(k))
        return red_cache[k]

    for name in TRAIN + HOLDOUT:
        old = _load(old_dir, name, old_gen if name == "biomodels" else None)
        new = _load(new_dir, name, new_gen if name == "biomodels" else None)
        f_old = Counter(k for ks in old.values() for k in ks)
        f_new = Counter(k for ks in new.values() for k in ks)
        mapping = {}
        for k in f_old:
            try:
                mapping[k] = reduce_key(k)
            except Exception:
                mapping[k] = k
        pre = defaultdict(set)
        for k, v in mapping.items():
            pre[v].add(k)
        merged = {v: ks for v, ks in pre.items() if len(ks) > 1}
        changed = {k: v for k, v in mapping.items() if v != k}
        models_changed = sum(1 for m in set(old) | set(new) if old.get(m, set()) != new.get(m, set()))
        mapped_in_new = sum(1 for v in set(mapping.values()) if v in f_new)
        r_old = [red(k) for k in f_old]
        r_new = [red(k) for k in f_new]
        per_source[name] = dict(
            old_forms=len(f_old), new_forms=len(f_new),
            gone=len(set(f_old) - set(f_new)), appeared=len(set(f_new) - set(f_old)),
            changed_by_reduce=len(changed), merged_groups=len(merged),
            merged_old_keys=sum(len(ks) for ks in merged.values()),
            models=len(set(old) | set(new)), models_changed=models_changed,
            mapped_forms=len(set(mapping.values())), mapped_in_new=mapped_in_new,
            red_report_old=sum(a for a, _ in r_old), red_report_new=sum(a for a, _ in r_new),
            red_eq_old=sum(b for _, b in r_old), red_eq_new=sum(b for _, b in r_new),
            pairs_old=sum(f_old.values()), pairs_new=sum(f_new.values()))
        for k, v in changed.items():
            ex = examples.setdefault((k, v), {"models": 0, "sources": set()})
            ex["models"] += f_old[k]
            ex["sources"].add(name)
        per_source[name]["_top_red_old"] = sorted(((n, k) for k, n in f_old.items() if red(k)[0]), reverse=True)[:12]
        per_source[name]["_top_red_new"] = [(n, k, red(k)[1]) for n, k in
                                            sorted(((n, k) for k, n in f_new.items() if red(k)[0]), reverse=True)[:15]]
        for k, n in f_old.items():
            all_old.setdefault(k, set()).add(name)
        for k, n in f_new.items():
            all_new.setdefault(k, set()).add(name)
        print(name, per_source[name], flush=True)

    def tot(names, field):
        return sum(per_source[n][field] for n in names)

    def union_red(names, which, new):
        pool = all_new if new else all_old
        ks = [k for k, srcs in pool.items() if srcs & set(names)]
        idx = 0 if which == "report" else 1
        return sum(red(k)[idx] for k in ks), len(ks)

    # numerical check of the most used changes
    top = sorted(examples.items(), key=lambda kv: (-kv[1]["models"], kv[0]))
    checks = []
    for (k, v), ex in top[:n_check]:
        fwd, bwd = _fit_residual(k, v), _fit_residual(v, k)
        checks.append({"old": k, "new": v, "models": ex["models"], "sources": sorted(ex["sources"]),
                       "consts_old": _n_consts(k), "consts_new": _n_consts(v),
                       "new_fits_old": fwd, "old_fits_new": bwd,
                       "new_redundant": red(v)[0], "new_redundant_eq": red(v)[1]})
        print(checks[-1], flush=True)

    L = ["# Canonicalisation changes (branch canon)", ""]
    L.append("Corpora rebuilt from the cached downloads with the new canonicalisation (hsr/canonical.py: "
             "deterministic per-law CPU-time limits, exp(A*log(g)) -> g**A, shared constant bases c**f(x) -> "
             "exp(c*f(x)), exp(a)*exp(b) -> exp(a + b), reduce_redundant and reduce_key), compared with the "
             "corpora committed on `themes` (bea2b95). BioModels curated keys are compared through their "
             "generalised forms (corpus_generalised.json), as the priors see them. Generated by "
             "workflow/scripts/canon_changes.py.")
    L.append("")
    L.append("Columns: forms = distinct forms (expressible keys); gone / new = forms present only before / only "
             "after; recanon. = old forms whose final form (hsr.canonical.reduce_key) differs; merged = groups of "
             "two or more old forms that now share one form (old forms in them); models chg. = models whose set "
             "of forms changed; redundant = forms whose constants are not identifiable, by the Jacobian-rank test "
             "of holdout_report.py / by the row-equilibrated test (hsr.canonical.is_redundant), before -> after.")
    L.append("")
    L.append("| source | models | models chg. | forms before | forms after | gone | new | recanon. | merged (old forms) "
             "| redundant (report test) | redundant (equilibrated) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in TRAIN + HOLDOUT:
        p = per_source[name]
        tag = " (hold-out)" if name in HOLDOUT else ""
        L.append(f"| {name}{tag} | {p['models']} | {p['models_changed']} | {p['old_forms']} | {p['new_forms']} | "
                 f"{p['gone']} | {p['appeared']} | {p['changed_by_reduce']} | {p['merged_groups']} "
                 f"({p['merged_old_keys']}) | {p['red_report_old']} -> {p['red_report_new']} | "
                 f"{p['red_eq_old']} -> {p['red_eq_new']} |")
    L.append("")
    for label, names in (("all training sources", TRAIN), ("hold-out sources", HOLDOUT), ("all sources", TRAIN + HOLDOUT)):
        ro, no = union_red(names, "report", False)
        rn, nn = union_red(names, "report", True)
        eo, _ = union_red(names, "eq", False)
        en, _ = union_red(names, "eq", True)
        L.append(f"- {label}: distinct forms {no} -> {nn}; redundant (report test) {ro} -> {rn}; "
                 f"redundant (equilibrated test) {eo} -> {en}.")
    L.append("")
    L.append("Model-form pairs (each model's distinct forms once): " + ", ".join(
        f"{n} {per_source[n]['pairs_old']} -> {per_source[n]['pairs_new']}" for n in TRAIN + HOLDOUT) + ".")
    L.append("")
    L.append("## Most used changes")
    L.append("")
    L.append("Old form -> new form, number of models using the old form (all sources), constants before -> after, "
             "and the numerical check: relative RMS misfit of the new form fitted to curves of the old one and "
             "of the old form fitted to curves of the new one (random positive constants; < 1e-4 means the same "
             "family up to reparametrisation; '-' no finite curve drawn), and whether the new form still fails "
             "the report's rank test.")
    L.append("")
    L.append("| old | new | models | consts | new fits old | old fits new | still redundant |")
    L.append("|---|---|---|---|---|---|---|")

    def f(r):
        return "-" if r is None else f"{r:.1e}"
    for c in checks:
        L.append(f"| `{c['old']}` | `{c['new']}` | {c['models']} | {c['consts_old']} -> {c['consts_new']} | "
                 f"{f(c['new_fits_old'])} | {f(c['old_fits_new'])} | {'yes' if c['new_redundant'] else 'no'} |")
    L.append("")
    groups = defaultdict(set)
    for (k, v) in examples:
        groups[v].add(k)
        if v in all_old:
            groups[v].add(v)
    big = sorted(((v, ks) for v, ks in groups.items() if len(ks) > 1), key=lambda t: (-len(t[1]), t[0]))[:15]
    if big:
        L.append("## Largest merges")
        L.append("")
        for v, ks in big:
            L.append(f"- `{v}` <- " + "; ".join(f"`{k}`" for k in sorted(ks)[:6]) + (" ..." if len(ks) > 6 else ""))
        L.append("")
    m = per_source["modeldb"]
    L.append("## ModelDB")
    L.append("")
    L.append(f"ModelDB is where redundancy matters most (over-parametrised gating and rate forms). Redundant forms "
             f"by the report test: {m['red_report_old']} of {m['old_forms']} before, {m['red_report_new']} of "
             f"{m['new_forms']} after; by the equilibrated test {m['red_eq_old']} -> {m['red_eq_new']}. A form "
             f"flagged by the report test but not by the equilibrated one is most likely a false positive of the "
             f"report test (its rows are dominated by the x where f is largest).")
    L.append("")
    L.append("Most used redundant forms before (models):")
    L.append("")
    for n, k in m["_top_red_old"]:
        L.append(f"- {n}: `{k}` -> `{reduce_key(k)}`")
    L.append("")
    L.append("Most used forms still flagged by the report test after (models; equilibrated test):")
    L.append("")
    for n, k, eq in m["_top_red_new"]:
        L.append(f"- {n}: `{k}` ({'redundant' if eq else 'identifiable'} by the equilibrated test)")
    L.append("")
    with open(out_md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    for p in per_source.values():
        for f in [f for f in p if f.startswith("_")]:
            p.pop(f)
    if out_json:
        with open(out_json, "w") as fh:
            json.dump({"per_source": per_source, "checks": checks}, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
