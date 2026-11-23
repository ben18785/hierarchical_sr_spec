"""Print the GitHub Actions job matrix for the benchmark as JSON.

main:        3 settings x n_j in {5,10,20,50} x noise in {0.05, 0.15}, default variant
families:    saturation and decay families x {parent, extended, mixture} x n_j in {10, 20}
             (families_full: every n_j), with --grow 1 --pooled
sensitivity: mixture, noise 0.05, every n_j, for each of
               --sensitivity (alpha and gamma-prior variants of the sampler)
               --scaling per_dataset (the literal spec's per-dataset standardisation)
               --tau 1 (narrower prior on constants)
               --expand (data-driven one-step edits added to C)
"""
import argparse
import json

NS = [5, 10, 20, 50]


def main_grid():
    out = []
    for setting in ["mixture", "all_A", "grammar"]:
        for noise in [0.05, 0.15]:
            for n in NS:
                out.append({"name": f"main_{setting}_n{n}_s{int(noise * 100):02d}", "setting": setting,
                            "n": n, "noise": noise, "extra": ""})
    return out


def sensitivity_grid():
    variants = {
        "sens": "--sensitivity",
        "perds": "--scaling per_dataset",
        "tau1": "--tau 1",
        "edits": "--expand",
    }
    out = []
    for label, extra in variants.items():
        for n in NS:
            out.append({"name": f"{label}_mixture_n{n}_s05", "setting": "mixture", "n": n, "noise": 0.05,
                        "extra": extra})
    return out


def families_grid(ns=(10, 20)):
    """Saturation and decay families (hsr/families.py), with one round of
    candidate growth and the pooled-constants model alongside the original."""
    from_settings = [f"{fam}_{pop}" for fam in ("saturation", "decay") for pop in ("parent", "extended", "mixture")]
    return [{"name": f"fam_{s}_n{n}_s05", "setting": s, "n": n, "noise": 0.05, "extra": "--grow 1 --pooled --themes"}
            for s in from_settings for n in ns]


def families_corpus_grid(ns=(10, 20), kappa=50):
    """The families grid twice: default grammar prior, and the corpus prior with
    every benchmark-family form held out of the corpus."""
    base = "--grow 1 --pooled --themes"
    out = []
    for job in families_grid(ns):
        out.append(job | {"extra": base})
        out.append(job | {"name": job["name"] + f"_corpusHO{kappa}",
                          "extra": base + f" --corpus hsr/data/corpus.json --corpus-heldout --kappa {kappa}"})
    return out


REAL = {"indometh": "pooled", "theoph": "pooled", "orange": "pooled", "dnase": "pooled", "co2": "pooled",
        "loblolly": "pooled", "forage_replaced": "per_dataset"}
REAL_SPLITS = ["full", "interp:0:3", "interp:1:3", "interp:2:3", "extrap"]


def realdata_grid(extra="--grow 1 --themes"):
    """hsr.realdata jobs: every dataset x split (FoRAGE curves are in different
    units, so they are scaled per dataset)."""
    out = []
    for d, sc in REAL.items():
        for sp in REAL_SPLITS:
            out.append({"name": f"real_{d}_{sp.replace(':', '')}", "module": "realdata",
                        "args": f"--data hsr/data/real/{d}.csv --split {sp} --scaling {sc} {extra}"})
    return out


def pysr_grid():
    """PySR baseline on the families cells (reps from the workflow input) and on
    every real-data job, with the same scaling and splits."""
    out = [{"name": "pysr_" + j["name"], "module": "pysr",
            "args": f"--setting {j['setting']} --n {j['n']} --noise {j['noise']}"} for j in families_grid()]
    out += [{"name": "pysr_" + j["name"], "module": "pysr",
             "args": " ".join(a for a in j["args"].split() if a not in ("--grow", "1", "--themes"))}
            for j in realdata_grid()]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="all", choices=["main", "sensitivity", "all", "families", "families_full",
                                                        "families_corpus", "realdata", "pysr"])
    ap.add_argument("--only", default="", help="regex; keep only jobs whose name matches")
    a = ap.parse_args()
    jobs = []
    if a.grid in ("main", "all"):
        jobs += main_grid()
    if a.grid in ("sensitivity", "all"):
        jobs += sensitivity_grid()
    if a.grid == "families":
        jobs += families_grid()
    if a.grid == "families_full":
        jobs += families_grid(NS)
    if a.grid == "families_corpus":
        jobs += families_corpus_grid()
    if a.grid == "realdata":
        jobs += realdata_grid()
    if a.grid == "pysr":
        jobs += pysr_grid()
    if a.only:
        import re
        jobs = [j for j in jobs if re.search(a.only, j["name"])]
    print(json.dumps({"include": jobs}))
