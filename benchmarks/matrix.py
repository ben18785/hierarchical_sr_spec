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
    return [{"name": f"fam_{s}_n{n}_s05", "setting": s, "n": n, "noise": 0.05, "extra": "--grow 1 --pooled"}
            for s in from_settings for n in ns]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="all", choices=["main", "sensitivity", "all", "families", "families_full"])
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
    print(json.dumps({"include": jobs}))
