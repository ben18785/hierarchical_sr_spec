"""Print the GitHub Actions job matrix for the benchmark as JSON.

main:        3 settings x n_j in {5,10,20,50} x noise in {0.05, 0.15}, default variant
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="all", choices=["main", "sensitivity", "all"])
    a = ap.parse_args()
    jobs = []
    if a.grid in ("main", "all"):
        jobs += main_grid()
    if a.grid in ("sensitivity", "all"):
        jobs += sensitivity_grid()
    print(json.dumps({"include": jobs}))
