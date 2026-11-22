"""A single worked example: one synthetic population, every method, readable output.

    python benchmarks/example.py --n 20 --noise 0.05 --J 30 --out results/example

Writes summary.md (method comparison + one row per dataset) and datasets.png
(each dataset with the fitted curve of its top hierarchical skeleton).
"""
import argparse
import os
from collections import Counter

import numpy as np

from hsr import baselines
from hsr.candidates import CandidateSet
from hsr.experiment import _compiled, evaluate, make_population, one_hot, scaled_arrays
from hsr.gibbs import GibbsConfig, run_gibbs
from hsr.likelihood import compute_likelihoods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="mixture")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--J", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=2000)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--out", default="results/example")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cands = CandidateSet.from_json(os.path.join(os.path.dirname(__file__), "..", "hsr", "data", "candidates.json"))
    rng = np.random.default_rng(a.seed)
    pop = make_population(a.setting, rng, a.n, a.noise, cands, a.J)
    X, Y, sx, sy = scaled_arrays(pop, "pooled")
    table = compute_likelihoods(cands.skeletons, X, Y, compiled_cache={})
    sks, counts, g = cands.skeletons, cands.counts, cands.grammar
    key_idx = {s.key: i for i, s in enumerate(sks)}
    true_idx = np.array([key_idx.get(k, -1) for k in pop.true_keys])
    cfg = GibbsConfig(n_sweeps=a.sweeps, burn_in=a.burn, n_chains=4)

    lines = [f"# Example: {a.setting}, J={a.J}, n_j={a.n}, noise={a.noise:g}, seed={a.seed}", "",
             "True forms in this population: " +
             ", ".join(f"`{k}` × {v}" for k, v in Counter(pop.true_keys).items()), ""]
    results = {}
    for kind in ["laplace", "bic"]:
        L = table.matrix(kind)
        res = run_gibbs(L, counts, g, cfg)
        results[kind] = res
        P_np = baselines.no_pooling_posterior(L, counts, g)
        methods = {
            "hier": (res.assign_probs, {sks[e].key: float(res.pop_weight[e]) for e in range(len(sks))}),
            "no_pool": (one_hot(baselines.no_pooling(L, counts, g), len(sks)), None),
            "no_pool_bma": (P_np, {sks[e].key: float(P_np[:, e].mean()) for e in range(len(sks))}),
            "complete_pool": (one_hot(np.full(a.J, baselines.complete_pooling(L)), len(sks)), None),
        }
        lines += [f"## Methods ({kind})", "",
                  "| method | recovery | ARI | weight error | lpd inside | lpd beyond | mean P(true) |",
                  "|---|---|---|---|---|---|---|"]
        for name, (P, w) in methods.items():
            r = evaluate(name, P, pop, sks, table, sx, sy, true_idx, w, calibration=False)
            lines.append(f"| {name} | {r['recovery']:.2f} | {r['ari']:.2f} | {r['weight_error']:.3f} | "
                         f"{r['lpd_in']:.2f} | {r['lpd_out']:.2f} | {r['p_true_mean']:.2f} |")
        lines += ["", f"Posterior over number of forms: "
                  + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(res.n_forms_distribution().items())),
                  f"; gamma mean {res.gamma.mean():.2f}; max chain disagreement {res.chain_disagreement():.2f}", "",
                  "Top population weights: " + ", ".join(
                      f"`{sks[e].display}` {res.pop_weight[e]:.2f}" for e in np.argsort(res.pop_weight)[::-1][:5]),
                  ""]

    res = results["laplace"]
    rb = results["bic"]
    z_np = baselines.no_pooling(table.laplace, counts, g)
    lines += ["## Per dataset", "", "| j | true | hier (Laplace) | p | hier (BIC) | p | no pooling (Laplace) |",
              "|---|---|---|---|---|---|---|"]
    for j in range(a.J):
        e, eb = res.top[j], rb.top[j]
        mark = lambda k: "✓ " if k == pop.true_keys[j] else ""
        lines.append(f"| {j} | `{pop.true_keys[j]}` | {mark(sks[e].key)}`{sks[e].display}` | "
                     f"{res.assign_probs[j, e]:.2f} | {mark(sks[eb].key)}`{sks[eb].display}` | "
                     f"{rb.assign_probs[j, eb]:.2f} | {mark(sks[z_np[j]].key)}`{sks[z_np[j]].display}` |")
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols = 6
    rows = int(np.ceil(a.J / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.2 * rows), squeeze=False)
    grid = np.linspace(0.1, 6.0, 200)
    for j, ds in enumerate(pop.datasets):
        ax = axes[j // cols, j % cols]
        ax.scatter(ds.x, ds.y, s=10, color="k", zorder=3)
        ax.scatter(ds.x_out, ds.y_out, s=8, color="0.6", zorder=2)
        for kind, colour in [("laplace", "tab:blue"), ("bic", "tab:orange")]:
            e = results[kind].top[j]
            cs = _compiled(sks[e])
            f = cs.f((grid / sx[j])[None, :, None], table.phi[e][j][None, :])[0] * sy[j]
            ax.plot(grid, f, color=colour, lw=1.2, label=kind)
        ok = "✓" if sks[res.top[j]].key == pop.true_keys[j] else "✗"
        ax.set_title(f"{j}: {ok} {sks[res.top[j]].display}"[:34], fontsize=7)
        ax.tick_params(labelsize=6)
    for k in range(a.J, rows * cols):
        axes[k // cols, k % cols].axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "datasets.png"), dpi=110)


if __name__ == "__main__":
    main()
