"""Summarise benchmark JSONL files into tables and plots.

    python benchmarks/summarise.py results/raw/*.jsonl --out results/summary

Writes:
  summary.md            main table (mean ± s.e. over replicates) and sensitivity tables
  summary.csv           every (variant, setting, noise, n, kind, method, metric) cell
  <metric>_<setting>.png  metric against n_j, one line per method, per noise level
  calibration.png       reliability curves for the posterior methods
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = ["recovery", "lpd_in", "lpd_out", "ari", "weight_error",
           "family_recovery", "variant_when_parent", "variant_detected",
           "extension_when_parent", "extension_detected", "theme_family", "theme_parent"]
METRIC_LABEL = {
    "recovery": "form recovery (fraction of datasets)",
    "lpd_in": "held-out log density, inside range",
    "lpd_out": "held-out log density, beyond range",
    "ari": "adjusted Rand index",
    "weight_error": "population-weight abs. error",
    "family_recovery": "family recovery (parent or variant chosen)",
    "variant_when_parent": "variant chosen when the truth is the parent",
    "variant_detected": "variant chosen when the truth is the variant",
    "extension_when_parent": "variant or fixed-exponent specialisation chosen when the truth is the parent",
    "extension_detected": "variant or fixed-exponent specialisation chosen when the truth is the variant",
    "theme_family": "most probable theme is in the family",
    "theme_parent": "most probable theme is the parent",
}
MAIN_METHODS = ["hier", "hier_pooled", "hier_themes", "no_pool", "no_pool_bma", "complete_pool", "oracle"]


def variant_of(r):
    parts = [f"scale={r.get('scaling', 'pooled')}", f"tau={r.get('tau', 3.0):g}"]
    if r.get("expand"):
        parts.append("edits")
    if r.get("grow_rounds"):
        parts.append(f"grow={r['grow_rounds']}")
    if r.get("themes"):
        parts.append("themes")
    return ",".join(parts)


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    for r in rows:
        r["variant"] = variant_of(r)
    return rows


def mean_se(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return float(v.mean()), float(se), len(v)


def aggregate(rows):
    cells = defaultdict(list)
    for r in rows:
        for m in METRICS + ["n_forms_mean", "chain_disagreement", "rhat_n_forms", "frac_datasets_true_in_C",
                            "n_themes_mean", "rho_mean"]:
            if m in r:
                cells[(r["variant"], r["setting"], r["noise"], r["n"], r["kind"], r["method"], m)].append(r[m])
    return {k: mean_se(v) for k, v in cells.items()}


def fmt(ms):
    m, se, n = ms
    if not np.isfinite(m):
        return "–"
    return f"{m:.2f} ± {se:.2f}" if np.isfinite(se) else f"{m:.2f}"


def write_tables(agg, rows, out):
    variants = sorted({k[0] for k in agg})
    settings = sorted({k[1] for k in agg})
    lines = ["# Benchmark summary", ""]
    n_reps = defaultdict(set)
    for r in rows:
        n_reps[(r["variant"], r["setting"], r["noise"], r["n"])].add(r["rep"])
    lines.append("Cells are mean ± standard error over replicate populations "
                 f"(replicates per cell: {min(len(v) for v in n_reps.values())}–{max(len(v) for v in n_reps.values())}).")
    lines.append("")
    for variant in variants:
        for setting in settings:
            for kind in ["laplace", "bic"]:
                keys = [k for k in agg if k[0] == variant and k[1] == setting and k[4] == kind]
                if not keys:
                    continue
                lines += [f"## {setting} — {kind} — {variant}", ""]
                noises = sorted({k[2] for k in keys})
                ns = sorted({k[3] for k in keys})
                methods = [m for m in MAIN_METHODS if any(k[5] == m for k in keys)]
                methods += sorted({k[5] for k in keys} - set(methods))
                for metric in METRICS:
                    if not any(k[6] == metric for k in keys):
                        continue
                    lines.append(f"**{METRIC_LABEL[metric]}**")
                    lines.append("")
                    header = "| method | " + " | ".join(f"σ={nz:g}, n={n}" for nz in noises for n in ns) + " |"
                    lines.append(header)
                    lines.append("|" + "---|" * (1 + len(noises) * len(ns)))
                    for meth in methods:
                        cells = [fmt(agg.get((variant, setting, nz, n, kind, meth, metric), (np.nan, np.nan, 0)))
                                 for nz in noises for n in ns]
                        lines.append(f"| {meth} | " + " | ".join(cells) + " |")
                    lines.append("")
                # diagnostics for the hierarchical sampler
                lines.append("**hier diagnostics** (posterior mean number of forms; max chain TV disagreement; "
                             "fraction of datasets whose true form is in C)")
                lines.append("")
                lines.append("| | " + " | ".join(f"σ={nz:g}, n={n}" for nz in noises for n in ns) + " |")
                lines.append("|" + "---|" * (1 + len(noises) * len(ns)))
                for m in ["n_forms_mean", "chain_disagreement", "frac_datasets_true_in_C"]:
                    cells = [fmt(agg.get((variant, setting, nz, n, kind, "hier", m), (np.nan, np.nan, 0)))
                             for nz in noises for n in ns]
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
                lines.append("")
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "setting", "noise", "n", "kind", "method", "metric", "mean", "se", "n_reps"])
        for k, (m, se, n) in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            w.writerow(list(k) + [m, se, n])


def plots(agg, rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    for variant in sorted({k[0] for k in agg}):
        _plots_for_variant(agg, rows, out, variant, plt)


def _plots_for_variant(agg, rows, out, variant, plt):
    slug = variant.replace("scale=", "").replace(",", "_").replace("=", "")
    settings = sorted({k[1] for k in agg if k[0] == variant})
    colours = {"hier": "#1f77b4", "hier_pooled": "#9467bd", "hier_themes": "#17becf", "no_pool": "#d62728", "no_pool_bma": "#ff9896",
               "complete_pool": "#7f7f7f", "oracle": "#2ca02c"}
    for setting in settings:
        for metric in METRICS:
            keys = [k for k in agg if k[0] == variant and k[1] == setting and k[6] == metric]
            if not keys:
                continue
            noises = sorted({k[2] for k in keys})
            fig, axes = plt.subplots(len(noises), 2, figsize=(10, 3.4 * len(noises)), squeeze=False,
                                     sharex=True)
            for i, nz in enumerate(noises):
                for jx, kind in enumerate(["laplace", "bic"]):
                    ax = axes[i, jx]
                    for meth in MAIN_METHODS:
                        pts = sorted((k[3], agg[k]) for k in keys if k[2] == nz and k[4] == kind and k[5] == meth)
                        if not pts:
                            continue
                        ns = [p[0] for p in pts]
                        m = np.array([p[1][0] for p in pts])
                        se = np.array([p[1][1] for p in pts])
                        ax.errorbar(ns, m, yerr=np.nan_to_num(se), marker="o", capsize=3, label=meth,
                                    color=colours.get(meth), linestyle="--" if meth == "oracle" else "-")
                    ax.set_xscale("log")
                    ax.set_xticks([5, 10, 20, 50])
                    ax.set_xticklabels(["5", "10", "20", "50"])
                    ax.set_xlim(4, 60)
                    ax.set_title(f"{setting}, noise {nz:g}, {kind}", fontsize=10)
                    ax.set_ylabel(METRIC_LABEL[metric], fontsize=9)
                    if i == len(noises) - 1:
                        ax.set_xlabel("points per dataset, n_j")
            axes[0, 0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"{metric}_{setting}_{slug}.png"), dpi=120)
            plt.close(fig)
    # calibration
    from hsr.metrics import reliability_curve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, kind in zip(axes, ["laplace", "bic"]):
        for meth in ["hier", "hier_pooled", "hier_themes", "no_pool_bma", "oracle"]:
            p, y = [], []
            for r in rows:
                if r["variant"] == variant and r["kind"] == kind and r["method"] == meth and "calib_p" in r:
                    p += r["calib_p"]
                    y += r["calib_y"]
            if not p:
                continue
            curve = reliability_curve(p, y, bins=10)
            ax.plot([c[0] for c in curve], [c[1] for c in curve], marker="o", label=meth, color=colours[meth])
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls=":")
        ax.set_xlabel("posterior probability of a skeleton")
        ax.set_ylabel("fraction that are the true form")
        ax.set_title(f"calibration ({kind}, all settings pooled)", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"calibration_{slug}.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="results/summary")
    args = ap.parse_args()
    paths = sorted({p for pat in args.paths for p in glob.glob(pat)})
    rows = load(paths)
    os.makedirs(args.out, exist_ok=True)
    agg = aggregate(rows)
    write_tables(agg, rows, args.out)
    plots(agg, rows, args.out)
    print(f"{len(rows)} rows from {len(paths)} files -> {args.out}")


if __name__ == "__main__":
    main()
