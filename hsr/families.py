"""Families of a familiar parent form plus a nested variant (Overleaf note, Section 6).

Each variant reduces to its parent at a null value of one extension
coefficient. Data are generated from natural parameterisations; skeleton
identity uses the canonical key, so the method never sees how the data were
written down.

First version: saturation (Michaelis-Menten -> Hill) and decay
(exponential -> stretched exponential), populations "parent" (all parent),
"extended" (all variant) and "mixture" (70% parent, 30% variant).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .canonical import canonicalise, const_symbol, make_vars
from .simulate import X_BEYOND, X_HIGH, X_LOW, Dataset, Population


def _ln(rng, median, sdlog):
    return float(np.exp(rng.normal(np.log(median), sdlog)))


def _key(expr_builder):
    x = make_vars(1)[0]
    c = [const_symbol(i) for i in range(4)]
    return canonicalise(expr_builder(x, c)).key


@dataclass(frozen=True)
class Form:
    name: str
    key_builder: object      # (x, consts) -> sympy expression (for the canonical key)
    f: object                # (x, params) -> numpy values
    draw: object             # rng -> params

    @property
    def key(self):
        return _key(self.key_builder)


FAMILIES = {
    "saturation": {
        "parent": Form(
            "michaelis_menten",
            lambda x, c: c[0] * x / (c[1] + x),
            lambda x, p: p[0] * x / (p[1] + x),
            lambda rng: np.array([_ln(rng, 3.0, 0.4), _ln(rng, 1.0, 0.4)])),
        "variant": Form(
            "hill",
            lambda x, c: c[0] * x ** c[1] / (c[2] + x ** c[1]),
            lambda x, p: p[0] * x ** p[2] / (p[1] ** p[2] + x ** p[2]),
            lambda rng: np.array([_ln(rng, 3.0, 0.4), _ln(rng, 1.0, 0.4), _ln(rng, 2.0, 0.15)])),
        "null": "Hill exponent h = 1",
    },
    "decay": {
        "parent": Form(
            "exponential",
            lambda x, c: c[0] * sp.exp(c[1] * x),
            lambda x, p: p[0] * np.exp(-p[1] * x),
            lambda rng: np.array([_ln(rng, 3.0, 0.4), _ln(rng, 0.6, 0.3)])),
        "variant": Form(
            "stretched_exponential",
            lambda x, c: c[0] * sp.exp(c[1] * x ** c[2]),
            lambda x, p: p[0] * np.exp(-(p[1] * x) ** p[2]),
            lambda rng: np.array([_ln(rng, 3.0, 0.4), _ln(rng, 0.6, 0.3), _ln(rng, 0.5, 0.15)])),
        "null": "stretching exponent beta = 1",
    },
}

POPULATIONS = {"parent": 0.0, "extended": 1.0, "mixture": 0.3}   # fraction of datasets from the variant
SETTINGS = [f"{fam}_{pop}" for fam in FAMILIES for pop in POPULATIONS]


def _dataset(rng, form: Form, key, n, noise_frac, n_heldout=20):
    params = form.draw(rng)
    grid = np.linspace(X_LOW, X_HIGH, 400)
    fgrid = form.f(grid, params)
    sd = noise_frac * (fgrid.max() - fgrid.min())
    x = rng.uniform(X_LOW, X_HIGH, n)
    x_in = rng.uniform(X_LOW, X_HIGH, n_heldout)
    x_out = rng.uniform(X_HIGH, X_BEYOND, n_heldout)

    def draw(xx):
        return form.f(xx, params) + rng.normal(0, sd, len(xx))

    return Dataset(x, draw(x), key, params, sd, x_in, draw(x_in), x_out, draw(x_out))


def family_population(rng, setting, n, noise_frac, J=30):
    family, popname = setting.split("_", 1)
    fam = FAMILIES[family]
    frac = POPULATIONS[popname]
    parent, variant = fam["parent"], fam["variant"]
    kp, kv = parent.key, variant.key
    is_variant = rng.random(J) < frac
    datasets = [_dataset(rng, variant if v else parent, kv if v else kp, n, noise_frac) for v in is_variant]
    keys = [d.true_key for d in datasets]
    if frac == 0.0:
        form_keys, weights = [kp], np.array([1.0])
    elif frac == 1.0:
        form_keys, weights = [kv], np.array([1.0])
    else:
        form_keys, weights = [kp, kv], np.array([1 - frac, frac])
    pop = Population(datasets, keys, form_keys, weights, setting)
    pop.parent_key, pop.variant_key = kp, kv
    return pop


def family_metrics(pop, top_keys):
    """Family-level recovery and the two nesting error rates."""
    kp, kv = pop.parent_key, pop.variant_key
    truth = np.array(pop.true_keys)
    top = np.array(top_keys)
    out = {"family_recovery": float(np.mean(np.isin(top, [kp, kv])))}
    par = truth == kp
    var = truth == kv
    out["variant_when_parent"] = float(np.mean(top[par] == kv)) if par.any() else None
    out["variant_detected"] = float(np.mean(top[var] == kv)) if var.any() else None
    return out
