"""Synthetic populations of datasets (spec section 6)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canonical import canonicalise
from .likelihood import CompiledSkeleton

X_LOW, X_HIGH = 0.1, 5.0
X_BEYOND = X_HIGH + 0.2 * (X_HIGH - X_LOW)   # 20% beyond the observed range


def _lognormal(rng, median, sdlog):
    return float(np.exp(rng.normal(np.log(median), sdlog)))


# The three benchmark forms. Constants are log-normal; medians chosen so that
# every curve is clearly non-linear on [0.1, 5] (see tests/test_simulate.py).
def _draw_A(rng):  # c0*exp(-c1*x): decay, rate in roughly [0.3, 1.2]
    return np.array([_lognormal(rng, 3.0, 0.4), -_lognormal(rng, 0.6, 0.3)])


def _draw_B(rng):  # c0*x/(c1+x): saturation, half-saturation near 1
    return np.array([_lognormal(rng, 3.0, 0.4), _lognormal(rng, 1.0, 0.4)])


def _draw_C(rng):  # c0*x/(c1+x^2): rise and fall, peak near sqrt(c1)
    return np.array([_lognormal(rng, 3.0, 0.4), _lognormal(rng, 2.0, 0.4)])


BENCHMARK_FORMS = {
    "A": ("C*exp(C*x1)", _draw_A),
    "B": ("C*x1/(C+x1)", _draw_B),
    "C": ("C*x1/(C+x1*x1)", _draw_C),
}


@dataclass
class Dataset:
    x: np.ndarray            # (n,)
    y: np.ndarray            # (n,)
    true_key: str
    true_phi: np.ndarray
    noise_sd: float
    x_in: np.ndarray = field(default=None)       # held-out inputs inside the range
    y_in: np.ndarray = field(default=None)
    x_out: np.ndarray = field(default=None)      # held-out inputs beyond the range
    y_out: np.ndarray = field(default=None)


@dataclass
class Population:
    datasets: list
    true_keys: list          # per dataset
    form_keys: list          # the distinct generating forms
    form_weights: np.ndarray  # population weights of form_keys
    setting: str

    @property
    def labels(self):
        index = {k: i for i, k in enumerate(self.form_keys)}
        return np.array([index[k] for k in self.true_keys])


def _make_dataset(rng, cs: CompiledSkeleton, phi, n, noise_frac, n_heldout=20):
    grid = np.linspace(X_LOW, X_HIGH, 400)
    fgrid = cs.f(grid[None, :, None], phi[None, :])[0]
    sd = noise_frac * (fgrid.max() - fgrid.min())
    x = rng.uniform(X_LOW, X_HIGH, n)
    x_in = rng.uniform(X_LOW, X_HIGH, n_heldout)
    x_out = rng.uniform(X_HIGH, X_BEYOND, n_heldout)

    def draw(xx):
        f = cs.f(xx[None, :, None], phi[None, :])[0]
        return f + rng.normal(0, sd, len(xx))

    return Dataset(x, draw(x), cs.skeleton.key, phi, sd, x_in, draw(x_in), x_out, draw(x_out))


_COMPILED = {}


def _compiled(key_string):
    if key_string not in _COMPILED:
        sk = canonicalise(key_string)
        _COMPILED[key_string] = CompiledSkeleton(sk)
    return _COMPILED[key_string]


def benchmark_population(rng, n, noise_frac, J=30, weights=(0.6, 0.3, 0.1), setting="mixture"):
    """Mixture of forms A, B, C with the given population weights."""
    names = list(BENCHMARK_FORMS)
    weights = np.asarray(weights, float)
    labels = rng.choice(len(names), size=J, p=weights / weights.sum())
    datasets, keys = [], []
    for lab in labels:
        string, draw = BENCHMARK_FORMS[names[lab]]
        cs = _compiled(string)
        phi = _canonical_phi(cs, string, draw(rng))
        ds = _make_dataset(rng, cs, phi, n, noise_frac)
        datasets.append(ds)
        keys.append(cs.skeleton.key)
    form_keys = [_compiled(BENCHMARK_FORMS[nm][0]).skeleton.key for nm in names]
    return Population(datasets, keys, form_keys, weights / weights.sum(), setting)


def _canonical_phi(cs, string, phi_spec):
    """Map constants in spec order (as written in BENCHMARK_FORMS) onto the
    canonical skeleton's constant order. All three forms keep the order
    (scale first), which tests/test_simulate.py checks numerically."""
    return np.asarray(phi_spec, float)


def all_A_population(rng, n, noise_frac, J=30):
    return benchmark_population(rng, n, noise_frac, J, weights=(1.0, 0.0, 0.0), setting="all_A")


def draw_constants(rng, cs: CompiledSkeleton, max_tries=200, sd=1.5):
    """Constants of a form drawn N(0, sd^2), rejected unless the curve is
    finite, bounded (|f| < 1e3) and clearly non-constant on the extended range
    [X_LOW, X_BEYOND]; None after max_tries rejections."""
    grid = np.linspace(X_LOW, X_BEYOND, 400)
    for _ in range(max_tries):
        phi = rng.normal(0, sd, cs.k)
        with np.errstate(all="ignore"):
            f = cs.f(grid[None, :, None], phi[None, :])[0]
        if np.all(np.isfinite(f)) and np.max(np.abs(f)) < 1e3 and np.ptp(f) > 0.05 * (1 + np.max(np.abs(f))):
            return phi
    return None


def grammar_population(rng, n, noise_frac, cands, J=30, max_tries=200):
    """J distinct forms drawn from the grammar prior (restricted to the
    candidate set, so recovery is possible) with N(0, 1.5^2) constants,
    rejected unless the curve is finite and non-constant on the extended range."""
    g = cands.grammar
    log_p = g.log_prob(cands.counts, g.prior_mean())
    ok = np.array([1 <= s.n_consts <= 3 for s in cands.skeletons])
    p = np.where(ok, np.exp(log_p - log_p.max()), 0.0)
    p /= p.sum()
    datasets, keys = [], []
    used = set()
    while len(datasets) < J:
        e = rng.choice(len(p), p=p)
        sk = cands.skeletons[e]
        if sk.key in used:
            continue
        cs = CompiledSkeleton(sk)
        phi = draw_constants(rng, cs, max_tries)
        if phi is None:
            continue
        used.add(sk.key)
        datasets.append(_make_dataset(rng, cs, phi, n, noise_frac))
        keys.append(sk.key)
    return Population(datasets, keys, list(keys), np.full(J, 1.0 / J), "grammar")
