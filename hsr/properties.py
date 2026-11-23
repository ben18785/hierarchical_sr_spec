"""Structural properties of skeletons, used to restrict the candidate set.

Scale invariance (units consistency)
------------------------------------
A skeleton f(x; phi) is *scale invariant* if changing the units of x or of y
can be absorbed by its constants: for every phi and a, b > 0 there is phi'
with f(a x; phi') = b f(x; phi) for all x. Scientific laws have this property
(a rate constant carries the units of 1/x, an amplitude the units of y);
forms such as ``x1``, ``sin(x1)``, ``tanh(2*x1)`` or ``(c0 + 3*x1)**c1`` do not:
they fit only because of the particular units (or scaling) of the data, and
they win marginal-likelihood comparisons because they have fewer constants.

The check is numerical: for random constants phi, refit phi' so that
f(a x; phi') matches f(x; phi) (a = 1.25, 0.8) and f(x; phi') matches
1.25 f(x; phi). Small factors keep the refit local (from phi); invariance
under them for every phi implies invariance under any positive factor. The
skeleton passes if both refits are exact (relative RMS below ``tol``) on a
grid of x. Results are cached by key in hsr/data/scale_invariant.json.

    python -m hsr.properties build      # (re)compute the cache for candidates.json
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
from scipy.optimize import least_squares

CACHE = os.path.join(os.path.dirname(__file__), "data", "scale_invariant.json")
CACHE_X = os.path.join(os.path.dirname(__file__), "data", "scale_invariant_x.json")
_GRID = np.linspace(0.2, 3.0, 25)
_FINE = np.linspace(0.15, 3.8, 400)   # covers a * _GRID for the factors below
_TRANSFORMS = ((1.25, 1.0), (1.0, 1.25), (0.8, 1.0))


def _eval(cs, x, phi):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.asarray(cs.f(x[None, :, None], np.asarray(phi, float)[None, :])[0], float)
    return np.broadcast_to(v, x.shape) if v.ndim == 0 or v.shape != x.shape else v


def _absorbs(cs, phi, a, b, tol, rng, n_starts=6):
    target = b * _eval(cs, _GRID, phi)
    scale = np.sqrt(np.mean(target ** 2))
    xa = a * _GRID

    def res(p):
        r = (_eval(cs, xa, p) - target) / scale
        return np.where(np.isfinite(r), r, 1e3)

    starts = [phi] + [phi + rng.normal(0, 0.5, len(phi)) for _ in range(n_starts - 2)] + \
             [rng.normal(0, 2, len(phi))]
    for p0 in starts:
        try:
            sol = least_squares(res, p0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=400 * (len(phi) + 1))
        except Exception:
            continue
        if np.all(np.abs(sol.fun) < 1e3) and np.sqrt(np.mean(sol.fun ** 2)) < tol:
            return True
    return False


def scale_invariant(skeleton, tol=1e-5, seed=0, n_phi=2, max_phi=6, y=True) -> bool:
    """True if units changes in x and y can be absorbed by the constants (see
    module docstring): the refits are exact for ``n_phi`` of up to ``max_phi``
    random constant vectors. A non-invariant skeleton cannot be refitted exactly
    for any generic phi, while an invariant one can fail for an unlucky phi
    (e.g. a pole near the grid), hence "n_phi successes" rather than "all".
    A skeleton that is never finite and free of blow-ups on the grid for random
    constants (e.g. exp(x1**2 + x1), sin(x1**2)**c0/x1) is rejected."""
    from .likelihood import CompiledSkeleton
    cs = CompiledSkeleton(skeleton)
    if cs.k == 0:
        return False   # nothing can absorb a change of units in x
    rng = np.random.default_rng(seed)
    tested = passed = 0
    for i in range(60):
        phi = rng.normal(0, 1.0 if i % 2 == 0 else 3.0, cs.k)
        v = _eval(cs, _GRID, phi)
        if not np.all(np.isfinite(v)) or np.std(v) < 1e-3 * (np.abs(v).mean() + 1e-12):
            continue
        # skip constants that put a pole or blow-up anywhere in the rescaled range
        vf = _eval(cs, _FINE, phi)
        if not np.all(np.isfinite(vf)) or np.abs(vf).max() > 50 * np.sqrt(np.mean(v ** 2)):
            continue
        tested += 1
        if all(_absorbs(cs, phi, a, b, tol, rng) for a, b in _TRANSFORMS if y or b == 1.0):
            passed += 1
            if passed >= n_phi:
                return True
        if tested >= max_phi:
            return False
    return passed > 0


def _worker(key, y=True):
    from .candidates import skeleton_from_key
    try:
        return key, bool(scale_invariant(skeleton_from_key(key), y=y))
    except Exception:
        return key, True


class ScaleFilter:
    """Callable skeleton -> bool with a persistent key cache."""

    def __init__(self, path=None, y=True):
        """y=False: require invariance to the units of x only (for data whose y
        scale is fixed, e.g. a bounded response)."""
        self.y = y
        path = path or (CACHE if y else CACHE_X)
        self.path = path
        self.cache = {}
        if path and os.path.exists(path):
            with open(path) as fh:
                self.cache = json.load(fh)

    def __call__(self, skeleton) -> bool:
        v = self.cache.get(skeleton.key)
        if v is None:
            v = bool(scale_invariant(skeleton, y=self.y))
            self.cache[skeleton.key] = v
        return v

    def filter(self, skeletons):
        return [s for s in skeletons if self(s)]


def build(candidates_path, out=CACHE, n_jobs=None, y=True):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    with open(candidates_path) as fh:
        keys = json.load(fh)["keys"]
    if os.path.exists(out):
        with open(out) as fh:
            old = json.load(fh)
    else:
        old = {}
    todo = [k for k in keys if k not in old]
    n_jobs = n_jobs or os.cpu_count()
    with ProcessPoolExecutor(n_jobs) as ex:
        res = dict(ex.map(partial(_worker, y=y), todo, chunksize=16))
    res = old | res
    with open(out, "w") as fh:
        json.dump(res, fh, indent=0, sort_keys=True)
    print(f"{sum(res.values())} of {len(res)} candidates are scale invariant -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--candidates", default=os.path.join(os.path.dirname(__file__), "data", "candidates.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--x-only", action="store_true", help="units of x only (cache scale_invariant_x.json)")
    a = ap.parse_args()
    build(a.candidates, a.out or (CACHE_X if a.x_only else CACHE), y=not a.x_only)
