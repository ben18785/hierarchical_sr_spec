"""Known properties of the functional form: bounds, limits, monotonicity.

A property spec is a string such as

    "range=0,1;limit_inf=1;monotone=increasing"

with any of
    range=lo,hi        lo <= f(x) <= hi on the checking grid (either side may be blank)
    nonneg             shorthand for range=0,
    limit_inf=v        f(x) -> v as x -> infinity
    value0=v           f(x) -> v as x -> 0+
    monotone=increasing | decreasing

Two mechanisms, matching the two kinds of property:

1. Exact properties (limit_inf, value0) change the candidate set. For each
   skeleton and each sign pattern of its constants, sympy computes the limit
   as a function of the constants and solves "limit = v" for one constant;
   the result is a skeleton with one constant fewer that has the property
   exactly (c0*x/(c1 + x) -> x/(c1 + x); c0*exp(c1*x) + c2 -> c0*exp(c1*x) + 1).
   Skeletons whose limit is already v are kept as they are; skeletons whose
   limit cannot be made v are dropped. The derivation holds only for one sign
   pattern, so the limit is also re-checked numerically below.

2. Inequalities (range, monotone, and the limits as a numerical check) truncate
   the prior on the constants to the region A where the property holds. The
   marginal likelihood of a skeleton under the truncated prior is

       M_A = M * P(A | data) / P(A | prior),

   estimated with draws from the Laplace posterior and from the N(0, tau^2)
   prior, checked on a grid that extends beyond the data. Held-out
   predictions use only posterior draws in A.

A property that fixes the y scale (a bound or a limit) means y must not be
rescaled, and only x-scale invariance is required of skeletons.
"""
from __future__ import annotations

import itertools
import signal
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .canonical import canonicalise, consts_of, make_vars


@dataclass(frozen=True)
class Spec:
    lower: float | None = None
    upper: float | None = None
    limit_inf: float | None = None
    value0: float | None = None
    monotone: str | None = None

    @classmethod
    def parse(cls, text: str | None) -> "Spec | None":
        if not text:
            return None
        kw = {}
        for part in filter(None, (p.strip() for p in text.split(";"))):
            name, _, val = part.partition("=")
            name = name.strip()
            if name == "nonneg":
                kw["lower"] = 0.0
            elif name == "range":
                lo, _, hi = val.partition(",")
                if lo.strip():
                    kw["lower"] = float(lo)
                if hi.strip():
                    kw["upper"] = float(hi)
            elif name in ("limit_inf", "value0"):
                kw[name] = float(val)
            elif name == "monotone":
                if val not in ("increasing", "decreasing"):
                    raise ValueError(f"monotone must be increasing or decreasing, not {val!r}")
                kw["monotone"] = val
            else:
                raise ValueError(f"unknown property {name!r}")
        return cls(**kw)

    @property
    def fixes_y_scale(self) -> bool:
        nz = lambda v: v is not None and v != 0
        return any(nz(v) for v in (self.lower, self.upper, self.limit_inf, self.value0))

    @property
    def exact(self) -> bool:
        return self.limit_inf is not None or self.value0 is not None

    @property
    def label(self) -> str:
        parts = []
        if self.lower is not None or self.upper is not None:
            parts.append(f"range{'' if self.lower is None else f'{self.lower:g}'}_"
                         f"{'' if self.upper is None else f'{self.upper:g}'}")
        if self.limit_inf is not None:
            parts.append(f"liminf{self.limit_inf:g}")
        if self.value0 is not None:
            parts.append(f"val0_{self.value0:g}")
        if self.monotone:
            parts.append("inc" if self.monotone == "increasing" else "dec")
        return "-".join(parts)


# ------------------------------------------------------- exact properties
class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _with_timeout(fn, seconds):
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn()
    except _Timeout:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _limit_solutions(expr, x, point, value, timeout):
    """Expressions obtained from ``expr`` (constants d_i > 0 with signs folded
    in) whose limit at ``point`` equals ``value``."""
    lim = _with_timeout(lambda: sp.limit(expr, x, point), timeout)
    if lim is None or lim.has(sp.Limit, sp.AccumBounds, sp.nan, sp.zoo) or lim in (sp.oo, -sp.oo):
        return []
    diff = sp.simplify(lim - value)
    if diff == 0:
        return [expr]
    out = []
    for d in sorted(lim.free_symbols, key=str):
        sols = _with_timeout(lambda: sp.solve(sp.Eq(lim, value), d, dict=True), timeout) or []
        for sol in (sd.get(d) for sd in sols if isinstance(sd, dict)):
            if sol is None or not isinstance(sol, sp.Expr) or sol.has(sp.I) or sol.is_positive is False:
                continue
            out.append(expr.xreplace({d: sol}))
    return out


def sign_normal_key(key, timeout=5):
    """Canonical key up to the signs of the constants: the smallest key over
    all flips c_i -> -c_i (the canonicaliser does not merge these)."""
    from .candidates import skeleton_from_key
    sk = skeleton_from_key(key)
    consts = sorted(consts_of(sk.expr), key=str)
    best = key
    for flips in itertools.product((1, -1), repeat=len(consts)):
        if all(f == 1 for f in flips):
            continue
        e = sk.expr.xreplace({c: f * c for c, f in zip(consts, flips)})
        sk2 = _with_timeout(lambda: canonicalise(e, 1, max_consts=None), timeout)
        if sk2 is not None and (len(sk2.key), sk2.key) < (len(best), best):
            best = sk2.key
    return best


def impose_exact(skeleton, spec: Spec, timeout=5, max_consts=4):
    """Canonical keys of skeletons derived from ``skeleton`` that satisfy the
    exact properties of ``spec`` (possibly the skeleton itself)."""
    x = make_vars(1)[0]
    consts = sorted(consts_of(skeleton.expr), key=lambda c: int(c.name[1:]))
    if len(consts) > max_consts:
        return set()
    targets = []
    if spec.limit_inf is not None:
        targets.append((sp.oo, sp.nsimplify(spec.limit_inf)))
    if spec.value0 is not None:
        targets.append((0, sp.nsimplify(spec.value0)))
    keys = set()
    d = [sp.Symbol(f"d{i}", positive=True) for i in range(len(consts))]
    for signs in itertools.product((1, -1), repeat=len(consts)):
        exprs = [skeleton.expr.xreplace({c: s * di for c, s, di in zip(consts, signs, d)})]
        try:
            for point, value in targets:
                exprs = [e2 for e in exprs for e2 in _limit_solutions(e, x, point, value, timeout)]
        except Exception:
            continue
        for e in exprs:
            back = {di: sp.Symbol(f"c{i}", real=True) for i, di in enumerate(d)}
            sk = _with_timeout(lambda: canonicalise(e.xreplace(back), 1, max_consts=None), timeout)
            if sk is not None:
                keys.add(sign_normal_key(sk.key, timeout))
    # a skeleton that already has the properties keeps its own key (the sign
    # normal form may rename it, e.g. Hill -> c0/(c1 + x1**c2) by c1 -> -c1,
    # and keys are matched by name elsewhere); only derived skeletons are
    # sign-normalised, to merge the copies produced by different sign patterns
    own = sign_normal_key(skeleton.key, timeout)
    if own in keys:
        keys.discard(own)
        keys.add(skeleton.key)
    return keys


def satisfies_exact(skeleton, spec: Spec, timeout=5) -> bool:
    """Does the skeleton have the exact properties already (for some signs of
    its constants)? Used to admit skeletons added during candidate growth."""
    if not spec.exact:
        return True
    return skeleton.key in impose_exact(skeleton, spec, timeout)


def _build_worker(args):
    key, text = args
    from .candidates import skeleton_from_key
    try:
        return key, sorted(impose_exact(skeleton_from_key(key), Spec.parse(text)))
    except Exception:
        return key, []


def build_constrained(candidates_path, spec_text, out, n_jobs=None):
    """Constrained base candidate set for an exact spec: every candidate's
    derived skeletons (impose_exact), deduplicated, as a candidates JSON."""
    import json
    import os
    from concurrent.futures import ProcessPoolExecutor
    with open(candidates_path) as fh:
        keys = json.load(fh)["keys"]
    with ProcessPoolExecutor(n_jobs or os.cpu_count()) as ex:
        res = dict(ex.map(_build_worker, [(k, spec_text) for k in keys], chunksize=8))
    new = sorted({k for v in res.values() for k in v}, key=lambda k: (len(k), k))
    with open(out, "w") as fh:
        json.dump({"n_vars": 1, "keys": new, "stats": {"source": os.path.basename(candidates_path),
                                                        "spec": spec_text, "n_source": len(keys)}}, fh, indent=1)
    print(f"{len(new)} constrained skeletons from {len(keys)} -> {out}")


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--candidates", default=os.path.join(os.path.dirname(__file__), "data", "candidates.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(__file__), "data", f"candidates_{Spec.parse(a.spec).label}.json")
    build_constrained(a.candidates, a.spec, out)


# ------------------------------------------------------ candidate set
def constrained_candidates_path(spec: "Spec"):
    import os
    return os.path.join(os.path.dirname(__file__), "data", f"candidates_{spec.label}.json")


def prepare(base, spec: "Spec | None", scale_invariant: bool):
    """Filter the base skeletons and return (base, admit) where ``admit``
    screens skeletons added later (growth, edit graph). With a spec that fixes
    the y scale, only x-scale invariance is required. With an exact spec, the
    base should already be the constrained set (constrained_candidates_path);
    added skeletons must have the exact properties already."""
    fns = []
    if scale_invariant:
        from .properties import ScaleFilter
        f = ScaleFilter(y=not (spec is not None and spec.fixes_y_scale))
        base = f.filter(base)
        fns.append(f)
    if spec is not None and spec.exact:
        # added skeletons must reach the limits for a set of constants of
        # non-negligible prior mass (>= 1% of prior draws). A skeleton that has
        # the limit only on a null set (c0*tanh(c1*x1) needs c0 = 1 exactly)
        # must be rejected: the tolerance in the numerical check would give it
        # a tiny P(A | prior) and so a large spurious truncation bonus.
        from .likelihood import CompiledSkeleton
        memo = {}
        rng = np.random.default_rng(0)
        lim_only = Spec(limit_inf=spec.limit_inf, value0=spec.value0)

        def exact_ok(sk):
            if sk.key not in memo:
                cs = CompiledSkeleton(sk)
                phis = rng.normal(0, 3.0, (2000, max(cs.k, 0)))
                memo[sk.key] = bool(satisfied(cs, phis, np.array([0.1, 1.0, 10.0]), lim_only).mean() >= 0.01)
            return memo[sk.key]
        fns.append(exact_ok)
    admit = (lambda sk: all(fn(sk) for fn in fns)) if fns else None
    return base, admit


def unscale_y(Y, sy):
    """Undo the y scaling (a spec that fixes the y scale needs original units)."""
    Y = [np.asarray(y) * s for y, s in zip(Y, sy)]
    return Y, [1.0] * len(sy)


# -------------------------------------------------------- truncation
def check_grid(x_obs, n=60):
    """Checking grid (scaled x) for one dataset: from 0+ to 3x the largest x."""
    hi = 3.0 * float(np.max(x_obs))
    lo = max(float(np.min(x_obs)) / 10.0, 1e-6)
    return np.concatenate([np.geomspace(lo, hi, n)])


def satisfied(cs, phis, grid, spec: Spec, y_scale=1.0, tol=1e-6):
    """Boolean mask over rows of ``phis`` (m, k): does f(.; phi) have the
    properties on ``grid`` (scaled x; y scaled by y_scale)? Limits are checked
    at a very large / very small x."""
    phis = np.atleast_2d(phis)
    with np.errstate(all="ignore"):
        F = np.asarray(cs.f(np.broadcast_to(grid[None, :, None], (len(phis), len(grid), 1)), phis), float)
    F = np.broadcast_to(F, (len(phis), len(grid))) * y_scale
    ok = np.all(np.isfinite(F), axis=1)
    if spec.lower is not None:
        ok &= np.all(F >= spec.lower - tol, axis=1)
    if spec.upper is not None:
        ok &= np.all(F <= spec.upper + tol, axis=1)
    if spec.monotone:
        dF = np.diff(F, axis=1)
        ok &= np.all(dF >= -tol, axis=1) if spec.monotone == "increasing" else np.all(dF <= tol, axis=1)
    # limits: f must be close to the value far out AND still approaching it
    # (error at the farther point at most half that at the nearer one, or at
    # rounding level). A fixed tolerance alone would accept near-limits such
    # as tanh(c0) ~ 1 for large c0.
    for (p1, p2), value in (((1e6 * grid[-1], 1e9 * grid[-1]), spec.limit_inf),
                            ((1e-6 * grid[0], 1e-9 * grid[0]), spec.value0)):
        if value is None:
            continue
        errs = []
        for point in (p1, p2):
            with np.errstate(all="ignore"):
                v = np.asarray(cs.f(np.full((len(phis), 1, 1), point), phis), float).reshape(len(phis), -1)[:, 0]
            errs.append(np.abs(v * y_scale - value))
        e1, e2 = errs
        scale = max(1.0, abs(value))
        ok &= np.isfinite(e1) & np.isfinite(e2) & (e2 < 1e-3 * scale) & ((e2 < 1e-10 * scale) | (e2 <= 0.5 * e1))
    return ok


def truncation_logratio(cs, phi_hat, cov, grid, spec: Spec, tau, y_scale=1.0, n_post=400, n_prior=4000,
                        rng=None, return_parts=False):
    """log P(A | data) - log P(A | prior) for one skeleton and dataset, with
    +1/2 smoothing. P(A | data) uses draws from the Laplace posterior of the
    constants (the mode alone if the covariance is unusable)."""
    rng = rng or np.random.default_rng(0)
    k = cs.k
    fail = (-np.inf, 0.0) if return_parts else -np.inf
    if k == 0:
        ok = satisfied(cs, np.zeros((1, 0)), grid, spec, y_scale)[0]
        return ((0.0, 0.0) if return_parts else 0.0) if ok else fail
    if not np.all(np.isfinite(phi_hat)):
        return fail
    C = np.asarray(cov)[:k, :k] if cov is not None else None
    if C is not None and np.all(np.isfinite(C)):
        try:
            L = np.linalg.cholesky(C + 1e-12 * np.eye(k))
            post = phi_hat + rng.standard_normal((n_post, k)) @ L.T
        except np.linalg.LinAlgError:
            post = phi_hat[None, :]
    else:
        post = phi_hat[None, :]
    a_post = satisfied(cs, post, grid, spec, y_scale)
    prior = rng.normal(0, tau, (n_prior, k))
    a_prior = satisfied(cs, prior, grid, spec, y_scale)
    if not a_post.any() and not satisfied(cs, phi_hat[None, :], grid, spec, y_scale)[0]:
        return fail
    lp = np.log((a_post.sum() + 0.5) / (len(post) + 1.0))
    lq = np.log((a_prior.sum() + 0.5) / (len(prior) + 1.0))
    return float(lp - lq) if not return_parts else (float(lp), float(lq))


def apply_truncation(table, skeletons, X, M, spec: Spec, tau, sy, compiled, seed=0):
    """Add the truncation log-ratio to table.laplace and table.bic in place.
    ``compiled`` maps a skeleton to its CompiledSkeleton. Returns the (J, E)
    matrix of log P(A | prior) terms, which the pooled-constants model removes
    again (its constants share a population prior that is concentrated inside
    A, so the normaliser of the truncated pooled prior is ~1 rather than the
    N(0, tau^2) mass P(A | prior); summing -log P(A | prior) over a cluster's
    members would hand forms with a small P(A | prior) a large false bonus)."""
    J = table.laplace.shape[0]
    E = len(skeletons)
    corr = np.zeros((J, E))
    logq = np.zeros((J, E))
    rng = np.random.default_rng(seed)
    grids = [check_grid(X[j][M[j] > 0, 0]) for j in range(J)]
    for e, sk in enumerate(skeletons):
        cs = compiled(sk)
        for j in range(J):
            if not np.isfinite(table.laplace[j, e]):
                continue
            lp, lq = truncation_logratio(cs, np.asarray(table.phi[e][j]), table.cov[e][j], grids[j], spec,
                                         tau, sy[j], rng=rng, return_parts=True)
            corr[j, e] = lp - lq
            logq[j, e] = lq
    table.laplace = table.laplace + corr
    table.bic = table.bic + corr
    return logq
