"""Stage A step 2: marginal likelihoods log p(y_j | e) for every skeleton.

Model for one dataset (after scaling, see ``scale_dataset``):

    y_i ~ Normal(f_e(x_i; phi), sigma^2),  phi_k ~ Normal(0, tau^2),  sigma ~ HalfNormal(1)

The MAP of (phi, s = log sigma) is found by a Levenberg-Marquardt iteration on
phi alternated with the closed-form optimum of s given phi. All datasets and
all multi-starts for one skeleton are fitted together as one batch, so a
skeleton costs one vectorised optimisation, not J * n_starts scalar ones.

Laplace:  log p(y|e) ~= -NLJ(MAP) + (K/2) log(2 pi) - 0.5 log det H,
          K = k + 1 (constants plus log sigma), H the exact Hessian of NLJ.
BIC:      max log-lik - (K/2) log n  (maximum found by a prior-free refit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.special import logsumexp

from .canonical import Skeleton, consts_of, make_vars

LPD_FLOOR = -50.0
LOG2PI = np.log(2 * np.pi)


# ------------------------------------------------------------------ scaling
def scale_dataset(x, y):
    """Divide x (per column) and y by their root-mean-square.

    The spec says "standardise". Centring is deliberately NOT done: shifting
    x or y changes the skeleton (C*exp(C*x) becomes C*exp(C*x)+C after
    centring y), whereas pure rescaling maps every skeleton whose variables
    and output carry a free multiplicative constant onto itself.
    Returns (x_scaled, y_scaled, x_scale, y_scale).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim == 1:
        x = x[:, None]
    sx = np.sqrt(np.mean(x**2, axis=0))
    sy = np.sqrt(np.mean(y**2))
    sx[sx == 0] = 1.0
    sy = sy if sy > 0 else 1.0
    return x / sx, y / sy, sx, sy


def scale_population(xs, ys, mode="pooled"):
    """Scale a list of datasets. Returns lists (x_scaled, y_scaled, x_scale, y_scale).

    mode="per_dataset": ``scale_dataset`` on each dataset (the literal spec).
        This leaks each dataset's amplitude into the data: after scaling every
        y has unit RMS, so constant-free skeletons such as ``x**(1/4)`` match
        the amplitude for free and are favoured unfairly (see REPORT.md).
    mode="pooled" (default): one RMS scale for x and one for y computed over
        all datasets together, so no per-dataset information leaks.
    mode="none": raw units.
    """
    xs = [np.asarray(x, float).reshape(len(x), -1) for x in xs]
    ys = [np.asarray(y, float) for y in ys]
    if mode == "per_dataset":
        out = [scale_dataset(x, y) for x, y in zip(xs, ys)]
        return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out], [o[3] for o in out]
    if mode == "pooled":
        allx = np.vstack(xs)
        ally = np.concatenate(ys)
        sx = np.sqrt(np.mean(allx**2, axis=0))
        sx[sx == 0] = 1.0
        sy = float(np.sqrt(np.mean(ally**2))) or 1.0
    elif mode == "none":
        sx = np.ones(xs[0].shape[1])
        sy = 1.0
    else:
        raise ValueError(f"unknown scaling mode {mode!r}")
    return [x / sx for x in xs], [y / sy for y in ys], [sx] * len(xs), [sy] * len(ys)


# --------------------------------------------------------------- compilation
class CompiledSkeleton:
    """numpy callables for f, its gradient and Hessian in the constants."""

    def __init__(self, skeleton: Skeleton, n_vars: int = 1):
        self.skeleton = skeleton
        expr = skeleton.expr
        self.consts = sorted(consts_of(expr), key=lambda c: int(c.name[1:]))
        self.k = len(self.consts)
        self.xs = make_vars(n_vars)
        args = list(self.xs) + self.consts
        grads = [sp.diff(expr, c) for c in self.consts]
        hess = [sp.diff(grads[a], self.consts[b]) for a in range(self.k) for b in range(a, self.k)]
        self._fg = sp.lambdify(args, [expr] + grads, modules="numpy", cse=True)
        self._h = sp.lambdify(args, hess, modules="numpy", cse=True) if self.k else None
        self._f = sp.lambdify(args, expr, modules="numpy", cse=True)
        # constants that appear as an exponent (x**c): their null value is 1
        exps = {a.exp for a in expr.atoms(sp.Pow)}
        self.exponent_mask = np.array([c in exps for c in self.consts], dtype=bool)

    def _call(self, fn, X, phi):
        # X: (B, n, d); phi: (B, k) -> list of (B, n) arrays
        xs = [X[..., i] for i in range(X.shape[-1])]
        cs = [phi[:, i:i + 1] for i in range(self.k)]
        out = fn(*xs, *cs)
        shape = X.shape[:-1]
        if not isinstance(out, (list, tuple)):
            return np.broadcast_to(np.asarray(out, float), shape)
        return [np.broadcast_to(np.asarray(o, float), shape) for o in out]

    def f(self, X, phi):
        return self._call(self._f, X, phi)

    def f_and_grad(self, X, phi):
        out = self._call(self._fg, X, phi)
        f = out[0]
        J = np.stack(out[1:], axis=-1) if self.k else np.zeros(f.shape + (0,))
        return f, J

    def hess(self, X, phi):
        """(B, n, k, k) second derivatives."""
        B, n = X.shape[:2]
        H = np.zeros((B, n, self.k, self.k))
        if not self.k:
            return H
        out = self._call(self._h, X, phi)
        idx = 0
        for a in range(self.k):
            for b in range(a, self.k):
                H[..., a, b] = out[idx]
                H[..., b, a] = out[idx]
                idx += 1
        return H


# ----------------------------------------------------------------- fitting
@dataclass
class FitResult:
    log_ml: np.ndarray        # (J,) Laplace log marginal likelihood (-inf on failure)
    bic: np.ndarray           # (J,) max log-lik - K/2 log n
    max_loglik: np.ndarray    # (J,)
    phi: np.ndarray           # (J, k) MAP constants
    log_sigma: np.ndarray     # (J,)
    cov: np.ndarray           # (J, k+1, k+1) inverse Hessian (nan on failure)


def _s_given_S(S, n, prior=True):
    """Optimal log sigma given residual sum of squares S."""
    if prior:
        # minimise (n-1)s + S e^{-2s}/2 + e^{2s}/2  ->  u^2 + (n-1)u - S = 0
        u = 0.5 * (-(n - 1) + np.sqrt((n - 1) ** 2 + 4 * S))
    else:
        u = S / n
    return 0.5 * np.log(np.maximum(u, 1e-300))


def _nlj(S, phi, s, n, tau, prior=True):
    """Negative log joint (prior=True) or negative log likelihood."""
    out = n * s + 0.5 * S * np.exp(-2 * s) + 0.5 * n * LOG2PI
    if prior:
        k = phi.shape[1]
        out = out + 0.5 * np.sum(phi**2, axis=1) / tau**2 + k * (0.5 * LOG2PI + np.log(tau))
        out = out - (np.log(2) - 0.5 * LOG2PI - 0.5 * np.exp(2 * s) + s)
    return out


def _lm(cs, X, Y, M, phi, n, tau, prior, max_iter, tol):
    """Batched Levenberg-Marquardt on phi with closed-form s updates.

    X: (B, n, d), Y and M (mask): (B, n), phi: (B, k), n: (B,).
    """
    B, k = phi.shape
    inv_t2 = 1.0 / tau**2 if prior else 0.0
    with np.errstate(all="ignore"):
        f, J = cs.f_and_grad(X, phi)
        r = (Y - f) * M
        S = np.sum(r**2, axis=1)
        S = np.where(np.isfinite(S), S, np.inf)
        s = _s_given_S(np.where(np.isfinite(S), S, 1.0), n, prior)
        obj = np.where(np.isfinite(S), _nlj(S, phi, s, n, tau, prior), np.inf)
        if k == 0:
            return phi, s, obj
        lam = np.full(B, 1e-2)
        active = np.isfinite(obj)
        for _ in range(max_iter):
            if not active.any():
                break
            w = np.exp(-2 * s)
            Jm = J * M[..., None]
            A = np.einsum("bni,bnj->bij", Jm, Jm) * w[:, None, None] + inv_t2 * np.eye(k)
            g = -np.einsum("bni,bn->bi", Jm, r) * w[:, None] + inv_t2 * phi
            diag = np.einsum("bii->bi", A)
            Ad = A + lam[:, None, None] * np.einsum("bi,ij->bij", np.maximum(diag, 1e-12), np.eye(k))
            Ad = np.where(np.isfinite(Ad), Ad, 0.0) + 1e-12 * np.eye(k)
            g_ = np.where(np.isfinite(g), g, 0.0)
            try:
                step = np.linalg.solve(Ad, -g_[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = np.zeros_like(phi)
                for b in range(B):
                    try:
                        step[b] = np.linalg.solve(Ad[b], -g_[b])
                    except np.linalg.LinAlgError:
                        pass
            new_phi = np.where(active[:, None], phi + step, phi)
            f2, J2 = cs.f_and_grad(X, new_phi)
            r2 = (Y - f2) * M
            S2 = np.sum(r2**2, axis=1)
            ok = np.isfinite(S2) & np.all(np.isfinite(J2 * M[..., None]), axis=(1, 2))
            S2s = np.where(ok, S2, 1.0)
            obj_phi = np.where(ok, _nlj(S2s, new_phi, s, n, tau, prior), np.inf)  # same s
            accept = active & ok & (obj_phi < obj)
            s2 = np.where(accept, _s_given_S(S2s, n, prior), s)
            new_obj = np.where(accept, _nlj(S2s, new_phi, s2, n, tau, prior), obj)
            improvement = obj - new_obj
            phi = np.where(accept[:, None], new_phi, phi)
            s = s2
            r = np.where(accept[:, None], r2, r)
            J = np.where(accept[:, None, None], J2, J)
            lam = np.where(accept, np.maximum(lam / 3, 1e-9), np.minimum(lam * 4, 1e10))
            converged = (accept & (improvement < tol * (1 + np.abs(new_obj)))) | (lam >= 1e10)
            obj = new_obj
            active = active & ~converged
    return phi, s, obj


def fit_skeleton(cs: CompiledSkeleton, X, Y, M=None, n_starts=10, tau=3.0, seed=0,
                 max_iter=200, tol=1e-10, start_sd=1.5, extra_starts=None) -> FitResult:
    """Laplace and BIC marginal likelihoods of one skeleton for J datasets.

    X: (J, n, d) scaled inputs, Y: (J, n) scaled outputs, M: (J, n) 0/1 mask
    (allows unequal n_j by padding).
    """
    X = np.asarray(X, float)
    if X.ndim == 2:
        X = X[..., None]
    Y = np.asarray(Y, float)
    J_, nmax = Y.shape
    M = np.ones_like(Y) if M is None else np.asarray(M, float)
    Xp = np.where(M[..., None] > 0, X, 1.0)  # padded points evaluated at a safe x
    Yp = np.where(M > 0, Y, 0.0)
    n = M.sum(axis=1)
    k = cs.k
    rng = np.random.default_rng(seed)
    S_ = n_starts if k else 1
    starts = rng.normal(0, start_sd, size=(J_, S_, k))
    if k:
        starts[:, 0, :] = 1.0
        if S_ > 1:
            starts[:, 1, :] = -1.0 if S_ > 2 else starts[:, 1, :]
        # exponents: start half of the random starts at the null value 1
        # (the parent form), with the other constants random
        em = getattr(cs, "exponent_mask", np.zeros(k, bool))
        if em.any() and S_ > 3:
            for s_i in range(2, S_, 2):
                starts[:, s_i, em] = 1.0
    if extra_starts is not None and k:
        starts = np.concatenate([starts, np.asarray(extra_starts).reshape(J_, -1, k)], axis=1)
        S_ = starts.shape[1]
    B = J_ * S_
    Xb = np.repeat(Xp, S_, axis=0)
    Yb = np.repeat(Yp, S_, axis=0)
    Mb = np.repeat(M, S_, axis=0)
    nb = np.repeat(n, S_)
    phi0 = starts.reshape(B, k)
    phi, s, obj = _lm(cs, Xb, Yb, Mb, phi0, nb, tau, True, max_iter, tol)
    obj = obj.reshape(J_, S_)
    best = np.argmin(np.where(np.isfinite(obj), obj, np.inf), axis=1)
    rows = np.arange(J_) * S_ + best
    phi_b, s_b, obj_b = phi[rows], s[rows], obj[np.arange(J_), best]

    # ---- Laplace
    log_ml = np.full(J_, -np.inf)
    cov = np.full((J_, k + 1, k + 1), np.nan)
    with np.errstate(all="ignore"):
        f, Jf = cs.f_and_grad(Xp, phi_b)
        r = (Yp - f) * M
        Hf = cs.hess(Xp, phi_b) * M[..., None, None]
        S = np.sum(r**2, axis=1)
        w = np.exp(-2 * s_b)
        Jm = Jf * M[..., None]
        H = np.zeros((J_, k + 1, k + 1))
        if k:
            H[:, :k, :k] = (np.einsum("jni,jnk->jik", Jm, Jm) - np.einsum("jn,jnik->jik", r, Hf)) * w[:, None, None] \
                + np.eye(k) / tau**2
            cross = 2 * np.einsum("jn,jni->ji", r, Jm) * w[:, None]
            H[:, :k, k] = cross
            H[:, k, :k] = cross
        H[:, k, k] = 2 * S * w + 2 * np.exp(2 * s_b)
        K = k + 1
        for j in range(J_):
            if not np.isfinite(obj_b[j]) or not np.all(np.isfinite(H[j])):
                continue
            try:
                L = np.linalg.cholesky(H[j])
            except np.linalg.LinAlgError:
                continue
            logdet = 2 * np.sum(np.log(np.diag(L)))
            log_ml[j] = -obj_b[j] + 0.5 * K * LOG2PI - 0.5 * logdet
            Linv = np.linalg.inv(L)
            cov[j] = Linv.T @ Linv

    # ---- BIC: prior-free refit from the MAP
    phi_m, s_m, nll = _lm(cs, Xp, Yp, M, phi_b.copy(), n, tau, False, max_iter, tol)
    nll = np.where(np.isfinite(nll), nll, np.inf)
    with np.errstate(all="ignore"):
        max_ll = -np.minimum(nll, np.where(np.isfinite(obj_b), _nll_at(cs, Xp, Yp, M, phi_b, n), np.inf))
        bic = max_ll - 0.5 * (k + 1) * np.log(n)
    bic = np.where(np.isfinite(bic), bic, -np.inf)
    max_ll = np.where(np.isfinite(max_ll), max_ll, -np.inf)
    return FitResult(log_ml, bic, max_ll, phi_b, s_b, cov)


def _nll_at(cs, X, Y, M, phi, n):
    f = cs.f(X, phi)
    S = np.sum(((Y - f) * M) ** 2, axis=1)
    s = _s_given_S(S, n, prior=False)
    return _nlj(S, phi, s, n, 1.0, prior=False)


# -------------------------------------------------- brute-force integration
def log_joint(cs: CompiledSkeleton, x, y, theta, tau=3.0):
    """log p(y, phi, s) for parameter draws theta (m, k+1); x (n, d), y (n,)."""
    theta = np.atleast_2d(theta)
    m = theta.shape[0]
    k = cs.k
    phi, s = theta[:, :k], theta[:, k]
    X = np.broadcast_to(np.asarray(x, float).reshape(1, len(y), -1), (m, len(y), cs.xs.__len__()))
    with np.errstate(all="ignore"):
        f = cs.f(X, phi)
        S = np.sum((y[None, :] - f) ** 2, axis=1)
        out = -_nlj(S, phi, s, len(y), tau, prior=True)
    return np.where(np.isfinite(out), out, -np.inf)


def log_ml_importance(cs: CompiledSkeleton, x, y, mode, cov, n_draws=20000, df=4, inflate=1.5,
                      tau=3.0, seed=0):
    """Importance-sampling estimate of log p(y|e) with a multivariate-t
    proposal centred on the Laplace mode. Returns (estimate, standard error)."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    dim = len(mode)
    scale = cov * inflate**2
    prop = stats.multivariate_t(loc=mode, shape=scale, df=df, seed=rng)
    draws = prop.rvs(size=n_draws).reshape(n_draws, dim)
    lw = log_joint(cs, x, y, draws, tau) - prop.logpdf(draws)
    est = logsumexp(lw) - np.log(n_draws)
    w = np.exp(lw - lw.max())
    ess = w.sum() ** 2 / np.sum(w**2)
    se = np.std(w) / (np.mean(w) * np.sqrt(n_draws))  # delta-method se of log mean
    return est, se, ess


# ----------------------------------------------------- posterior predictive
def predictive_logpdf(cs: CompiledSkeleton, x_new, y_new, phi, log_sigma, cov, x_scale, y_scale,
                      n_draws=200, seed=0, accept=None):
    """Pointwise log predictive density of y_new (original units) under the
    Laplace posterior of one skeleton for one dataset. Falls back to the MAP
    plug-in if the covariance is unusable."""
    rng = np.random.default_rng(seed)
    x_new = np.asarray(x_new, float).reshape(len(y_new), -1) / x_scale
    y_s = np.asarray(y_new, float) / y_scale
    mode = np.concatenate([phi, [log_sigma]])
    if np.all(np.isfinite(cov)):
        try:
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(mode)))
            draws = mode + rng.standard_normal((n_draws, len(mode))) @ L.T
        except np.linalg.LinAlgError:
            draws = mode[None, :]
    else:
        draws = mode[None, :]
    if accept is not None and draws.shape[0] > 1:
        # truncated prior (hsr.constraints): keep only draws with the properties
        keep = accept(draws[:, :cs.k])
        if keep.any():
            draws = draws[keep]
    m = draws.shape[0]
    X = np.broadcast_to(x_new[None], (m,) + x_new.shape)
    with np.errstate(all="ignore"):
        f = cs.f(X, draws[:, :cs.k])               # (m, n_new)
        sig = np.exp(draws[:, cs.k])[:, None]
        lp = -0.5 * LOG2PI - np.log(sig) - 0.5 * ((y_s[None, :] - f) / sig) ** 2
    lp = np.where(np.isfinite(lp), lp, -np.inf)
    out = logsumexp(lp, axis=0) - np.log(m) - np.log(y_scale)
    # floor each point's log density: a skeleton that blows up beyond the
    # observed range should cost a lot, not dominate every average
    return np.where(np.isfinite(out), np.maximum(out, LPD_FLOOR), LPD_FLOOR)


# --------------------------------------------------------- whole L matrix
@dataclass
class LikelihoodTable:
    laplace: np.ndarray     # (J, n_cand)
    bic: np.ndarray         # (J, n_cand)
    phi: list               # per candidate: (J, k) MAP constants
    log_sigma: np.ndarray   # (J, n_cand)
    cov: list               # per candidate: (J, k+1, k+1)

    def matrix(self, kind="laplace"):
        return self.laplace if kind == "laplace" else self.bic

    def extend(self, other: "LikelihoodTable"):
        return LikelihoodTable(np.hstack([self.laplace, other.laplace]), np.hstack([self.bic, other.bic]),
                               self.phi + other.phi, np.hstack([self.log_sigma, other.log_sigma]),
                               self.cov + other.cov)


def compute_likelihoods(skeletons, X, Y, M=None, n_vars=1, n_starts=10, tau=3.0, seed=0,
                        compiled_cache=None, progress=False) -> LikelihoodTable:
    """L[j, e] for all skeletons (list of Skeleton) and scaled datasets."""
    Y = np.asarray(Y)
    J_ = Y.shape[0]
    nc = len(skeletons)
    lap = np.full((J_, nc), -np.inf)
    bic = np.full((J_, nc), -np.inf)
    logsig = np.full((J_, nc), np.nan)
    phis, covs = [], []
    for e, sk in enumerate(skeletons):
        if progress and e % 500 == 0:
            print(f"  likelihoods {e}/{nc}", flush=True)
        cs = None if compiled_cache is None else compiled_cache.get(sk.key)
        if cs is None:
            cs = CompiledSkeleton(sk, n_vars)
            if compiled_cache is not None:
                compiled_cache[sk.key] = cs
        try:
            res = fit_skeleton(cs, X, Y, M, n_starts=n_starts, tau=tau, seed=seed + e)
        except Exception:
            phis.append(np.full((J_, cs.k), np.nan))
            covs.append(np.full((J_, cs.k + 1, cs.k + 1), np.nan))
            continue
        lap[:, e] = res.log_ml
        bic[:, e] = res.bic
        logsig[:, e] = res.log_sigma
        phis.append(res.phi)
        covs.append(res.cov)
    return LikelihoodTable(lap, bic, phis, logsig, covs)
