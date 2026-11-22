"""Pooled constants within a skeleton (Overleaf note, Section 5.1).

Datasets assigned to the same skeleton e share a population distribution for
their constants:

    phi_j | e, mu_e, s_e ~ N(mu_e, s_e^2 I),    mu_e ~ N(0, tau_mu^2 I),
    s_e ~ HalfNormal(s0), integrated on a fixed grid.

The likelihood of each (dataset, skeleton) pair is represented by a Gaussian
in phi, obtained by dividing the Laplace posterior under the reference prior
N(0, tau_ref^2 I) (the fit already computed by ``compute_likelihoods``) by that
prior:

    p(y_j | e, phi) ~ Z_je N(phi; m_je, V_je) / N(phi; 0, tau_ref^2 I),

with Z_je = exp(L_je), m_je the MAP and V_je the phi block of the Laplace
covariance (sigma already integrated out). Every integral below is then a
Gaussian integral, so the cluster marginal

    M_e(S) = log ∫∫ prod_{j in S} p(y_j | e, mu_e, s_e) N(mu_e; 0, tau_mu^2 I) p(s_e) dmu_e ds_e

is available in closed form (mu_e) and by exact summation over the grid (s_e).
The Gibbs sampler needs nothing else: there is no per-cluster continuous state.

Isotropic s_e is a simplification (one spread for all constants of a
skeleton); constants are on the scale of the pooled-scaled data.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

LOG2PI = np.log(2 * np.pi)
DEFAULT_GRID = np.array([0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0])


def grid_log_weights(grid, s0=1.0):
    """Log quadrature weights of HalfNormal(s0) on a log-spaced grid."""
    grid = np.asarray(grid, float)
    w = grid * np.exp(-0.5 * (grid / s0) ** 2)   # density x d(log s)
    return np.log(w / w.sum())


def member_terms(L, m, V, s, tau_ref, min_precision=1e-6, min_variance=1e-8):
    """Quadratic form in mu of log p(y | e, mu, s^2 I) for a batch of members.

    L: (B,), m: (B, k), V: (B, k, k), s: scalar.
    Returns A (B, k, k), a (B, k), c (B,) with
        log p = -1/2 mu' A mu + a' mu + c.

    The likelihood in phi is the Laplace posterior divided by the reference
    prior: precision Lam = V^-1 - I/tau_ref^2, mean m_l = Lam^-1 V^-1 m. Where
    the posterior is wider than the prior (weakly identified directions, or a
    Laplace fit at a saddle) Lam has non-positive eigenvalues and the implied
    likelihood would grow without bound; those eigenvalues are clipped to
    ``min_precision`` (a flat direction). The likelihood is
        K N(phi; m_l, Lam^-1),  log K = L - log N(m_l; 0, Lam^-1 + tau_ref^2 I),
    so that integrating it against the reference prior returns exp(L); and
        p(y | mu, s) = K N(m_l; mu, Lam^-1 + s^2 I).
    Invalid members (non-finite L or V) get c = -inf and A = a = 0.
    """
    B, k = m.shape
    ok = np.isfinite(L) & np.all(np.isfinite(V.reshape(B, -1)), axis=1) & np.all(np.isfinite(m), axis=1)
    A = np.zeros((B, k, k))
    a = np.zeros((B, k))
    c = np.full(B, -np.inf)
    if not ok.any():
        return A, a, c
    I = np.eye(k)
    Vo, mo, Lo = V[ok], m[ok], L[ok]
    Vo = 0.5 * (Vo + np.swapaxes(Vo, 1, 2))
    # floor the covariance eigenvalues: a near-singular Laplace covariance
    # (e.g. a power whose exponent is pinned at a pole) otherwise gives
    # precisions ~1e26 and destroys the Gaussian algebra below
    ev_v, U_v = np.linalg.eigh(Vo)
    ev_v = np.maximum(ev_v, min_variance)
    L1 = np.einsum("bij,bj,bkj->bik", U_v, 1.0 / ev_v, U_v)
    Lam = L1 - I / tau_ref**2
    Lam = 0.5 * (Lam + np.swapaxes(Lam, 1, 2))
    ev, U = np.linalg.eigh(Lam)
    ev = np.maximum(ev, min_precision)
    Lam_inv = np.einsum("bij,bj,bkj->bik", U, 1.0 / ev, U)
    Lam = np.einsum("bij,bj,bkj->bik", U, ev, U)
    m_l = np.einsum("bij,bjk,bk->bi", Lam_inv, L1, mo)
    # log K
    Wref = Lam_inv + tau_ref**2 * I
    _, ld_ref = np.linalg.slogdet(Wref)
    q_ref = np.einsum("bi,bi->b", m_l, np.linalg.solve(Wref, m_l[..., None])[..., 0])
    logK = Lo + 0.5 * (k * LOG2PI + ld_ref) + 0.5 * q_ref
    # Gaussian in mu with covariance W = Lam^-1 + s^2 I
    W = Lam_inv + s**2 * I
    Winv = np.linalg.inv(W)
    _, ld_W = np.linalg.slogdet(W)
    A[ok] = Winv
    a[ok] = np.einsum("bij,bj->bi", Winv, m_l)
    c[ok] = logK - 0.5 * (k * LOG2PI + ld_W) - 0.5 * np.einsum("bi,bij,bj->b", m_l, Winv, m_l)
    return A, a, c


def _integrate_mu(Asum, asum, csum, k, tau_mu):
    """log ∫ N(mu; 0, tau_mu^2 I) exp(-1/2 mu'Asum mu + asum'mu + csum) dmu (batched)."""
    Q = Asum + np.eye(k) / tau_mu**2
    sign, logdetQ = np.linalg.slogdet(Q)
    sol = np.linalg.solve(Q, asum[..., None])[..., 0]
    out = csum - k * np.log(tau_mu) - 0.5 * logdetQ + 0.5 * np.einsum("...i,...i->...", asum, sol)
    return np.where((sign > 0) & np.isfinite(csum), out, -np.inf)


class PooledModel:
    """Cluster marginals M_e(S) for the pooled-constants model.

    ``table`` is a LikelihoodTable from ``compute_likelihoods`` with the
    Laplace fit under prior N(0, tau_ref^2) (tau_ref = the ``tau`` used there).
    """

    def __init__(self, table, n_consts, tau_ref=3.0, tau_mu=3.0, grid=DEFAULT_GRID, s0=1.0):
        self.tau_ref, self.tau_mu = tau_ref, tau_mu
        self.grid = np.asarray(grid, float)
        self.logw = grid_log_weights(self.grid, s0)
        L = table.laplace
        self.J, self.C = L.shape
        self.k = np.asarray(n_consts, int)
        G = len(self.grid)
        # group candidates by number of constants; precompute member terms
        self.groups = {}
        for k in np.unique(self.k):
            idx = np.flatnonzero(self.k == k)
            if k == 0:
                self.groups[0] = {"idx": idx, "c": L[:, idx]}
                continue
            A = np.zeros((self.J, len(idx), G, k, k))
            a = np.zeros((self.J, len(idx), G, k))
            c = np.zeros((self.J, len(idx), G))
            m = np.stack([table.phi[e] for e in idx], axis=1)               # (J, Ck, k)
            V = np.stack([table.cov[e][:, :k, :k] for e in idx], axis=1)    # (J, Ck, k, k)
            Lk = L[:, idx]
            for g, s in enumerate(self.grid):
                Ag, ag, cg = member_terms(Lk.reshape(-1), m.reshape(-1, k), V.reshape(-1, k, k), s, tau_ref)
                A[:, :, g] = Ag.reshape(self.J, len(idx), k, k)
                a[:, :, g] = ag.reshape(self.J, len(idx), k)
                c[:, :, g] = cg.reshape(self.J, len(idx))
            self.groups[int(k)] = {"idx": idx, "A": A, "a": a, "c": c}
        self.pos = np.empty(self.C, int)
        for k, grp in self.groups.items():
            self.pos[grp["idx"]] = np.arange(len(grp["idx"]))
        self._singletons = None

    # ------------------------------------------------------------------ API
    def marginal(self, e: int, members) -> float:
        """M_e(S) for one skeleton and a list of member datasets."""
        members = np.asarray(members, int)
        if len(members) == 0:
            return 0.0
        k = int(self.k[e])
        grp = self.groups[k]
        p = self.pos[e]
        if k == 0:
            return float(grp["c"][members, p].sum())
        Asum = grp["A"][members, p].sum(axis=0)       # (G, k, k)
        asum = grp["a"][members, p].sum(axis=0)       # (G, k)
        csum = grp["c"][members, p].sum(axis=0)       # (G,)
        Mg = _integrate_mu(Asum, asum, csum, k, self.tau_mu)
        return float(logsumexp(self.logw + Mg))

    def marginal_all(self, members) -> np.ndarray:
        """M_e(S) for every candidate e (S fixed)."""
        members = np.asarray(members, int)
        out = np.empty(self.C)
        for k, grp in self.groups.items():
            if k == 0:
                out[grp["idx"]] = grp["c"][members].sum(axis=0)
                continue
            Asum = grp["A"][members].sum(axis=0)        # (Ck, G, k, k)
            asum = grp["a"][members].sum(axis=0)
            csum = grp["c"][members].sum(axis=0)
            Mg = _integrate_mu(Asum, asum, csum, k, self.tau_mu)
            out[grp["idx"]] = logsumexp(self.logw + Mg, axis=-1)
        return out

    def singletons(self) -> np.ndarray:
        """(J, C) matrix of M_e({j})."""
        if self._singletons is None:
            out = np.empty((self.J, self.C))
            for k, grp in self.groups.items():
                if k == 0:
                    out[:, grp["idx"]] = grp["c"]
                    continue
                Mg = _integrate_mu(grp["A"], grp["a"], grp["c"], k, self.tau_mu)   # (J, Ck, G)
                out[:, grp["idx"]] = logsumexp(self.logw + Mg, axis=-1)
            self._singletons = out
        return self._singletons

    def posterior_mu(self, e: int, members):
        """Posterior mean and sd of mu_e and posterior over the grid of s_e."""
        members = np.asarray(members, int)
        k = int(self.k[e])
        if k == 0:
            return np.zeros(0), np.zeros(0), np.exp(self.logw)
        grp = self.groups[k]
        p = self.pos[e]
        Asum = grp["A"][members, p].sum(axis=0)
        asum = grp["a"][members, p].sum(axis=0)
        csum = grp["c"][members, p].sum(axis=0)
        Mg = _integrate_mu(Asum, asum, csum, k, self.tau_mu)
        wg = np.exp(self.logw + Mg - logsumexp(self.logw + Mg))
        Q = Asum + np.eye(k) / self.tau_mu**2
        means = np.linalg.solve(Q, asum[..., None])[..., 0]
        covs = np.linalg.inv(Q)
        mean = np.einsum("g,gi->i", wg, means)
        second = np.einsum("g,gij->ij", wg, covs + np.einsum("gi,gj->gij", means, means))
        sd = np.sqrt(np.maximum(np.diag(second) - mean**2, 0))
        return mean, sd, wg


class IndependentModel:
    """The original model: constants independent across datasets, M_e(S) = sum_j L[j, e]."""

    def __init__(self, L):
        self.L = np.where(np.isfinite(L), L, -np.inf)
        self.J, self.C = L.shape

    def marginal(self, e, members):
        members = np.asarray(members, int)
        return float(self.L[members, e].sum()) if len(members) else 0.0

    def marginal_all(self, members):
        return self.L[np.asarray(members, int)].sum(axis=0)

    def singletons(self):
        return self.L
