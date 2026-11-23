"""Baselines from spec section 6, all using the same candidate set and L matrix."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .gibbs import GibbsConfig, GibbsResult, run_gibbs
from .grammar import Grammar


def no_pooling(L: np.ndarray, counts: np.ndarray, grammar: Grammar) -> np.ndarray:
    """z_j = argmax_e log P_theta0(e) + L[j, e], theta0 = prior mean."""
    log_p = grammar.log_prob(counts, grammar.prior_mean())
    return np.argmax(log_p[None, :] + L, axis=1)


def no_pooling_posterior(L: np.ndarray, counts: np.ndarray, grammar: Grammar) -> np.ndarray:
    """Independent per-dataset posterior P(e | y_j) ∝ P_theta0(e) exp(L[j, e]).
    Not in the spec; included to separate the effect of pooling from the
    effect of averaging over skeletons."""
    log_p = grammar.log_prob(counts, grammar.prior_mean())
    lw = log_p[None, :] + L
    return np.exp(lw - logsumexp(lw, axis=1, keepdims=True))


def complete_pooling(L: np.ndarray) -> int:
    """The single skeleton maximising sum_j L[j, e] (no prior)."""
    Lf = np.where(np.isfinite(L), L, -1e300)
    return int(np.argmax(Lf.sum(axis=0)))


def complete_pooling_posterior(L: np.ndarray, counts: np.ndarray, grammar: Grammar) -> np.ndarray:
    """One form shared by every dataset, constants fitted per dataset:
    P(e | all data) ∝ P_theta0(e) prod_j exp(L[j, e]). Returns (n_cand,)."""
    log_p = grammar.log_prob(counts, grammar.prior_mean())
    Lf = np.where(np.isfinite(L), L, -np.inf).sum(axis=0)
    lw = log_p + Lf
    if not np.isfinite(lw).any():
        return np.full(len(lw), 1.0 / len(lw))
    return np.exp(lw - logsumexp(lw))


def single_curve_posterior(L1: np.ndarray, counts: np.ndarray, grammar: Grammar) -> np.ndarray:
    """One curve (form and constants) for all datasets: ``L1`` (n_cand,) is
    the evidence of the concatenated data. P(e | data) ∝ P_theta0(e) exp(L1[e])."""
    log_p = grammar.log_prob(counts, grammar.prior_mean())
    lw = log_p + np.where(np.isfinite(L1), L1, -np.inf)
    if not np.isfinite(lw).any():
        return np.full(len(lw), 1.0 / len(lw))
    return np.exp(lw - logsumexp(lw))


def broadcast_table(table1, J):
    """A LikelihoodTable for J datasets that all use the single-curve fit
    ``table1`` (1 dataset): every dataset gets the same constants, sigma and
    covariance, so held-out predictions use the common curve."""
    from .likelihood import LikelihoodTable
    rep = lambda a: np.repeat(np.asarray(a), J, axis=0)
    return LikelihoodTable(rep(table1.laplace), rep(table1.bic), [rep(p) for p in table1.phi],
                           rep(table1.log_sigma), [rep(c) for c in table1.cov])


def oracle(L: np.ndarray, counts: np.ndarray, grammar: Grammar, true_idx, cfg: GibbsConfig | None = None):
    """Gibbs sampler restricted to the true forms. Returns (GibbsResult, idx)
    where result columns refer to ``true_idx`` in order."""
    idx = np.asarray(true_idx)
    return run_gibbs(L[:, idx], counts[idx], grammar, cfg), idx
