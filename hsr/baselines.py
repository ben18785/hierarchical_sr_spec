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
    """The single skeleton maximising sum_j L[j, e]."""
    Lf = np.where(np.isfinite(L), L, -1e300)
    return int(np.argmax(Lf.sum(axis=0)))


def oracle(L: np.ndarray, counts: np.ndarray, grammar: Grammar, true_idx, cfg: GibbsConfig | None = None):
    """Gibbs sampler restricted to the true forms. Returns (GibbsResult, idx)
    where result columns refer to ``true_idx`` in order."""
    idx = np.asarray(true_idx)
    return run_gibbs(L[:, idx], counts[idx], grammar, cfg), idx
