"""Stage A step 3: collapsed Gibbs sampler over skeleton assignments.

State: z_j (index into the candidate set) for each dataset, theta, gamma.
With G integrated out (Chinese restaurant process):

    P(z_j = e | z_-j, y_j, theta, gamma) ∝ (n_e^{-j} + gamma P_theta(e)) exp(L[j, e])

theta | z ~ Dirichlet(alpha + sum over distinct skeletons in use of counts(e))
gamma | K  by the Escobar & West (1995) auxiliary-variable step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .grammar import Grammar


@dataclass
class GibbsConfig:
    n_sweeps: int = 2000
    burn_in: int = 500
    n_chains: int = 4
    gamma_a: float = 1.0          # Gamma(a, b) prior on gamma (b is a rate)
    gamma_b: float = 1.0
    gamma_init: float = 1.0
    update_theta: bool = True
    update_gamma: bool = True
    init: str = "argmax"          # chain 0 always argmax; "argmax" | "random" for the others
    thin: int = 1
    seed: int = 0


@dataclass
class GibbsResult:
    assign_probs: np.ndarray      # (J, n_cand) posterior P(z_j = e), pooled over chains
    chain_assign_probs: np.ndarray  # (n_chains, J, n_cand)
    pop_weight: np.ndarray        # (n_cand,) posterior mean of G({e})
    n_forms: np.ndarray           # (n_chains, n_kept) number of distinct skeletons in use
    gamma: np.ndarray             # (n_chains, n_kept)
    theta: dict                   # nt -> (n_chains, n_kept, n_options)
    z_samples: np.ndarray         # (n_chains, n_kept, J)

    @property
    def top(self) -> np.ndarray:
        return np.argmax(self.assign_probs, axis=1)

    def n_forms_distribution(self):
        vals, cnt = np.unique(self.n_forms, return_counts=True)
        return dict(zip(vals.tolist(), (cnt / cnt.sum()).tolist()))

    def rhat(self, x: np.ndarray) -> float:
        """Split-free Gelman-Rubin R-hat for a (n_chains, n_kept) array."""
        m, n = x.shape
        if m < 2 or n < 2:
            return np.nan
        means = x.mean(axis=1)
        W = x.var(axis=1, ddof=1).mean()
        Bn = means.var(ddof=1)
        if W == 0:
            return 1.0 if Bn == 0 else np.inf
        var = (n - 1) / n * W + Bn
        return float(np.sqrt(var / W))

    def chain_disagreement(self) -> float:
        """Max over datasets and chain pairs of the total-variation distance
        between chain-specific assignment posteriors (0 = perfect agreement)."""
        P = self.chain_assign_probs
        worst = 0.0
        for a in range(P.shape[0]):
            for b in range(a + 1, P.shape[0]):
                worst = max(worst, float(0.5 * np.abs(P[a] - P[b]).sum(axis=1).max()))
        return worst


def _log_crp_weight(n_minus: np.ndarray, log_gamma: float, log_p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        log_n = np.log(n_minus)
    return np.logaddexp(log_n, log_gamma + log_p)


def escobar_west_gamma(rng, gamma: float, K: int, n: int, a: float, b: float) -> float:
    """One Escobar & West (1995) update of the DP concentration (b = rate)."""
    eta = rng.beta(gamma + 1.0, n)
    rate = b - np.log(eta)
    odds = (a + K - 1.0) / (n * rate)
    pi = odds / (1.0 + odds)
    shape = a + K if rng.random() < pi else a + K - 1.0
    return rng.gamma(shape, 1.0 / rate)


def run_chain(L: np.ndarray, counts: np.ndarray, grammar: Grammar, cfg: GibbsConfig, seed: int,
              init: str = "argmax", theta0: dict | None = None):
    rng = np.random.default_rng(seed)
    J, C = L.shape
    theta = grammar.prior_mean() if theta0 is None else theta0
    gamma = cfg.gamma_init
    log_p = grammar.log_prob(counts, theta)
    Lf = np.where(np.isfinite(L), L, -np.inf)
    if init == "argmax":
        z = np.argmax(log_p[None, :] + Lf, axis=1)
    else:
        lw = log_p[None, :] + Lf
        z = np.array([rng.choice(C, p=np.exp(row - logsumexp(row))) for row in lw])
    n_e = np.bincount(z, minlength=C).astype(float)

    kept = []
    nt_names = grammar.nonterminals
    assign_counts = np.zeros((J, C))
    pop = np.zeros(C)
    n_forms, gammas = [], []
    thetas = {nt: [] for nt in nt_names}
    for sweep in range(cfg.n_sweeps):
        log_gamma = np.log(gamma)
        for j in rng.permutation(J):
            n_e[z[j]] -= 1
            lw = _log_crp_weight(n_e, log_gamma, log_p) + Lf[j]
            top = lw.max()
            if not np.isfinite(top):  # every candidate failed for this dataset
                lw = np.zeros(C)
                top = 0.0
            p = np.exp(lw - top)
            p /= p.sum()
            z[j] = rng.choice(C, p=p)
            n_e[z[j]] += 1
        in_use = np.flatnonzero(n_e > 0)
        if cfg.update_theta:
            theta = grammar.sample_theta(rng, counts[in_use].sum(axis=0))
            log_p = grammar.log_prob(counts, theta)
        if cfg.update_gamma:
            gamma = escobar_west_gamma(rng, gamma, len(in_use), J, cfg.gamma_a, cfg.gamma_b)
        if sweep >= cfg.burn_in and (sweep - cfg.burn_in) % cfg.thin == 0:
            assign_counts[np.arange(J), z] += 1
            pop += (n_e + gamma * np.exp(log_p)) / (gamma + J)
            n_forms.append(len(in_use))
            gammas.append(gamma)
            for nt in nt_names:
                thetas[nt].append(theta[nt])
            kept.append(z.copy())
    n_kept = len(kept)
    return {
        "assign_probs": assign_counts / max(n_kept, 1),
        "pop": pop / max(n_kept, 1),
        "n_forms": np.array(n_forms),
        "gamma": np.array(gammas),
        "theta": {nt: np.array(v) for nt, v in thetas.items()},
        "z": np.array(kept),
    }


def run_gibbs(L: np.ndarray, counts: np.ndarray, grammar: Grammar, cfg: GibbsConfig | None = None) -> GibbsResult:
    """Run cfg.n_chains chains and pool them."""
    cfg = cfg or GibbsConfig()
    if cfg.burn_in >= cfg.n_sweeps:
        raise ValueError("burn_in must be smaller than n_sweeps")
    chains = []
    for c in range(cfg.n_chains):
        init = "argmax" if c == 0 else cfg.init
        chains.append(run_chain(L, counts, grammar, cfg, seed=cfg.seed * 1000 + c, init=init))
    cap = np.stack([ch["assign_probs"] for ch in chains])
    return GibbsResult(
        assign_probs=cap.mean(axis=0),
        chain_assign_probs=cap,
        pop_weight=np.mean([ch["pop"] for ch in chains], axis=0),
        n_forms=np.stack([ch["n_forms"] for ch in chains]),
        gamma=np.stack([ch["gamma"] for ch in chains]),
        theta={nt: np.stack([ch["theta"][nt] for ch in chains]) for nt in grammar.nonterminals},
        z_samples=np.stack([ch["z"] for ch in chains]),
    )
