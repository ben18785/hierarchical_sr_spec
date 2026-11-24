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
    init: str = "random"          # chain 0 always argmax; others "random" (each dataset's form drawn from
                                  # its own posterior: dispersed starts) or "argmax"
    relabel: bool = True          # block move: resample the skeleton of a whole cluster
    relabel_every: int = 1        # sweeps between block moves
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
    log_lik: np.ndarray = None    # (n_chains, n_kept) sum of cluster marginals (label-invariant diagnostic)
    log_post: np.ndarray = None   # (n_chains, n_kept) log_lik + log P(partition, labels | gamma, theta)

    @property
    def top(self) -> np.ndarray:
        return np.argmax(self.assign_probs, axis=1)

    def coassignment(self) -> np.ndarray:
        """(J, J) posterior probability that datasets i and j use the same
        skeleton (clusters are identified by their skeleton), over all chains
        and kept sweeps."""
        z = self.z_samples.reshape(-1, self.z_samples.shape[-1])     # (draws, J)
        return (z[:, :, None] == z[:, None, :]).mean(axis=0)

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


def _relabel_blocks(rng, z, n_e, members, Mc, model, log_p, gamma):
    """For each cluster (the datasets sharing one skeleton), resample its
    skeleton label with the partition held fixed.

    With the partition fixed and the cluster's label e unused by any other
    cluster, the Polya-urn probability of the assignment depends on e only
    through gamma P(e) prod_{k=1}^{m-1} (k + gamma P(e)) (m = cluster size),
    so the exact conditional is
        P(e) ∝ P(e) prod_{k=1}^{m-1} (k + gamma P(e)) exp(M_e(cluster))
    over every label not used by another cluster, where M_e is the cluster
    marginal (sum_j L[j, e] for independent constants). This lets a whole
    cluster move between skeletons, which single-site updates can only do
    through low-probability intermediate states (spec section 8).
    """
    C = len(n_e)
    gp = gamma * np.exp(log_p)
    for e in rng.permutation(list(members)):
        e = int(e)
        S = sorted(members[e])
        m = len(S)
        lw = log_p + model.marginal_all(S)
        if m > 1:
            k = np.arange(1, m)[:, None]
            lw = lw + np.log(k + gp[None, :]).sum(axis=0)
        other = n_e > 0
        other[e] = False
        lw[other] = -np.inf
        top = lw.max()
        if not np.isfinite(top):
            continue
        p = np.exp(lw - top)
        p /= p.sum()
        new = int(rng.choice(C, p=p))
        if new != e:
            z[S] = new
            n_e[new] = m
            n_e[e] = 0
            members[new] = members.pop(e)
            Mc.pop(e, None)
            Mc[new] = model.marginal(new, S)


def partition_log_prior(n_e, log_p, gamma) -> float:
    """log P(z | gamma, theta) under the DP with discrete base P_theta: each
    cluster b of size m with label e contributes
    log(gamma P(e)) + sum_{k=1}^{m-1} log(k + gamma P(e)), and the whole
    assignment is divided by prod_{i=0}^{J-1} (gamma + i)."""
    used = np.flatnonzero(n_e > 0)
    J = int(n_e.sum())
    out = -float(np.sum(np.log(gamma + np.arange(J))))
    for e in used:
        gp = gamma * np.exp(log_p[e])
        m = int(n_e[e])
        out += float(np.log(gp) + np.sum(np.log(np.arange(1, m) + gp)))
    return out


def escobar_west_gamma(rng, gamma: float, K: int, n: int, a: float, b: float) -> float:
    """One Escobar & West (1995) update of the DP concentration (b = rate)."""
    eta = rng.beta(gamma + 1.0, n)
    rate = b - np.log(eta)
    odds = (a + K - 1.0) / (n * rate)
    pi = odds / (1.0 + odds)
    shape = a + K if rng.random() < pi else a + K - 1.0
    return rng.gamma(shape, 1.0 / rate)


def run_chain(L: np.ndarray, counts: np.ndarray, grammar: Grammar, cfg: GibbsConfig, seed: int,
              init: str = "argmax", theta0: dict | None = None, z_init=None, model=None):
    """One chain. ``model`` supplies cluster marginals M_e(S) (see hsr.pooled);
    the default is the independent-constants model, M_e(S) = sum_j L[j, e]."""
    from .pooled import IndependentModel
    rng = np.random.default_rng(seed)
    model = IndependentModel(L) if model is None else model
    S1 = model.singletons()
    J, C = S1.shape
    theta = grammar.prior_mean() if theta0 is None else theta0
    gamma = cfg.gamma_init
    log_p = grammar.log_prob(counts, theta)
    if z_init is not None:
        z = np.asarray(z_init, dtype=int).copy()
    elif init == "argmax":
        z = np.argmax(log_p[None, :] + S1, axis=1)
    else:
        lw = log_p[None, :] + S1
        z = np.array([rng.choice(C, p=np.exp(row - logsumexp(row))) for row in lw])
    n_e = np.bincount(z, minlength=C).astype(float)
    members = {}
    for j, e in enumerate(z):
        members.setdefault(int(e), set()).add(j)
    Mc = {e: model.marginal(e, sorted(S)) for e, S in members.items()}

    kept = []
    nt_names = grammar.nonterminals
    assign_counts = np.zeros((J, C))
    pop = np.zeros(C)
    n_forms, gammas, log_liks, log_posts = [], [], [], []
    thetas = {nt: [] for nt in nt_names}
    for sweep in range(cfg.n_sweeps):
        log_gamma = np.log(gamma)
        for j in rng.permutation(J):
            e0 = int(z[j])
            members[e0].discard(j)
            n_e[e0] -= 1
            if members[e0]:
                Mc[e0] = model.marginal(e0, sorted(members[e0]))
            else:
                del members[e0]
                Mc.pop(e0, None)
            # every skeleton as a new cluster ...
            lw = log_gamma + log_p + S1[j]
            # ... and the occupied ones as existing clusters
            joint = {}
            for e, S in members.items():
                joint[e] = model.marginal(e, sorted(S | {j}))
                lw[e] = np.log(n_e[e] + gamma * np.exp(log_p[e])) + joint[e] - Mc[e]
            top = lw.max()
            if not np.isfinite(top):  # every candidate failed for this dataset
                lw = np.zeros(C)
                top = 0.0
            p = np.exp(lw - top)
            p /= p.sum()
            e1 = int(rng.choice(C, p=p))
            z[j] = e1
            n_e[e1] += 1
            if e1 in members:
                members[e1].add(j)
                Mc[e1] = joint[e1]
            else:
                members[e1] = {j}
                Mc[e1] = S1[j, e1]
        if cfg.relabel and sweep % cfg.relabel_every == 0:
            _relabel_blocks(rng, z, n_e, members, Mc, model, log_p, gamma)
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
            log_liks.append(float(sum(Mc.values())))
            log_posts.append(log_liks[-1] + partition_log_prior(n_e, log_p, gamma))
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
        "log_lik": np.array(log_liks),
        "log_post": np.array(log_posts),
    }


def run_gibbs(L: np.ndarray, counts: np.ndarray, grammar: Grammar, cfg: GibbsConfig | None = None,
              z_init=None, model=None) -> GibbsResult:
    """Run cfg.n_chains chains and pool them. ``z_init`` (J,) fixes the
    starting assignment of every chain (default: cfg.init)."""
    cfg = cfg or GibbsConfig()
    if cfg.burn_in >= cfg.n_sweeps:
        raise ValueError("burn_in must be smaller than n_sweeps")
    chains = []
    for c in range(cfg.n_chains):
        init = "argmax" if c == 0 else cfg.init
        chains.append(run_chain(L, counts, grammar, cfg, seed=cfg.seed * 1000 + c, init=init, z_init=z_init, model=model))
    cap = np.stack([ch["assign_probs"] for ch in chains])
    return GibbsResult(
        assign_probs=cap.mean(axis=0),
        chain_assign_probs=cap,
        pop_weight=np.mean([ch["pop"] for ch in chains], axis=0),
        n_forms=np.stack([ch["n_forms"] for ch in chains]),
        gamma=np.stack([ch["gamma"] for ch in chains]),
        theta={nt: np.stack([ch["theta"][nt] for ch in chains]) for nt in grammar.nonterminals},
        z_samples=np.stack([ch["z"] for ch in chains]),
        log_lik=np.stack([ch["log_lik"] for ch in chains]),
        log_post=np.stack([ch["log_post"] for ch in chains]),
    )
