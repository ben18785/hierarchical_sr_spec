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
    split_merge: int = 3          # split-merge proposals per sweep (0 = off)
    single_site: bool = True      # single-dataset Gibbs updates (off only for testing other moves)
    n_temps: int = 1              # > 1: non-reversible parallel tempering with this many replicas (hsr.tempering)
    pt_tune_rounds: int = 6       # schedule-tuning rounds of 1, 2, 4, ... iterations (discarded)
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
    seconds: float = None         # wall time of the sampler
    pt: list = None               # per chain: parallel-tempering diagnostics (betas, barrier, swaps, round trips)

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


def _cluster_label_weights(model, T, log_p, gamma, allowed):
    """log g_T(e) = log gamma P(e) + sum_{k=1}^{m-1} log(k + gamma P(e)) + M_e(T)
    for every label e (-inf where not allowed): the prior-times-marginal
    weight of cluster T carrying label e."""
    m = len(T)
    gp = gamma * np.exp(log_p)
    lg = np.log(gp) + model.marginal_all(sorted(T))
    if m > 1:
        lg = lg + np.log(np.arange(1, m)[:, None] + gp[None, :]).sum(axis=0)
    return np.where(allowed, lg, -np.inf)


def _log_w_pair(lg1, lg2):
    """log sum_{e1 != e2} exp(lg1[e1] + lg2[e2])."""
    a1, a2 = logsumexp(lg1), logsumexp(lg2)
    if not (np.isfinite(a1) and np.isfinite(a2)):
        return -np.inf
    r = logsumexp(lg1 + lg2) - a1 - a2
    if r >= 0:
        return -np.inf
    with np.errstate(divide="ignore"):
        return a1 + a2 + np.log1p(-np.exp(r))


def _sequential_allocation(rng, model, S, i, j, label_score, target=None):
    """Sequential allocation (Dahl 2003) of the members of S other than i and j
    to two sides seeded with i and j, in random order. Each member k goes to
    side s with probability proportional to |T_s| exp(M_{e_s}(T_s + k) - M_{e_s}(T_s)),
    where e_s is the side's best label under ``label_score`` (refreshed when
    the side's size reaches 1, 2, 4, 8, ...), so placements follow the groups
    as they form rather than the two seeds alone.

    With ``target`` (dict k -> 0/1) the placements are forced, which gives the
    probability of proposing a given split (for the reverse of a merge).
    Returns (side sets, log proposal probability)."""
    T = [{int(i)}, {int(j)}]
    lab = [int(np.argmax(label_score(T[0]))), int(np.argmax(label_score(T[1])))]
    cur = [model.marginal(lab[0], sorted(T[0])), model.marginal(lab[1], sorted(T[1]))]
    nxt_refresh = [2, 2]
    logq = 0.0
    for k in rng.permutation(sorted(S - {i, j})):
        k = int(k)
        new = [model.marginal(lab[s], sorted(T[s] | {k})) for s in (0, 1)]
        lw = np.array([np.log(len(T[s])) + new[s] - cur[s] for s in (0, 1)])
        if not np.all(np.isfinite(lw)):
            lw = np.where(np.isfinite(lw), 0.0, -np.inf) if np.isfinite(lw).any() else np.zeros(2)
        lp = lw - logsumexp(lw)
        side = (0 if rng.random() < np.exp(lp[0]) else 1) if target is None else target[k]
        logq += lp[side]
        T[side].add(k)
        cur[side] = new[side]
        if len(T[side]) >= nxt_refresh[side]:
            nxt_refresh[side] *= 2
            lab[side] = int(np.argmax(label_score(T[side])))
            cur[side] = model.marginal(lab[side], sorted(T[side]))
    return T[0], T[1], logq


def _split_merge(rng, z, n_e, members, Mc, model, log_p, gamma, logsim, n_prop):
    """Split-merge Metropolis-Hastings moves (in the spirit of Jain & Neal 2004
    and Dahl 2003), with the labels of the clusters involved integrated out.

    Pick two datasets i != j at random. If they share a cluster S, propose
    splitting it into S_i (with i) and S_j (with j), the other members placed
    by sequential allocation (_sequential_allocation). If they are in
    different clusters, propose merging them; the reverse probability is that
    of the sequential allocation producing the current split. The target is the posterior of the partition of the datasets
    involved with the labels of the clusters involved summed out (labels used
    by other clusters are excluded), so the acceptance ratio uses
        W(S) = sum_e g_S(e)  and  W(S_i, S_j) = sum_{e1 != e2} g_{S_i}(e1) g_{S_j}(e2).
    After an accepted move the new labels are drawn from their exact
    conditional; a rejected move leaves the state unchanged."""
    J = len(z)
    C = len(n_e)
    for _ in range(n_prop):
        i, j = rng.choice(J, size=2, replace=False)
        ci, cj = int(z[i]), int(z[j])
        allowed = n_e == 0
        allowed[ci] = True
        allowed[cj] = True
        score = lambda T: _cluster_label_weights(model, T, log_p, gamma, allowed)
        if ci == cj:
            S = members[ci]
            Si, Sj, logq = _sequential_allocation(rng, model, S, i, j, score)
            lgS = _cluster_label_weights(model, S, log_p, gamma, allowed)
            lg1 = _cluster_label_weights(model, Si, log_p, gamma, allowed)
            lg2 = _cluster_label_weights(model, Sj, log_p, gamma, allowed)
            log_a = _log_w_pair(lg1, lg2) - logsumexp(lgS) - logq
            if np.isnan(log_a) or np.log(rng.random()) >= log_a:
                continue
            # labels (e1, e2), e1 != e2, from g1(e1) g2(e2)
            a2 = logsumexp(lg2)
            with np.errstate(divide="ignore", invalid="ignore"):
                m1 = lg1 + a2 + np.log1p(-np.exp(np.minimum(lg2 - a2, 0.0)))
            m1 = np.where(np.isfinite(m1), m1, -np.inf)
            e1 = int(rng.choice(C, p=np.exp(m1 - logsumexp(m1))))
            lg2b = lg2.copy()
            lg2b[e1] = -np.inf
            e2 = int(rng.choice(C, p=np.exp(lg2b - logsumexp(lg2b))))
            del members[ci]
            Mc.pop(ci, None)
            n_e[ci] = 0
            for e, T in ((e1, Si), (e2, Sj)):
                members[e] = set(T)
                n_e[e] = len(T)
                z[sorted(T)] = e
                Mc[e] = model.marginal(e, sorted(T))
        else:
            Si, Sj = members[ci], members[cj]
            S = Si | Sj
            _, _, logq = _sequential_allocation(rng, model, S, i, j, score,
                                                target={k: (0 if k in Si else 1) for k in S})
            lgS = _cluster_label_weights(model, S, log_p, gamma, allowed)
            lg1 = _cluster_label_weights(model, Si, log_p, gamma, allowed)
            lg2 = _cluster_label_weights(model, Sj, log_p, gamma, allowed)
            log_a = logsumexp(lgS) - _log_w_pair(lg1, lg2) + logq
            if np.isnan(log_a) or np.log(rng.random()) >= log_a:
                continue
            e = int(rng.choice(C, p=np.exp(lgS - logsumexp(lgS))))
            for c in (ci, cj):
                del members[c]
                Mc.pop(c, None)
                n_e[c] = 0
            members[e] = set(S)
            n_e[e] = len(S)
            z[sorted(S)] = e
            Mc[e] = model.marginal(e, sorted(S))


def _log_similarity(S1, log_p):
    """(J, J) log P(datasets k and l use the same form | each fitted alone)."""
    lw = log_p[None, :] + S1
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    P = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(P @ P.T, 1e-300))


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


class ChainState:
    """State of one collapsed-Gibbs chain (assignments, theta, gamma) with
    one-sweep updates, so that several chains can be advanced together (as in
    parallel tempering, hsr.tempering). ``model`` supplies the cluster
    marginals M_e(S) (hsr.pooled); the default is the independent-constants
    model, M_e(S) = sum_j L[j, e]."""

    def __init__(self, L, counts, grammar: Grammar, cfg: GibbsConfig, seed: int, init="argmax", theta0=None,
                 z_init=None, model=None):
        from .pooled import IndependentModel
        self.rng = np.random.default_rng(seed)
        self.cfg, self.grammar, self.counts = cfg, grammar, counts
        self.set_model(IndependentModel(L) if model is None else model, rebuild=False)
        J, C = self.S1.shape
        self.J, self.C = J, C
        self.theta = grammar.prior_mean() if theta0 is None else theta0
        self.gamma = cfg.gamma_init
        self.log_p = grammar.log_prob(counts, self.theta)
        rng = self.rng
        if z_init is not None:
            z = np.asarray(z_init, dtype=int).copy()
        elif init == "argmax":
            z = np.argmax(self.log_p[None, :] + self.S1, axis=1)
        else:
            lw = self.log_p[None, :] + self.S1
            z = np.array([rng.choice(C, p=np.exp(row - logsumexp(row))) for row in lw])
        self.z = z
        self.n_e = np.bincount(z, minlength=C).astype(float)
        self.members = {}
        for j, e in enumerate(z):
            self.members.setdefault(int(e), set()).add(j)
        self.Mc = {e: self.model.marginal(e, sorted(S)) for e, S in self.members.items()}
        self.n_sweeps_done = 0

    def set_model(self, model, rebuild=True):
        """Change the cluster-marginal model (e.g. its temperature) and
        recompute the cached marginals of the current clusters."""
        self.model = model
        self.S1 = model.singletons()
        if rebuild:
            self.Mc = {e: model.marginal(e, sorted(S)) for e, S in self.members.items()}

    def sweep(self):
        cfg, rng, model = self.cfg, self.rng, self.model
        z, n_e, members, Mc, S1 = self.z, self.n_e, self.members, self.Mc, self.S1
        J, C = self.J, self.C
        gamma, log_p = self.gamma, self.log_p
        log_gamma = np.log(gamma)
        for j in (rng.permutation(J) if cfg.single_site else ()):
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
        if cfg.split_merge and J > 1:
            _split_merge(rng, z, n_e, members, Mc, model, log_p, gamma, _log_similarity(S1, log_p),
                         cfg.split_merge)
        if cfg.relabel and self.n_sweeps_done % cfg.relabel_every == 0:
            _relabel_blocks(rng, z, n_e, members, Mc, model, log_p, gamma)
        in_use = np.flatnonzero(n_e > 0)
        if cfg.update_theta:
            self.theta = self.grammar.sample_theta(rng, self.counts[in_use].sum(axis=0))
            self.log_p = self.grammar.log_prob(self.counts, self.theta)
        if cfg.update_gamma:
            self.gamma = escobar_west_gamma(rng, gamma, len(in_use), J, cfg.gamma_a, cfg.gamma_b)
        self.n_sweeps_done += 1

    def swap_state(self, other):
        """Exchange the sampler state (not the model or rng) with another chain."""
        for a in ("z", "n_e", "members", "theta", "gamma", "log_p"):
            va, vb = getattr(self, a), getattr(other, a)
            setattr(self, a, vb)
            setattr(other, a, va)
        self.Mc = {e: self.model.marginal(e, sorted(S)) for e, S in self.members.items()}
        other.Mc = {e: other.model.marginal(e, sorted(S)) for e, S in other.members.items()}


class _Recorder:
    def __init__(self, J, C, nt_names):
        self.assign = np.zeros((J, C))
        self.pop = np.zeros(C)
        self.n_forms, self.gammas, self.log_liks, self.log_posts, self.kept = [], [], [], [], []
        self.thetas = {nt: [] for nt in nt_names}

    def record(self, st: ChainState, log_lik):
        J = st.J
        in_use = np.flatnonzero(st.n_e > 0)
        self.assign[np.arange(J), st.z] += 1
        self.pop += (st.n_e + st.gamma * np.exp(st.log_p)) / (st.gamma + J)
        self.n_forms.append(len(in_use))
        self.gammas.append(st.gamma)
        self.log_liks.append(float(log_lik))
        self.log_posts.append(self.log_liks[-1] + partition_log_prior(st.n_e, st.log_p, st.gamma))
        for nt in self.thetas:
            self.thetas[nt].append(st.theta[nt])
        self.kept.append(st.z.copy())

    def result(self):
        n = max(len(self.kept), 1)
        return {
            "assign_probs": self.assign / n,
            "pop": self.pop / n,
            "n_forms": np.array(self.n_forms),
            "gamma": np.array(self.gammas),
            "theta": {nt: np.array(v) for nt, v in self.thetas.items()},
            "z": np.array(self.kept),
            "log_lik": np.array(self.log_liks),
            "log_post": np.array(self.log_posts),
        }


def run_chain(L: np.ndarray, counts: np.ndarray, grammar: Grammar, cfg: GibbsConfig, seed: int,
              init: str = "argmax", theta0: dict | None = None, z_init=None, model=None):
    """One chain. ``model`` supplies cluster marginals M_e(S) (see hsr.pooled);
    the default is the independent-constants model, M_e(S) = sum_j L[j, e]."""
    st = ChainState(L, counts, grammar, cfg, seed, init, theta0, z_init, model)
    rec = _Recorder(st.J, st.C, grammar.nonterminals)
    for sweep in range(cfg.n_sweeps):
        st.sweep()
        if sweep >= cfg.burn_in and (sweep - cfg.burn_in) % cfg.thin == 0:
            rec.record(st, sum(st.Mc.values()))
    return rec.result()


def run_gibbs(L: np.ndarray, counts: np.ndarray, grammar: Grammar, cfg: GibbsConfig | None = None,
              z_init=None, model=None) -> GibbsResult:
    """Run cfg.n_chains chains and pool them. ``z_init`` (J,) fixes the
    starting assignment of every chain (default: cfg.init)."""
    cfg = cfg or GibbsConfig()
    if cfg.burn_in >= cfg.n_sweeps:
        raise ValueError("burn_in must be smaller than n_sweeps")
    import time
    t0 = time.time()
    chains = []
    for c in range(cfg.n_chains):
        init = "argmax" if c == 0 else cfg.init
        if cfg.n_temps > 1:
            from .tempering import run_pt_chain
            chains.append(run_pt_chain(L, counts, grammar, cfg, seed=cfg.seed * 1000 + c, init=init, model=model,
                                       z_init=z_init))
        else:
            chains.append(run_chain(L, counts, grammar, cfg, seed=cfg.seed * 1000 + c, init=init, z_init=z_init,
                                    model=model))
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
        seconds=time.time() - t0,
        pt=[ch["pt"] for ch in chains] if cfg.n_temps > 1 else None,
    )
