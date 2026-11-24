"""Non-reversible parallel tempering for the DP samplers.

K replicas target pi_beta(state) ∝ prior(state) * exp(beta * sum_b M_{e_b}(S_b))
for 0 = beta_0 < ... < beta_{K-1} = 1: only the cluster marginals (the
likelihood part) are tempered, so the beta = 0 replica samples the prior over
partitions, labels, theta and gamma, which the Gibbs moves explore easily.
Failed fits (M = -inf) stay impossible at every beta, so all replicas share
one support.

Each iteration every replica does one ordinary sweep (hsr.gibbs.ChainState),
then adjacent replicas propose to exchange states with the deterministic
even-odd (DEO) scheme of Syed, Bouchard-Cote, Deligiannidis & Doucet (2022,
JRSSB): on even iterations the pairs (0,1), (2,3), ...; on odd iterations
(1,2), (3,4), .... The exchange of states x_k, x_{k+1} is accepted with
probability min(1, exp((beta_{k+1} - beta_k) (l(x_k) - l(x_{k+1})))), with
l the untempered sum of cluster marginals. Deterministic alternation makes
replicas move along the ladder with momentum: round trips take O(K) rather
than O(K^2) iterations.

The schedule is tuned in rounds of 1, 2, 4, ... iterations (Syed et al.,
Algorithm 4): the swap rejection rates r_k estimate the communication
barrier Lambda(beta) (piecewise linear in the cumulative r_k), and the new
betas equalise Lambda between neighbours. Tuning iterations are discarded.
Samples are recorded from the beta = 1 replica only.
"""
from __future__ import annotations

import numpy as np

from .gibbs import ChainState, GibbsConfig, _Recorder


class TemperedModel:
    """Cluster-marginal model with marginals multiplied by beta (-inf kept)."""

    def __init__(self, base, beta: float):
        self.base, self.beta = base, float(beta)
        self._S1 = None

    def _t(self, v):
        v = np.asarray(v, float)
        return np.where(np.isfinite(v), self.beta * v, v)

    def marginal(self, e, members):
        v = self.base.marginal(e, members)
        return v if not np.isfinite(v) else self.beta * v

    def marginal_all(self, members):
        return self._t(self.base.marginal_all(members))

    def singletons(self):
        if self._S1 is None:
            self._S1 = self._t(self.base.singletons())
        return self._S1


def _loglik(st: ChainState, base):
    return float(sum(base.marginal(e, sorted(S)) for e, S in st.members.items()))


def _new_schedule(betas, rej):
    """Betas equalising the cumulative rejection (communication barrier)."""
    lam = np.r_[0.0, np.cumsum(rej)]
    if lam[-1] <= 0:
        return betas, 0.0
    targets = np.linspace(0, lam[-1], len(betas))
    nb = np.interp(targets, lam, betas)
    nb[0], nb[-1] = 0.0, 1.0
    nb = np.maximum.accumulate(nb)
    return nb, float(lam[-1])


def run_pt_chain(L, counts, grammar, cfg: GibbsConfig, seed: int, init="argmax", model=None, z_init=None):
    """One parallel-tempering run; returns the recorder dict of hsr.gibbs.run_chain
    plus 'pt' diagnostics."""
    from .pooled import IndependentModel
    base = IndependentModel(L) if model is None else model
    K = max(int(cfg.n_temps), 2)
    betas = np.linspace(0, 1, K) ** 3
    states = [ChainState(L, counts, grammar, cfg, seed * 7919 + k, init if k == K - 1 else "random",
                         z_init=z_init, model=TemperedModel(base, b)) for k, b in enumerate(betas)]
    # replica identities, for round-trip counting (a trip: beta=0 -> beta=1 -> beta=0)
    ident = list(range(K))
    leg = {i: None for i in range(K)}          # None, "up" (left 0, heading to 1), "down"
    trips = 0
    it = 0

    def communicate(rej_acc):
        nonlocal it, trips
        ll = [_loglik(st, base) for st in states]
        for k in range(it % 2, K - 1, 2):
            log_a = (betas[k + 1] - betas[k]) * (ll[k] - ll[k + 1])
            a = 1.0 if log_a >= 0 else float(np.exp(log_a))
            if rej_acc is not None:
                rej_acc[k] += 1.0 - a
            if states[k].rng.random() < a:
                states[k].swap_state(states[k + 1])
                ll[k], ll[k + 1] = ll[k + 1], ll[k]
                ident[k], ident[k + 1] = ident[k + 1], ident[k]
        for pos, rid in ((0, ident[0]), (K - 1, ident[K - 1])):
            if pos == 0:
                if leg[rid] == "down":
                    trips += 1
                leg[rid] = "up"
            elif leg[rid] == "up":
                leg[rid] = "down"
        it += 1
        return ll

    # ---- tuning rounds
    barrier = None
    for r in range(cfg.pt_tune_rounds):
        n = 2 ** r
        rej = np.zeros(K - 1)
        cnt = np.zeros(K - 1)
        for _ in range(n):
            for st in states:
                st.sweep()
            before = it
            communicate(rej)
            cnt[np.arange(before % 2, K - 1, 2)] += 1
        rate = np.where(cnt > 0, rej / np.maximum(cnt, 1), np.nan)
        rate = np.where(np.isfinite(rate), rate, np.nanmean(rate) if np.isfinite(rate).any() else 0.0)
        new, barrier = _new_schedule(betas, rate)
        if not np.allclose(new, betas):
            betas = new
            for st, b in zip(states, betas):
                st.set_model(TemperedModel(base, b))
    # ---- sampling
    rec = _Recorder(states[-1].J, states[-1].C, grammar.nonterminals)
    rej = np.zeros(K - 1)
    cnt = np.zeros(K - 1)
    trips_start = trips
    for sweep in range(cfg.n_sweeps):
        for st in states:
            st.sweep()
        before = it
        ll = communicate(rej)
        cnt[np.arange(before % 2, K - 1, 2)] += 1
        if sweep >= cfg.burn_in and (sweep - cfg.burn_in) % cfg.thin == 0:
            rec.record(states[-1], ll[-1])
    out = rec.result()
    out["pt"] = {"betas": betas.tolist(), "barrier": barrier,
                 "swap_rejection": (rej / np.maximum(cnt, 1)).tolist(),
                 "round_trips": trips - trips_start}
    return out
